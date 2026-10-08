"""Independent-process probes of SQLite's actual POSIX file locks."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

_PROBE = """
import fcntl, json, os, sys
result = {}
locks = [("main", "", 1073741826, 510), ("shm", "-shm", 128, 1)]
if sys.argv[2] == "all":
    locks.extend((("wal", "-wal", 0, 1), ("journal", "-journal", 0, 1)))
for name, suffix, offset, length in locks:
    descriptor = os.open(sys.argv[1] + suffix, os.O_RDWR)
    try:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB, length, offset)
        except BlockingIOError:
            result[name] = "protected"
        else:
            result[name] = "available"
    finally:
        os.close(descriptor)
print(json.dumps(result))
"""

_DIRECTORY_PROBE = """
import fcntl, os, sys
descriptor = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)
try:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("protected")
    else:
        print("available")
finally:
    os.close(descriptor)
"""


def sqlite_lock_state(path: Path, *, include_wal_journal: bool = False) -> dict[str, str]:
    """A separate process must observe the kernel locks, not its own lock table."""
    result = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", _PROBE, str(path), "all" if include_wal_journal else "sqlite"], text=True
        )
    )
    assert isinstance(result, dict)
    return {str(key): str(value) for key, value in result.items()}


def directory_flock_state(path: Path) -> str:
    """Probe the actual directory flock from an independent process."""
    return subprocess.check_output([sys.executable, "-c", _DIRECTORY_PROBE, str(path)], text=True).strip()


@contextmanager
def retained_sqlite_namespace_locks(path: Path) -> Iterator[sqlite3.Connection]:
    """Retain native main/SHM locks and POSIX sentinels on WAL/journal.

    SQLite's Unix VFS locks main and SHM, not WAL/journal. Those two sentinels
    exercise the same kernel close law on each other SQLite namespace inode.
    They do not stand in for an invented SQLite lock range.
    """
    with ExitStack() as stack:
        connection = stack.enter_context(closing(sqlite3.connect(path)))
        with closing(connection.execute("PRAGMA journal_mode=WAL")) as rows:
            assert rows.fetchone() == ("wal",)
        with closing(connection.execute("CREATE TABLE evidence(value TEXT)")):
            pass
        with closing(connection.execute("INSERT INTO evidence VALUES ('neutral')")):
            pass
        connection.commit()
        with closing(connection.execute("BEGIN")):
            pass
        with closing(connection.execute("SELECT value FROM evidence")) as rows:
            assert rows.fetchone() == ("neutral",)
        for suffix in ("-wal", "-journal"):
            flags = os.O_RDWR | os.O_NOFOLLOW
            if suffix == "-journal":
                flags |= os.O_CREAT | os.O_EXCL
            descriptor = os.open(str(path) + suffix, flags, 0o600)
            stack.callback(os.close, descriptor)
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0)
        yield connection
