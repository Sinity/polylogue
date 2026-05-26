"""Canonical SQLite connection profiles and factory functions shared by sync and async backends.

Factories
---------
``open_connection(path)`` returns a read-write connection with write pragmas applied.
``open_readonly_connection(path)`` returns a uri=ro connection with read pragmas applied.
``connection_context(path)`` is a context manager for a single-use read-write connection.

These are lightweight one-shot wrappers around ``sqlite3.connect()``.  For the
thread-local cached connection used by the async runtime, use the factories in
``connection.py`` instead.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class SQLiteConnectionProfile:
    """SQLite timeout and PRAGMA profile for one connection role."""

    role: Literal["read", "write"]
    timeout_seconds: int
    busy_timeout_ms: int
    cache_size_kib: int
    mmap_size_bytes: int
    foreign_keys: bool = False
    journal_mode: str | None = None
    synchronous: str | None = None
    temp_store: str = "MEMORY"
    wal_autocheckpoint_pages: int | None = None

    @property
    def pragma_statements(self) -> tuple[str, ...]:
        statements: list[str] = []
        if self.foreign_keys:
            statements.append("PRAGMA foreign_keys = ON")
        if self.journal_mode is not None:
            statements.append(f"PRAGMA journal_mode={self.journal_mode}")
        statements.extend(
            (
                f"PRAGMA busy_timeout = {self.busy_timeout_ms}",
                f"PRAGMA cache_size = -{self.cache_size_kib}",
            )
        )
        if self.synchronous is not None:
            statements.append(f"PRAGMA synchronous = {self.synchronous}")
        statements.extend(
            (
                f"PRAGMA mmap_size = {self.mmap_size_bytes}",
                f"PRAGMA temp_store = {self.temp_store}",
            )
        )
        if self.wal_autocheckpoint_pages is not None:
            statements.append(f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages}")
        return tuple(statements)


DB_TIMEOUT = 30
READ_DB_TIMEOUT = 1
WRITE_CACHE_SIZE_KIB = 131072  # 128 MiB
READ_CACHE_SIZE_KIB = 32768  # 32 MiB
WRITE_MMAP_SIZE_BYTES = 1073741824  # 1 GiB
READ_MMAP_SIZE_BYTES = 134217728  # 128 MiB
WAL_AUTOCHECKPOINT_PAGES = 10000

WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
)

READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=READ_DB_TIMEOUT,
    busy_timeout_ms=READ_DB_TIMEOUT * 1000,
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
)

WRITE_CONNECTION_PRAGMA_STATEMENTS = WRITE_CONNECTION_PROFILE.pragma_statements
READ_CONNECTION_PRAGMA_STATEMENTS = READ_CONNECTION_PROFILE.pragma_statements


# ---------------------------------------------------------------------------
# Lightweight factory functions — open + apply pragmas, no caching / schema / vec
# ---------------------------------------------------------------------------


def open_connection(path: str | Path, *, timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
    """Open a read-write SQLite connection with canonical write pragmas applied.

    This is a lightweight one-shot factory: it opens the file, applies the
    write-time PRAGMA profile, and returns the connection.  The caller owns
    the connection lifecycle (must close it).

    For the thread-local cached archive connection used by the async runtime,
    use ``connection_context`` from ``connection.py`` instead.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    for stmt in WRITE_CONNECTION_PRAGMA_STATEMENTS:
        conn.execute(stmt)
    return conn


def open_readonly_connection(path: str | Path, *, timeout: float = READ_DB_TIMEOUT) -> sqlite3.Connection:
    """Open a read-only SQLite connection with canonical read pragmas applied.

    Uses ``file:...?mode=ro`` URI mode to guarantee no write locks are taken.
    Returns ``None`` / raises ``sqlite3.OperationalError`` if the database file
    does not exist.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    for stmt in READ_CONNECTION_PRAGMA_STATEMENTS:
        conn.execute(stmt)
    return conn


@contextmanager
def connection_context(path: str | Path, *, timeout: float = DB_TIMEOUT) -> Iterator[sqlite3.Connection]:
    """Context manager for a single-use read-write connection.

    Opens a connection with write pragmas, yields it, and closes on exit.
    """
    conn = open_connection(path, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


__all__ = [
    "DB_TIMEOUT",
    "READ_CACHE_SIZE_KIB",
    "READ_CONNECTION_PRAGMA_STATEMENTS",
    "READ_CONNECTION_PROFILE",
    "READ_DB_TIMEOUT",
    "READ_MMAP_SIZE_BYTES",
    "SQLiteConnectionProfile",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "WRITE_CONNECTION_PROFILE",
    "WRITE_MMAP_SIZE_BYTES",
    "connection_context",
    "open_connection",
    "open_readonly_connection",
]
