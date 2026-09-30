"""Independent-process probes of SQLite's actual POSIX file locks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROBE = """
import fcntl, json, os, sys
result = {}
for name, suffix, offset, length in (("main", "", 1073741826, 510), ("shm", "-shm", 128, 1)):
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


def sqlite_lock_state(path: Path) -> dict[str, str]:
    """A separate process must observe the kernel locks, not its own lock table."""
    return json.loads(subprocess.check_output([sys.executable, "-c", _PROBE, str(path)], text=True))
