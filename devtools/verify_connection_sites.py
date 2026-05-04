"""Verify no bare ``sqlite3.connect()`` calls remain in production code.

Bare ``sqlite3.connect()`` calls miss the canonical connection-profile
PRAGMA settings (busy_timeout, cache_size, mmap_size, WAL, etc.) and
must be replaced with ``open_connection()`` or ``open_readonly_connection()``
from ``polylogue.storage.sqlite.connection_profile``.

The following files are exempt:
- ``connection_profile.py`` — contains the factory functions themselves
- ``connection.py`` — contains the thread-local cached factories
- All files under ``tests/``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_POLYLOGUE_DIR = _REPO_ROOT / "polylogue"

_EXCLUDED_FILES = {
    "polylogue/storage/sqlite/connection_profile.py",
    "polylogue/storage/sqlite/connection.py",
}


def main() -> int:
    """Run the check and return 0 if clean, 1 if violations found."""
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            r"sqlite3\.connect(",
            str(_POLYLOGUE_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        print(f"grep failed: {result.stderr}", file=sys.stderr)
        return 2

    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    violations: list[str] = []
    for line in lines:
        # Format: polylogue/path/file.py:NN: code...
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path = parts[0].replace(str(_REPO_ROOT) + "/", "")
        # Normalize for matching
        rel = str(Path(file_path))
        if rel in _EXCLUDED_FILES or "/tests/" in rel:
            continue
        violations.append(line)

    if violations:
        print("Bare sqlite3.connect() calls found in production code:", file=sys.stderr)
        print(file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            f"Count: {len(violations)}. Route through connection_profile.py factories instead.",
            file=sys.stderr,
        )
        return 1

    print("OK: No bare sqlite3.connect() calls in production code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
