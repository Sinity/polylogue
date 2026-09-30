"""Physical SQLite reads whose descriptor custody preserves caller locks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.file_identity import SQLiteFileIdentity, open_sqlite_identity, require_sqlite_identity


@dataclass(frozen=True, slots=True)
class SQLiteFileRead:
    sha256: str
    size_bytes: int
    metadata: os.stat_result


def read_sqlite_file_in_lock_isolated_process(
    path: Path,
    *,
    copy_to: Path | None = None,
    opened_identity: SQLiteFileIdentity | None = None,
    copy_directory_fd: int | None = None,
    copy_exclusive: bool = False,
) -> SQLiteFileRead:
    """Stream hash/copy in child custody; caller owns stability and partial copy cleanup."""
    identity = open_sqlite_identity(path) if opened_identity is None else opened_identity
    try:
        require_sqlite_identity(identity)
        metadata = identity.stat()
        result = identity.physical_read(
            copy_to=copy_to, copy_exclusive=copy_exclusive, copy_directory_fd=copy_directory_fd
        )
        return SQLiteFileRead(str(result["sha256"]), int(result["size_bytes"]), metadata)
    finally:
        if opened_identity is None:
            identity.close()
