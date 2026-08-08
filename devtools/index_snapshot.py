"""Shared selected-index file-set observation for evidence reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SNAPSHOT_HASH_CHUNK_BYTES = 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_SNAPSHOT_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_index_file_set(index_db: Path) -> dict[str, Any]:
    """Capture one selected index and its SQLite sidecars under one contract.

    Each present file is hashed between two metadata reads. A disappearing or
    changing file makes ``observation_complete`` false, while the file-set
    digest remains useful evidence for the caller's stability comparison.
    """
    paths = (index_db, Path(f"{index_db}-wal"), Path(f"{index_db}-shm"), Path(f"{index_db}-journal"))
    files: list[dict[str, Any]] = []
    complete = True
    for path in paths:
        try:
            metadata_before = path.stat()
        except FileNotFoundError:
            if path == index_db:
                # A missing sidecar is a normal quiescent SQLite state; a
                # missing selected database is not evidence of an observed
                # snapshot, even when an already-open connection can still
                # serve reads from the unlinked inode.
                complete = False
            files.append({"path": str(path), "present": False})
            continue
        try:
            digest = _file_sha256(path)
            metadata_after = path.stat()
        except FileNotFoundError:
            complete = False
            files.append({"path": str(path), "present": False, "changed_during_observation": True})
            continue
        unchanged = (
            metadata_before.st_dev,
            metadata_before.st_ino,
            metadata_before.st_size,
            metadata_before.st_mtime_ns,
        ) == (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_size,
            metadata_after.st_mtime_ns,
        )
        complete = complete and unchanged
        files.append(
            {
                "path": str(path),
                "present": True,
                "size": metadata_after.st_size,
                "mtime_ns": metadata_after.st_mtime_ns,
                "inode": metadata_after.st_ino,
                "sha256": digest,
                "changed_during_observation": not unchanged,
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    main = files[0]
    return {
        "path": str(index_db),
        "index_db": str(index_db),
        "present": main["present"],
        "size": main.get("size"),
        "files": files,
        "observation_complete": complete,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
