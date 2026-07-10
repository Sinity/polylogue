"""Consistent acquisition of live SQLite databases."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.blob_store import BlobStore, Heartbeat

_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")


@dataclass(frozen=True, slots=True)
class SQLiteBlobSnapshot:
    """One immutable SQLite backup stored in the content-addressed blob store."""

    blob_hash: str
    blob_size: int
    source_revision: str


def is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in _SQLITE_SUFFIXES


def sqlite_database_for_sidecar(path: Path) -> Path | None:
    """Map a SQLite WAL/SHM path back to its main database path."""
    lowered = path.name.lower()
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        if not lowered.endswith(suffix):
            continue
        database = path.with_name(path.name[: -len(suffix)])
        return database if is_sqlite_path(database) else None
    return None


def sqlite_source_revision(path: Path) -> str:
    """Fingerprint main/WAL filesystem state without reading mutable DB bytes."""
    hasher = hashlib.sha256()
    for candidate in (path, path.with_name(f"{path.name}-wal")):
        hasher.update(candidate.name.encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\0")
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            hasher.update(b"missing")
        else:
            hasher.update(f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create a consistent standalone backup without writing to the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_conn, sqlite3.connect(destination) as destination_conn:
        source_conn.backup(destination_conn)


def snapshot_sqlite_to_blob(
    source: Path,
    blob_store: BlobStore,
    *,
    heartbeat: Heartbeat | None = None,
) -> SQLiteBlobSnapshot:
    """Back up *source*, then hash/store only those consistent snapshot bytes."""
    source_revision = sqlite_source_revision(source)
    blob_store.root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=blob_store.root,
        prefix=".sqlite-snapshot.",
        suffix=source.suffix or ".db",
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        snapshot_sqlite_database(source, temporary_path)
        blob_hash, blob_size = blob_store.write_from_path(temporary_path, heartbeat=heartbeat)
        return SQLiteBlobSnapshot(
            blob_hash=blob_hash,
            blob_size=blob_size,
            source_revision=source_revision,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "SQLiteBlobSnapshot",
    "is_sqlite_path",
    "snapshot_sqlite_database",
    "snapshot_sqlite_to_blob",
    "sqlite_database_for_sidecar",
    "sqlite_source_revision",
]
