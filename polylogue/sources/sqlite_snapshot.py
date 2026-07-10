"""Consistent acquisition of live SQLite databases."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.blob_store import BlobStore, Heartbeat

_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")
_STAGING_METADATA_SUFFIX = ".polylogue-import"
_STAGING_METADATA_VERSION = 1
_HERMES_RAW_ID_DOMAIN = b"polylogue:hermes-profile-raw:v1\0"


@dataclass(frozen=True, slots=True)
class SQLiteBlobSnapshot:
    """One immutable SQLite backup stored in the content-addressed blob store."""

    blob_hash: str
    blob_size: int
    source_revision: str
    blob_publication_receipt_id: str | None = None


def hermes_profile_raw_id(source_path: Path | str, source_index: int, blob_hash: str) -> str:
    """Identify one Hermes snapshot without conflating it with its blob.

    Hermes session IDs are only unique within a profile.  Raw acquisition
    identity therefore includes the stable original profile path, while
    ``blob_hash`` continues to address the exact retained SQLite bytes.
    """
    normalized_profile = str(Path(source_path).expanduser().resolve(strict=False).parent)
    digest = hashlib.sha256()
    digest.update(_HERMES_RAW_ID_DOMAIN)
    digest.update(normalized_profile.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(str(source_index).encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(blob_hash))
    return digest.hexdigest()


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


def sqlite_staging_metadata_path(staged_path: Path) -> Path:
    """Return the non-ingestible provenance sidecar for a staged database."""
    return staged_path.with_name(f"{staged_path.name}{_STAGING_METADATA_SUFFIX}")


def original_sqlite_source_path(staged_path: Path) -> Path | None:
    """Read the original source path recorded for a staged SQLite snapshot."""
    metadata_path = sqlite_staging_metadata_path(staged_path)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _STAGING_METADATA_VERSION:
        return None
    original = payload.get("original_source_path")
    if not isinstance(original, str) or not original:
        return None
    return Path(original)


def stage_sqlite_snapshot(source: Path, destination: Path) -> None:
    """Atomically publish a snapshot and its original-path provenance."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".staging",
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    metadata_path = sqlite_staging_metadata_path(destination)
    metadata_temporary_path = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        snapshot_sqlite_database(source, temporary_path)
        metadata_temporary_path.write_text(
            json.dumps(
                {
                    "version": _STAGING_METADATA_VERSION,
                    "original_source_path": str(source.expanduser().resolve()),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.chmod(metadata_temporary_path, 0o600)
        os.replace(metadata_temporary_path, metadata_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
        metadata_temporary_path.unlink(missing_ok=True)


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
        from polylogue.storage.blob_publication import publication_receipt_id

        return SQLiteBlobSnapshot(
            blob_hash=blob_hash,
            blob_size=blob_size,
            source_revision=source_revision,
            blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
        )
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "SQLiteBlobSnapshot",
    "hermes_profile_raw_id",
    "is_sqlite_path",
    "original_sqlite_source_path",
    "snapshot_sqlite_database",
    "snapshot_sqlite_to_blob",
    "sqlite_staging_metadata_path",
    "stage_sqlite_snapshot",
    "sqlite_database_for_sidecar",
    "sqlite_source_revision",
]
