"""Consistent acquisition of live SQLite databases."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.binary_signatures import SQLITE_MAGIC_HEADER
from polylogue.core.binary_signatures import looks_like_sqlite_bytes as _looks_like_sqlite_bytes
from polylogue.logging import get_logger
from polylogue.sources.sqlite_export import (
    MemberExportScope,
    logical_export_digest,
    looks_like_logical_export_path,
    write_logical_export,
)
from polylogue.storage.blob_store import BlobStore, Heartbeat

if TYPE_CHECKING:
    from polylogue.sources.origin_specs import DatabaseMemberBinding

logger = get_logger(__name__)

_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_STAGING_METADATA_SUFFIX = ".polylogue-import"
_STAGING_METADATA_VERSION = 1
_HERMES_RAW_ID_DOMAIN = b"polylogue:hermes-profile-raw:v2\0"
_CODEX_STATE_RAW_ID_DOMAIN = b"polylogue:codex-state-raw:v2\0"
# Re-exported for existing call sites; canonical constant now lives on the
# shared, provider-agnostic detector in ``core.binary_signatures`` so it is
# never redefined in more than one place (polylogue-hbtj2).
_SQLITE_MAGIC_HEADER = SQLITE_MAGIC_HEADER


@dataclass(frozen=True, slots=True)
class SQLiteBlobSnapshot:
    """One immutable SQLite backup stored in the content-addressed blob store."""

    blob_hash: str
    blob_size: int
    source_revision: str
    source_fingerprint: str
    blob_publication_receipt_id: str | None = None


def hermes_profile_raw_id(source_path: Path | str, source_index: int, logical_revision: str) -> str:
    """Identify one Hermes snapshot by profile, member, and logical content.

    Hermes session IDs are only unique within a profile, and the two declared
    members of a profile can hold identical logical content while empty, so
    identity carries the profile directory and the member filename alongside
    :func:`sqlite_logical_revision`.

    ``logical_revision`` -- never the blob hash -- is the content term. A
    backup of an unchanged database differs in its page image after any
    commit, checkpoint or vacuum; keying identity on those bytes mints a new
    raw revision for a source that did not change.
    """
    normalized = Path(source_path).expanduser().resolve(strict=False)
    digest = hashlib.sha256()
    digest.update(_HERMES_RAW_ID_DOMAIN)
    digest.update(str(normalized.parent).encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(normalized.name.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(str(source_index).encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(logical_revision))
    return digest.hexdigest()


def codex_state_raw_id(source_path: Path | str, logical_revision: str) -> str:
    """Identify one acquired Codex state-db snapshot by path and logical content.

    Codex keeps exactly one instance of each declared database per ``~/.codex``
    install, so the stable absolute source path distinguishes the members and
    no profile index is needed. The content term is
    :func:`sqlite_logical_revision`, for the reason given on
    :func:`hermes_profile_raw_id`.
    """
    normalized_path = str(Path(source_path).expanduser().resolve(strict=False))
    digest = hashlib.sha256()
    digest.update(_CODEX_STATE_RAW_ID_DOMAIN)
    digest.update(normalized_path.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(logical_revision))
    return digest.hexdigest()


def is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in _SQLITE_SUFFIXES


def looks_like_sqlite_bytes(payload: bytes) -> bool:
    """Return whether *payload* starts with the SQLite file-format magic header.

    Path-suffix detection (``is_sqlite_path``) is useless once bytes have
    already been read into memory as a raw revision (e.g. replay/backfill
    call sites, which only have ``payload: bytes`` -- see
    ``revision_backfill.py``). This is a thin re-export of the shared,
    provider-agnostic sniffer in ``core.binary_signatures`` (kept here too
    since most call sites already import it from this module); do not
    duplicate the magic-byte constant itself.
    """
    return _looks_like_sqlite_bytes(payload)


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


def declared_database_member(path: Path) -> DatabaseMemberBinding | None:
    """Return the declared member rule acquisition applies to *path*.

    A staged import copy carries its original path in a provenance sidecar, so
    the declaration is resolved from the name the operator's install uses.
    """
    from polylogue.sources.origin_specs import database_member_for_filename

    original = original_sqlite_source_path(path)
    return database_member_for_filename((original or path).name)


def declared_logical_tables(path: Path) -> tuple[str, ...] | None:
    """Return the logical tables *path* is acquired for, or ``None`` for all.

    An undeclared database still retains an export rather than a page image:
    without a declared product every ordinary table is its logical content.
    """
    binding = declared_database_member(path)
    if binding is None or not binding.member.logical_tables:
        return None
    return binding.member.logical_tables


def member_export_scope(path: Path) -> MemberExportScope:
    """Return the export scope and header *path* is acquired under.

    Acquisition and the freshness gate must produce byte-identical exports for
    one database state, so both derive their scope here rather than each
    naming the member on its own.
    """
    binding = declared_database_member(path)
    return MemberExportScope(
        tables=declared_logical_tables(path),
        member=None if binding is None else binding.member.filename,
        origin=None if binding is None else binding.origin.value,
        kind=None if binding is None else binding.member.kind,
    )


def sqlite_logical_revision(
    path: Path,
    *,
    tables: Sequence[str] | None = None,
    immutable: bool = False,
) -> str:
    """Digest SQLite schema and logical rows, independent of page layout.

    Filesystem metadata and SQLite page images change for ordinary commits,
    checkpoints, and vacuuming, so source continuity is the digest of the
    canonical logical export (``sources/sqlite_export.py``) -- the same bytes
    acquisition retains. ``tables`` scopes the digest to a declared member's
    logical product; ``None`` digests every ordinary table.

    ``immutable`` reads a retained blob, which no writer can reach and whose
    directory need not be writable for a WAL-mode page image.
    """
    return logical_export_digest(path, tables=tables, immutable=immutable)


def sqlite_member_revision(path: Path, *, immutable: bool = False) -> str:
    """Digest the logical revision acquisition records for *path*.

    Acquisition retains one export per declared member, so the freshness gate
    and the raw identity must both be scoped to that member's logical tables.
    A whole-database digest would move for a commit in a table nothing reads.
    """
    return logical_export_digest(path, scope=member_export_scope(path), immutable=immutable)


def retained_content_revision(blob_path: Path, blob_hash: str) -> str:
    """Return the content term identifying one retained acquisition.

    Live acquisition of a mutable database identifies it by
    :func:`sqlite_logical_revision`, so the import and replay routes must
    derive the same term from the retained blob or the two routes mint
    different raw identities for one database state. Material that is not a
    SQLite database is already identified by its bytes, and its blob hash is
    that term.

    A retained export is already the canonical form of its member's logical
    revision, so its blob hash -- sha256 over exactly those bytes -- is that
    term with nothing to recompute.

    An unreadable or damaged blob falls back to the blob hash: identity must
    stay derivable so the raw remains addressable and its parse failure is
    reported as a parse failure rather than as a missing acquisition.
    """
    if looks_like_logical_export_path(blob_path):
        return blob_hash
    try:
        with blob_path.open("rb") as handle:
            header = handle.read(len(SQLITE_MAGIC_HEADER))
    except OSError:
        logger.warning(
            "sqlite_snapshot: retained blob %s is unreadable; identifying the acquisition by its blob hash",
            blob_hash,
        )
        return blob_hash
    if header != SQLITE_MAGIC_HEADER:
        return blob_hash
    try:
        return sqlite_logical_revision(blob_path, immutable=True)
    except (sqlite3.Error, OSError, UnicodeDecodeError):
        return blob_hash


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create a consistent standalone backup without writing to the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source_conn,
        closing(sqlite3.connect(destination)) as destination_conn,
    ):
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
    """Retain *source* as one canonical logical export and return its revision.

    The retained material is the export, never a page image: a page image
    differs after every commit, checkpoint and vacuum, so an unchanged
    database would mint a second whole copy of content the archive already
    holds, and no reader could prove those bytes against the live database.

    The export is taken inside one SQLite read transaction, so a concurrent
    commit cannot make it a combination of two source states, and its blob
    hash -- sha256 over exactly the exported bytes -- is the member's logical
    revision with nothing to recompute.
    """
    temporary_path = blob_store.allocate_staging_path(prefix=".sqlite-export.", suffix=".jsonl")
    try:
        with temporary_path.open("wb") as handle:
            write_logical_export(source, handle, scope=member_export_scope(source))
        source_fingerprint = sqlite_source_revision(source)
        blob_hash, blob_size = blob_store.write_from_path(temporary_path, heartbeat=heartbeat)
        from polylogue.storage.blob_publication import publication_receipt_id

        return SQLiteBlobSnapshot(
            blob_hash=blob_hash,
            blob_size=blob_size,
            source_revision=blob_hash,
            source_fingerprint=source_fingerprint,
            blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
        )
    finally:
        blob_store.discard_staging_path(
            temporary_path,
            companion_suffixes=_SQLITE_SIDECAR_SUFFIXES,
        )


__all__ = [
    "SQLiteBlobSnapshot",
    "codex_state_raw_id",
    "declared_database_member",
    "declared_logical_tables",
    "member_export_scope",
    "hermes_profile_raw_id",
    "is_sqlite_path",
    "original_sqlite_source_path",
    "retained_content_revision",
    "snapshot_sqlite_database",
    "snapshot_sqlite_to_blob",
    "sqlite_staging_metadata_path",
    "stage_sqlite_snapshot",
    "sqlite_database_for_sidecar",
    "sqlite_logical_revision",
    "sqlite_member_revision",
    "sqlite_source_revision",
]
