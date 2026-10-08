"""Consistent acquisition of live SQLite databases."""

from __future__ import annotations

import errno
import hashlib
import os
import sqlite3
import stat
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from polylogue.core.binary_signatures import SQLITE_MAGIC_HEADER
from polylogue.core.binary_signatures import looks_like_sqlite_bytes as _looks_like_sqlite_bytes
from polylogue.core.provider_identity import captured_hermes_profile_key
from polylogue.sources.source_staging import SourceInputBinding, bind_source_input
from polylogue.sources.sqlite_export import (
    MemberExportScope,
    _write_logical_export_bound,
    logical_export_digest,
    looks_like_logical_export_path,
    read_export_header,
)
from polylogue.storage.blob_store import BlobStore, Heartbeat

if TYPE_CHECKING:
    from polylogue.sources.origin_specs import DatabaseMemberBinding

_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_HERMES_RAW_ID_DOMAIN = b"polylogue:hermes-profile-raw:v3\0"
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
    source_path: Path
    identity_path: Path
    captured_profile_key: str
    captured_profile_root: Path
    captured_profile_source_path: Path
    blob_publication_receipt_id: str | None = None


class _SQLiteSnapshotFailureAsOSError:
    """Adapt SQLite acquisition failures to the live-file failure contract.

    The live batch already treats ``OSError`` as a per-file acquisition
    failure.  Keeping this adapter as an exception context manager lets that
    route account for SQLite corruption/lock errors without widening every
    batch catch site, while direct callers still receive the original typed
    ``sqlite3.Error`` from :func:`snapshot_sqlite_to_blob`.
    """

    def __enter__(self) -> _SQLiteSnapshotFailureAsOSError:
        return self

    def __exit__(self, _exc_type: object, exc: BaseException | None, _traceback: object) -> Literal[False]:
        if isinstance(exc, sqlite3.Error):
            raise OSError(f"SQLite snapshot acquisition failed: {exc}") from exc
        return False


def sqlite_snapshot_failure_as_oserror() -> _SQLiteSnapshotFailureAsOSError:
    """Translate a snapshot's SQLite error for per-file live acquisition."""
    return _SQLiteSnapshotFailureAsOSError()


def hermes_profile_raw_id(
    source_path: Path | str,
    source_index: int,
    logical_revision: str,
    *,
    identity_path: Path,
    profile_identity: str,
) -> str:
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
    from polylogue.core.provider_identity import profile_root_for_artifact

    normalized = identity_path
    if captured_hermes_profile_key(profile_root_for_artifact(normalized)) != profile_identity:
        raise ValueError("Hermes raw identity requires its matching captured profile namespace")
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


def codex_state_raw_id(source_path: Path | str, logical_revision: str, *, identity_path: Path | None = None) -> str:
    """Identify one acquired Codex state-db snapshot by path and logical content.

    Codex keeps exactly one instance of each declared database per ``~/.codex``
    install, so the stable absolute source path distinguishes the members and
    no profile index is needed. The content term is
    :func:`sqlite_logical_revision`, for the reason given on
    :func:`hermes_profile_raw_id`.
    """
    normalized_path = str(
        identity_path if identity_path is not None else Path(source_path).expanduser().resolve(strict=False)
    )
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


def sqlite_source_revision(path: Path, *, source_binding: SourceInputBinding | None = None) -> str:
    """Fingerprint the accepted physical main/WAL under its anchored parent."""
    if source_binding is None:
        with bind_source_input(path) as binding:
            return sqlite_source_revision(path, source_binding=binding)
    hasher = hashlib.sha256()
    for suffix in ("", "-wal"):
        name = source_binding.physical_path.name + suffix
        hasher.update((source_binding.source_path.name + suffix).encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\0")
        try:
            info = os.stat(name, dir_fd=source_binding.parent_anchor, follow_symlinks=False)
        except FileNotFoundError:
            if not suffix:
                raise
            hasher.update(b"missing")
        else:
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.ELOOP, "SQLite source observation requires a regular file", name)
            if not suffix and (info.st_dev, info.st_ino) != source_binding.main_identity:
                raise OSError(errno.ESTALE, "SQLite main identity changed", name)
            hasher.update(f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}".encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def declared_database_member(path: Path) -> DatabaseMemberBinding | None:
    """Return the declared member rule acquisition applies to *path*.

    This coordinate is already accepted acquisition evidence. Live staged
    inputs resolve their provenance through ``bind_source_input`` first;
    retained evidence never consults a mutable filesystem sidecar.
    """
    from polylogue.sources.origin_specs import database_member_for_filename

    return database_member_for_filename(path.name)


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


def sqlite_member_revision(
    path: Path, *, immutable: bool = False, source_binding: SourceInputBinding | None = None
) -> str:
    """Digest the logical revision acquisition records for *path*.

    Acquisition retains one export per declared member, so the freshness gate
    and the raw identity must both be scoped to that member's logical tables.
    A whole-database digest would move for a commit in a table nothing reads.
    """
    return sqlite_member_revision_and_size(path, immutable=immutable, source_binding=source_binding)[0]


def sqlite_member_revision_and_size(
    path: Path, *, immutable: bool = False, source_binding: SourceInputBinding | None = None
) -> tuple[str, int]:
    """Return the retained logical export's revision and byte length together."""
    if source_binding is None:
        with bind_source_input(path) as binding:
            return sqlite_member_revision_and_size(path, immutable=immutable, source_binding=binding)
    from polylogue.sources.sqlite_export import _HashingSink

    sink = _HashingSink()
    _write_logical_export_bound(
        path,
        sink,
        scope=member_export_scope(source_binding.source_path),
        immutable=immutable,
        source_binding=source_binding,
        parent_anchor=source_binding.parent_anchor,
    )
    return sink.hexdigest(), sink.byte_count


def is_sqlite_page_image(blob_path: Path) -> bool:
    """Return whether *blob_path* is a raw SQLite page image, not an export.

    Pre-logical-export acquisition retained the database file itself. A page
    image cannot be proven against the live database it was copied from and
    re-snapshots on every commit, so it is never replay authority -- but it is
    deliberately retained material that a rebuild must still account for.
    """
    try:
        with blob_path.open("rb") as handle:
            prefix = handle.read(len(SQLITE_MAGIC_HEADER))
    except OSError:
        return False
    return prefix.startswith(SQLITE_MAGIC_HEADER)


def is_undeclared_logical_export(blob_path: Path, source_path: Path | str) -> bool:
    """Return whether *blob_path* is a well-framed export with no declared member.

    A database acquired under a noncanonical filename (``backup.db``) resolves
    to no :func:`database_member_for_filename` binding, so
    :func:`is_declared_logical_export` is false even though the retained bytes
    are a real logical export. Such material is replayable only with the
    unbound full-scope header used at acquisition; a declared member export
    cannot be relabeled under a noncanonical source filename.
    """
    if declared_database_member(Path(source_path)) is not None:
        return False
    if not looks_like_logical_export_path(blob_path):
        return False
    try:
        header = read_export_header(blob_path)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return header.member is None and header.origin is None and header.kind is None and not header.missing


def is_declared_logical_export(blob_path: Path, source_path: Path | str) -> bool:
    """Return whether *blob_path* is the canonical export for *source_path*.

    A SQLite page image is neither an acquisition product nor a replay input
    for a declared mutable member. The header binds the retained bytes to the
    member declaration, so an export from a sibling database cannot be parsed
    under a copied filename either.
    """
    binding = declared_database_member(Path(source_path))
    if binding is None or not looks_like_logical_export_path(blob_path):
        return False
    try:
        header = read_export_header(blob_path)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    member = binding.member
    expected_tables = tuple(member.logical_tables)
    return (
        header.member == member.filename
        and header.origin == binding.origin.value
        and header.kind == member.kind
        and tuple(sorted((*header.tables, *header.missing))) == tuple(sorted(expected_tables))
    )


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create a consistent standalone backup without writing to the source."""
    _snapshot_sqlite_database_bound(source, destination)


def _snapshot_sqlite_database_bound(
    source: Path,
    destination: Path,
    *,
    source_binding: SourceInputBinding | None = None,
    expected_identity: tuple[int, int] | None = None,
    heartbeat: Heartbeat | None = None,
) -> dict[str, Any]:
    """Carry the actual backup owner's accepted coordinate to staging provenance."""
    from polylogue.sources.sqlite_export import _backup_source_database

    with ExitStack() as owners:
        if source_binding is None:
            source_binding = owners.enter_context(bind_source_input(source))
        if source_binding.physical_path == destination.resolve():
            raise ValueError("a SQLite backup cannot replace its source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        try:
            return _backup_source_database(
                source,
                destination,
                source_binding=source_binding,
                expected_identity=expected_identity,
                heartbeat=heartbeat,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def snapshot_sqlite_to_blob(
    source: Path,
    blob_store: BlobStore,
    *,
    heartbeat: Heartbeat | None = None,
    source_binding: SourceInputBinding | None = None,
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
    # Capture the mutable filesystem observation *before* opening the export
    # transaction.  A WAL writer can commit after the logical export finishes
    # but before the old implementation sampled this token.  Recording that
    # post-export token alongside the older retained bytes lets the cursor
    # claim that it already observed the newer commit, and the watcher then
    # silently skips it.  The pre-export token is conservative: a commit that
    # races the export remains dirty on the next pass, while a commit that was
    # already present is either included in the export or causes one harmless
    # extra acquisition when the token was sampled just before it.
    if source_binding is None:
        with bind_source_input(source) as binding:
            return snapshot_sqlite_to_blob(source, blob_store, heartbeat=heartbeat, source_binding=binding)
    source_fingerprint = sqlite_source_revision(source, source_binding=source_binding)
    # The export is written straight into the blob store's own staging file:
    # exporting to a work file and copying it would hold two full-size copies
    # on the blob filesystem at once, which capacity preflight never budgets.
    blob_hash, blob_size = blob_store.write_from_writer(
        lambda handle: _write_logical_export_bound(
            source,
            handle,
            scope=member_export_scope(source_binding.source_path),
            source_binding=source_binding,
            parent_anchor=source_binding.parent_anchor,
        ),
        heartbeat=heartbeat,
    )
    from polylogue.storage.blob_publication import publication_receipt_id

    return SQLiteBlobSnapshot(
        blob_hash=blob_hash,
        blob_size=blob_size,
        source_revision=blob_hash,
        source_fingerprint=source_fingerprint,
        source_path=source_binding.source_path,
        identity_path=source_binding.identity_path,
        captured_profile_key=source_binding.captured_profile_key,
        captured_profile_root=source_binding.captured_profile_root,
        captured_profile_source_path=source_binding.captured_profile_source_path,
        blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
    )


__all__ = [
    "SQLiteBlobSnapshot",
    "codex_state_raw_id",
    "declared_database_member",
    "declared_logical_tables",
    "member_export_scope",
    "hermes_profile_raw_id",
    "is_declared_logical_export",
    "is_sqlite_page_image",
    "is_undeclared_logical_export",
    "is_sqlite_path",
    "snapshot_sqlite_database",
    "snapshot_sqlite_to_blob",
    "sqlite_snapshot_failure_as_oserror",
    "sqlite_database_for_sidecar",
    "sqlite_logical_revision",
    "sqlite_member_revision",
    "sqlite_source_revision",
]
