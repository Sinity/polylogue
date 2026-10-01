"""Consistent acquisition of live SQLite databases."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from polylogue.core.binary_signatures import SQLITE_MAGIC_HEADER
from polylogue.core.binary_signatures import looks_like_sqlite_bytes as _looks_like_sqlite_bytes
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
_STAGING_METADATA_SUFFIX = ".polylogue-import"
_STAGING_METADATA_VERSION = 2
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


@dataclass(frozen=True, slots=True)
class SQLiteSourceBinding:
    """Declaration and provenance for the exact main inode the reader must open."""

    source: Path
    physical_path: Path
    source_path: Path
    identity_path: Path
    captured_profile_key: str
    captured_profile_root: Path
    captured_profile_source_path: Path
    parent_anchor: int
    metadata_anchor: int
    main_identity: tuple[int, int]
    provenance: dict[str, Any]
    staged: bool


def _staging_provenance(
    directory: int, name: str, main: tuple[int, int]
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    from polylogue.sources.sqlite_export import _identity, _named_identity

    metadata_name = name + _STAGING_METADATA_SUFFIX
    try:
        descriptor = os.open(metadata_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return None, {"name": metadata_name, "identity": None}
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise OSError(errno.ESTALE, "SQLite staging provenance is not a regular file", metadata_name)
        payload_bytes = handle.read()
        after = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino, before.st_ctime_ns, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_ctime_ns,
            after.st_size,
        ) or _named_identity(directory, metadata_name) != _identity(before):
            raise OSError(errno.ESTALE, "SQLite staging provenance changed", metadata_name)
    try:
        payload = json.loads(payload_bytes)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _STAGING_METADATA_VERSION
            or not isinstance(payload.get("original_source_path"), str)
            or not payload["original_source_path"]
            or not Path(payload["original_source_path"]).is_absolute()
            or type(payload.get("version")) is not int
            or not isinstance(payload.get("database_identity"), list)
            or any(type(value) is not int for value in payload["database_identity"])
            or payload.get("database_identity") != list(main)
            or not isinstance(payload.get("declared_source_path"), str)
            or not Path(payload["declared_source_path"]).is_absolute()
            or not isinstance(payload.get("profile_root"), str)
            or not Path(payload["profile_root"]).is_absolute()
            or not isinstance(payload.get("profile_key"), str)
            or len(payload["profile_key"]) != 12
            or any(character not in "0123456789abcdef" for character in payload["profile_key"])
            or not isinstance(payload.get("profile_source_path"), str)
            or not Path(payload["profile_source_path"]).is_absolute()
        ):
            raise ValueError("invalid or mismatched staging provenance")
    except (UnicodeDecodeError, ValueError) as exc:
        raise OSError(errno.ESTALE, "invalid SQLite staging provenance", metadata_name) from exc
    return {
        "source_path": payload["declared_source_path"],
        "identity_path": payload["original_source_path"],
        "profile_root": payload["profile_root"],
        "profile_key": payload["profile_key"],
        "profile_source_path": payload["profile_source_path"],
    }, {
        "name": metadata_name,
        "identity": list(_identity(before)),
        "digest": hashlib.sha256(payload_bytes).hexdigest(),
        "observation": [before.st_ctime_ns, before.st_mtime_ns, before.st_size],
    }


def _verify_staging_provenance(directory: int, expected: dict[str, Any], main: tuple[int, int]) -> None:
    _, current = _staging_provenance(directory, expected["name"][: -len(_STAGING_METADATA_SUFFIX)], main)
    if current != expected:
        raise OSError(errno.ESTALE, "SQLite staging provenance changed", expected["name"])


def _verify_staging_metadata_name(directory: int, expected: dict[str, Any]) -> None:
    """Parent metadata-only proof; an ordinary FD close could release SQLite locks."""
    from polylogue.sources.sqlite_export import _identity

    identity = expected.get("identity")
    absent = identity is None
    if (
        set(expected) != ({"name", "identity"} if absent else {"name", "identity", "digest", "observation"})
        or not isinstance(expected.get("name"), str)
        or (
            not absent
            and (
                not isinstance(identity, list)
                or len(identity) != 3
                or any(type(value) is not int for value in identity)
                or not isinstance(expected.get("observation"), list)
                or len(expected["observation"]) != 3
                or any(type(value) is not int for value in expected["observation"])
                or not isinstance(expected.get("digest"), str)
                or len(expected["digest"]) != 64
            )
        )
    ):
        raise OSError(errno.EPROTO, "invalid SQLite staging metadata proof")
    try:
        current = os.stat(expected["name"], dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        if expected["identity"] is None:
            return
        raise
    if (
        expected["identity"] is None
        or list(_identity(current)) != expected["identity"]
        or [current.st_ctime_ns, current.st_mtime_ns, current.st_size] != expected["observation"]
    ):
        raise OSError(errno.ESTALE, "SQLite staging metadata changed", expected["name"])


@contextmanager
def bind_sqlite_source(
    path: Path, *, parent_anchor: int | None = None, semantic_parent: Path | None = None
) -> Iterator[SQLiteSourceBinding]:
    """Capture provenance once without opening an ordinary parent database FD."""
    from polylogue.sources.sqlite_export import _exchange_source_worker, _identity, _named_identity

    path = path.absolute()
    physical_path = path.resolve(strict=True) if parent_anchor is None else path
    if parent_anchor is not None and semantic_parent is None:
        raise OSError(errno.EINVAL, "anchored SQLite source requires its captured semantic parent", str(path))
    declared_parent = path.parent.resolve(strict=True) if parent_anchor is None else semantic_parent
    assert declared_parent is not None
    semantic_path = declared_parent / path.name
    parent = physical_path.parent
    descriptor = (
        os.open(parent, getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW)
        if parent_anchor is None
        else os.dup(parent_anchor)
    )
    try:
        metadata_descriptor = (
            os.open(
                declared_parent,
                getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            if parent_anchor is None
            else os.dup(parent_anchor)
        )
    except BaseException:
        os.close(descriptor)
        raise
    try:
        if _identity(os.fstat(descriptor)) != _identity(physical_path.parent.stat()):
            raise OSError(errno.ESTALE, "SQLite source parent changed", str(path))
        if _identity(os.fstat(metadata_descriptor)) != _identity(path.parent.stat()):
            raise OSError(errno.ESTALE, "SQLite declared parent changed", str(path))
        main = _named_identity(descriptor, physical_path.name)
        result = _exchange_source_worker(
            {
                "operation": "binding",
                "source": str(physical_path),
                "directory": descriptor,
                "metadata_directory": metadata_descriptor,
                "metadata_name": path.name,
                "semantic_source": str(semantic_path),
                "identities": {"": main},
            }
        )
        if (
            set(result) != {"source_path", "provenance", "staged", "profile"}
            or not isinstance(result["source_path"], str)
            or not Path(result["source_path"]).is_absolute()
            or not isinstance(result["provenance"], dict)
            or type(result["staged"]) is not bool
        ):
            raise OSError(errno.EPROTO, "invalid SQLite source binding result")
        provenance = result["provenance"]
        if provenance.get("name") != path.name + _STAGING_METADATA_SUFFIX:
            raise OSError(errno.EPROTO, "invalid SQLite source binding metadata name")
        _verify_staging_metadata_name(metadata_descriptor, provenance)
        if result["staged"] != (provenance["identity"] is not None):
            raise OSError(errno.EPROTO, "inconsistent SQLite source binding provenance")
        if not result["staged"] and Path(result["source_path"]) != semantic_path:
            raise OSError(errno.EPROTO, "inconsistent SQLite source binding coordinate")
        actual = path.stat()
        if (actual.st_dev, actual.st_ino) != main[:2]:
            raise OSError(errno.ESTALE, "SQLite declared root changed", str(path))
        from polylogue.sources.parsers.hermes_identity import capture_profile_namespace

        with ExitStack() as profile_stack:
            if result["staged"]:
                from polylogue.sources.parsers.hermes_identity import CapturedHermesProfile

                receipt = result["profile"]
                if (
                    not isinstance(receipt, dict)
                    or set(receipt)
                    != {
                        "source_path",
                        "identity_path",
                        "profile_root",
                        "profile_key",
                        "profile_source_path",
                    }
                    or any(not isinstance(value, str) for value in receipt.values())
                ):
                    raise OSError(errno.EPROTO, "invalid staged profile identity receipt")
                from polylogue.sources.parsers.hermes_identity import _captured_profile_key, profile_root_for_artifact

                profile_root = Path(receipt["profile_root"])
                profile_source = Path(receipt["profile_source_path"])
                if (
                    not profile_root.is_absolute()
                    or not profile_source.is_absolute()
                    or not Path(receipt["identity_path"]).is_absolute()
                    or receipt["source_path"] != result["source_path"]
                    or profile_root_for_artifact(profile_source) != profile_root
                    or _captured_profile_key(profile_root) != receipt["profile_key"]
                ):
                    raise OSError(errno.EPROTO, "inconsistent staged profile identity receipt")
                profile = CapturedHermesProfile(
                    Path(receipt["profile_root"]), receipt["profile_key"], Path(receipt["profile_source_path"])
                )
                identity_path = Path(receipt["identity_path"])
            else:
                profile = profile_stack.enter_context(capture_profile_namespace(path, metadata_descriptor))
                identity_path = physical_path if parent_anchor is None else declared_parent / path.name
            yield SQLiteSourceBinding(
                path,
                physical_path,
                Path(result["source_path"]),
                identity_path,
                profile.key,
                profile.root,
                profile.source_path,
                descriptor,
                metadata_descriptor,
                main[:2],
                provenance,
                result["staged"],
            )
    finally:
        os.close(metadata_descriptor)
        os.close(descriptor)


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
    from polylogue.sources.parsers.hermes_identity import _captured_profile_key, profile_root_for_artifact

    normalized = identity_path
    if _captured_profile_key(profile_root_for_artifact(normalized)) != profile_identity:
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

    This coordinate is already accepted acquisition evidence. Live staged
    inputs resolve their provenance through ``bind_sqlite_source`` first;
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


def sqlite_member_revision(path: Path, *, immutable: bool = False) -> str:
    """Digest the logical revision acquisition records for *path*.

    Acquisition retains one export per declared member, so the freshness gate
    and the raw identity must both be scoped to that member's logical tables.
    A whole-database digest would move for a commit in a table nothing reads.
    """
    return sqlite_member_revision_and_size(path, immutable=immutable)[0]


def sqlite_member_revision_and_size(
    path: Path, *, immutable: bool = False, source_binding: SQLiteSourceBinding | None = None
) -> tuple[str, int]:
    """Return the retained logical export's revision and byte length together."""
    if source_binding is None:
        with bind_sqlite_source(path) as binding:
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
    are a real logical export. Such material is replayable: its own header
    carries the scope it was written under.
    """
    if declared_database_member(Path(source_path)) is not None:
        return False
    if not looks_like_logical_export_path(blob_path):
        return False
    try:
        read_export_header(blob_path)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return True


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


def retained_content_revision(blob_path: Path, blob_hash: str) -> str:
    """Return the content term identifying one retained acquisition.

    Live acquisition of a mutable database identifies it by the canonical
    logical export's digest, so import and replay use that digest directly.
    Material that is not a canonical export is identified by its bytes.

    A retained export is already the canonical form of its member's logical
    revision, so its blob hash -- sha256 over exactly those bytes -- is that
    term with nothing to recompute.

    In particular, an old SQLite page image remains addressable by its blob
    hash but cannot regain logical-source identity. It is historical opaque
    material, not a compatibility input for the current source contract.
    """
    if looks_like_logical_export_path(blob_path):
        return blob_hash
    return blob_hash


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create a consistent standalone backup without writing to the source."""
    _snapshot_sqlite_database_bound(source, destination)


def _snapshot_sqlite_database_bound(source: Path, destination: Path) -> dict[str, Any]:
    """Carry the actual backup owner's accepted coordinate to staging provenance."""
    if source.resolve() == destination.resolve():
        raise ValueError("a SQLite backup cannot replace its source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    from polylogue.sources.sqlite_export import _backup_source_database

    try:
        return _backup_source_database(source, destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def sqlite_staging_metadata_path(staged_path: Path) -> Path:
    """Return the non-ingestible provenance sidecar for a staged database."""
    return staged_path.with_name(f"{staged_path.name}{_STAGING_METADATA_SUFFIX}")


def stage_sqlite_snapshot(source: Path, destination: Path) -> None:
    """Publish a proven snapshot; readers refuse either incomplete replacement."""
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
        accepted = _snapshot_sqlite_database_bound(source, temporary_path)
        accepted_source = Path(accepted["source_path"])
        database_identity = tuple(accepted["database_identity"])
        metadata_temporary_path.write_text(
            json.dumps(
                {
                    "version": _STAGING_METADATA_VERSION,
                    "original_source_path": str(accepted_source),
                    "database_identity": list(database_identity),
                    "declared_source_path": accepted["declared_source_path"],
                    "profile_key": accepted["profile_key"],
                    "profile_root": accepted["profile_root"],
                    "profile_source_path": accepted["profile_source_path"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.chmod(metadata_temporary_path, 0o600)
        os.replace(metadata_temporary_path, metadata_path)
        os.replace(temporary_path, destination)
        with bind_sqlite_source(destination) as published:
            if (
                published.main_identity != database_identity
                or published.identity_path != accepted_source
                or str(published.source_path) != accepted["declared_source_path"]
                or published.captured_profile_key != accepted["profile_key"]
            ):
                raise OSError(errno.ESTALE, "staged SQLite publication changed", str(destination))
    finally:
        temporary_path.unlink(missing_ok=True)
        metadata_temporary_path.unlink(missing_ok=True)


def snapshot_sqlite_to_blob(
    source: Path,
    blob_store: BlobStore,
    *,
    heartbeat: Heartbeat | None = None,
    source_binding: SQLiteSourceBinding | None = None,
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
        with bind_sqlite_source(source) as binding:
            return snapshot_sqlite_to_blob(source, blob_store, heartbeat=heartbeat, source_binding=binding)
    source_fingerprint = sqlite_source_revision(source)
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
    "SQLiteSourceBinding",
    "bind_sqlite_source",
    "retained_content_revision",
    "snapshot_sqlite_database",
    "snapshot_sqlite_to_blob",
    "sqlite_snapshot_failure_as_oserror",
    "sqlite_staging_metadata_path",
    "stage_sqlite_snapshot",
    "sqlite_database_for_sidecar",
    "sqlite_logical_revision",
    "sqlite_member_revision",
    "sqlite_source_revision",
]
