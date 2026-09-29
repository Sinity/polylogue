"""Read-only blob-store integrity probes.

The probes in this module classify blob-store health without deleting files.
Garbage collection remains owned by :mod:`polylogue.storage.blob_gc`; this
surface exists so daemon health and ``polylogue ops doctor`` can report the same
integrity classes with bounded default cost.
"""

from __future__ import annotations

import hashlib
import sqlite3
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from polylogue.archive.zip_admission import (
    MAX_UNCOMPRESSED_SIZE,
    ZIP_JSON_SUFFIXES,
    ZipAdmission,
    ZipBombError,
    open_bounded_zip_entry,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError as CoreJSONDecodeError
from polylogue.core.json import dumps_bytes as json_dumps_bytes
from polylogue.core.json import loads as json_loads
from polylogue.core.raw_coordinates import zip_member_coordinate, zip_member_identity_coordinate
from polylogue.core.sqlite_introspection import column_exists as _column_exists
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.logging import get_logger
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.storage.blob_liveness import (
    BlobLivenessProjection,
    acquired_attachment_missing_ref_predicate,
    project_index_blob_hashes,
    project_live_blob_hashes,
)
from polylogue.storage.blob_store import BlobNamespaceEntry, BlobStore
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

BlobIntegrityKind = Literal[
    "orphan_blobs",
    "missing_referenced_blobs",
    "hash_mismatch",
    "invalid_namespace_entries",
]
BlobIntegritySeverity = Literal["warning", "critical"]

_DEFAULT_SAMPLE_SIZE = 100
_MAX_FINDING_SAMPLE = 10

SourceBlobSchemaKind = Literal[
    "current_versioned",
    "current_unversioned",
    "legacy_raw_only",
    "legacy",
    "mixed_transitional",
    "unreadable",
]


@dataclass(frozen=True, slots=True)
class SourceBlobCapabilityProjection:
    """The source-tier schema evidence used by integrity read routes.

    ``user_version`` is a useful positive current-schema signal, but imported
    files and minimal fixtures commonly leave it at zero. The catalog shape
    therefore decides whether canonical liveness is available. Fallback
    carriers are selected by columns, not by a version guess, and are used
    only when canonical liveness is unavailable for a genuinely historical
    or transitional source.
    """

    kind: SourceBlobSchemaKind
    user_version: int | None
    catalog_readable: bool
    current_authority: bool
    current_blob_refs: bool
    legacy_carriers: tuple[str, ...]
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlobIntegrityFinding:
    kind: BlobIntegrityKind
    severity: BlobIntegritySeverity
    count: int
    sample: tuple[str, ...]
    suggested_action: str
    bytes_total: int = 0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "severity": self.severity,
            "count": self.count,
            "sample": list(self.sample),
            "suggested_action": self.suggested_action,
        }
        if self.bytes_total:
            payload["bytes_total"] = self.bytes_total
        return payload


@dataclass(frozen=True)
class BlobIntegrityReport:
    full_scan: bool
    sample_size: int
    scanned_blobs: int
    scanned_references: int
    total_blobs_seen: int
    total_references_seen: int
    findings: tuple[BlobIntegrityFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def worst_severity(self) -> BlobIntegritySeverity | None:
        if any(finding.severity == "critical" for finding in self.findings):
            return "critical"
        if self.findings:
            return "warning"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "full_scan": self.full_scan,
            "sample_size": self.sample_size,
            "scanned_blobs": self.scanned_blobs,
            "scanned_references": self.scanned_references,
            "total_blobs_seen": self.total_blobs_seen,
            "total_references_seen": self.total_references_seen,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class BlobReferenceDebtReport:
    """Exact, read-only debt report for DB references with no blob file.

    Unlike :func:`scan_blob_integrity`, this path does not walk every blob file
    and does not re-hash blob contents. It checks every referenced blob hash
    from source evidence, counts the missing files exactly, and keeps only a
    bounded sample for operator output.
    """

    total_references_seen: int
    missing_referenced_blobs: int
    sample: tuple[str, ...]
    reference_sources: dict[str, int]

    @property
    def ok(self) -> bool:
        return self.missing_referenced_blobs == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "total_references_seen": self.total_references_seen,
            "missing_referenced_blobs": self.missing_referenced_blobs,
            "sample": list(self.sample),
            "reference_sources": dict(self.reference_sources),
        }


@dataclass(frozen=True)
class BlobReferenceDebtSample:
    blob_hash: str
    tables: tuple[str, ...]
    ref_types: tuple[str, ...]
    origins: tuple[str, ...]
    reference_rows: int
    sample_ref_id: str | None
    sample_ref_id_has_raw_session: bool
    sample_source_path: str | None
    sample_source_outer_path: str | None
    sample_source_available: bool | None
    sample_size_bytes: int | None
    sample_parse_error: str | None
    sample_validation_status: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "tables": list(self.tables),
            "ref_types": list(self.ref_types),
            "origins": list(self.origins),
            "reference_rows": self.reference_rows,
            "sample_ref_id": self.sample_ref_id,
            "sample_ref_id_has_raw_session": self.sample_ref_id_has_raw_session,
            "sample_source_path": self.sample_source_path,
            "sample_source_outer_path": self.sample_source_outer_path,
            "sample_source_available": self.sample_source_available,
            "sample_size_bytes": self.sample_size_bytes,
            "sample_parse_error": self.sample_parse_error,
            "sample_validation_status": self.sample_validation_status,
        }


@dataclass(frozen=True)
class BlobReferenceDebtClassificationReport:
    """Grouped, exact read-only classifier for missing referenced blobs."""

    source_db: str
    blob_root: str
    distinct_referenced_blobs: int
    reference_rows: int
    missing_distinct_blobs: int
    missing_by_table: dict[str, int]
    missing_by_ref_type: dict[str, int]
    missing_by_origin: dict[str, int]
    missing_ref_id_join: dict[str, int]
    missing_source_path_presence: dict[str, int]
    missing_validation_status: dict[str, int]
    missing_parse_error: dict[str, int]
    top_groups: tuple[dict[str, object], ...]
    samples: tuple[BlobReferenceDebtSample, ...]

    @property
    def ok(self) -> bool:
        return self.missing_distinct_blobs == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "source_db": self.source_db,
            "blob_root": self.blob_root,
            "distinct_referenced_blobs": self.distinct_referenced_blobs,
            "reference_rows": self.reference_rows,
            "missing_distinct_blobs": self.missing_distinct_blobs,
            "missing_by_table": dict(self.missing_by_table),
            "missing_by_ref_type": dict(self.missing_by_ref_type),
            "missing_by_origin": dict(self.missing_by_origin),
            "missing_ref_id_join": dict(self.missing_ref_id_join),
            "missing_source_path_presence": dict(self.missing_source_path_presence),
            "missing_validation_status": dict(self.missing_validation_status),
            "missing_parse_error": dict(self.missing_parse_error),
            "top_groups": [dict(group) for group in self.top_groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _blob_hash_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) == 32:
            return value.hex()
        return None
    text = str(value)
    return text if text else None


_LEGACY_DIRECT_BLOB_CARRIERS = ("raw_sessions", "raw_hook_events", "history_sidecars")
_CURRENT_SOURCE_COLUMNS = {
    "raw_sessions": ("raw_id", "blob_hash"),
    "raw_hook_events": ("hook_event_id", "blob_hash"),
    "blob_refs": ("blob_hash", "ref_id", "ref_type"),
}


def _source_schema_capabilities(conn: sqlite3.Connection) -> SourceBlobCapabilityProjection:
    """Project source blob-reference capabilities from readable catalog facts.

    The projection intentionally describes only the evidence needed by the
    integrity routes. A source schema can be older than the runtime and still
    carry a complete typed blob-ref ledger, or it can be a minimal imported
    fixture whose ``blob_refs`` relation has only a legacy ``raw_id`` key.
    Those are different contracts even when both report ``user_version=0``.
    """

    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    # The only version a source tier of this format lineage carries is the one
    # bootstrap stamps. ``user_version`` was renumbered from one by the archive
    # format floor, so it no longer orders across lineages: a historical 29 is
    # not "newer" than a current 1. Exact equality is therefore the only honest
    # version statement -- any other stamp is a file this runtime did not write,
    # and its catalog, not its integer, has to earn authority below.
    stamped_source_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]

    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        user_version = int(row[0] or 0) if row is not None else 0
        columns = {
            table: all(_table_exists(conn, table) and _column_exists(conn, table, column) for column in required)
            for table, required in _CURRENT_SOURCE_COLUMNS.items()
        }
        current_blob_refs = columns["blob_refs"]
        current_capabilities = all(columns.values())
        legacy_carriers = [
            table
            for table in _LEGACY_DIRECT_BLOB_CARRIERS
            if _table_exists(conn, table) and _column_exists(conn, table, "blob_hash")
        ]
        # A typed blob_refs table is a valid conservative carrier for an old
        # source, even when a later referent relation is absent. It is added to
        # fallback only when canonical authority is not selected below.
        # The stamp alone cannot confer authority. ``user_version`` was
        # renumbered from one by the archive format floor, so a historical
        # source database can carry the integer this runtime stamps today
        # while holding only the legacy direct-carrier shape -- and this
        # projection has no format marker to separate the two lineages
        # (polylogue, PR #5369). Trusting the integer sent such a file to the
        # canonical query, which is blocked for it, so its existing blob
        # references became unavailable to integrity and recovery tooling.
        # ``blob_refs`` is the typed ledger only this lineage writes, so the
        # stamp earns authority alongside it -- or where the catalog offers no
        # historical carrier at all to contradict it, since then there is no
        # legacy classification to preserve.
        current_authority = current_capabilities or (
            user_version == stamped_source_version and (current_blob_refs or not legacy_carriers)
        )
        if not current_authority and current_blob_refs:
            legacy_carriers.append("blob_refs")
        if current_authority:
            kind: SourceBlobSchemaKind = (
                "current_versioned" if user_version == stamped_source_version else "current_unversioned"
            )
        elif current_blob_refs and legacy_carriers:
            kind = "mixed_transitional"
        elif legacy_carriers == ["raw_sessions"]:
            kind = "legacy_raw_only"
        else:
            kind = "legacy"
        return SourceBlobCapabilityProjection(
            kind=kind,
            user_version=user_version,
            catalog_readable=True,
            current_authority=current_authority,
            current_blob_refs=current_blob_refs,
            legacy_carriers=tuple(dict.fromkeys(legacy_carriers)),
        )
    except sqlite3.Error as exc:
        return SourceBlobCapabilityProjection(
            kind="unreadable",
            user_version=None,
            catalog_readable=False,
            current_authority=True,
            current_blob_refs=False,
            legacy_carriers=(),
            blockers=(f"source schema catalog is unreadable: {exc}",),
        )


def _legacy_source_owner_hashes(conn: sqlite3.Connection) -> dict[str, frozenset[str]]:
    """Read each proven historical carrier once, retaining all its hashes."""

    capabilities = _source_schema_capabilities(conn)
    if not capabilities.catalog_readable:
        raise RuntimeError("; ".join(capabilities.blockers))
    owner_hashes: dict[str, frozenset[str]] = {}
    for table in capabilities.legacy_carriers:
        try:
            rows = conn.execute(f"SELECT DISTINCT blob_hash FROM {table} WHERE blob_hash IS NOT NULL")
            hashes = frozenset(value.hex() for (value,) in rows if isinstance(value, bytes) and len(value) == 32)
        except sqlite3.Error as exc:
            raise RuntimeError(f"historical source blob carrier {table}.blob_hash is unreadable: {exc}") from exc
        if hashes:
            owner_hashes[f"source.db.{table}"] = hashes
    return owner_hashes


def _historical_projection(
    conn: sqlite3.Connection,
    projection: BlobLivenessProjection,
    *,
    index_conn: sqlite3.Connection | None = None,
    require_index: bool = False,
) -> BlobLivenessProjection:
    """Complete a historical source fallback with readable index ownership."""

    if not projection.blockers:
        return projection
    capabilities = _source_schema_capabilities(conn)
    if capabilities.current_authority or not capabilities.catalog_readable:
        blockers = capabilities.blockers or projection.blockers
        raise RuntimeError(f"canonical blob liveness projection blocked: {'; '.join(blockers)}")
    owner_hashes = _legacy_source_owner_hashes(conn)
    if index_conn is None:
        if require_index:
            raise RuntimeError("canonical blob liveness projection blocked: index tier is unavailable")
    else:
        index_projection = project_index_blob_hashes(index_conn)
        if index_projection.blockers:
            if require_index:
                raise RuntimeError(
                    "canonical blob liveness projection blocked: "
                    f"historical source fallback cannot resolve index ownership: {'; '.join(index_projection.blockers)}"
                )
        else:
            owner_hashes.update(dict(index_projection.owner_hashes))
    return BlobLivenessProjection(
        frozenset().union(*owner_hashes.values()) if owner_hashes else frozenset(),
        owner_hashes=tuple(sorted(owner_hashes.items())),
    )


def project_source_blob_liveness(
    source_db: Path,
    *,
    index_db: Path | None = None,
    immutable: bool = False,
) -> BlobLivenessProjection:
    """Return a complete canonical source projection or refuse incomplete evidence."""

    with closing(open_readonly_connection(source_db, immutable=immutable, validate_schema=False)) as source_conn:
        if index_db is None:
            projection = project_live_blob_hashes(source_conn)
            index_conn = None
        else:
            with closing(open_readonly_connection(index_db, immutable=immutable, validate_schema=False)) as index_conn:
                projection = project_live_blob_hashes(source_conn, index_conn=index_conn, require_index=True)
                return _historical_projection(source_conn, projection, index_conn=index_conn, require_index=True)
        return _historical_projection(source_conn, projection, index_conn=index_conn)


def _referenced_blob_hashes(
    db_path: Path,
    conn: sqlite3.Connection,
    *,
    configured_root: Path | None = None,
    require_index: bool = True,
    index_db: Path | None = None,
    immutable: bool = False,
) -> list[str]:
    source_db = (configured_root / "source.db") if configured_root is not None else db_path.with_name("source.db")
    if source_db != db_path and source_db.exists():
        try:
            source_conn = open_readonly_connection(source_db, timeout_class="background-read", validate_schema=False)
            try:
                projection = project_live_blob_hashes(source_conn, index_conn=conn, require_index=True)
                if projection.blockers:
                    logger.warning(
                        "blob integrity using non-current fixture schema: %s", "; ".join(projection.blockers)
                    )
                    historical = _historical_projection(source_conn, projection, index_conn=conn)
                    return sorted(historical.live_hashes)
                return sorted(projection.live_hashes)
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            raise RuntimeError(f"source tier referenced-hash query failed for {source_db}: {exc}") from exc

    if index_db is None and configured_root is not None:
        index_db = configured_root / "index.db"
    elif index_db is None:
        from polylogue.storage.archive_identity import ArchiveLocation

        index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if require_index and db_path.name == "source.db" and index_db.exists():
        try:
            with closing(open_readonly_connection(index_db, immutable=immutable, validate_schema=False)) as index_conn:
                projection = project_live_blob_hashes(conn, index_conn=index_conn, require_index=True)
                if projection.blockers:
                    logger.warning(
                        "blob integrity using non-current fixture schema: %s", "; ".join(projection.blockers)
                    )
                    return sorted(_historical_projection(conn, projection, index_conn=index_conn).live_hashes)
        except sqlite3.Error as exc:
            raise RuntimeError(f"source tier referenced-hash query failed for {db_path}: {exc}") from exc
    else:
        try:
            projection = project_live_blob_hashes(conn, require_index=require_index)
        except sqlite3.Error as exc:
            raise RuntimeError(f"source tier referenced-hash query failed for {db_path}: {exc}") from exc
    if projection.blockers:
        logger.warning("blob integrity using non-current fixture schema: %s", "; ".join(projection.blockers))
        return sorted(_historical_projection(conn, projection).live_hashes)
    return sorted(projection.live_hashes)


def _reference_source_counts(
    db_path: Path, conn: sqlite3.Connection, *, configured_root: Path | None = None
) -> dict[str, int]:
    source_db = (configured_root / "source.db") if configured_root is not None else db_path.with_name("source.db")
    if source_db != db_path and source_db.exists():
        try:
            source_conn = open_readonly_connection(source_db, timeout_class="background-read", validate_schema=False)
            try:
                projection = project_live_blob_hashes(source_conn, index_conn=conn, require_index=True)
                if not projection.blockers:
                    return {owner: len(hashes) for owner, hashes in projection.owner_hashes}
                historical = _historical_projection(source_conn, projection, index_conn=conn)
                return {owner: len(hashes) for owner, hashes in historical.owner_hashes}
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            logger.warning(
                "blob integrity: source.db reference-count query failed for %s: %s", source_db, exc, exc_info=True
            )

    if configured_root is not None:
        index_db = configured_root / "index.db"
    else:
        from polylogue.storage.archive_identity import ArchiveLocation

        index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if db_path.name == "source.db" and index_db.exists():
        with closing(
            open_readonly_connection(index_db, timeout_class="background-read", validate_schema=False)
        ) as index_conn:
            projection = project_live_blob_hashes(conn, index_conn=index_conn, require_index=True)
            if projection.blockers:
                historical = _historical_projection(conn, projection, index_conn=index_conn)
                return {owner: len(hashes) for owner, hashes in historical.owner_hashes}
    else:
        projection = project_live_blob_hashes(conn, require_index=True)
    if projection.blockers:
        historical = _historical_projection(conn, projection)
        return {owner: len(hashes) for owner, hashes in historical.owner_hashes}
    return {owner: len(hashes) for owner, hashes in projection.owner_hashes}


def referenced_blob_hashes(
    db_path: str | Path,
    *,
    immutable: bool = False,
    require_index: bool = True,
    index_db: Path | None = None,
) -> list[str]:
    """Return distinct blob hashes referenced by archive source evidence."""

    resolved_db_path = Path(db_path)
    try:
        with closing(open_readonly_connection(resolved_db_path, immutable=immutable, validate_schema=False)) as conn:
            return _referenced_blob_hashes(
                resolved_db_path,
                conn,
                require_index=require_index,
                index_db=index_db,
                immutable=immutable,
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"source tier referenced-hash query failed for {resolved_db_path}: {exc}") from exc


def _source_db_for_blob_reference_report(db_path: str | Path) -> Path:
    resolved = Path(db_path)
    sibling_source = resolved.with_name("source.db")
    if sibling_source.exists():
        return sibling_source
    return resolved


# Directories the archive owns and acquires material into. A recorded path
# that runs through one of them was written under some archive root, so its
# tail from that segment re-anchors onto the root in force.
_ARCHIVE_OWNED_DIRECTORIES = ("inbox", "browser-capture", "hooks")


def _reanchored_archive_path(path: Path, archive_root: Path | None) -> Path | None:
    """Re-anchor a path recorded under a previous archive root, if it is one."""
    if archive_root is None:
        return None
    parts = path.parts
    for name in _ARCHIVE_OWNED_DIRECTORIES:
        if name not in parts:
            continue
        tail = parts[parts.index(name) :]
        candidate = archive_root.joinpath(*tail)
        if candidate != path:
            return candidate
    return None


def _source_path_availability(path: str | None, archive_root: Path | None = None) -> tuple[bool | None, str | None]:
    """Report whether a recorded source path still resolves to material on disk.

    An archive root moves, and acquisition records absolute paths, so a path
    written under a previous root names material that is present under the
    current one. Reporting those as missing is false loss, and this number
    decides whether a prune was safe.
    """
    if not path:
        return None, None
    direct = Path(path)
    if direct.exists():
        return True, str(direct)
    if ":" in path:
        outer, _inner = path.split(":", 1)
        outer_path = Path(outer)
        if outer_path.exists():
            return True, str(outer_path)
        reanchored_outer = _reanchored_archive_path(outer_path, archive_root)
        if reanchored_outer is not None and reanchored_outer.exists():
            return True, str(reanchored_outer)
        return False, str(outer_path)
    reanchored = _reanchored_archive_path(direct, archive_root)
    if reanchored is not None and reanchored.exists():
        return True, str(reanchored)
    return False, str(direct)


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _blob_ref_source_path_column(conn: sqlite3.Connection) -> str:
    return "source_path" if _column_exists(conn, "blob_refs", "source_path") else "NULL"


def _blob_ref_size_column(conn: sqlite3.Connection) -> str:
    return "size_bytes" if _column_exists(conn, "blob_refs", "size_bytes") else "0"


def _blob_ref_id_column(conn: sqlite3.Connection) -> str:
    if _column_exists(conn, "blob_refs", "ref_id"):
        return "ref_id"
    if _column_exists(conn, "blob_refs", "raw_id"):
        return "raw_id"
    return "NULL"


def _raw_session_reference_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "raw_sessions"):
        return []
    conn.row_factory = sqlite3.Row
    origin_column = "origin" if _column_exists(conn, "raw_sessions", "origin") else "NULL"
    detected_provider_column = (
        "detected_provider" if _column_exists(conn, "raw_sessions", "detected_provider") else "NULL"
    )
    native_id_column = "native_id" if _column_exists(conn, "raw_sessions", "native_id") else "NULL"
    source_path_column = "source_path" if _column_exists(conn, "raw_sessions", "source_path") else "NULL"
    source_index_column = "source_index" if _column_exists(conn, "raw_sessions", "source_index") else "NULL"
    revision_kind_column = "revision_kind" if _column_exists(conn, "raw_sessions", "revision_kind") else "NULL"
    append_start_offset_column = (
        "append_start_offset" if _column_exists(conn, "raw_sessions", "append_start_offset") else "NULL"
    )
    append_end_offset_column = (
        "append_end_offset" if _column_exists(conn, "raw_sessions", "append_end_offset") else "NULL"
    )
    capture_mode_column = "capture_mode" if _column_exists(conn, "raw_sessions", "capture_mode") else "NULL"
    acquired_at_ms_column = "acquired_at_ms" if _column_exists(conn, "raw_sessions", "acquired_at_ms") else "NULL"
    has_container_coordinates = _table_exists(conn, "raw_container_coordinates")
    coordinate_join = (
        "LEFT JOIN raw_container_coordinates coordinate ON coordinate.raw_id = raw_sessions.raw_id"
        if has_container_coordinates
        else ""
    )
    coordinate_format_column = "coordinate.coordinate_format" if has_container_coordinates else "NULL"
    entry_ordinal_column = "coordinate.entry_ordinal" if has_container_coordinates else "NULL"
    split_index_column = "coordinate.split_index" if has_container_coordinates else "NULL"
    addressing_mode_column = (
        "coordinate.addressing_mode"
        if has_container_coordinates and _column_exists(conn, "raw_container_coordinates", "addressing_mode")
        else "NULL"
    )
    content_identity_column = (
        "coordinate.content_identity"
        if has_container_coordinates and _column_exists(conn, "raw_container_coordinates", "content_identity")
        else "NULL"
    )
    blob_size_column = "blob_size" if _column_exists(conn, "raw_sessions", "blob_size") else "0"
    parse_error_column = "parse_error" if _column_exists(conn, "raw_sessions", "parse_error") else "NULL"
    validation_status_column = (
        "validation_status" if _column_exists(conn, "raw_sessions", "validation_status") else "NULL"
    )
    rows = conn.execute(
        f"""
        SELECT lower(hex(raw_sessions.blob_hash)) AS blob_hash,
               'raw_sessions' AS table_name,
               'raw_payload' AS ref_type,
               raw_sessions.raw_id AS ref_id,
               raw_sessions.raw_id AS raw_id,
               {origin_column} AS origin,
               {detected_provider_column} AS detected_provider,
               {native_id_column} AS native_id,
               {capture_mode_column} AS capture_mode,
               {acquired_at_ms_column} AS acquired_at_ms,
               {source_path_column} AS source_path,
               {source_index_column} AS source_index,
               {revision_kind_column} AS revision_kind,
               {append_start_offset_column} AS append_start_offset,
               {append_end_offset_column} AS append_end_offset,
               {blob_size_column} AS size_bytes,
               {parse_error_column} AS parse_error,
               {validation_status_column} AS validation_status,
               {coordinate_format_column} AS coordinate_format,
               {entry_ordinal_column} AS entry_ordinal,
               {split_index_column} AS split_index,
               {addressing_mode_column} AS addressing_mode,
               {content_identity_column} AS content_identity,
               1 AS ref_id_has_raw_session
        FROM raw_sessions
        {coordinate_join}
        WHERE raw_sessions.blob_hash IS NOT NULL
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _blob_ref_reference_rows(
    conn: sqlite3.Connection,
    *,
    raw_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "blob_refs"):
        return []
    conn.row_factory = sqlite3.Row
    ref_id_column = _blob_ref_id_column(conn)
    source_path_column = _blob_ref_source_path_column(conn)
    size_column = _blob_ref_size_column(conn)
    rows = conn.execute(
        f"""
        SELECT lower(hex(blob_hash)) AS blob_hash,
               'blob_refs' AS table_name,
               ref_type,
               {ref_id_column} AS ref_id,
               {source_path_column} AS source_path,
               {size_column} AS size_bytes
        FROM blob_refs
        WHERE blob_hash IS NOT NULL
        """
    ).fetchall()
    refs: list[dict[str, Any]] = []
    for row in rows:
        ref = dict(row)
        raw = raw_by_id.get(str(ref.get("ref_id")))
        ref["origin"] = raw.get("origin") if raw else None
        ref["native_id"] = raw.get("native_id") if raw else None
        ref["parse_error"] = raw.get("parse_error") if raw else None
        ref["validation_status"] = raw.get("validation_status") if raw else None
        ref["source_index"] = raw.get("source_index") if raw else None
        ref["ref_id_has_raw_session"] = raw is not None
        refs.append(ref)
    return refs


def _group_reference_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blob_hash = str(row.get("blob_hash") or "")
        if blob_hash:
            by_hash[blob_hash].append(row)
    return by_hash


def _reference_rows_for_blob_debt(source_db: Path) -> list[dict[str, Any]]:
    with closing(open_readonly_connection(source_db, timeout_class="background-read", validate_schema=False)) as conn:
        raw_refs = _raw_session_reference_rows(conn)
        raw_by_id = {str(row["ref_id"]): row for row in raw_refs if row.get("ref_id")}
        return [*raw_refs, *_blob_ref_reference_rows(conn, raw_by_id=raw_by_id)]


def classify_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = 30,
    group_limit: int = 20,
) -> BlobReferenceDebtClassificationReport:
    """Classify missing referenced blobs without mutating archive state."""

    source_db = _source_db_for_blob_reference_report(db_path)
    archive_root = source_db.parent
    blob_store = store if store is not None else BlobStore(source_db.parent / "blob")
    refs = _reference_rows_for_blob_debt(source_db)
    by_hash = _group_reference_rows(refs)
    missing = [(blob_hash, group) for blob_hash, group in by_hash.items() if not blob_store.exists(blob_hash)]

    by_table: Counter[str] = Counter()
    by_ref_type: Counter[str] = Counter()
    by_origin: Counter[str] = Counter()
    ref_id_join: Counter[str] = Counter()
    source_path_presence: Counter[str] = Counter()
    validation_status: Counter[str] = Counter()
    parse_error: Counter[str] = Counter()
    grouped: Counter[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = Counter()
    samples: list[BlobReferenceDebtSample] = []

    for blob_hash, group in missing:
        tables = tuple(sorted({str(row.get("table_name")) for row in group if row.get("table_name")}))
        ref_types = tuple(sorted({str(row.get("ref_type")) for row in group if row.get("ref_type")}))
        origins = tuple(sorted({str(row.get("origin")) for row in group if row.get("origin")}))
        for table in tables:
            by_table[table] += 1
        for ref_type in ref_types:
            by_ref_type[ref_type] += 1
        for origin in origins or ("(none)",):
            by_origin[origin] += 1
        if any(bool(row.get("ref_id_has_raw_session")) for row in group):
            ref_id_join["ref_id_has_raw_session"] += 1
        else:
            ref_id_join["ref_id_without_raw_session"] += 1

        source_availability = [
            _source_path_availability(_optional_str(row.get("source_path")), archive_root)[0] for row in group
        ]
        known_source_availability = [value for value in source_availability if value is not None]
        if not known_source_availability:
            source_path_presence["no_source_path_recorded"] += 1
        elif any(known_source_availability):
            source_path_presence["recoverable_source_path_exists"] += 1
        else:
            source_path_presence["source_path_missing"] += 1

        statuses = {str(row.get("validation_status")) for row in group if row.get("validation_status")}
        validation_status[",".join(sorted(statuses)) if statuses else "(none)"] += 1
        errors = {str(row.get("parse_error")) for row in group if row.get("parse_error")}
        parse_error["has_parse_error" if errors else "no_parse_error"] += 1
        grouped[(tables, ref_types, origins or ("(none)",))] += 1

        if len(samples) < max(0, sample_size):
            sample = group[0]
            sample_source_path = _optional_str(sample.get("source_path"))
            available, outer = _source_path_availability(sample_source_path, archive_root)
            size = sample.get("size_bytes")
            samples.append(
                BlobReferenceDebtSample(
                    blob_hash=blob_hash,
                    tables=tables,
                    ref_types=ref_types,
                    origins=origins,
                    reference_rows=len(group),
                    sample_ref_id=_optional_str(sample.get("ref_id")),
                    sample_ref_id_has_raw_session=any(bool(row.get("ref_id_has_raw_session")) for row in group),
                    sample_source_path=sample_source_path,
                    sample_source_outer_path=outer,
                    sample_source_available=available,
                    sample_size_bytes=int(size) if size is not None else None,
                    sample_parse_error=_optional_str(sample.get("parse_error")),
                    sample_validation_status=_optional_str(sample.get("validation_status")),
                )
            )

    top_groups: list[dict[str, object]] = []
    for (tables, ref_types, origins), count in grouped.most_common(max(0, group_limit)):
        top_groups.append(
            {
                "tables": list(tables),
                "ref_types": list(ref_types),
                "origins": list(origins),
                "count": count,
            }
        )

    return BlobReferenceDebtClassificationReport(
        source_db=str(source_db),
        blob_root=str(blob_store.root),
        distinct_referenced_blobs=len(by_hash),
        reference_rows=len(refs),
        missing_distinct_blobs=len(missing),
        missing_by_table=_counter_dict(by_table),
        missing_by_ref_type=_counter_dict(by_ref_type),
        missing_by_origin=_counter_dict(by_origin),
        missing_ref_id_join=_counter_dict(ref_id_join),
        missing_source_path_presence=_counter_dict(source_path_presence),
        missing_validation_status=_counter_dict(validation_status),
        missing_parse_error=_counter_dict(parse_error),
        top_groups=tuple(top_groups),
        samples=tuple(samples),
    )


def _path_is_container_member(path: str) -> bool:
    return ":" in path


def _split_container_source_path(source_path: str) -> tuple[Path, str] | None:
    outer, sep, member = source_path.partition(":")
    if not sep or not outer or not member:
        return None
    # A container path may itself hold a colon; prefer the prefix that is a real ZIP.
    return zip_member_coordinate(source_path) or (Path(outer), member)


def _jsonl_payloads(raw_bytes: bytes) -> list[object]:
    return [json_loads(line) for line in raw_bytes.splitlines() if line.strip()]


def _member_payload_by_content(
    decoded_payload: object,
    *,
    split_index: int,
    blob_hash: str | None,
    content_identity: str | None = None,
    addressing_mode: str | None = None,
    positional_when_content_is_gone: bool = False,
) -> bytes:
    """Return the member value the recorded reference names.

    ``split_index`` chooses which value is checked first and never which value
    is returned: an export that reorders or inserts elements leaves a valid but
    unrelated conversation at the recorded position.

    ``positional_when_content_is_gone`` belongs to the one caller whose
    contract is that the recorded content is stale by construction --
    recanonicalizing a row onto whatever its source holds now. Every caller
    that is proving a recorded reference leaves it false.
    """
    if addressing_mode == "whole_member":
        elements = [decoded_payload]
    elif isinstance(decoded_payload, list):
        elements = list(decoded_payload)
    elif split_index == 0:
        elements = [decoded_payload]
    else:
        raise IndexError("non-array JSON payload only supports source_index 0")
    hinted: bytes | None = None
    if 0 <= split_index < len(elements):
        hinted = json_dumps_bytes(elements[split_index])
        if _payload_matches(hinted, blob_hash=blob_hash, content_identity=content_identity):
            return hinted
    elif blob_hash is None:
        raise IndexError(f"source_index {split_index} outside member array")
    for element in elements:
        encoded = json_dumps_bytes(element)
        if _payload_matches(encoded, blob_hash=blob_hash, content_identity=content_identity):
            return encoded
    if positional_when_content_is_gone and hinted is not None:
        return hinted
    raise IndexError(f"no member value matches the content identity of {blob_hash}")


def _payload_matches(payload: bytes, *, blob_hash: str | None, content_identity: str | None) -> bool:
    if content_identity is not None:
        from polylogue.core.content_identity import ContentIdentityRefusal, payload_content_identity

        try:
            return payload_content_identity(payload) == content_identity
        except ContentIdentityRefusal:
            # Acquisition refuses such a value, so no recorded identity names
            # it: this candidate is not the referenced one.
            return False
    return blob_hash is None or hashlib.sha256(payload).hexdigest() == blob_hash


def _current_raw_payload_bytes(
    source_path: str,
    source_index: int | None,
    *,
    raw_id: str | None = None,
    blob_hash: str | None = None,
    content_identity: str | None = None,
    addressing_mode: str | None = None,
    zip_coordinate: tuple[int, int] | None = None,
    source_bytes_cache: dict[str, bytes] | None = None,
    decoded_payload_cache: dict[str, object] | None = None,
    provider_hint: str | None = None,
    positional_when_content_is_gone: bool = False,
) -> tuple[bytes | None, str | None]:
    if _path_is_container_member(source_path):
        split = _split_container_source_path(source_path)
        if split is None:
            return None, "unsupported_container_path"
        zip_path, member = split
        if not zip_path.exists():
            return None, "source_missing"
        entry_ordinal: int | None = zip_coordinate[0] if zip_coordinate is not None else None
        split_index = zip_coordinate[1] if zip_coordinate is not None else source_index
        if zip_coordinate is None and raw_id is not None and blob_hash is not None and source_index is not None:
            coordinate = zip_member_identity_coordinate(
                raw_id=raw_id,
                source_path=source_path,
                source_index=source_index,
                blob_hash=blob_hash,
            )
            if coordinate is not None:
                entry_ordinal, split_index = coordinate
        cache_key = source_path if entry_ordinal is None else f"{source_path}\0{entry_ordinal}"
        provider = None
        declared_rule = None
        if provider_hint is not None:
            try:
                provider = Provider.from_string(provider_hint)
            except ValueError:
                provider = None
            if provider is not None:
                declared_rule = artifact_rule_for_path(provider, member)
        try:
            if source_bytes_cache is not None and cache_key in source_bytes_cache:
                member_bytes = source_bytes_cache[cache_key]
            else:
                with zipfile.ZipFile(zip_path) as archive:
                    central_directory = archive.infolist()
                    if entry_ordinal is None:
                        matching = [info for info in central_directory if info.filename == member]
                    elif entry_ordinal >= len(central_directory):
                        return None, "container_coordinate_mismatch"
                    else:
                        coordinated = central_directory[entry_ordinal]
                        matching = [coordinated] if coordinated.filename == member else []
                    if len(matching) != 1:
                        reason = (
                            "ambiguous_container_member" if entry_ordinal is None else "container_coordinate_mismatch"
                        )
                        return None, reason
                    allowed_path = (
                        (lambda name: artifact_rule_for_path(provider, name) is not None) if provider else None
                    )
                    admitted = list(
                        ZipAdmission(zip_path=zip_path).filter_entries(
                            matching,
                            allowed_suffixes=ZIP_JSON_SUFFIXES,
                            allowed_path=allowed_path,
                        )
                    )
                    if len(admitted) != 1:
                        return None, "container_member_rejected"
                    with open_bounded_zip_entry(archive, admitted[0]) as handle:
                        member_bytes = handle.read(MAX_UNCOMPRESSED_SIZE + 1)
                if source_bytes_cache is not None:
                    source_bytes_cache[cache_key] = member_bytes
        except KeyError:
            return None, "source_missing"
        except (OSError, zipfile.BadZipFile) as exc:
            return None, f"error:{exc}"
        except ZipBombError:
            return None, "container_member_rejected"
        if split_index is None:
            return None, "source_index_missing"
        if blob_hash is not None and hashlib.sha256(member_bytes).hexdigest() == blob_hash:
            return member_bytes, None
        if declared_rule is not None and declared_rule.parse_policy == "raw-only":
            return member_bytes, None
        try:
            if decoded_payload_cache is not None and cache_key in decoded_payload_cache:
                decoded_payload = decoded_payload_cache[cache_key]
            elif member.endswith(".jsonl"):
                decoded_payload = _jsonl_payloads(member_bytes)
            else:
                decoded_payload = json_loads(member_bytes)
            if decoded_payload_cache is not None:
                decoded_payload_cache[cache_key] = decoded_payload
            payload_bytes = _member_payload_by_content(
                decoded_payload,
                split_index=int(split_index),
                blob_hash=blob_hash,
                content_identity=content_identity,
                addressing_mode=addressing_mode,
                positional_when_content_is_gone=positional_when_content_is_gone,
            )
        except (IndexError, CoreJSONDecodeError, UnicodeDecodeError) as exc:
            return None, f"source_index:{exc}"
        return payload_bytes, None

    path = Path(source_path)
    if not path.exists():
        return None, "source_missing"
    try:
        # The recorded blob size is not a bound on the file that is here now:
        # a source can have grown arbitrarily since acquisition, and this
        # fallback runs on every prefix mismatch. Read one byte past the same
        # ceiling the container branch above applies and refuse an oversized
        # source by name, so the caller reports a bounded refusal rather than
        # silently hashing a truncated prefix.
        with path.open("rb") as handle:
            payload = handle.read(MAX_UNCOMPRESSED_SIZE + 1)
    except OSError as exc:
        return None, f"error:{exc}"
    if len(payload) > MAX_UNCOMPRESSED_SIZE:
        return None, "source_too_large"
    return payload, None


def _sample(values: list[str], *, full: bool, sample_size: int) -> list[str]:
    if full:
        return values
    return list(islice(values, max(0, sample_size)))


def _blob_sample(blob_store: BlobStore, *, full: bool, sample_size: int) -> list[str]:
    hashes = blob_store.iter_all()
    if full:
        return list(hashes)
    return list(islice(hashes, max(0, sample_size)))


def _blob_size(store: BlobStore, blob_hash: str) -> int:
    try:
        return int(store.blob_path(blob_hash).stat().st_size)
    except (OSError, ValueError):
        return 0


def blob_reference_debt_from_projection(
    projection: BlobLivenessProjection,
    *,
    store: BlobStore,
    sample_size: int = _MAX_FINDING_SAMPLE,
) -> BlobReferenceDebtReport:
    """Report missing bytes from one already-validated canonical projection."""

    if projection.blockers:
        raise RuntimeError(f"canonical blob liveness projection blocked: {'; '.join(projection.blockers)}")
    missing: list[str] = []
    for blob_hash in sorted(projection.live_hashes):
        if not store.exists(blob_hash):
            missing.append(blob_hash)
    return BlobReferenceDebtReport(
        total_references_seen=len(projection.live_hashes),
        missing_referenced_blobs=len(missing),
        sample=tuple(missing[: max(0, sample_size)]),
        reference_sources={owner: len(hashes) for owner, hashes in projection.owner_hashes},
    )


def scan_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = _MAX_FINDING_SAMPLE,
    immutable: bool = False,
    configured_root: Path | None = None,
) -> BlobReferenceDebtReport:
    """Count missing referenced blob files exactly without mutating state.

    ``configured_root`` names the durable-tier archive root explicitly for
    callers whose ``db_path`` was resolved through a ``.index-active-pointer``
    generation (``resolve_active_index_path``) and may therefore live outside
    the root that actually houses ``source.db``; it defaults to
    ``db_path.with_name("source.db")`` for callers that never diverge from
    the plain convention.
    """

    resolved_db_path = Path(db_path)
    # The blob root follows ``configured_root`` for exactly the reason the
    # reference side does: a generation-resolved ``db_path`` can be
    # ``<archive>/.index-generations/<gen>/index.db`` while the durable blobs
    # stay at ``<archive>/blob``. Deriving the store from ``db_path`` alone
    # pointed the scan at a directory that does not exist, so every reference
    # counted as missing and the health check reported debt that was not there.
    blob_root_parent = configured_root if configured_root is not None else resolved_db_path.parent
    blob_store = store if store is not None else BlobStore(Path(blob_root_parent) / "blob")
    with closing(open_readonly_connection(resolved_db_path, immutable=immutable, validate_schema=False)) as conn:
        referenced = _referenced_blob_hashes(resolved_db_path, conn, configured_root=configured_root)
        reference_sources = _reference_source_counts(resolved_db_path, conn, configured_root=configured_root)

    missing_count = 0
    sample: list[str] = []
    sample_limit = max(0, sample_size)
    for blob_hash in referenced:
        if blob_store.exists(blob_hash):
            continue
        missing_count += 1
        if len(sample) < sample_limit:
            sample.append(blob_hash)
    return BlobReferenceDebtReport(
        total_references_seen=len(referenced),
        missing_referenced_blobs=missing_count,
        sample=tuple(sample),
        reference_sources=reference_sources,
    )


@dataclass(frozen=True)
class AttachmentCoverageReport:
    """Index-tier attachment coverage used by archive verification.

    Deliberately separate from :class:`BlobReferenceDebtReport`, which
    classifies source-tier backup debt from ``source.db``/``raw_sessions``
    references and never queries the index-tier ``attachments`` table.
    ``unfetched`` (``blob_hash IS NULL``) is an honest floor -- bytes were
    never fetched, not a broken reference -- and must never be conflated
    with missing referenced blobs. Only an ``acquired`` row whose blob file
    is absent from the store is genuine attachment acquisition debt.

    ``acquisition_status='acquired'`` alone overstates real coverage
    (polylogue-w06b): it says bytes were fetched, not that anything can
    read them back. ``acquired_unreachable_count``/``acquired_reachable_count``
    split the ``acquired`` bucket by whether the attachment also has at
    least one ``attachment_refs`` row -- without one it cannot be returned
    by any session/message read path (``get_attachments``, MCP get/read,
    CLI ``read --view``).

    ``acquired_count`` is the status tally -- the claim the rows make. The
    three terms below partition it by what the blob store can say about that
    claim, so no single figure in this report can be read as measured
    coverage (polylogue-o0uw5):

    ``acquired_with_bytes_count``
        Corroborated: the store holds an object that still re-hashes to the
        recorded hash. Presence alone is not corroboration -- ``exists()``
        answers "a file sits at that path", while ``acquired`` is the
        positive claim that these exact bytes were fetched and stored, which
        is why the re-bind probe (``_surviving_blob_ref``) decides survival
        by re-hashing too.
    ``acquired_corrupt_count``
        Contradicted in place: an object sits at the recorded hash's path but
        its content hashes to something else. Counting it as corroborated
        certified the one state the re-bind probe rejects, so an archive
        holding decayed attachment bytes reported full coverage.
    ``acquired_missing_blob_count``
        Contradicted: the row names a hash the store does not hold. This is
        the 2026-09-14 pre-wipe census shape (1240 of 1446 acquired hashes
        with no bytes behind them).
    ``acquired_unverifiable_count``
        Unverifiable: ``acquired`` with no ``blob_hash`` at all, so there is
        nothing to check it against. This bucket used to be skipped outright,
        which counted an unchecked claim as a verified one and left ``ok``
        true; it is debt now, exactly as the contradicted bucket is. No
        current writer produces the shape (``write.py``'s
        ``_acquire_attachment_blob`` returns ``unfetched`` whenever it has no
        hash) but the column is nullable, and the whole point of this report
        is that stored state is not taken on trust.
    """

    total_attachments: int
    acquired_count: int
    acquired_with_bytes_count: int
    acquired_missing_blob_count: int
    acquired_missing_blob_sample: tuple[str, ...]
    acquired_unverifiable_count: int
    acquired_unverifiable_sample: tuple[str, ...]
    unavailable_count: int
    unfetched_count: int
    # polylogue-w06b: `acquisition_status='acquired'` alone overstates real
    # coverage -- an attachment row with zero `attachment_refs` rows cannot
    # be surfaced through any session/message read path (`get_attachments`,
    # MCP get/read, CLI read --view all INNER JOIN attachment_refs). Track
    # the reachable subset separately so this report can't be read as "N
    # attachments are queryable" when it means "N attachments have bytes on
    # disk somewhere, most of which nothing can ever reach".
    acquired_unreachable_count: int
    acquired_unreachable_sample: tuple[str, ...]
    #: Acquired attachments the writer retained with an ambiguous owner
    #: (ref_count 0, never swept). Unreferenced by construction, so never
    #: coverage debt -- and not reachable either, so they are their own term.
    acquired_unowned_count: int = 0
    #: Acquired rows whose stored object no longer hashes to the recorded
    #: ``blob_hash``. Its own bucket rather than part of
    #: ``acquired_missing_blob_count``: the object is present, so "re-fetch
    #: the missing bytes" is the wrong remedy and the decay is evidence in
    #: its own right.
    acquired_corrupt_count: int = 0
    acquired_corrupt_sample: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.acquired_missing_blob_count == 0
            and self.acquired_unverifiable_count == 0
            and self.acquired_corrupt_count == 0
        )

    @property
    def acquired_reachable_count(self) -> int:
        """Acquired rows a session/message read path can return.

        This is the reference axis, not the bytes axis: a contradicted or
        unverifiable row with a live ``attachment_refs`` row is still counted
        here. Read it together with ``acquired_with_bytes_count``, never on
        its own as coverage.
        """
        return self.acquired_count - self.acquired_unreachable_count - self.acquired_unowned_count

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "total_attachments": self.total_attachments,
            "acquired_count": self.acquired_count,
            "acquired_with_bytes_count": self.acquired_with_bytes_count,
            "acquired_reachable_count": self.acquired_reachable_count,
            "acquired_unreachable_count": self.acquired_unreachable_count,
            "acquired_unreachable_sample": list(self.acquired_unreachable_sample),
            "acquired_unowned_count": self.acquired_unowned_count,
            "acquired_missing_blob_count": self.acquired_missing_blob_count,
            "acquired_missing_blob_sample": list(self.acquired_missing_blob_sample),
            "acquired_corrupt_count": self.acquired_corrupt_count,
            "acquired_corrupt_sample": list(self.acquired_corrupt_sample),
            "acquired_unverifiable_count": self.acquired_unverifiable_count,
            "acquired_unverifiable_sample": list(self.acquired_unverifiable_sample),
            "unavailable_count": self.unavailable_count,
            "unfetched_count": self.unfetched_count,
        }


def scan_attachment_coverage(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = _MAX_FINDING_SAMPLE,
) -> AttachmentCoverageReport:
    """Project index-tier attachment coverage without mutating state.

    ``db_path`` is the index-tier database (``index.db``), not ``source.db``.
    """

    resolved_db_path = Path(db_path)
    blob_store = store if store is not None else BlobStore(resolved_db_path.parent / "blob")
    with closing(
        open_readonly_connection(resolved_db_path, timeout_class="background-read", validate_schema=False)
    ) as conn:
        conn.row_factory = sqlite3.Row
        status_counts: dict[str, int] = dict(
            conn.execute("SELECT acquisition_status, COUNT(*) FROM attachments GROUP BY acquisition_status").fetchall()
        )
        acquired_rows = conn.execute(
            "SELECT attachment_id, blob_hash FROM attachments WHERE acquisition_status = 'acquired'"
        ).fetchall()
        # polylogue-w06b: an `acquired` attachment with no `attachment_refs`
        # row is unreachable from every session/message read path -- track it
        # as its own dimension, distinct from "bytes missing from the blob
        # store" (acquired_missing_blob_count, below).
        unreachable_rows = conn.execute(
            f"""
            SELECT a.attachment_id AS attachment_id
            FROM attachments a
            WHERE {acquired_attachment_missing_ref_predicate()}
            ORDER BY a.attachment_id
            """
        ).fetchall()
        # The writer's owner-ambiguous retention: unreferenced by construction,
        # so it is reported as its own dimension rather than as debt.
        unowned_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM attachments a
                WHERE a.acquisition_status = 'acquired'
                  AND a.ref_count = 0
                  AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
                """
            ).fetchone()[0]
        )

    missing_sample: list[str] = []
    missing_count = 0
    unverifiable_sample: list[str] = []
    unverifiable_count = 0
    corrupt_sample: list[str] = []
    corrupt_count = 0
    with_bytes_count = 0
    for row in acquired_rows:
        blob_hash = row["blob_hash"]
        if blob_hash is None:
            # polylogue-o0uw5: an `acquired` claim with no blob identity has
            # no evidence to reconcile against. Skipping it counted it as
            # covered; it is its own state, not a corroborated one and not a
            # contradicted one.
            unverifiable_count += 1
            if len(unverifiable_sample) < sample_size:
                unverifiable_sample.append(str(row["attachment_id"]))
            continue
        hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        # Re-hash rather than probe for the path. ``exists()`` answers "a file
        # sits there", and an object overwritten with bytes the recorded hash
        # no longer names passed that probe, incremented the corroborated
        # bucket, and left both debt counters at zero -- so this check
        # certified the exact contradicted-object state the attachment re-bind
        # probe rejects (polylogue-o0uw5, PR #5378).
        if blob_store.verify(hash_hex):
            with_bytes_count += 1
            continue
        if blob_store.exists(hash_hex):
            corrupt_count += 1
            if len(corrupt_sample) < sample_size:
                corrupt_sample.append(str(row["attachment_id"]))
            continue
        missing_count += 1
        if len(missing_sample) < sample_size:
            missing_sample.append(str(row["attachment_id"]))

    unreachable_sample = tuple(str(row["attachment_id"]) for row in unreachable_rows[:sample_size])

    return AttachmentCoverageReport(
        total_attachments=sum(status_counts.values()),
        acquired_count=status_counts.get("acquired", 0),
        acquired_with_bytes_count=with_bytes_count,
        acquired_missing_blob_count=missing_count,
        acquired_missing_blob_sample=tuple(missing_sample),
        acquired_unverifiable_count=unverifiable_count,
        acquired_unverifiable_sample=tuple(unverifiable_sample),
        acquired_corrupt_count=corrupt_count,
        acquired_corrupt_sample=tuple(corrupt_sample),
        unavailable_count=status_counts.get("unavailable", 0),
        unfetched_count=status_counts.get("unfetched", 0),
        acquired_unreachable_count=len(unreachable_rows),
        acquired_unreachable_sample=unreachable_sample,
        acquired_unowned_count=unowned_count,
    )


def scan_blob_integrity(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    full: bool = False,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    configured_root: Path | None = None,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> BlobIntegrityReport:
    """Classify blob-store integrity without mutating disk or database state.

    ``full=False`` bounds the filesystem and hash-verification scan to
    ``sample_size`` blobs and references. ``full=True`` scans every blob and
    every raw-session reference.

    ``configured_root`` names the durable-tier archive root explicitly; see
    :func:`scan_blob_reference_debt` for why this matters when ``db_path``
    was resolved through a ``.index-active-pointer`` generation.
    """

    resolved_db_path = Path(db_path)
    blob_store = store if store is not None else BlobStore(resolved_db_path.parent / "blob")
    # This is an evidence scan, not a read of the active archive API. It must
    # remain usable while a candidate generation or an older active generation
    # is being inspected, so do not impose the current canonical schema gate.
    with closing(
        open_readonly_connection(resolved_db_path, timeout_class="background-read", validate_schema=False)
    ) as conn:
        referenced = _referenced_blob_hashes(
            resolved_db_path,
            conn,
            configured_root=configured_root,
            require_index=active_index_context == "required",
        )

    referenced_set = set(referenced)
    disk_sample = _blob_sample(blob_store, full=full, sample_size=sample_size)
    reference_sample = _sample(referenced, full=full, sample_size=sample_size)

    findings: list[BlobIntegrityFinding] = []

    missing = [blob_hash for blob_hash in reference_sample if not blob_store.exists(blob_hash)]
    if missing:
        findings.append(
            BlobIntegrityFinding(
                kind="missing_referenced_blobs",
                severity="critical",
                count=len(missing),
                sample=tuple(missing[:_MAX_FINDING_SAMPLE]),
                suggested_action="restore the missing blob files from backup or re-ingest the affected raw sources",
            )
        )

    orphan_hashes = [blob_hash for blob_hash in disk_sample if blob_hash not in referenced_set]
    if orphan_hashes:
        findings.append(
            BlobIntegrityFinding(
                kind="orphan_blobs",
                severity="warning",
                count=len(orphan_hashes),
                sample=tuple(orphan_hashes[:_MAX_FINDING_SAMPLE]),
                bytes_total=sum(_blob_size(blob_store, blob_hash) for blob_hash in orphan_hashes),
                suggested_action="allow the daemon's bounded periodic blob-GC pass to reclaim aged orphan files",
            )
        )

    hash_mismatches = [
        blob_hash for blob_hash in disk_sample if blob_store.exists(blob_hash) and not blob_store.verify(blob_hash)
    ]
    if hash_mismatches:
        findings.append(
            BlobIntegrityFinding(
                kind="hash_mismatch",
                severity="critical",
                count=len(hash_mismatches),
                sample=tuple(hash_mismatches[:_MAX_FINDING_SAMPLE]),
                suggested_action="replace corrupted blob files from backup or re-ingest the affected raw sources",
            )
        )

    namespace_entries: Iterable[BlobNamespaceEntry] = blob_store.iter_namespace()
    if not full:
        namespace_entries = islice(namespace_entries, max(0, sample_size))
    namespace_issues = [entry for entry in namespace_entries if entry.hash_hex is None]
    if namespace_issues:
        findings.append(
            BlobIntegrityFinding(
                kind="invalid_namespace_entries",
                severity="critical",
                count=len(namespace_issues),
                sample=tuple(entry.relative_path for entry in namespace_issues[:_MAX_FINDING_SAMPLE]),
                suggested_action=(
                    "quiesce blob writers and classify the invalid entries; do not delete them without "
                    "a verified cleanup plan and durable receipt"
                ),
            )
        )

    return BlobIntegrityReport(
        full_scan=full,
        sample_size=sample_size,
        scanned_blobs=len(disk_sample),
        scanned_references=len(reference_sample),
        total_blobs_seen=len(disk_sample),
        total_references_seen=len(referenced),
        findings=tuple(findings),
    )


__all__ = [
    "AttachmentCoverageReport",
    "BlobIntegrityFinding",
    "BlobIntegrityKind",
    "BlobIntegrityReport",
    "BlobReferenceDebtClassificationReport",
    "BlobReferenceDebtReport",
    "BlobReferenceDebtSample",
    "BlobLivenessProjection",
    "blob_reference_debt_from_projection",
    "classify_blob_reference_debt",
    "project_source_blob_liveness",
    "referenced_blob_hashes",
    "scan_attachment_coverage",
    "scan_blob_reference_debt",
    "scan_blob_integrity",
]
