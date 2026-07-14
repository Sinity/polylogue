"""Read-only blob-store integrity probes.

The probes in this module classify blob-store health without deleting files.
Garbage collection remains owned by :mod:`polylogue.storage.blob_gc`; this
surface exists so daemon health and ``polylogue ops doctor`` can report the same
integrity classes with bounded default cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from polylogue.core.json import dumps_bytes as json_dumps_bytes
from polylogue.core.json import loads as json_loads
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.connection import open_read_connection

logger = get_logger(__name__)

BlobIntegrityKind = Literal["orphan_blobs", "missing_referenced_blobs", "hash_mismatch"]
BlobIntegritySeverity = Literal["warning", "critical"]

_DEFAULT_SAMPLE_SIZE = 100
_MAX_FINDING_SAMPLE = 10


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


@dataclass(frozen=True)
class BlobReferenceDebtRestoreSample:
    blob_hash: str
    source_path: str | None
    action: str
    reason: str | None = None
    bytes_restored: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "source_path": self.source_path,
            "action": self.action,
            "reason": self.reason,
            "bytes_restored": self.bytes_restored,
        }


@dataclass(frozen=True)
class BlobReferenceDebtRestoreReport:
    """Dry-run/apply report for direct-file blob debt restoration."""

    source_db: str
    blob_root: str
    dry_run: bool
    missing_distinct_blobs: int
    candidate_count: int
    restored_count: int
    restored_bytes: int
    skipped_existing: int
    skipped_no_source_path: int
    skipped_container_member: int
    skipped_source_missing: int
    skipped_size_mismatch: int
    skipped_hash_mismatch: int
    skipped_error: int
    samples: tuple[BlobReferenceDebtRestoreSample, ...]

    @property
    def ok(self) -> bool:
        return self.skipped_hash_mismatch == 0 and self.skipped_error == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "source_db": self.source_db,
            "blob_root": self.blob_root,
            "dry_run": self.dry_run,
            "missing_distinct_blobs": self.missing_distinct_blobs,
            "candidate_count": self.candidate_count,
            "restored_count": self.restored_count,
            "restored_bytes": self.restored_bytes,
            "skipped_existing": self.skipped_existing,
            "skipped_no_source_path": self.skipped_no_source_path,
            "skipped_container_member": self.skipped_container_member,
            "skipped_source_missing": self.skipped_source_missing,
            "skipped_size_mismatch": self.skipped_size_mismatch,
            "skipped_hash_mismatch": self.skipped_hash_mismatch,
            "skipped_error": self.skipped_error,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class BlobReferenceOrphanPruneSample:
    blob_hash: str
    ref_id: str | None
    ref_type: str | None
    source_path: str | None
    action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "ref_id": self.ref_id,
            "ref_type": self.ref_type,
            "source_path": self.source_path,
            "action": self.action,
        }


@dataclass(frozen=True)
class BlobReferenceOrphanPruneReport:
    """Dry-run/apply report for quarantine-backed orphan blob-ref pruning."""

    source_db: str
    blob_root: str
    dry_run: bool
    quarantine_path: str | None
    scanned_blob_refs: int
    missing_orphan_refs: int
    missing_orphan_distinct_blobs: int
    pruned_refs: int
    pruned_distinct_blobs: int
    skipped_existing_blob: int
    skipped_raw_session_present: int
    samples: tuple[BlobReferenceOrphanPruneSample, ...]

    @property
    def ok(self) -> bool:
        return self.missing_orphan_refs == 0 or (not self.dry_run and self.pruned_refs == self.missing_orphan_refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "source_db": self.source_db,
            "blob_root": self.blob_root,
            "dry_run": self.dry_run,
            "quarantine_path": self.quarantine_path,
            "scanned_blob_refs": self.scanned_blob_refs,
            "missing_orphan_refs": self.missing_orphan_refs,
            "missing_orphan_distinct_blobs": self.missing_orphan_distinct_blobs,
            "pruned_refs": self.pruned_refs,
            "pruned_distinct_blobs": self.pruned_distinct_blobs,
            "skipped_existing_blob": self.skipped_existing_blob,
            "skipped_raw_session_present": self.skipped_raw_session_present,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class BlobReferenceRecoveryPlanRow:
    blob_hash: str
    raw_id: str
    origin: str | None
    native_id: str | None
    source_path: str | None
    source_index: int | None
    expected_size_bytes: int | None
    action: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "raw_id": self.raw_id,
            "origin": self.origin,
            "native_id": self.native_id,
            "source_path": self.source_path,
            "source_index": self.source_index,
            "expected_size_bytes": self.expected_size_bytes,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BlobReferenceRecoveryPlanReport:
    """Read-only plan for raw-session-backed missing referenced blobs."""

    source_db: str
    blob_root: str
    missing_raw_backed_blobs: int
    manifest_path: str | None
    by_action: dict[str, int]
    by_origin: dict[str, int]
    by_source_shape: dict[str, int]
    rows: tuple[BlobReferenceRecoveryPlanRow, ...]
    samples: tuple[BlobReferenceRecoveryPlanRow, ...]

    @property
    def ok(self) -> bool:
        return self.missing_raw_backed_blobs == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "source_db": self.source_db,
            "blob_root": self.blob_root,
            "missing_raw_backed_blobs": self.missing_raw_backed_blobs,
            "manifest_path": self.manifest_path,
            "by_action": dict(self.by_action),
            "by_origin": dict(self.by_origin),
            "by_source_shape": dict(self.by_source_shape),
            "rows": [row.to_dict() for row in self.rows],
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class BlobReferenceSourceReplaceSample:
    raw_id: str
    old_blob_hash: str
    new_blob_hash: str | None
    source_path: str | None
    action: str
    reason: str | None = None
    bytes_written: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_id": self.raw_id,
            "old_blob_hash": self.old_blob_hash,
            "new_blob_hash": self.new_blob_hash,
            "source_path": self.source_path,
            "action": self.action,
            "reason": self.reason,
            "bytes_written": self.bytes_written,
        }


@dataclass(frozen=True)
class BlobReferenceSourceReplaceReport:
    """Current-source replacement report for raw-session-backed blob debt."""

    source_db: str
    blob_root: str
    dry_run: bool
    manifest_path: str | None
    scanned_rows: int
    candidate_rows: int
    replaced_rows: int
    written_blobs: int
    written_bytes: int
    skipped_existing_blob: int
    skipped_no_source_path: int
    skipped_source_missing: int
    skipped_source_index: int
    skipped_unsupported_source: int
    skipped_error: int
    by_origin: dict[str, int]
    by_source_shape: dict[str, int]
    samples: tuple[BlobReferenceSourceReplaceSample, ...]

    @property
    def ok(self) -> bool:
        return self.replaced_rows > 0 if not self.dry_run else self.candidate_rows > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "source_db": self.source_db,
            "blob_root": self.blob_root,
            "dry_run": self.dry_run,
            "manifest_path": self.manifest_path,
            "scanned_rows": self.scanned_rows,
            "candidate_rows": self.candidate_rows,
            "replaced_rows": self.replaced_rows,
            "written_blobs": self.written_blobs,
            "written_bytes": self.written_bytes,
            "skipped_existing_blob": self.skipped_existing_blob,
            "skipped_no_source_path": self.skipped_no_source_path,
            "skipped_source_missing": self.skipped_source_missing,
            "skipped_source_index": self.skipped_source_index,
            "skipped_unsupported_source": self.skipped_unsupported_source,
            "skipped_error": self.skipped_error,
            "by_origin": dict(self.by_origin),
            "by_source_shape": dict(self.by_source_shape),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _blob_hash_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) == 32:
            return value.hex()
        return None
    text = str(value)
    return text if text else None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _archive_source_blob_hashes(conn: sqlite3.Connection) -> list[str]:
    hashes_by_table = _archive_source_blob_hashes_by_table(conn)
    hashes: set[str] = set()
    for table_hashes in hashes_by_table.values():
        hashes.update(table_hashes)
    return sorted(hashes)


def _archive_source_blob_hashes_by_table(conn: sqlite3.Connection) -> dict[str, list[str]]:
    hashes_by_table: dict[str, list[str]] = {}
    if _table_exists(conn, "raw_sessions"):
        rows = conn.execute("SELECT blob_hash FROM raw_sessions").fetchall()
        raw_hashes = {hash_text for row in rows if (hash_text := _blob_hash_text(row[0])) is not None}
        if raw_hashes:
            hashes_by_table["raw_sessions"] = sorted(raw_hashes)
    if _table_exists(conn, "blob_refs"):
        rows = conn.execute("SELECT blob_hash FROM blob_refs").fetchall()
        ref_hashes = {hash_text for row in rows if (hash_text := _blob_hash_text(row[0])) is not None}
        if ref_hashes:
            hashes_by_table["blob_refs"] = sorted(ref_hashes)
    return hashes_by_table


def _raw_session_hashes(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "raw_sessions"):
        return []
    # No ORDER BY: the result is consumed unordered into a set
    # (scan_blob_integrity builds ``set(referenced)``), so sorting the full
    # raw_sessions scan on unindexed ``acquired_at`` was pure overhead.
    rows = conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
    return [str(row[0]) for row in rows if row[0]]


def _referenced_blob_hashes(db_path: Path, conn: sqlite3.Connection) -> list[str]:
    direct_archive_hashes = _archive_source_blob_hashes(conn)
    if direct_archive_hashes:
        return direct_archive_hashes

    source_db = db_path.with_name("source.db")
    if source_db != db_path and source_db.exists():
        try:
            source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
            try:
                source_archive_hashes = _archive_source_blob_hashes(source_conn)
                if source_archive_hashes:
                    return source_archive_hashes
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            # Falls through to _raw_session_hashes(conn), a less-complete
            # reference set — a report consuming this could mis-classify a
            # still-referenced blob as orphaned if source.db's evidence was
            # silently dropped here (polylogue-cpf.4).
            logger.warning(
                "blob integrity: source.db referenced-hash query failed for %s: %s", source_db, exc, exc_info=True
            )

    return _raw_session_hashes(conn)


def _reference_source_counts(db_path: Path, conn: sqlite3.Connection) -> dict[str, int]:
    direct = _archive_source_blob_hashes_by_table(conn)
    if direct:
        return {table: len(hashes) for table, hashes in direct.items()}

    source_db = db_path.with_name("source.db")
    if source_db != db_path and source_db.exists():
        try:
            source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
            try:
                source = _archive_source_blob_hashes_by_table(source_conn)
                if source:
                    return {f"source.db:{table}": len(hashes) for table, hashes in source.items()}
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            logger.warning(
                "blob integrity: source.db reference-count query failed for %s: %s", source_db, exc, exc_info=True
            )

    fallback_count = len(_raw_session_hashes(conn))
    return {"raw_sessions.raw_id": fallback_count} if fallback_count else {}


def referenced_blob_hashes(db_path: str | Path, *, immutable: bool = False) -> list[str]:
    """Return distinct blob hashes referenced by archive source evidence."""

    resolved_db_path = Path(db_path)
    immutable_query = "&immutable=1" if immutable else ""
    with closing(sqlite3.connect(f"file:{resolved_db_path}?mode=ro{immutable_query}", uri=True)) as conn:
        return _referenced_blob_hashes(resolved_db_path, conn)


def _source_db_for_blob_reference_report(db_path: str | Path) -> Path:
    resolved = Path(db_path)
    sibling_source = resolved.with_name("source.db")
    if sibling_source.exists():
        return sibling_source
    return resolved


def _source_path_availability(path: str | None) -> tuple[bool | None, str | None]:
    if not path:
        return None, None
    direct = Path(path)
    if direct.exists():
        return True, str(direct)
    if ":" in path:
        outer, _inner = path.split(":", 1)
        outer_path = Path(outer)
        return outer_path.exists(), str(outer_path)
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


def _blob_ref_acquired_at_column(conn: sqlite3.Connection) -> str:
    if _column_exists(conn, "blob_refs", "acquired_at_ms"):
        return "acquired_at_ms"
    if _column_exists(conn, "blob_refs", "acquired_at"):
        return "acquired_at"
    return "NULL"


def _raw_session_reference_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "raw_sessions"):
        return []
    conn.row_factory = sqlite3.Row
    origin_column = "origin" if _column_exists(conn, "raw_sessions", "origin") else "NULL"
    native_id_column = "native_id" if _column_exists(conn, "raw_sessions", "native_id") else "NULL"
    source_path_column = "source_path" if _column_exists(conn, "raw_sessions", "source_path") else "NULL"
    source_index_column = "source_index" if _column_exists(conn, "raw_sessions", "source_index") else "NULL"
    blob_size_column = "blob_size" if _column_exists(conn, "raw_sessions", "blob_size") else "0"
    parse_error_column = "parse_error" if _column_exists(conn, "raw_sessions", "parse_error") else "NULL"
    validation_status_column = (
        "validation_status" if _column_exists(conn, "raw_sessions", "validation_status") else "NULL"
    )
    rows = conn.execute(
        f"""
        SELECT lower(hex(blob_hash)) AS blob_hash,
               'raw_sessions' AS table_name,
               'raw_payload' AS ref_type,
               raw_id AS ref_id,
               {origin_column} AS origin,
               {native_id_column} AS native_id,
               {source_path_column} AS source_path,
               {source_index_column} AS source_index,
               {blob_size_column} AS size_bytes,
               {parse_error_column} AS parse_error,
               {validation_status_column} AS validation_status,
               1 AS ref_id_has_raw_session
        FROM raw_sessions
        WHERE blob_hash IS NOT NULL
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
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
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

        source_availability = [_source_path_availability(_optional_str(row.get("source_path")))[0] for row in group]
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
            available, outer = _source_path_availability(sample_source_path)
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


def _direct_restore_candidate(group: list[dict[str, Any]]) -> tuple[Path | None, str]:
    saw_source_path = False
    saw_container_member = False
    saw_missing = False
    saw_size_mismatch = False
    for row in group:
        source_path = _optional_str(row.get("source_path"))
        if not source_path:
            continue
        saw_source_path = True
        if _path_is_container_member(source_path):
            saw_container_member = True
            continue
        path = Path(source_path)
        if not path.exists():
            saw_missing = True
            continue
        size = row.get("size_bytes")
        if size is not None and path.stat().st_size != int(size):
            saw_size_mismatch = True
            continue
        return path, "candidate"
    if not saw_source_path:
        return None, "no_source_path"
    if saw_container_member:
        return None, "container_member"
    if saw_missing:
        return None, "source_missing"
    if saw_size_mismatch:
        return None, "size_mismatch"
    return None, "no_candidate"


def _restore_expected_hash_from_path(blob_store: BlobStore, expected_hash: str, source_path: Path) -> tuple[str, int]:
    blob_store.root.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=blob_store.root, prefix=".restore.")
        hasher = hashlib.sha256()
        size = 0
        with source_path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                os.write(fd, chunk)
                size += len(chunk)
        os.close(fd)
        fd = None
        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            return actual_hash, size
        destination = blob_store.blob_path(expected_hash)
        if destination.exists():
            return actual_hash, size
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, destination)
        tmp_path = None
        return actual_hash, size
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _source_span_restore_candidate(group: list[dict[str, Any]]) -> tuple[Path, int, str] | None:
    """Find an exact-hash span candidate inside a larger append-only source.

    Live Claude/Codex session files are append-only JSONL streams. A raw row may
    point at a historical prefix snapshot (``source_index == 0``) or append-tail
    chunk (``source_index == -1``) whose bytes still exist inside the current
    source file even though the full source path is now larger than the stored
    raw blob. Only those two shapes are recovered here, and only after the
    caller verifies the SHA-256 against the expected content address.
    """

    for row in group:
        source_path = _optional_str(row.get("source_path"))
        if not source_path or _path_is_container_member(source_path):
            continue
        expected_size_value = row.get("size_bytes")
        if expected_size_value is None:
            continue
        try:
            expected_size = int(expected_size_value)
        except (TypeError, ValueError):
            continue
        if expected_size <= 0:
            continue
        source_index_value = row.get("source_index")
        if source_index_value is None:
            continue
        try:
            source_index = int(source_index_value)
        except (TypeError, ValueError):
            continue
        if source_index not in (0, -1):
            continue
        path = Path(source_path)
        try:
            source_size = path.stat().st_size
        except OSError:
            continue
        if source_size < expected_size:
            continue
        position = "prefix" if source_index == 0 else "suffix"
        return path, expected_size, position
    return None


def _restore_expected_hash_from_source_span(
    blob_store: BlobStore,
    expected_hash: str,
    source_path: Path,
    *,
    size_bytes: int,
    position: str,
    dry_run: bool,
) -> tuple[str, int]:
    blob_store.root.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: str | None = None
    try:
        if dry_run:
            hasher = hashlib.sha256()
            size = 0
            with source_path.open("rb") as source:
                if position == "suffix":
                    source.seek(-size_bytes, os.SEEK_END)
                remaining = size_bytes
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    hasher.update(chunk)
                    size += len(chunk)
                    remaining -= len(chunk)
            return hasher.hexdigest(), size

        fd, tmp_path = tempfile.mkstemp(dir=blob_store.root, prefix=".restore.")
        hasher = hashlib.sha256()
        size = 0
        with source_path.open("rb") as source:
            if position == "suffix":
                source.seek(-size_bytes, os.SEEK_END)
            remaining = size_bytes
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                hasher.update(chunk)
                os.write(fd, chunk)
                size += len(chunk)
                remaining -= len(chunk)
        os.close(fd)
        fd = None
        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            return actual_hash, size
        destination = blob_store.blob_path(expected_hash)
        if destination.exists():
            return actual_hash, size
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, destination)
        tmp_path = None
        return actual_hash, size
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _blob_ref_rows_for_orphan_prune(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "blob_refs"):
        return []
    conn.row_factory = sqlite3.Row
    ref_id_column = _blob_ref_id_column(conn)
    source_path_column = _blob_ref_source_path_column(conn)
    size_column = _blob_ref_size_column(conn)
    acquired_at_column = _blob_ref_acquired_at_column(conn)
    raw_join = (
        f"EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = {ref_id_column})"
        if _table_exists(conn, "raw_sessions") and ref_id_column != "NULL"
        else "0"
    )
    rows = conn.execute(
        f"""
        SELECT lower(hex(blob_hash)) AS blob_hash,
               ref_type,
               {ref_id_column} AS ref_id,
               {source_path_column} AS source_path,
               {size_column} AS size_bytes,
               {acquired_at_column} AS acquired_at,
               {raw_join} AS raw_session_present
        FROM blob_refs
        WHERE blob_hash IS NOT NULL
        ORDER BY source_path, ref_id, ref_type, blob_hash
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _missing_raw_backed_blob_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "raw_sessions"):
        return []
    conn.row_factory = sqlite3.Row
    origin_column = "origin" if _column_exists(conn, "raw_sessions", "origin") else "NULL"
    native_id_column = "native_id" if _column_exists(conn, "raw_sessions", "native_id") else "NULL"
    source_path_column = "source_path" if _column_exists(conn, "raw_sessions", "source_path") else "NULL"
    source_index_column = "source_index" if _column_exists(conn, "raw_sessions", "source_index") else "NULL"
    blob_size_column = "blob_size" if _column_exists(conn, "raw_sessions", "blob_size") else "NULL"
    acquired_at_ms_column = "acquired_at_ms" if _column_exists(conn, "raw_sessions", "acquired_at_ms") else "NULL"
    file_mtime_ms_column = "file_mtime_ms" if _column_exists(conn, "raw_sessions", "file_mtime_ms") else "NULL"
    rows = conn.execute(
        f"""
        SELECT lower(hex(blob_hash)) AS blob_hash,
               raw_id,
               {origin_column} AS origin,
               {native_id_column} AS native_id,
               {source_path_column} AS source_path,
               {source_index_column} AS source_index,
               {blob_size_column} AS expected_size_bytes,
               {acquired_at_ms_column} AS acquired_at_ms,
               {file_mtime_ms_column} AS file_mtime_ms
        FROM raw_sessions
        WHERE blob_hash IS NOT NULL
        ORDER BY origin, source_path, source_index, raw_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _raw_backed_recovery_action(blob_hash: str, source_path: str | None, expected_size: int | None) -> tuple[str, str]:
    if not source_path:
        return "no_source_path", "raw row does not record a source path"
    if _path_is_container_member(source_path):
        outer_path, _inner = source_path.split(":", 1)
        if not Path(outer_path).exists():
            return "source_missing", "container source path is missing"
        return "container_member_reacquire_required", "container member requires exact source-aware re-acquisition"
    path = Path(source_path)
    if not path.exists():
        return "source_missing", "source file is missing"
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        return "direct_source_size_mismatch", f"source size is {actual_size}, expected {expected_size}"
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    if actual_hash == blob_hash:
        return "direct_exact_restore_candidate", "source bytes match the expected blob hash"
    return "direct_source_hash_mismatch", f"source hash is {actual_hash}"


def _split_container_source_path(source_path: str) -> tuple[Path, str] | None:
    outer, sep, member = source_path.partition(":")
    if not sep or not outer or not member:
        return None
    return Path(outer), member


def _json_payload_at_index(raw_bytes: bytes, source_index: int) -> object:
    loaded = json_loads(raw_bytes)
    if isinstance(loaded, list):
        try:
            return loaded[source_index]
        except IndexError as exc:
            raise IndexError(f"source_index {source_index} outside JSON array of {len(loaded)} payloads") from exc
    if source_index == 0:
        return loaded
    raise IndexError("non-array JSON payload only supports source_index 0")


def _jsonl_payload_at_index(raw_bytes: bytes, source_index: int) -> object:
    for idx, line in enumerate(raw_bytes.splitlines()):
        if idx == source_index:
            return json_loads(line)
    raise IndexError(f"source_index {source_index} outside JSONL stream")


def _current_raw_payload_bytes(
    source_path: str,
    source_index: int | None,
    *,
    source_bytes_cache: dict[str, bytes] | None = None,
    decoded_payload_cache: dict[str, object] | None = None,
) -> tuple[bytes | None, str | None]:
    if _path_is_container_member(source_path):
        split = _split_container_source_path(source_path)
        if split is None:
            return None, "unsupported_container_path"
        zip_path, member = split
        if not zip_path.exists():
            return None, "source_missing"
        try:
            if source_bytes_cache is not None and source_path in source_bytes_cache:
                member_bytes = source_bytes_cache[source_path]
            else:
                with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
                    member_bytes = handle.read()
                if source_bytes_cache is not None:
                    source_bytes_cache[source_path] = member_bytes
        except KeyError:
            return None, "source_missing"
        if source_index is None:
            return None, "source_index_missing"
        try:
            if decoded_payload_cache is not None and source_path in decoded_payload_cache:
                decoded_payload = decoded_payload_cache[source_path]
                if isinstance(decoded_payload, list):
                    payload = decoded_payload[int(source_index)]
                elif int(source_index) == 0:
                    payload = decoded_payload
                else:
                    raise IndexError("non-array JSON payload only supports source_index 0")
            else:
                if member.endswith(".jsonl"):
                    payload = _jsonl_payload_at_index(member_bytes, int(source_index))
                else:
                    decoded_payload = json_loads(member_bytes)
                    if decoded_payload_cache is not None:
                        decoded_payload_cache[source_path] = decoded_payload
                    if isinstance(decoded_payload, list):
                        payload = decoded_payload[int(source_index)]
                    elif int(source_index) == 0:
                        payload = decoded_payload
                    else:
                        raise IndexError("non-array JSON payload only supports source_index 0")
        except (IndexError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return None, f"source_index:{exc}"
        return json_dumps_bytes(payload), None

    path = Path(source_path)
    if not path.exists():
        return None, "source_missing"
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, f"error:{exc}"


def _delete_blob_refs_for_raw_id(conn: sqlite3.Connection, raw_id: str) -> None:
    ref_id_column = "ref_id" if _column_exists(conn, "blob_refs", "ref_id") else "raw_id"
    conn.execute(f"DELETE FROM blob_refs WHERE {ref_id_column} = ?", (raw_id,))


def _insert_raw_payload_blob_ref(
    conn: sqlite3.Connection,
    *,
    blob_hash: str,
    raw_id: str,
    source_path: str,
    size_bytes: int,
    acquired_at_ms: int,
) -> None:
    ref_id_column = "ref_id" if _column_exists(conn, "blob_refs", "ref_id") else "raw_id"
    columns = ["blob_hash", ref_id_column, "ref_type"]
    values: list[object] = [bytes.fromhex(blob_hash), raw_id, "raw_payload"]
    if _column_exists(conn, "blob_refs", "source_path"):
        columns.append("source_path")
        values.append(source_path)
    if _column_exists(conn, "blob_refs", "size_bytes"):
        columns.append("size_bytes")
        values.append(size_bytes)
    if _column_exists(conn, "blob_refs", "acquired_at_ms"):
        columns.append("acquired_at_ms")
        values.append(acquired_at_ms)
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO blob_refs ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values),
    )


def _update_raw_session_blob_ref(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    new_blob_hash: str,
    new_blob_size: int,
    acquired_at_ms: int,
    file_mtime_ms: int | None,
) -> None:
    assignments = ["blob_hash = ?", "blob_size = ?"]
    values: list[object] = [bytes.fromhex(new_blob_hash), new_blob_size]
    if _column_exists(conn, "raw_sessions", "acquired_at_ms"):
        assignments.append("acquired_at_ms = ?")
        values.append(acquired_at_ms)
    if file_mtime_ms is not None and _column_exists(conn, "raw_sessions", "file_mtime_ms"):
        assignments.append("file_mtime_ms = ?")
        values.append(file_mtime_ms)
    values.append(raw_id)
    conn.execute(f"UPDATE raw_sessions SET {', '.join(assignments)} WHERE raw_id = ?", tuple(values))


def _default_blob_ref_quarantine_path(source_db: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return source_db.parent / ".maintenance-state" / "blob-ref-quarantine" / f"orphan-blob-refs-{stamp}.jsonl"


def _write_blob_ref_quarantine(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_jsonl_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _delete_blob_ref_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    ref_id_column = _blob_ref_id_column(conn)
    if ref_id_column == "NULL":
        return 0
    deleted = 0
    for row in rows:
        blob_hash = str(row["blob_hash"])
        ref_id = row.get("ref_id")
        ref_type = row.get("ref_type")
        before = conn.total_changes
        conn.execute(
            f"""
            DELETE FROM blob_refs
            WHERE blob_hash = ?
              AND {ref_id_column} IS ?
              AND ref_type IS ?
            """,
            (bytes.fromhex(blob_hash), ref_id, ref_type),
        )
        deleted += conn.total_changes - before
    return deleted


def prune_orphan_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    dry_run: bool = True,
    quarantine_path: str | Path | None = None,
    max_count: int | None = None,
    sample_size: int = 30,
) -> BlobReferenceOrphanPruneReport:
    """Prune stale missing ``blob_refs`` only after writing quarantine JSONL.

    This intentionally does not touch refs whose ``ref_id`` still has a
    ``raw_sessions`` row, because those rows may still be reconstructable from
    source exports or browser-capture files.
    """

    source_db = _source_db_for_blob_reference_report(db_path)
    blob_store = store if store is not None else BlobStore(source_db.parent / "blob")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as read_conn:
        rows = _blob_ref_rows_for_orphan_prune(read_conn)

    skipped_existing_blob = 0
    skipped_raw_session_present = 0
    candidates: list[dict[str, Any]] = []
    for row in rows:
        blob_hash = str(row.get("blob_hash") or "")
        if not blob_hash or blob_store.exists(blob_hash):
            skipped_existing_blob += 1
            continue
        if bool(row.get("raw_session_present")):
            skipped_raw_session_present += 1
            continue
        candidates.append(row)

    limited_candidates = candidates if max_count is None else candidates[: max(0, max_count)]
    samples = tuple(
        BlobReferenceOrphanPruneSample(
            blob_hash=str(row.get("blob_hash") or ""),
            ref_id=_optional_str(row.get("ref_id")),
            ref_type=_optional_str(row.get("ref_type")),
            source_path=_optional_str(row.get("source_path")),
            action="would_prune" if dry_run else "pruned",
        )
        for row in limited_candidates[: max(0, sample_size)]
    )

    resolved_quarantine_path: Path | None = None
    pruned_refs = 0
    pruned_distinct_blobs = 0
    if not dry_run and limited_candidates:
        resolved_quarantine_path = (
            Path(quarantine_path) if quarantine_path is not None else _default_blob_ref_quarantine_path(source_db)
        )
        _write_blob_ref_quarantine(resolved_quarantine_path, limited_candidates)
        with closing(sqlite3.connect(source_db)) as write_conn:
            pruned_refs = _delete_blob_ref_rows(write_conn, limited_candidates)
            write_conn.commit()
        pruned_distinct_blobs = len({str(row["blob_hash"]) for row in limited_candidates})
    elif quarantine_path is not None:
        resolved_quarantine_path = Path(quarantine_path)

    return BlobReferenceOrphanPruneReport(
        source_db=str(source_db),
        blob_root=str(blob_store.root),
        dry_run=dry_run,
        quarantine_path=str(resolved_quarantine_path) if resolved_quarantine_path is not None else None,
        scanned_blob_refs=len(rows),
        missing_orphan_refs=len(candidates),
        missing_orphan_distinct_blobs=len({str(row["blob_hash"]) for row in candidates}),
        pruned_refs=pruned_refs,
        pruned_distinct_blobs=pruned_distinct_blobs,
        skipped_existing_blob=skipped_existing_blob,
        skipped_raw_session_present=skipped_raw_session_present,
        samples=samples,
    )


def plan_raw_backed_blob_reference_recovery(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    manifest_path: str | Path | None = None,
    sample_size: int = 30,
    include_rows: bool = False,
) -> BlobReferenceRecoveryPlanReport:
    """Classify raw-backed missing blob refs without mutating archive state."""

    source_db = _source_db_for_blob_reference_report(db_path)
    blob_store = store if store is not None else BlobStore(source_db.parent / "blob")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        rows = _missing_raw_backed_blob_rows(conn)

    plan_rows: list[BlobReferenceRecoveryPlanRow] = []
    by_action: Counter[str] = Counter()
    by_origin: Counter[str] = Counter()
    by_source_shape: Counter[str] = Counter()
    for row in rows:
        blob_hash = str(row.get("blob_hash") or "")
        if not blob_hash or blob_store.exists(blob_hash):
            continue
        source_path = _optional_str(row.get("source_path"))
        source_shape = "container_member" if source_path and _path_is_container_member(source_path) else "direct_file"
        expected_size_value = row.get("expected_size_bytes")
        expected_size = int(expected_size_value) if expected_size_value is not None else None
        action, reason = _raw_backed_recovery_action(blob_hash, source_path, expected_size)
        origin = _optional_str(row.get("origin"))
        plan_row = BlobReferenceRecoveryPlanRow(
            blob_hash=blob_hash,
            raw_id=str(row.get("raw_id") or ""),
            origin=origin,
            native_id=_optional_str(row.get("native_id")),
            source_path=source_path,
            source_index=int(row["source_index"]) if row.get("source_index") is not None else None,
            expected_size_bytes=expected_size,
            action=action,
            reason=reason,
        )
        plan_rows.append(plan_row)
        by_action[action] += 1
        by_origin[origin or "(none)"] += 1
        by_source_shape[source_shape] += 1

    resolved_manifest_path: Path | None = Path(manifest_path) if manifest_path is not None else None
    if resolved_manifest_path is not None:
        _write_jsonl_rows(resolved_manifest_path, (row.to_dict() for row in plan_rows))

    return BlobReferenceRecoveryPlanReport(
        source_db=str(source_db),
        blob_root=str(blob_store.root),
        missing_raw_backed_blobs=len(plan_rows),
        manifest_path=str(resolved_manifest_path) if resolved_manifest_path is not None else None,
        by_action=_counter_dict(by_action),
        by_origin=_counter_dict(by_origin),
        by_source_shape=_counter_dict(by_source_shape),
        rows=tuple(plan_rows) if include_rows else (),
        samples=tuple(plan_rows[: max(0, sample_size)]),
    )


def replace_raw_backed_blob_reference_debt_from_source(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    dry_run: bool = True,
    manifest_path: str | Path | None = None,
    max_count: int | None = None,
    sample_size: int = 30,
) -> BlobReferenceSourceReplaceReport:
    """Replace missing raw-payload blob refs with current source-derived bytes.

    This does not restore the historical missing blob. It records the old hash
    in the manifest, writes the current source payload to the blob store, and
    updates the existing raw_id to point at that current evidence.
    """

    if not dry_run and manifest_path is None:
        raise ValueError("manifest_path is required when applying source replacement")

    source_db = _source_db_for_blob_reference_report(db_path)
    blob_store = store if store is not None else BlobStore(source_db.parent / "blob")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        rows = _missing_raw_backed_blob_rows(conn)

    samples: list[BlobReferenceSourceReplaceSample] = []
    manifest_rows: list[dict[str, object]] = []
    by_origin: Counter[str] = Counter()
    by_source_shape: Counter[str] = Counter()
    candidate_updates: list[tuple[dict[str, Any], str, int, int | None, int]] = []
    skipped_existing_blob = 0
    skipped_no_source_path = 0
    skipped_source_missing = 0
    skipped_source_index = 0
    skipped_unsupported_source = 0
    skipped_error = 0
    source_bytes_cache: dict[str, bytes] = {}
    decoded_payload_cache: dict[str, object] = {}

    for row in rows:
        old_blob_hash = str(row.get("blob_hash") or "")
        raw_id = str(row.get("raw_id") or "")
        if not old_blob_hash or not raw_id:
            continue
        if blob_store.exists(old_blob_hash):
            skipped_existing_blob += 1
            continue
        if max_count is not None and len(candidate_updates) >= max_count:
            break

        source_path = _optional_str(row.get("source_path"))
        origin = _optional_str(row.get("origin")) or "(none)"
        by_origin[origin] += 1
        source_shape = "container_member" if source_path and _path_is_container_member(source_path) else "direct_file"
        by_source_shape[source_shape] += 1
        if not source_path:
            skipped_no_source_path += 1
            reason: str | None = "no_source_path"
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceSourceReplaceSample(raw_id, old_blob_hash, None, source_path, "skipped", reason)
                )
            continue

        try:
            payload_bytes, reason = _current_raw_payload_bytes(
                source_path,
                int(row["source_index"]) if row.get("source_index") is not None else None,
                source_bytes_cache=source_bytes_cache,
                decoded_payload_cache=decoded_payload_cache,
            )
        except OSError as exc:
            payload_bytes, reason = None, f"error:{exc}"
        if payload_bytes is None:
            if reason == "source_missing":
                skipped_source_missing += 1
            elif reason and reason.startswith("source_index"):
                skipped_source_index += 1
            elif reason and reason.startswith("error:"):
                skipped_error += 1
            else:
                skipped_unsupported_source += 1
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceSourceReplaceSample(raw_id, old_blob_hash, None, source_path, "skipped", reason)
                )
            manifest_rows.append(
                {
                    "action": "skipped",
                    "reason": reason,
                    "raw_id": raw_id,
                    "old_blob_hash": old_blob_hash,
                    "origin": row.get("origin"),
                    "native_id": row.get("native_id"),
                    "source_path": source_path,
                    "source_index": row.get("source_index"),
                    "old_blob_size": row.get("expected_size_bytes"),
                }
            )
            continue

        new_blob_hash = hashlib.sha256(payload_bytes).hexdigest()
        new_blob_size = len(payload_bytes)
        acquired_at_ms = int(time.time() * 1000)
        base_path = source_path.partition(":")[0] if _path_is_container_member(source_path) else source_path
        try:
            file_mtime_ms: int | None = int(Path(base_path).stat().st_mtime * 1000)
        except OSError:
            file_mtime_ms = None
        manifest_row = {
            "action": "would_replace" if dry_run else "replaced",
            "raw_id": raw_id,
            "origin": row.get("origin"),
            "native_id": row.get("native_id"),
            "source_path": source_path,
            "source_index": row.get("source_index"),
            "old_blob_hash": old_blob_hash,
            "old_blob_size": row.get("expected_size_bytes"),
            "new_blob_hash": new_blob_hash,
            "new_blob_size": new_blob_size,
            "new_equals_old": new_blob_hash == old_blob_hash,
        }
        manifest_rows.append(manifest_row)
        candidate_updates.append((row, new_blob_hash, new_blob_size, file_mtime_ms, acquired_at_ms))
        if len(samples) < max(0, sample_size):
            samples.append(
                BlobReferenceSourceReplaceSample(
                    raw_id=raw_id,
                    old_blob_hash=old_blob_hash,
                    new_blob_hash=new_blob_hash,
                    source_path=source_path,
                    action="would_replace" if dry_run else "replaced",
                    bytes_written=new_blob_size,
                )
            )

    resolved_manifest_path: Path | None = Path(manifest_path) if manifest_path is not None else None
    if resolved_manifest_path is not None:
        _write_jsonl_rows(resolved_manifest_path, manifest_rows)

    replaced_rows = 0
    written_blobs = 0
    written_bytes = 0
    if not dry_run and candidate_updates:
        from polylogue.storage.blob_publication import (
            ArchiveBlobPublisher,
            consume_blob_publication_receipt,
        )

        apply_source_bytes_cache: dict[str, bytes] = {}
        apply_decoded_payload_cache: dict[str, object] = {}
        publisher = ArchiveBlobPublisher(source_db, blob_store.root, store=blob_store)
        publication_receipts: list[str | None] = []
        for row, new_blob_hash, _new_blob_size, _file_mtime_ms, _acquired_at_ms in candidate_updates:
            receipt_id: str | None = None
            existed_before_publication = blob_store.exists(new_blob_hash)
            source_path = str(row["source_path"])
            payload_bytes, _reason = _current_raw_payload_bytes(
                source_path,
                int(row["source_index"]) if row.get("source_index") is not None else None,
                source_bytes_cache=apply_source_bytes_cache,
                decoded_payload_cache=apply_decoded_payload_cache,
            )
            if payload_bytes is not None:
                published_hash, _published_size = publisher.write_from_bytes(payload_bytes)
                receipt_id = publisher.receipt_id(published_hash)
                if not existed_before_publication:
                    written_blobs += 1
                    written_bytes += len(payload_bytes)
            publication_receipts.append(receipt_id)
        publisher.flush()
        with closing(sqlite3.connect(source_db)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                for (
                    row,
                    new_blob_hash,
                    new_blob_size,
                    file_mtime_ms,
                    acquired_at_ms,
                ), publication_receipt_id in zip(candidate_updates, publication_receipts, strict=True):
                    raw_id = str(row["raw_id"])
                    source_path = str(row["source_path"])
                    if not blob_store.exists(new_blob_hash):
                        skipped_error += 1
                        continue
                    _delete_blob_refs_for_raw_id(conn, raw_id)
                    _update_raw_session_blob_ref(
                        conn,
                        raw_id=raw_id,
                        new_blob_hash=new_blob_hash,
                        new_blob_size=new_blob_size,
                        acquired_at_ms=acquired_at_ms,
                        file_mtime_ms=file_mtime_ms,
                    )
                    _insert_raw_payload_blob_ref(
                        conn,
                        blob_hash=new_blob_hash,
                        raw_id=raw_id,
                        source_path=source_path,
                        size_bytes=new_blob_size,
                        acquired_at_ms=acquired_at_ms,
                    )
                    consume_blob_publication_receipt(
                        conn,
                        publication_receipt_id,
                        bytes.fromhex(new_blob_hash),
                    )
                    replaced_rows += 1

    return BlobReferenceSourceReplaceReport(
        source_db=str(source_db),
        blob_root=str(blob_store.root),
        dry_run=dry_run,
        manifest_path=str(resolved_manifest_path) if resolved_manifest_path is not None else None,
        scanned_rows=len(rows),
        candidate_rows=len(candidate_updates),
        replaced_rows=replaced_rows,
        written_blobs=written_blobs,
        written_bytes=written_bytes,
        skipped_existing_blob=skipped_existing_blob,
        skipped_no_source_path=skipped_no_source_path,
        skipped_source_missing=skipped_source_missing,
        skipped_source_index=skipped_source_index,
        skipped_unsupported_source=skipped_unsupported_source,
        skipped_error=skipped_error,
        by_origin=_counter_dict(by_origin),
        by_source_shape=_counter_dict(by_source_shape),
        samples=tuple(samples),
    )


def restore_direct_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    dry_run: bool = True,
    max_count: int | None = None,
    sample_size: int = 30,
) -> BlobReferenceDebtRestoreReport:
    """Restore direct-file missing blob refs after exact hash verification."""

    source_db = _source_db_for_blob_reference_report(db_path)
    blob_store = store if store is not None else BlobStore(source_db.parent / "blob")
    refs = _reference_rows_for_blob_debt(source_db)
    by_hash = _group_reference_rows(refs)
    missing = {blob_hash: group for blob_hash, group in by_hash.items() if not blob_store.exists(blob_hash)}

    candidate_count = 0
    restored_count = 0
    restored_bytes = 0
    skipped_existing = 0
    skipped_no_source_path = 0
    skipped_container_member = 0
    skipped_source_missing = 0
    skipped_size_mismatch = 0
    skipped_hash_mismatch = 0
    skipped_error = 0
    samples: list[BlobReferenceDebtRestoreSample] = []

    for blob_hash, group in missing.items():
        if max_count is not None and candidate_count >= max_count:
            break
        if blob_store.exists(blob_hash):
            skipped_existing += 1
            continue
        source_path, reason = _direct_restore_candidate(group)
        if source_path is None:
            span_candidate = _source_span_restore_candidate(group) if reason == "size_mismatch" else None
            if span_candidate is None:
                if reason == "no_source_path":
                    skipped_no_source_path += 1
                elif reason == "container_member":
                    skipped_container_member += 1
                elif reason == "source_missing":
                    skipped_source_missing += 1
                elif reason == "size_mismatch":
                    skipped_size_mismatch += 1
                else:
                    skipped_no_source_path += 1
                if len(samples) < max(0, sample_size):
                    samples.append(
                        BlobReferenceDebtRestoreSample(
                            blob_hash=blob_hash,
                            source_path=None,
                            action="skipped",
                            reason=reason,
                        )
                    )
                continue

            source_path, span_size, span_position = span_candidate
            candidate_count += 1
            try:
                actual_hash, size = _restore_expected_hash_from_source_span(
                    blob_store,
                    blob_hash,
                    source_path,
                    size_bytes=span_size,
                    position=span_position,
                    dry_run=dry_run,
                )
            except OSError as exc:
                skipped_error += 1
                if len(samples) < max(0, sample_size):
                    samples.append(
                        BlobReferenceDebtRestoreSample(
                            blob_hash=blob_hash,
                            source_path=str(source_path),
                            action="skipped",
                            reason=f"error:{exc}",
                        )
                    )
                continue
            if actual_hash != blob_hash:
                skipped_hash_mismatch += 1
                if len(samples) < max(0, sample_size):
                    samples.append(
                        BlobReferenceDebtRestoreSample(
                            blob_hash=blob_hash,
                            source_path=str(source_path),
                            action="skipped",
                            reason=f"{span_position}_hash_mismatch:{actual_hash}",
                        )
                    )
                continue
            if dry_run:
                if len(samples) < max(0, sample_size):
                    samples.append(
                        BlobReferenceDebtRestoreSample(
                            blob_hash=blob_hash,
                            source_path=str(source_path),
                            action="would_restore",
                            reason=span_position,
                            bytes_restored=size,
                        )
                    )
                continue
            restored_count += 1
            restored_bytes += size
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceDebtRestoreSample(
                        blob_hash=blob_hash,
                        source_path=str(source_path),
                        action="restored",
                        reason=span_position,
                        bytes_restored=size,
                    )
                )
            continue

        candidate_count += 1
        if dry_run:
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceDebtRestoreSample(
                        blob_hash=blob_hash,
                        source_path=str(source_path),
                        action="would_restore",
                    )
                )
            continue

        try:
            actual_hash, size = _restore_expected_hash_from_path(blob_store, blob_hash, source_path)
        except OSError as exc:
            skipped_error += 1
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceDebtRestoreSample(
                        blob_hash=blob_hash,
                        source_path=str(source_path),
                        action="skipped",
                        reason=f"error:{exc}",
                    )
                )
            continue
        if actual_hash != blob_hash:
            skipped_hash_mismatch += 1
            if len(samples) < max(0, sample_size):
                samples.append(
                    BlobReferenceDebtRestoreSample(
                        blob_hash=blob_hash,
                        source_path=str(source_path),
                        action="skipped",
                        reason=f"hash_mismatch:{actual_hash}",
                    )
                )
            continue
        restored_count += 1
        restored_bytes += size
        if len(samples) < max(0, sample_size):
            samples.append(
                BlobReferenceDebtRestoreSample(
                    blob_hash=blob_hash,
                    source_path=str(source_path),
                    action="restored",
                    bytes_restored=size,
                )
            )

    return BlobReferenceDebtRestoreReport(
        source_db=str(source_db),
        blob_root=str(blob_store.root),
        dry_run=dry_run,
        missing_distinct_blobs=len(missing),
        candidate_count=candidate_count,
        restored_count=restored_count,
        restored_bytes=restored_bytes,
        skipped_existing=skipped_existing,
        skipped_no_source_path=skipped_no_source_path,
        skipped_container_member=skipped_container_member,
        skipped_source_missing=skipped_source_missing,
        skipped_size_mismatch=skipped_size_mismatch,
        skipped_hash_mismatch=skipped_hash_mismatch,
        skipped_error=skipped_error,
        samples=tuple(samples),
    )


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


def scan_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = _MAX_FINDING_SAMPLE,
    immutable: bool = False,
) -> BlobReferenceDebtReport:
    """Count missing referenced blob files exactly without mutating state."""

    resolved_db_path = Path(db_path)
    blob_store = store if store is not None else BlobStore(resolved_db_path.parent / "blob")
    immutable_query = "&immutable=1" if immutable else ""
    with closing(sqlite3.connect(f"file:{resolved_db_path}?mode=ro{immutable_query}", uri=True)) as conn:
        referenced = _referenced_blob_hashes(resolved_db_path, conn)
        reference_sources = _reference_source_counts(resolved_db_path, conn)

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
class AttachmentAcquisitionDebtReport:
    """Index-tier attachment acquisition state (#83u.4).

    Deliberately separate from :class:`BlobReferenceDebtReport`, which
    classifies source-tier backup debt from ``source.db``/``raw_sessions``
    references and never queries the index-tier ``attachments`` table.
    ``unfetched`` (``blob_hash IS NULL``) is an honest floor -- bytes were
    never fetched, not a broken reference -- and must never be conflated
    with missing referenced blobs. Only an ``acquired`` row whose blob file
    is absent from the store is genuine attachment acquisition debt.
    """

    total_attachments: int
    acquired_count: int
    acquired_missing_blob_count: int
    acquired_missing_blob_sample: tuple[str, ...]
    unavailable_count: int
    unfetched_count: int

    @property
    def ok(self) -> bool:
        return self.acquired_missing_blob_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "total_attachments": self.total_attachments,
            "acquired_count": self.acquired_count,
            "acquired_missing_blob_count": self.acquired_missing_blob_count,
            "acquired_missing_blob_sample": list(self.acquired_missing_blob_sample),
            "unavailable_count": self.unavailable_count,
            "unfetched_count": self.unfetched_count,
        }


def scan_attachment_acquisition_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = _MAX_FINDING_SAMPLE,
) -> AttachmentAcquisitionDebtReport:
    """Classify index-tier attachment acquisition debt without mutating state.

    ``db_path`` is the index-tier database (``index.db``), not ``source.db``.
    """

    resolved_db_path = Path(db_path)
    blob_store = store if store is not None else BlobStore(resolved_db_path.parent / "blob")
    with closing(sqlite3.connect(f"file:{resolved_db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        status_counts: dict[str, int] = dict(
            conn.execute("SELECT acquisition_status, COUNT(*) FROM attachments GROUP BY acquisition_status").fetchall()
        )
        acquired_rows = conn.execute(
            "SELECT attachment_id, blob_hash FROM attachments WHERE acquisition_status = 'acquired'"
        ).fetchall()

    missing_sample: list[str] = []
    missing_count = 0
    for row in acquired_rows:
        blob_hash = row["blob_hash"]
        if blob_hash is None:
            continue
        hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        if blob_store.exists(hash_hex):
            continue
        missing_count += 1
        if len(missing_sample) < sample_size:
            missing_sample.append(str(row["attachment_id"]))

    return AttachmentAcquisitionDebtReport(
        total_attachments=sum(status_counts.values()),
        acquired_count=status_counts.get("acquired", 0),
        acquired_missing_blob_count=missing_count,
        acquired_missing_blob_sample=tuple(missing_sample),
        unavailable_count=status_counts.get("unavailable", 0),
        unfetched_count=status_counts.get("unfetched", 0),
    )


def scan_blob_integrity(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    full: bool = False,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> BlobIntegrityReport:
    """Classify blob-store integrity without mutating disk or database state.

    ``full=False`` bounds the filesystem and hash-verification scan to
    ``sample_size`` blobs and references. ``full=True`` scans every blob and
    every raw-session reference.
    """

    resolved_db_path = Path(db_path)
    blob_store = store if store is not None else BlobStore(resolved_db_path.parent / "blob")
    with open_read_connection(resolved_db_path) as conn:
        referenced = _referenced_blob_hashes(resolved_db_path, conn)

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
                suggested_action="preview cleanup with `polylogue ops doctor --cleanup --preview --target orphaned_blobs`",
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
    "AttachmentAcquisitionDebtReport",
    "BlobIntegrityFinding",
    "BlobIntegrityKind",
    "BlobIntegrityReport",
    "BlobReferenceDebtClassificationReport",
    "BlobReferenceOrphanPruneSample",
    "BlobReferenceOrphanPruneReport",
    "BlobReferenceRecoveryPlanReport",
    "BlobReferenceRecoveryPlanRow",
    "BlobReferenceSourceReplaceReport",
    "BlobReferenceSourceReplaceSample",
    "BlobReferenceDebtReport",
    "BlobReferenceDebtRestoreReport",
    "BlobReferenceDebtRestoreSample",
    "BlobReferenceDebtSample",
    "classify_blob_reference_debt",
    "plan_raw_backed_blob_reference_recovery",
    "prune_orphan_blob_reference_debt",
    "referenced_blob_hashes",
    "replace_raw_backed_blob_reference_debt_from_source",
    "restore_direct_blob_reference_debt",
    "scan_attachment_acquisition_debt",
    "scan_blob_reference_debt",
    "scan_blob_integrity",
]
