"""Read-only blob-store integrity probes.

The probes in this module classify blob-store health without deleting files.
Garbage collection remains owned by :mod:`polylogue.storage.blob_gc`; this
surface exists so daemon health and ``polylogue ops doctor`` can report the same
integrity classes with bounded default cost.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from polylogue.storage.blob_store import BlobStore, get_blob_store
from polylogue.storage.sqlite.connection import open_read_connection

BlobIntegrityKind = Literal["orphan_blobs", "missing_referenced_blobs", "hash_mismatch", "stale_leases"]
BlobIntegritySeverity = Literal["warning", "critical"]

_DEFAULT_SAMPLE_SIZE = 100
_DEFAULT_STALE_LEASE_S = 3600
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
    active_lease_count: int
    stale_lease_count: int
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
            "active_lease_count": self.active_lease_count,
            "stale_lease_count": self.stale_lease_count,
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
        except sqlite3.Error:
            pass

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
        except sqlite3.Error:
            pass

    fallback_count = len(_raw_session_hashes(conn))
    return {"raw_sessions.raw_id": fallback_count} if fallback_count else {}


def referenced_blob_hashes(db_path: str | Path) -> list[str]:
    """Return distinct blob hashes referenced by archive source evidence."""

    resolved_db_path = Path(db_path)
    with sqlite3.connect(f"file:{resolved_db_path}?mode=ro", uri=True) as conn:
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


def _raw_session_reference_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "raw_sessions"):
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT lower(hex(blob_hash)) AS blob_hash,
               'raw_sessions' AS table_name,
               'raw_payload' AS ref_type,
               raw_id AS ref_id,
               origin,
               native_id,
               source_path,
               blob_size AS size_bytes,
               parse_error,
               validation_status,
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


def classify_blob_reference_debt(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    sample_size: int = 30,
    group_limit: int = 20,
) -> BlobReferenceDebtClassificationReport:
    """Classify missing referenced blobs without mutating archive state."""

    blob_store = store if store is not None else get_blob_store()
    source_db = _source_db_for_blob_reference_report(db_path)
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        raw_refs = _raw_session_reference_rows(conn)
        raw_by_id = {str(row["ref_id"]): row for row in raw_refs if row.get("ref_id")}
        refs = [*raw_refs, *_blob_ref_reference_rows(conn, raw_by_id=raw_by_id)]

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


def _pending_blob_leases(conn: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(conn, "pending_blob_refs"):
        return {}
    rows = conn.execute("SELECT blob_hash, MIN(acquired_at) FROM pending_blob_refs GROUP BY blob_hash").fetchall()
    leases: dict[str, int] = {}
    for blob_hash, acquired_at in rows:
        if blob_hash is None:
            continue
        try:
            leases[str(blob_hash)] = int(acquired_at)
        except (TypeError, ValueError):
            leases[str(blob_hash)] = 0
    return leases


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
) -> BlobReferenceDebtReport:
    """Count missing referenced blob files exactly without mutating state."""

    blob_store = store if store is not None else get_blob_store()
    resolved_db_path = Path(db_path)
    with sqlite3.connect(f"file:{resolved_db_path}?mode=ro", uri=True) as conn:
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


def scan_blob_integrity(
    db_path: str | Path,
    *,
    store: BlobStore | None = None,
    full: bool = False,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    stale_lease_s: int = _DEFAULT_STALE_LEASE_S,
) -> BlobIntegrityReport:
    """Classify blob-store integrity without mutating disk or database state.

    ``full=False`` bounds the filesystem and hash-verification scan to
    ``sample_size`` blobs and references while still loading the reference and
    lease sets needed to avoid false orphan reports. ``full=True`` scans every
    blob and every raw-session reference.
    """

    blob_store = store if store is not None else get_blob_store()
    resolved_db_path = Path(db_path)
    with open_read_connection(resolved_db_path) as conn:
        referenced = _referenced_blob_hashes(resolved_db_path, conn)
        leases = _pending_blob_leases(conn)

    referenced_set = set(referenced)
    active_lease_hashes = set(leases)
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

    orphan_hashes = [
        blob_hash
        for blob_hash in disk_sample
        if blob_hash not in referenced_set and blob_hash not in active_lease_hashes
    ]
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

    now = int(time.time())
    stale_leases = [
        blob_hash
        for blob_hash, acquired_at in leases.items()
        if acquired_at <= 0 or now - acquired_at >= max(0, stale_lease_s)
    ]
    if stale_leases:
        findings.append(
            BlobIntegrityFinding(
                kind="stale_leases",
                severity="warning",
                count=len(stale_leases),
                sample=tuple(stale_leases[:_MAX_FINDING_SAMPLE]),
                suggested_action="inspect the owning ingest process; stale pending_blob_refs usually indicate an interrupted write",
            )
        )

    return BlobIntegrityReport(
        full_scan=full,
        sample_size=sample_size,
        scanned_blobs=len(disk_sample),
        scanned_references=len(reference_sample),
        total_blobs_seen=len(disk_sample),
        total_references_seen=len(referenced),
        active_lease_count=len(leases),
        stale_lease_count=len(stale_leases),
        findings=tuple(findings),
    )


__all__ = [
    "BlobIntegrityFinding",
    "BlobIntegrityKind",
    "BlobIntegrityReport",
    "BlobReferenceDebtClassificationReport",
    "BlobReferenceDebtReport",
    "BlobReferenceDebtSample",
    "classify_blob_reference_debt",
    "referenced_blob_hashes",
    "scan_blob_reference_debt",
    "scan_blob_integrity",
]
