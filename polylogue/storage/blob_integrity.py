"""Read-only blob-store integrity probes.

The probes in this module classify blob-store health without deleting files.
Garbage collection remains owned by :mod:`polylogue.storage.blob_gc`; this
surface exists so daemon health and ``polylogue ops doctor`` can report the same
integrity classes with bounded default cost.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

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
    "BlobReferenceDebtReport",
    "referenced_blob_hashes",
    "scan_blob_reference_debt",
    "scan_blob_integrity",
]
