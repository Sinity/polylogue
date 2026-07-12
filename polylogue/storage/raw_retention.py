"""Retention cleanup for superseded live raw payload snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from polylogue.storage.blob_store import BlobStore, get_blob_store

_V1_RAW_CANDIDATE_SQL = """
WITH ranked AS (
    SELECT
        raw_id,
        source_path,
        source_index,
        blob_hash,
        blob_size,
        acquired_at_ms,
        ROW_NUMBER() OVER (
            PARTITION BY source_path, source_index
            ORDER BY acquired_at_ms DESC, raw_id DESC
        ) AS recency
    FROM raw_sessions
    WHERE source_index IN (-1, 0)
      AND (? IS NULL OR source_path = ?)
      AND (? IS NULL OR acquired_at_ms >= ?)
)
SELECT raw_id, source_path, source_index, blob_hash, blob_size
FROM ranked AS r
WHERE r.recency > CASE WHEN r.source_index = 0 THEN ? ELSE ? END
ORDER BY r.blob_size DESC, r.acquired_at_ms ASC, r.raw_id ASC
LIMIT ?
"""

_RAW_REVISION_CHAIN_COLUMNS = """
raw_id, source_index, logical_source_key, revision_kind, source_revision,
predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
append_start_offset, append_end_offset, acquisition_generation,
revision_authority, blob_size
"""


class RawRetentionSafetyError(RuntimeError):
    """Raised when active raw evidence cannot be proven safe for retention."""


@dataclass(frozen=True)
class RawSnapshotCleanupCandidate:
    raw_id: str
    source_path: str
    source_index: int
    blob_size: int
    blob_hash: str | None = None

    @property
    def blob_store_hash(self) -> str:
        return self.blob_hash or self.raw_id


@dataclass(frozen=True)
class RawSnapshotCleanupResult:
    candidate_count: int
    deleted_raw_count: int
    deleted_blob_count: int
    deleted_raw_bytes: int
    deleted_blob_bytes: int
    skipped_missing_source_count: int
    skipped_referenced_count: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawRetentionAuthority:
    """Index-proven source rows to preserve and rows authorized for deletion."""

    protected_raw_ids: frozenset[str]
    eligible_raw_ids: frozenset[str]


@dataclass(frozen=True)
class _IndexRawRevisionHead:
    logical_source_key: str
    accepted_raw_id: str
    accepted_source_revision: str
    accepted_frontier_kind: str
    accepted_frontier: int
    acquisition_generation: int
    append_end_offset: int | None


@dataclass(frozen=True)
class _EligibleRawReceipt:
    raw_id: str
    logical_source_key: str
    source_revision: str
    baseline_raw_id: str | None
    predecessor_raw_id: str | None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _blob_hash_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex() if len(value) == 32 else None
    text = str(value)
    return text if text else None


def _timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _active_index_raw_authority(
    index_db_path: Path,
) -> tuple[frozenset[str], tuple[_IndexRawRevisionHead, ...], tuple[_EligibleRawReceipt, ...]]:
    """Read current raw references and explicit deletion receipts read-only."""
    if not index_db_path.is_file():
        raise RawRetentionSafetyError(f"index tier is unavailable: {index_db_path}")
    try:
        uri = f"{index_db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            session_rows = conn.execute("SELECT DISTINCT raw_id FROM sessions WHERE raw_id IS NOT NULL").fetchall()
            head_rows = conn.execute(
                """SELECT logical_source_key, accepted_raw_id, accepted_source_revision,
                          accepted_frontier_kind, accepted_frontier,
                          acquisition_generation, append_end_offset
                   FROM raw_revision_heads"""
            ).fetchall()
            eligible_rows = conn.execute(
                """SELECT DISTINCT application.raw_id,
                          application.logical_source_key,
                          application.source_revision,
                          application.baseline_raw_id,
                          application.predecessor_raw_id
                   FROM raw_revision_applications AS application
                   JOIN raw_revision_heads AS head
                     ON head.logical_source_key = application.logical_source_key
                    AND head.session_id = application.session_id
                    AND head.accepted_raw_id = application.accepted_raw_id
                    AND head.accepted_source_revision = application.accepted_source_revision
                    AND head.accepted_content_hash = application.accepted_content_hash
                    AND head.acquisition_generation = application.acquisition_generation
                    AND head.append_end_offset IS application.append_end_offset
                    AND head.decided_at_ms = application.decided_at_ms
                   WHERE application.decision = 'superseded'
                     AND head.accepted_frontier_kind = 'byte'"""
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"index tier raw authority is unreadable: {exc}") from exc
    session_raw_ids = frozenset(str(row[0]) for row in session_rows if row[0] is not None and str(row[0]))
    heads = tuple(
        _IndexRawRevisionHead(
            logical_source_key=str(row[0]),
            accepted_raw_id=str(row[1]),
            accepted_source_revision=str(row[2]),
            accepted_frontier_kind=str(row[3]),
            accepted_frontier=int(row[4]),
            acquisition_generation=int(row[5]),
            append_end_offset=int(row[6]) if row[6] is not None else None,
        )
        for row in head_rows
    )
    eligible_receipts = tuple(
        _EligibleRawReceipt(
            raw_id=str(row[0]),
            logical_source_key=str(row[1]),
            source_revision=str(row[2]),
            baseline_raw_id=str(row[3]) if row[3] is not None else None,
            predecessor_raw_id=str(row[4]) if row[4] is not None else None,
        )
        for row in eligible_rows
    )
    return session_raw_ids, heads, eligible_receipts


def _raw_revision_rows(
    conn: sqlite3.Connection,
    raw_ids: set[str],
) -> dict[str, sqlite3.Row]:
    rows_by_id: dict[str, sqlite3.Row] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        try:
            rows = conn.execute(
                f"SELECT {_RAW_REVISION_CHAIN_COLUMNS} FROM raw_sessions WHERE raw_id IN ({placeholders})",
                batch,
            ).fetchall()
        except sqlite3.Error as exc:
            raise RawRetentionSafetyError(f"source raw revision authority is unreadable: {exc}") from exc
        found = {str(row["raw_id"]): row for row in rows}
        missing = set(batch).difference(found)
        if missing:
            rendered = ", ".join(sorted(missing)[:3])
            raise RawRetentionSafetyError(f"active index raw is missing from source tier: {rendered}")
        rows_by_id.update(found)
        for row in rows:
            if str(row["revision_kind"]) != "append":
                continue
            predecessor = row["predecessor_raw_id"]
            if predecessor is not None and str(predecessor) not in rows_by_id:
                pending.add(str(predecessor))
    return rows_by_id


def _validate_active_revision_chain(
    rows_by_id: dict[str, sqlite3.Row],
    seed_raw_id: str,
) -> frozenset[str]:
    protected: set[str] = set()
    current_raw_id = seed_raw_id
    chain_baseline_raw_id: str | None = None
    while True:
        if current_raw_id in protected:
            raise RawRetentionSafetyError(f"active raw revision chain contains a cycle at {current_raw_id}")
        protected.add(current_raw_id)
        row = rows_by_id[current_raw_id]
        kind = str(row["revision_kind"])
        if kind == "full":
            if str(row["revision_authority"]) != "byte_proven":
                raise RawRetentionSafetyError(f"active full raw lacks byte-proven authority: {current_raw_id}")
            if chain_baseline_raw_id is not None and current_raw_id != chain_baseline_raw_id:
                raise RawRetentionSafetyError(f"active append chain terminates at the wrong baseline: {current_raw_id}")
            # A full payload is a self-contained reset even when historical
            # classification records an older full predecessor.
            return frozenset(protected)
        if kind == "unknown":
            if int(row["source_index"]) == -1:
                raise RawRetentionSafetyError(f"active append raw lacks revision authority: {current_raw_id}")
            return frozenset(protected)
        if kind != "append":
            raise RawRetentionSafetyError(f"active raw has unsupported revision kind {kind!r}: {current_raw_id}")
        if str(row["revision_authority"]) != "byte_proven":
            raise RawRetentionSafetyError(f"active append raw lacks byte-proven authority: {current_raw_id}")
        logical_source_key = row["logical_source_key"]
        predecessor_raw_id = row["predecessor_raw_id"]
        predecessor_revision = row["predecessor_source_revision"]
        declared_baseline_value = row["baseline_raw_id"]
        if not all((logical_source_key, predecessor_raw_id, predecessor_revision, declared_baseline_value)):
            raise RawRetentionSafetyError(f"active append raw has an incomplete predecessor envelope: {current_raw_id}")
        declared_baseline = str(row["baseline_raw_id"])
        if chain_baseline_raw_id is None:
            chain_baseline_raw_id = declared_baseline
        elif declared_baseline != chain_baseline_raw_id:
            raise RawRetentionSafetyError(f"active append chain changes baseline identity: {current_raw_id}")
        parent_id = str(predecessor_raw_id)
        parent = rows_by_id.get(parent_id)
        if parent is None:
            raise RawRetentionSafetyError(f"active append predecessor is missing from source tier: {parent_id}")
        if parent["logical_source_key"] != logical_source_key:
            raise RawRetentionSafetyError(f"active append predecessor crosses logical sources: {current_raw_id}")
        if parent["source_revision"] != predecessor_revision:
            raise RawRetentionSafetyError(f"active append predecessor revision does not match: {current_raw_id}")
        parent_kind = str(parent["revision_kind"])
        if str(parent["revision_authority"]) != "byte_proven":
            raise RawRetentionSafetyError(f"active append predecessor lacks byte-proven authority: {parent_id}")
        if parent_kind == "append" and str(parent["baseline_raw_id"] or "") != chain_baseline_raw_id:
            raise RawRetentionSafetyError(f"active append predecessor changes baseline identity: {current_raw_id}")
        expected_offset = parent["blob_size"] if parent_kind == "full" else parent["append_end_offset"]
        if parent_kind not in {"full", "append"} or expected_offset != row["append_start_offset"]:
            raise RawRetentionSafetyError(f"active append predecessor is not byte-contiguous: {current_raw_id}")
        parent_generation = parent["acquisition_generation"]
        child_generation = row["acquisition_generation"]
        if parent_generation is None or child_generation is None or int(child_generation) != int(parent_generation) + 1:
            raise RawRetentionSafetyError(f"active append predecessor generation does not match: {current_raw_id}")
        current_raw_id = parent_id


def _validate_byte_head(row: sqlite3.Row, head: _IndexRawRevisionHead) -> None:
    if str(row["revision_kind"]) not in {"full", "append"}:
        raise RawRetentionSafetyError(
            f"byte head references a raw without typed revision authority: {head.accepted_raw_id}"
        )
    if row["logical_source_key"] != head.logical_source_key:
        raise RawRetentionSafetyError(f"accepted raw logical source disagrees with index head: {head.accepted_raw_id}")
    if row["source_revision"] != head.accepted_source_revision:
        raise RawRetentionSafetyError(f"accepted raw revision disagrees with index head: {head.accepted_raw_id}")
    if row["acquisition_generation"] != head.acquisition_generation:
        raise RawRetentionSafetyError(f"accepted raw generation disagrees with index head: {head.accepted_raw_id}")
    is_append = str(row["revision_kind"]) == "append"
    raw_frontier = row["append_end_offset"] if is_append else row["blob_size"]
    if raw_frontier != head.accepted_frontier:
        raise RawRetentionSafetyError(f"accepted raw frontier disagrees with index head: {head.accepted_raw_id}")
    if (is_append and raw_frontier != head.append_end_offset) or (not is_append and head.append_end_offset is not None):
        raise RawRetentionSafetyError(f"accepted raw append end disagrees with index head: {head.accepted_raw_id}")


def _validate_eligible_receipt(row: sqlite3.Row, receipt: _EligibleRawReceipt) -> None:
    if str(row["revision_kind"]) not in {"full", "append"}:
        raise RawRetentionSafetyError(f"superseded receipt references an untyped raw: {receipt.raw_id}")
    if str(row["revision_authority"]) != "byte_proven":
        raise RawRetentionSafetyError(f"superseded receipt references unproven raw evidence: {receipt.raw_id}")
    if row["logical_source_key"] != receipt.logical_source_key:
        raise RawRetentionSafetyError(f"superseded receipt crosses logical sources: {receipt.raw_id}")
    if row["source_revision"] != receipt.source_revision:
        raise RawRetentionSafetyError(f"superseded receipt revision disagrees with source evidence: {receipt.raw_id}")
    if row["baseline_raw_id"] != receipt.baseline_raw_id:
        raise RawRetentionSafetyError(f"superseded receipt baseline disagrees with source evidence: {receipt.raw_id}")
    if row["predecessor_raw_id"] != receipt.predecessor_raw_id:
        raise RawRetentionSafetyError(
            f"superseded receipt predecessor disagrees with source evidence: {receipt.raw_id}"
        )


def active_raw_retention_authority(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
) -> RawRetentionAuthority:
    """Return current protection plus explicitly authorized deletion rows.

    The index selects active leaves. Source-tier predecessor evidence proves
    which append fragments are required to reconstruct each leaf. Only an
    immutable ``superseded`` receipt tied to the current head authorizes raw
    deletion. Callers must serialize this read with source deletion under the
    daemon's single-writer contract, or stop the daemon for manual cleanup.
    """
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        session_raw_ids, heads, eligible_receipts = _active_index_raw_authority(index_db_path)
        seeds = set(session_raw_ids)
        seeds.update(head.accepted_raw_id for head in heads)
        if not seeds:
            if conn.execute("SELECT 1 FROM raw_sessions LIMIT 1").fetchone() is not None:
                raise RawRetentionSafetyError("source tier contains raw evidence but index has no raw authority")
            return RawRetentionAuthority(protected_raw_ids=frozenset(), eligible_raw_ids=frozenset())
        authority_raw_ids = seeds.union(receipt.raw_id for receipt in eligible_receipts)
        rows_by_id = _raw_revision_rows(conn, authority_raw_ids)
        protected: set[str] = set()
        for seed_raw_id in sorted(session_raw_ids):
            protected.update(_validate_active_revision_chain(rows_by_id, seed_raw_id))
        for head in heads:
            row = rows_by_id[head.accepted_raw_id]
            if head.accepted_frontier_kind == "byte":
                _validate_byte_head(row, head)
            protected.update(_validate_active_revision_chain(rows_by_id, head.accepted_raw_id))
        eligible: set[str] = set()
        for receipt in eligible_receipts:
            _validate_eligible_receipt(rows_by_id[receipt.raw_id], receipt)
            eligible.add(receipt.raw_id)
        protected_ids = frozenset(protected)
        return RawRetentionAuthority(
            protected_raw_ids=protected_ids,
            eligible_raw_ids=frozenset(eligible.difference(protected_ids)),
        )
    finally:
        conn.row_factory = original_row_factory


def protected_active_raw_revision_ids(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
) -> frozenset[str]:
    """Compatibility projection for callers that only inspect protection."""
    return active_raw_retention_authority(conn, index_db_path=index_db_path).protected_raw_ids


# ---------------------------------------------------------------------------
# Raw-frontier integrity readiness (polylogue-yla8.7)
# ---------------------------------------------------------------------------
#
# ``active_raw_retention_authority`` above answers "what may retention delete
# right now?" and fails *closed* (raises) the moment it finds one broken
# chain, because that is the correct behaviour for a cleanup gate. Ordinary
# process health and raw-materialization candidate counts never call it, so a
# broken accepted append head or a cursor that has committed past the
# material actually accepted into the index can sit invisible until an
# operator runs manual SQL (as happened before yla8.6). This section reuses
# the exact same chain validator (``_validate_active_revision_chain``) to
# build a *reporting* projection instead: it never raises on a per-head
# violation, it counts and samples it, so readiness and retention safety
# cannot drift apart.

RawFrontierIntegrityStatus = Literal["healthy", "unknown", "violated"]
"""Typed check outcome. ``"unknown"`` means the check's authority tier could
not be read — it is never collapsed into a false ``"healthy"`` zero."""


@dataclass(frozen=True)
class BrokenAppendHeadSample:
    """One accepted append head whose predecessor chain failed validation."""

    logical_source_key: str
    accepted_raw_id: str
    reason: str


@dataclass(frozen=True)
class CursorAheadSample:
    """One ingest cursor committed past the byte frontier accepted into the index."""

    source_path: str
    logical_source_key: str
    cursor_byte_offset: int
    accepted_frontier: int


@dataclass(frozen=True)
class RawFrontierIntegritySnapshot:
    """Two of the three yla8.7 raw-frontier integrity facts.

    The third fact — an index ``sessions.raw_id`` absent from
    ``source.raw_sessions`` — is intentionally **not** recomputed here: it is
    already exact-projected by
    :func:`polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot`
    as ``lost_source_evidence_count`` / ``lost_source_evidence_samples`` and
    already gates ``raw_materialization_ready()`` / claim-guard ``converged``.
    Callers compose that existing signal alongside this snapshot instead of
    re-querying it (no duplicated SQL/semantics).
    """

    broken_head_status: RawFrontierIntegrityStatus
    broken_head_count: int
    broken_head_checked_count: int
    broken_head_samples: tuple[BrokenAppendHeadSample, ...]
    broken_head_reason: str

    cursor_ahead_status: RawFrontierIntegrityStatus
    cursor_ahead_count: int
    cursor_ahead_checked_count: int
    cursor_ahead_samples: tuple[CursorAheadSample, ...]
    cursor_ahead_reason: str

    @property
    def overall_status(self) -> RawFrontierIntegrityStatus:
        statuses = (self.broken_head_status, self.cursor_ahead_status)
        if "unknown" in statuses:
            return "unknown"
        if "violated" in statuses:
            return "violated"
        return "healthy"


def raw_frontier_integrity_snapshot(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
    ops_db_path: Path,
    sample_limit: int = 10,
) -> RawFrontierIntegritySnapshot:
    """Read-only substrate integrity projection over source/index/ops.

    Reports two authority gaps that ordinary process health and
    raw-materialization candidate counts cannot see on their own
    (polylogue-yla8.7):

    * ``broken_head`` — a current accepted append head
      (``index.raw_revision_heads``) whose transitive predecessor chain is
      missing or invalid. Uses the exact same
      :func:`_validate_active_revision_chain` that
      :func:`active_raw_retention_authority` uses to protect retention, so
      cleanup safety and readiness visibility cannot drift.
    * ``cursor_ahead`` — an ``ops.ingest_cursor`` committed byte frontier past
      the byte frontier actually accepted into the index for that logical
      source — the exact symptom yla8.6 found only via manual SQL.

    Each check independently degrades to ``"unknown"`` (never a false
    ``"healthy"`` zero) when its authority tier cannot be read.
    """
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        try:
            _session_raw_ids, heads, _eligible = _active_index_raw_authority(index_db_path)
        except RawRetentionSafetyError as exc:
            return _unavailable_frontier_integrity_snapshot(str(exc))

        source_reason = _source_tier_unavailable_reason(conn)
        if source_reason is not None:
            return _unavailable_frontier_integrity_snapshot(source_reason)

        broken_status, broken_count, broken_checked, broken_samples, broken_reason = _check_broken_append_heads(
            conn, heads, sample_limit=sample_limit
        )
        cursor_status, cursor_count, cursor_checked, cursor_samples, cursor_reason = _check_cursor_ahead_of_accepted(
            conn, ops_db_path, heads, sample_limit=sample_limit
        )
        return RawFrontierIntegritySnapshot(
            broken_head_status=broken_status,
            broken_head_count=broken_count,
            broken_head_checked_count=broken_checked,
            broken_head_samples=broken_samples,
            broken_head_reason=broken_reason,
            cursor_ahead_status=cursor_status,
            cursor_ahead_count=cursor_count,
            cursor_ahead_checked_count=cursor_checked,
            cursor_ahead_samples=cursor_samples,
            cursor_ahead_reason=cursor_reason,
        )
    finally:
        conn.row_factory = original_row_factory


def _unavailable_frontier_integrity_snapshot(reason: str) -> RawFrontierIntegritySnapshot:
    return RawFrontierIntegritySnapshot(
        broken_head_status="unknown",
        broken_head_count=0,
        broken_head_checked_count=0,
        broken_head_samples=(),
        broken_head_reason=reason,
        cursor_ahead_status="unknown",
        cursor_ahead_count=0,
        cursor_ahead_checked_count=0,
        cursor_ahead_samples=(),
        cursor_ahead_reason=reason,
    )


def _source_tier_unavailable_reason(conn: sqlite3.Connection) -> str | None:
    try:
        conn.execute("SELECT 1 FROM raw_sessions LIMIT 1")
    except sqlite3.Error as exc:
        return f"source raw revision authority is unreadable: {exc}"
    return None


def _check_broken_append_heads(
    conn: sqlite3.Connection,
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    sample_limit: int,
) -> tuple[RawFrontierIntegrityStatus, int, int, tuple[BrokenAppendHeadSample, ...], str]:
    samples: list[BrokenAppendHeadSample] = []
    broken_count = 0
    for head in heads:
        try:
            rows_by_id = _raw_revision_rows(conn, {head.accepted_raw_id})
            _validate_active_revision_chain(rows_by_id, head.accepted_raw_id)
        except RawRetentionSafetyError as exc:
            broken_count += 1
            if len(samples) < sample_limit:
                samples.append(
                    BrokenAppendHeadSample(
                        logical_source_key=head.logical_source_key,
                        accepted_raw_id=head.accepted_raw_id,
                        reason=str(exc),
                    )
                )
    status: RawFrontierIntegrityStatus = "violated" if broken_count else "healthy"
    reason = "" if broken_count == 0 else f"{broken_count} accepted append head(s) have a broken predecessor chain"
    return status, broken_count, len(heads), tuple(samples), reason


def _check_cursor_ahead_of_accepted(
    conn: sqlite3.Connection,
    ops_db_path: Path,
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    sample_limit: int,
) -> tuple[RawFrontierIntegrityStatus, int, int, tuple[CursorAheadSample, ...], str]:
    byte_heads = [head for head in heads if head.accepted_frontier_kind == "byte"]
    if not byte_heads:
        return "healthy", 0, 0, (), ""
    try:
        source_path_by_raw_id = _source_paths_for_raw_ids(conn, {head.accepted_raw_id for head in byte_heads})
    except sqlite3.Error as exc:
        return "unknown", 0, 0, (), f"source raw path lookup failed: {exc}"

    source_path_by_key = {
        head.logical_source_key: source_path_by_raw_id[head.accepted_raw_id]
        for head in byte_heads
        if head.accepted_raw_id in source_path_by_raw_id
    }
    try:
        cursor_map = _ops_cursor_byte_offsets(ops_db_path, set(source_path_by_key.values()))
    except RawRetentionSafetyError as exc:
        return "unknown", 0, 0, (), str(exc)

    samples: list[CursorAheadSample] = []
    ahead_count = 0
    checked = 0
    for head in byte_heads:
        path = source_path_by_key.get(head.logical_source_key)
        if path is None:
            continue
        cursor_offset = cursor_map.get(path)
        if cursor_offset is None:
            continue
        checked += 1
        if cursor_offset > head.accepted_frontier:
            ahead_count += 1
            if len(samples) < sample_limit:
                samples.append(
                    CursorAheadSample(
                        source_path=path,
                        logical_source_key=head.logical_source_key,
                        cursor_byte_offset=cursor_offset,
                        accepted_frontier=head.accepted_frontier,
                    )
                )
    status: RawFrontierIntegrityStatus = "violated" if ahead_count else "healthy"
    reason = "" if ahead_count == 0 else f"{ahead_count} ingest cursor(s) committed past accepted raw material"
    return status, ahead_count, checked, tuple(samples), reason


def _source_paths_for_raw_ids(conn: sqlite3.Connection, raw_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT raw_id, source_path FROM raw_sessions WHERE raw_id IN ({placeholders})", batch
        ).fetchall()
        for row in rows:
            result[str(row[0])] = str(row[1])
    return result


def _ops_cursor_byte_offsets(ops_db_path: Path, source_paths: set[str]) -> dict[str, int]:
    if not source_paths:
        return {}
    if not ops_db_path.is_file():
        raise RawRetentionSafetyError(f"ops tier is unavailable: {ops_db_path}")
    try:
        uri = f"{ops_db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'ingest_cursor'"
            ).fetchone()
            if has_table is None:
                raise RawRetentionSafetyError(f"ops tier has no ingest_cursor table: {ops_db_path}")
            placeholders = ", ".join("?" for _ in source_paths)
            rows = conn.execute(
                f"SELECT source_path, byte_offset FROM ingest_cursor WHERE source_path IN ({placeholders})",
                tuple(source_paths),
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"ops tier raw cursor authority is unreadable: {exc}") from exc
    return {str(row[0]): int(row[1]) for row in rows if row[1] is not None}


def _superseded_archive_raw_session_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None,
    keep_full_snapshots: int,
    keep_append_snapshots: int,
    min_acquired_at: str | None,
    limit: int,
) -> list[RawSnapshotCleanupCandidate]:
    source_path_str = str(source_path) if source_path is not None else None
    min_acquired_at_ms = _timestamp_ms(min_acquired_at)
    rows = conn.execute(
        _V1_RAW_CANDIDATE_SQL,
        (
            source_path_str,
            source_path_str,
            min_acquired_at_ms,
            min_acquired_at_ms,
            max(1, keep_full_snapshots),
            max(1, keep_append_snapshots),
            limit,
        ),
    ).fetchall()
    candidates: list[RawSnapshotCleanupCandidate] = []
    for row in rows:
        row_source_path = str(row[1])
        if not Path(row_source_path).exists():
            continue
        blob_hash = _blob_hash_text(row[3])
        if blob_hash is None:
            continue
        candidates.append(
            RawSnapshotCleanupCandidate(
                raw_id=str(row[0]),
                source_path=row_source_path,
                source_index=int(row[2] or 0),
                blob_size=int(row[4] or 0),
                blob_hash=blob_hash,
            )
        )
    return candidates


def superseded_raw_snapshot_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
) -> list[RawSnapshotCleanupCandidate]:
    """Return redundant live raw snapshots that are safe to compact.

    The source file must still exist on disk before a row is returned. If
    the source disappeared, the raw blob may be the only remaining copy and
    is deliberately preserved.
    """
    if limit <= 0:
        return []

    if not _table_exists(conn, "raw_sessions"):
        return []
    return _superseded_archive_raw_session_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )


def cleanup_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
    dry_run: bool = True,
    blob_store: BlobStore | None = None,
    protected_raw_ids: set[str] | frozenset[str] | None = None,
    eligible_raw_ids: set[str] | frozenset[str] | None = None,
) -> RawSnapshotCleanupResult:
    all_candidates = superseded_raw_snapshot_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )
    protected = protected_raw_ids or frozenset()
    eligible = eligible_raw_ids
    candidates = [
        candidate
        for candidate in all_candidates
        if candidate.raw_id not in protected and (eligible is None or candidate.raw_id in eligible)
    ]
    skipped_referenced_count = sum(candidate.raw_id in protected for candidate in all_candidates)
    if not candidates:
        return RawSnapshotCleanupResult(
            candidate_count=0,
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=0,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
            skipped_referenced_count=skipped_referenced_count,
        )

    raw_ids = [candidate.raw_id for candidate in candidates]
    raw_bytes = sum(candidate.blob_size for candidate in candidates)
    if dry_run:
        return RawSnapshotCleanupResult(
            candidate_count=len(candidates),
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=raw_bytes,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
            skipped_referenced_count=skipped_referenced_count,
        )

    placeholders = ", ".join("?" for _ in raw_ids)
    if _column_exists(conn, "blob_refs", "ref_id"):
        conn.execute(f"DELETE FROM blob_refs WHERE ref_id IN ({placeholders})", raw_ids)
    elif _column_exists(conn, "blob_refs", "raw_id"):
        conn.execute(f"DELETE FROM blob_refs WHERE raw_id IN ({placeholders})", raw_ids)
    conn.execute(f"DELETE FROM raw_sessions WHERE raw_id IN ({placeholders})", raw_ids)
    conn.commit()

    store = blob_store if blob_store is not None else get_blob_store()
    deleted_blob_count = 0
    deleted_blob_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        if _column_exists(conn, "blob_refs", "blob_hash"):
            try:
                blob_hash = bytes.fromhex(candidate.blob_store_hash)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if conn.execute("SELECT 1 FROM blob_refs WHERE blob_hash = ? LIMIT 1", (blob_hash,)).fetchone():
                continue
        else:
            # Without a reference catalog, row cleanup cannot prove that this
            # process owns the final CAS reference. Leave bytes for typed GC.
            continue
        try:
            path = store.blob_path(candidate.blob_store_hash)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists():
            continue
        with suppress(OSError):
            deleted_blob_bytes += path.stat().st_size
        try:
            path.unlink()
            deleted_blob_count += 1
        except OSError as exc:
            errors.append(f"{candidate.raw_id[:16]}: {exc}")

    return RawSnapshotCleanupResult(
        candidate_count=len(candidates),
        deleted_raw_count=len(candidates),
        deleted_blob_count=deleted_blob_count,
        deleted_raw_bytes=raw_bytes,
        deleted_blob_bytes=deleted_blob_bytes,
        skipped_missing_source_count=0,
        skipped_referenced_count=skipped_referenced_count,
        errors=tuple(errors),
    )


def compact_paths_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    source_paths: Iterable[Path],
    *,
    limit_per_path: int = 25,
    min_acquired_at: str | None = None,
    dry_run: bool = False,
    protected_raw_ids: set[str] | frozenset[str] | None = None,
    eligible_raw_ids: set[str] | frozenset[str] | None = None,
) -> RawSnapshotCleanupResult:
    totals = RawSnapshotCleanupResult(
        candidate_count=0,
        deleted_raw_count=0,
        deleted_blob_count=0,
        deleted_raw_bytes=0,
        deleted_blob_bytes=0,
        skipped_missing_source_count=0,
    )
    errors: list[str] = []
    for path in source_paths:
        result = cleanup_superseded_raw_snapshots(
            conn,
            source_path=path,
            keep_full_snapshots=1_000_000,
            min_acquired_at=min_acquired_at,
            limit=limit_per_path,
            dry_run=dry_run,
            protected_raw_ids=protected_raw_ids,
            eligible_raw_ids=eligible_raw_ids,
        )
        errors.extend(result.errors)
        totals = RawSnapshotCleanupResult(
            candidate_count=totals.candidate_count + result.candidate_count,
            deleted_raw_count=totals.deleted_raw_count + result.deleted_raw_count,
            deleted_blob_count=totals.deleted_blob_count + result.deleted_blob_count,
            deleted_raw_bytes=totals.deleted_raw_bytes + result.deleted_raw_bytes,
            deleted_blob_bytes=totals.deleted_blob_bytes + result.deleted_blob_bytes,
            skipped_missing_source_count=totals.skipped_missing_source_count + result.skipped_missing_source_count,
            skipped_referenced_count=totals.skipped_referenced_count + result.skipped_referenced_count,
            errors=tuple(errors),
        )
    return totals


__all__ = [
    "BrokenAppendHeadSample",
    "CursorAheadSample",
    "RawFrontierIntegritySnapshot",
    "RawFrontierIntegrityStatus",
    "RawSnapshotCleanupCandidate",
    "RawSnapshotCleanupResult",
    "RawRetentionAuthority",
    "RawRetentionSafetyError",
    "active_raw_retention_authority",
    "cleanup_superseded_raw_snapshots",
    "compact_paths_superseded_raw_snapshots",
    "protected_active_raw_revision_ids",
    "raw_frontier_integrity_snapshot",
    "superseded_raw_snapshot_candidates",
]
