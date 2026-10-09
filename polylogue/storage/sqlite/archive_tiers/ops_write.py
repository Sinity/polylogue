"""Minimal ops-tier archive read/write helpers.

Writer module: ops.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from polylogue.core.enums import (
    IngestOutcome,
    OperationStatus,
    Origin,
)
from polylogue.core.types import (
    ConvergenceDebtStatus,
    CursorLagSeverity,
    DaemonTerminationClass,
    JudgmentSchedulerStatus,
    OperationRunStatus,
    require_literal,
)
from polylogue.pipeline.ingest_outcomes import IngestAttemptDisposition
from polylogue.storage.sqlite.archive_tiers.ops import McpCallSessionRelation

MCP_CALL_LOG_RETENTION_MS = 90 * 24 * 60 * 60 * 1000
# polylogue-1xc.12: a 30 day window
# (long enough to see week-over-week drift trend) capped at 5,000 rows (one
# sample per surface per convergence/startup pass keeps this table tiny in
# practice; the cap is a hard backstop against a runaway sampling loop).
FTS_DRIFT_SAMPLE_RETENTION_MS = 30 * 24 * 60 * 60 * 1000
FTS_DRIFT_SAMPLE_ROW_CAP = 5_000
# polylogue-da1: the sentinel's rate must be windowed since a date, not
# lifetime, or an old archive's many historical clean records permanently
# dilute a recent provider-shape regression. 30 days matches the FTS drift
# sample precedent above; the row cap is a hard backstop against a runaway
# per-record sampling loop (one row per drifted record, not per session).
SCHEMA_DRIFT_SAMPLE_RETENTION_MS = 30 * 24 * 60 * 60 * 1000
SCHEMA_DRIFT_SAMPLE_ROW_CAP = 20_000


@dataclass(frozen=True, slots=True)
class OpsCompactState:
    """Compact one-row status snapshot from OPS-tier tables."""

    cursor_count: int
    ingest_attempt_total: int
    ingest_attempt_running: int
    ingest_attempt_completed: int
    ingest_attempt_failed: int
    convergence_debt_count: int
    latest_attempt_id: str | None
    latest_attempt_status: str | None
    latest_cursor_path: str | None
    latest_debt_stage: str | None
    latest_debt_priority: int


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingCatchupRun:
    """Compact read-back row for one embedding catchup run."""

    run_id: str
    started_at_ms: int
    finished_at_ms: int | None
    status: str
    origin: str | None
    scanned_sessions: int
    embedded_sessions: int
    skipped_sessions: int
    error_count: int
    embedded_messages: int
    estimated_cost_usd: float | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ArchiveCursorLagSample:
    """Compact read-back row for one cursor lag sample."""

    sample_id: str
    family: str
    source_path: str | None
    lag_ms: int
    stuck_file_count: int
    p50_lag_ms: int
    p95_lag_ms: int
    severity: str
    sampled_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveDaemonStageEvent:
    """Compact read-back row for one daemon stage event."""

    event_id: str
    attempt_id: str | None
    stage: str
    status: str
    observed_at_ms: int
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class ArchiveDaemonLifecycle:
    """Forensic lifecycle record for one daemon process instance."""

    run_id: str
    started_at_ms: int
    stopped_at_ms: int | None
    last_heartbeat_at_ms: int
    signal: str | None
    exit_kind: str | None
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class ArchiveJudgmentSchedulerReceipt:
    """Typed scheduler receipt persisted in the disposable ops tier."""

    operation_id: str
    observed_at_ms: int
    status: str
    reason: str
    retryable: bool
    retry_route: str
    batch_limit: int
    considered: int = 0
    accepted: int = 0
    rejected: int = 0
    escalated: int = 0
    idempotent: int = 0
    failed: int = 0
    receipt_persistence_degraded: bool = False
    receipt_persistence_recovered: bool = False


_JUDGMENT_SCHEDULER_RECEIPT_COUNTERS = (
    "considered",
    "accepted",
    "rejected",
    "escalated",
    "idempotent",
    "failed",
)


def _is_exact_bool(value: object) -> bool:
    return type(value) is bool


def _validate_judgment_scheduler_receipt(receipt: ArchiveJudgmentSchedulerReceipt) -> None:
    if not receipt.operation_id:
        raise ValueError("judgment scheduler receipt operation_id must not be empty")
    require_literal(receipt.status, JudgmentSchedulerStatus, name="judgment scheduler status")
    if not receipt.reason:
        raise ValueError("judgment scheduler receipt reason must not be empty")
    if not _is_exact_bool(receipt.retryable) or not _is_exact_bool(receipt.receipt_persistence_degraded):
        raise ValueError("judgment scheduler receipt boolean fields must be bool")
    if not _is_exact_bool(receipt.receipt_persistence_recovered):
        raise ValueError("judgment scheduler receipt boolean fields must be bool")
    if not receipt.retry_route:
        raise ValueError("judgment scheduler receipt retry_route must not be empty")
    if type(receipt.observed_at_ms) is not int:
        raise ValueError("judgment scheduler receipt observed_at_ms must be an integer")
    if type(receipt.batch_limit) is not int or receipt.batch_limit <= 0:
        raise ValueError("judgment scheduler receipt batch_limit must be positive")
    counters = tuple(getattr(receipt, name) for name in _JUDGMENT_SCHEDULER_RECEIPT_COUNTERS)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
        raise ValueError("judgment scheduler receipt counters must be non-negative integers")
    if receipt.considered != sum(counters[1:]):
        raise ValueError("judgment scheduler receipt counters do not sum to considered")


def record_judgment_scheduler_receipt(
    conn: sqlite3.Connection,
    receipt: ArchiveJudgmentSchedulerReceipt,
) -> None:
    """Upsert one operation-owned scheduler receipt without JSON decoding."""

    _validate_judgment_scheduler_receipt(receipt)
    conn.execute(
        """
        INSERT INTO judgment_scheduler_receipts (
            operation_id, observed_at_ms, status, reason, retryable, retry_route,
            batch_limit, considered, accepted, rejected, escalated, idempotent,
            failed, receipt_persistence_degraded, receipt_persistence_recovered
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(operation_id) DO UPDATE SET
            observed_at_ms = excluded.observed_at_ms,
            status = excluded.status,
            reason = excluded.reason,
            retryable = excluded.retryable,
            retry_route = excluded.retry_route,
            batch_limit = excluded.batch_limit,
            considered = excluded.considered,
            accepted = excluded.accepted,
            rejected = excluded.rejected,
            escalated = excluded.escalated,
            idempotent = excluded.idempotent,
            failed = excluded.failed,
            receipt_persistence_degraded = excluded.receipt_persistence_degraded,
            receipt_persistence_recovered = excluded.receipt_persistence_recovered
        """,
        (
            receipt.operation_id,
            receipt.observed_at_ms,
            receipt.status,
            receipt.reason,
            int(receipt.retryable),
            receipt.retry_route,
            receipt.batch_limit,
            receipt.considered,
            receipt.accepted,
            receipt.rejected,
            receipt.escalated,
            receipt.idempotent,
            receipt.failed,
            int(receipt.receipt_persistence_degraded),
            int(receipt.receipt_persistence_recovered),
        ),
    )


def _read_latest_judgment_scheduler_receipt(
    conn: sqlite3.Connection,
    *,
    operation_id: str | None = None,
) -> ArchiveJudgmentSchedulerReceipt | None:
    """Read the newest typed scheduler receipt, optionally for one operation."""

    query = """
        SELECT operation_id, observed_at_ms, status, reason, retryable, retry_route,
               batch_limit, considered, accepted, rejected, escalated, idempotent,
               failed, receipt_persistence_degraded, receipt_persistence_recovered
        FROM judgment_scheduler_receipts
    """
    params: tuple[object, ...] = ()
    if operation_id is not None:
        query += " WHERE operation_id = ?"
        params = (operation_id,)
    query += " ORDER BY rowid DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    if row is None:
        return None
    receipt = ArchiveJudgmentSchedulerReceipt(
        operation_id=str(row[0]),
        observed_at_ms=int(row[1]),
        status=str(row[2]),
        reason=str(row[3]),
        retryable=bool(row[4]),
        retry_route=str(row[5]),
        batch_limit=int(row[6]),
        considered=int(row[7]),
        accepted=int(row[8]),
        rejected=int(row[9]),
        escalated=int(row[10]),
        idempotent=int(row[11]),
        failed=int(row[12]),
        receipt_persistence_degraded=bool(row[13]),
        receipt_persistence_recovered=bool(row[14]),
    )
    _validate_judgment_scheduler_receipt(receipt)
    return receipt


@dataclass(frozen=True, slots=True)
class ArchiveMcpCallLogEntry:
    """Compact read-back row for one durable MCP tool call-log entry."""

    call_id: str
    tool_name: str
    session_id: str | None
    started_at_ms: int
    finished_at_ms: int
    duration_ms: int
    success: bool
    error_detail: str | None


@dataclass(frozen=True, slots=True)
class ArchiveFtsDriftSample:
    """One bounded drift-magnitude sample for an FTS-backed surface."""

    sample_id: str
    surface: str
    state: str
    source_rows: int
    indexed_rows: int
    missing_rows: int
    excess_rows: int
    duplicate_rows: int
    identity_mismatch_rows: int
    sampled_at_ms: int


def record_fts_drift_sample(
    conn: sqlite3.Connection,
    *,
    surface: str,
    state: str,
    source_rows: int,
    indexed_rows: int,
    missing_rows: int,
    excess_rows: int,
    duplicate_rows: int,
    identity_mismatch_rows: int,
    sampled_at_ms: int,
    sample_id: str | None = None,
) -> str:
    """Record one bounded FTS drift-magnitude sample and return its id.

    polylogue-1xc.12: the ``fts_freshness_state`` ledger in index.db (see
    ``storage/fts/freshness.py``) is O(1) current state, not history -- this
    writer appends a time-series snapshot of the same counters to ops.db (a
    separate database file/connection) so an operator can see drift
    MAGNITUDE trend, not just today's boolean ready/stale. Best-effort
    telemetry delivered without the MCP call outbox: a plain direct INSERT,
    pruned by both time (``FTS_DRIFT_SAMPLE_RETENTION_MS``) and row count
    (``FTS_DRIFT_SAMPLE_ROW_CAP``) so it cannot grow unbounded.
    """
    if sample_id is None:
        sample_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            """
            INSERT INTO fts_drift_samples (
                sample_id, surface, state, source_rows, indexed_rows,
                missing_rows, excess_rows, duplicate_rows, identity_mismatch_rows, sampled_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                surface,
                state,
                max(0, int(source_rows)),
                max(0, int(indexed_rows)),
                max(0, int(missing_rows)),
                max(0, int(excess_rows)),
                max(0, int(duplicate_rows)),
                max(0, int(identity_mismatch_rows)),
                sampled_at_ms,
            ),
        )
        conn.execute(
            "DELETE FROM fts_drift_samples WHERE sampled_at_ms < ?",
            (sampled_at_ms - FTS_DRIFT_SAMPLE_RETENTION_MS,),
        )
        row_count = int(conn.execute("SELECT COUNT(*) FROM fts_drift_samples").fetchone()[0])
        if row_count > FTS_DRIFT_SAMPLE_ROW_CAP:
            excess = row_count - FTS_DRIFT_SAMPLE_ROW_CAP
            conn.execute(
                """
                DELETE FROM fts_drift_samples WHERE sample_id IN (
                    SELECT sample_id FROM fts_drift_samples
                    ORDER BY sampled_at_ms ASC LIMIT ?
                )
                """,
                (excess,),
            )
    return sample_id


def list_fts_drift_samples(
    conn: sqlite3.Connection,
    *,
    surface: str | None = None,
    since_ms: int | None = None,
    limit: int = 1000,
) -> tuple[ArchiveFtsDriftSample, ...]:
    """Return FTS drift samples newest-first, optionally filtered."""
    query = """
        SELECT sample_id, surface, state, source_rows, indexed_rows,
               missing_rows, excess_rows, duplicate_rows, identity_mismatch_rows, sampled_at_ms
        FROM fts_drift_samples
    """
    clauses: list[str] = []
    params: list[object] = []
    if surface is not None:
        clauses.append("surface = ?")
        params.append(surface)
    if since_ms is not None:
        clauses.append("sampled_at_ms >= ?")
        params.append(since_ms)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY sampled_at_ms DESC, sample_id DESC LIMIT ?"
    params.append(limit)
    return tuple(_fts_drift_sample_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def _fts_drift_sample_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveFtsDriftSample:
    return ArchiveFtsDriftSample(
        sample_id=str(row[0]),
        surface=str(row[1]),
        state=str(row[2]),
        source_rows=_int_value(row[3]),
        indexed_rows=_int_value(row[4]),
        missing_rows=_int_value(row[5]),
        excess_rows=_int_value(row[6]),
        duplicate_rows=_int_value(row[7]),
        identity_mismatch_rows=_int_value(row[8]),
        sampled_at_ms=_int_value(row[9]),
    )


@dataclass(frozen=True, slots=True)
class ArchiveSchemaDriftSample:
    """One classified format-drift sample for a single ingested record."""

    sample_id: str
    origin: str
    element_kind: str
    classification: str
    signature_byte_count: int
    native_id_example: str
    raw_id: str
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class SchemaDriftOriginSummary:
    """Windowed per-origin drift rate for the ``polylogue ops status`` line."""

    origin: str
    total: int
    risky: int
    benign: int
    since_ms: int
    example_native_ids: tuple[str, ...]

    @property
    def risky_rate(self) -> float:
        return self.risky / self.total if self.total else 0.0


def record_schema_drift_sample(
    conn: sqlite3.Connection,
    *,
    origin: str,
    element_kind: str,
    classification: str,
    signature_chunks: Iterable[bytes],
    signature_byte_count: int,
    native_id_example: str,
    raw_id: str,
    observed_at_ms: int,
    sample_id: str | None = None,
) -> str:
    """Record one format-drift sample and its exact bounded signature chunks.

    Best-effort telemetry like ``record_fts_drift_sample``: a plain direct
    INSERT, pruned by both time (``SCHEMA_DRIFT_SAMPLE_RETENTION_MS``) and
    row count (``SCHEMA_DRIFT_SAMPLE_ROW_CAP``) so it cannot grow unbounded
    across a long-running archive (polylogue-da1).
    """
    if sample_id is None:
        sample_id = str(uuid.uuid4())
    if signature_byte_count < 0:
        raise ValueError("signature_byte_count must be non-negative")
    with conn:
        conn.execute(
            """
            INSERT INTO schema_drift_samples (
                sample_id, origin, element_kind, classification,
                signature_byte_count, native_id_example, raw_id, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                origin,
                element_kind,
                classification,
                signature_byte_count,
                native_id_example,
                raw_id,
                observed_at_ms,
            ),
        )
        pending = bytearray()
        written_bytes = 0
        ordinal = 0
        for incoming in signature_chunks:
            if not isinstance(incoming, bytes):
                raise TypeError("signature chunks must be bytes")
            written_bytes += len(incoming)
            view = memoryview(incoming)
            offset = 0
            while offset < len(view):
                take = min(4096 - len(pending), len(view) - offset)
                pending.extend(view[offset : offset + take])
                offset += take
                if len(pending) == 4096:
                    payload = bytes(pending)
                    pending.clear()
                    conn.execute(
                        "INSERT INTO schema_drift_signature_chunks(sample_id, chunk_ordinal, chunk_bytes) VALUES (?, ?, ?)",
                        (sample_id, ordinal, payload),
                    )
                    ordinal += 1
        if pending:
            conn.execute(
                "INSERT INTO schema_drift_signature_chunks(sample_id, chunk_ordinal, chunk_bytes) VALUES (?, ?, ?)",
                (sample_id, ordinal, bytes(pending)),
            )
        if written_bytes != signature_byte_count:
            raise ValueError(
                f"signature byte count mismatch: declared {signature_byte_count}, received {written_bytes}"
            )
        conn.execute(
            "DELETE FROM schema_drift_samples WHERE observed_at_ms < ?",
            (observed_at_ms - SCHEMA_DRIFT_SAMPLE_RETENTION_MS,),
        )
        row_count = int(conn.execute("SELECT COUNT(*) FROM schema_drift_samples").fetchone()[0])
        if row_count > SCHEMA_DRIFT_SAMPLE_ROW_CAP:
            excess = row_count - SCHEMA_DRIFT_SAMPLE_ROW_CAP
            conn.execute(
                """
                DELETE FROM schema_drift_samples WHERE sample_id IN (
                    SELECT sample_id FROM schema_drift_samples
                    ORDER BY observed_at_ms ASC LIMIT ?
                )
                """,
                (excess,),
            )
    return sample_id


def list_schema_drift_samples(
    conn: sqlite3.Connection,
    *,
    origin: str | None = None,
    since_ms: int | None = None,
    limit: int = 1000,
) -> tuple[ArchiveSchemaDriftSample, ...]:
    """Return schema-drift samples newest-first, optionally filtered."""
    query = """
        SELECT sample_id, origin, element_kind, classification,
               signature_byte_count, native_id_example, raw_id, observed_at_ms
        FROM schema_drift_samples
    """
    clauses: list[str] = []
    params: list[object] = []
    if origin is not None:
        clauses.append("origin = ?")
        params.append(origin)
    if since_ms is not None:
        clauses.append("observed_at_ms >= ?")
        params.append(since_ms)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY observed_at_ms DESC, sample_id DESC LIMIT ?"
    params.append(limit)
    return tuple(_schema_drift_sample_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def _schema_drift_sample_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveSchemaDriftSample:
    return ArchiveSchemaDriftSample(
        sample_id=str(row[0]),
        origin=str(row[1]),
        element_kind=str(row[2]),
        classification=str(row[3]),
        signature_byte_count=_int_value(row[4]),
        native_id_example=str(row[5]),
        raw_id=str(row[6]),
        observed_at_ms=_int_value(row[7]),
    )


def iter_schema_drift_signature(
    conn: sqlite3.Connection,
    sample_id: str,
    *,
    schema: str = "main",
) -> Iterator[bytes]:
    """Yield exact stored signature bytes while borrowing the caller's connection."""
    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported schema-drift reader schema: {schema!r}")
    table = f"{schema}.schema_drift_samples"
    chunks_table = f"{schema}.schema_drift_signature_chunks"
    row = conn.execute(f"SELECT signature_byte_count FROM {table} WHERE sample_id = ?", (sample_id,)).fetchone()
    if row is None:
        raise KeyError(sample_id)
    expected_bytes = _int_value(row[0])
    cursor = conn.execute(
        f"SELECT chunk_ordinal, chunk_bytes FROM {chunks_table} WHERE sample_id = ? ORDER BY chunk_ordinal",
        (sample_id,),
    )
    seen_bytes = 0
    expected_ordinal = 0
    try:
        for ordinal, payload in cursor:
            if _int_value(ordinal) != expected_ordinal:
                raise ValueError(f"schema-drift signature chunk sequence is incomplete for {sample_id}")
            chunk = bytes(payload)
            if len(chunk) > 4096:
                raise ValueError(f"schema-drift signature chunk exceeds storage size for {sample_id}")
            seen_bytes += len(chunk)
            expected_ordinal += 1
            yield chunk
    finally:
        cursor.close()
    if seen_bytes != expected_bytes:
        raise ValueError(f"schema-drift signature byte count mismatch for {sample_id}")


def summarize_schema_drift_since(
    conn: sqlite3.Connection,
    *,
    since_ms: int,
    example_limit: int = 5,
    schema: str = "main",
) -> tuple[SchemaDriftOriginSummary, ...]:
    """Return one windowed drift-rate summary per origin since ``since_ms``.

    ``risky`` counts ``field_changed``/``unseen_shape`` samples (a known
    field disappeared/changed type, or the shape matched no committed
    package at all); ``benign`` counts ``new_field`` samples (an
    additional, schema-permitted field). Origins with zero samples in the
    window are omitted. ``example_native_ids`` is bounded to
    ``example_limit`` and favors risky samples over benign ones so an
    operator sees the more actionable examples first.
    """
    from polylogue.schemas.drift_sentinel import RISKY_CLASSIFICATIONS

    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported schema-drift reader schema: {schema!r}")
    table = f"{schema}.schema_drift_samples"
    origins = [
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT origin FROM {table} WHERE observed_at_ms >= ? ORDER BY origin",
            (since_ms,),
        ).fetchall()
    ]
    summaries: list[SchemaDriftOriginSummary] = []
    for origin in origins:
        rows = conn.execute(
            f"""
            SELECT classification, native_id_example FROM {table}
            WHERE origin = ? AND observed_at_ms >= ?
            ORDER BY observed_at_ms DESC
            """,
            (origin, since_ms),
        ).fetchall()
        total = len(rows)
        risky = sum(1 for classification, _ in rows if classification in RISKY_CLASSIFICATIONS)
        benign = total - risky
        risky_examples = [
            str(native_id) for classification, native_id in rows if classification in RISKY_CLASSIFICATIONS
        ]
        other_examples = [
            str(native_id) for classification, native_id in rows if classification not in RISKY_CLASSIFICATIONS
        ]
        examples: list[str] = []
        for native_id in risky_examples + other_examples:
            if native_id not in examples:
                examples.append(native_id)
            if len(examples) >= example_limit:
                break
        summaries.append(
            SchemaDriftOriginSummary(
                origin=origin,
                total=total,
                risky=risky,
                benign=benign,
                since_ms=since_ms,
                example_native_ids=tuple(examples),
            )
        )
    return tuple(summaries)


def upsert_ingest_cursor(
    conn: sqlite3.Connection,
    *,
    source_path: str,
    canonical_source_path: str | None = None,
    captured_profile_key: str | None = None,
    updated_at_ms: int,
    origin: Origin | str | None = None,
    stat_size: int | None = None,
    byte_offset: int | None = None,
    last_complete_newline: int | None = None,
    record_count: int = 0,
    last_record_ts_ms: int | None = None,
    parser_fingerprint: str | None = None,
    content_fingerprint: str | None = None,
    tail_hash: str | None = None,
    st_dev: int | None = None,
    st_ino: int | None = None,
    mtime_ns: int | None = None,
    failure_count: int = 0,
    next_retry_at: str | None = None,
    excluded: bool = False,
    deferred_end_offset: int | None = None,
    manage_transaction: bool = True,
) -> None:
    """Create or refresh one cursor row in ``ingest_cursor``.

    ``manage_transaction=False`` is required whenever the caller already owns a
    transaction: committing here would end it and leave every later write in the
    batch running in autocommit (polylogue-5pv1p).
    """
    conn.execute(
        """
        INSERT INTO ingest_cursor (
            source_path,
            canonical_source_path,
            captured_profile_key,
            origin,
            stat_size,
            byte_offset,
            last_complete_newline,
            record_count,
            last_record_ts_ms,
            parser_fingerprint,
            content_fingerprint,
            tail_hash,
            st_dev,
            st_ino,
            mtime_ns,
            failure_count,
            next_retry_at,
            excluded,
            deferred_end_offset,
            updated_at_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (source_path) DO UPDATE SET
            canonical_source_path = excluded.canonical_source_path,
            captured_profile_key = excluded.captured_profile_key,
            origin = excluded.origin,
            stat_size = excluded.stat_size,
            byte_offset = excluded.byte_offset,
            last_complete_newline = excluded.last_complete_newline,
            record_count = excluded.record_count,
            last_record_ts_ms = excluded.last_record_ts_ms,
            parser_fingerprint = excluded.parser_fingerprint,
            content_fingerprint = excluded.content_fingerprint,
            tail_hash = excluded.tail_hash,
            st_dev = excluded.st_dev,
            st_ino = excluded.st_ino,
            mtime_ns = excluded.mtime_ns,
            failure_count = excluded.failure_count,
            next_retry_at = excluded.next_retry_at,
            excluded = excluded.excluded,
            deferred_end_offset = excluded.deferred_end_offset,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            source_path,
            canonical_source_path,
            captured_profile_key,
            _origin_value(origin),
            stat_size,
            byte_offset,
            last_complete_newline,
            record_count,
            last_record_ts_ms,
            parser_fingerprint,
            content_fingerprint,
            tail_hash,
            st_dev,
            st_ino,
            mtime_ns,
            failure_count,
            next_retry_at,
            1 if excluded else 0,
            deferred_end_offset,
            updated_at_ms,
        ),
    )
    if manage_transaction:
        conn.commit()


def record_ingest_attempt(
    conn: sqlite3.Connection,
    *,
    status: OperationStatus | OperationRunStatus | str,
    source_path: str | None = None,
    origin: Origin | str | None = None,
    phase: str | None = None,
    started_at_ms: int,
    heartbeat_at_ms: int | None = None,
    finished_at_ms: int | None = None,
    parsed_raw_count: int = 0,
    materialized_count: int = 0,
    error_message: str | None = None,
    source_paths_json: str = "[]",
    storage_route: str | None = None,
    attempt_id: str | None = None,
    disposition: IngestAttemptDisposition | None = None,
) -> str:
    """Create or replace one ``ingest_attempts`` row and return its ``attempt_id``.

    ``disposition`` (polylogue-cnu3) is the typed, structurally-classified
    outcome for this attempt. Omitting it (legacy callers) writes
    ``outcome_code='legacy_unknown'`` with unknown retryability -- never a
    guessed real class (AC4).
    """
    if attempt_id is None:
        attempt_id = str(uuid.uuid4())
    status_value = require_literal(status, OperationRunStatus, name="ingest attempt status")
    outcome_code = disposition.outcome_code if disposition is not None else IngestOutcome.LEGACY_UNKNOWN.value
    retryable = disposition.retryable if disposition is not None else None
    retryable_int = None if retryable is None else (1 if retryable else 0)

    conn.execute(
        """
        INSERT INTO ingest_attempts (
            attempt_id,
            source_path,
            origin,
            status,
            phase,
            storage_route,
            outcome_code, retryable, evidence_ref, diagnostic, remediation,
            started_at_ms,
            heartbeat_at_ms,
            finished_at_ms,
            parsed_raw_count,
            materialized_count,
            error_message,
            source_paths_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (attempt_id) DO UPDATE SET
            source_path = excluded.source_path,
            origin = excluded.origin,
            status = excluded.status,
            phase = excluded.phase,
            storage_route = COALESCE(excluded.storage_route, ingest_attempts.storage_route),
            outcome_code = excluded.outcome_code,
            retryable = excluded.retryable,
            evidence_ref = excluded.evidence_ref,
            diagnostic = excluded.diagnostic,
            remediation = excluded.remediation,
            started_at_ms = excluded.started_at_ms,
            heartbeat_at_ms = excluded.heartbeat_at_ms,
            finished_at_ms = excluded.finished_at_ms,
            parsed_raw_count = excluded.parsed_raw_count,
            materialized_count = excluded.materialized_count,
            error_message = excluded.error_message,
            source_paths_json = excluded.source_paths_json
        """,
        (
            attempt_id,
            source_path,
            _origin_value(origin),
            status_value,
            phase,
            storage_route,
            outcome_code,
            retryable_int,
            disposition.evidence_ref if disposition is not None else None,
            disposition.diagnostic if disposition is not None else None,
            disposition.remediation if disposition is not None else None,
            started_at_ms,
            heartbeat_at_ms,
            finished_at_ms,
            parsed_raw_count,
            materialized_count,
            error_message,
            source_paths_json,
        ),
    )
    conn.commit()
    return attempt_id


def add_convergence_debt(
    conn: sqlite3.Connection,
    *,
    stage: str,
    target_type: str,
    target_id: str,
    status: str = "failed",
    priority: int = 0,
    attempts: int = 1,
    last_error: str | None = None,
    next_retry_at: str | None = None,
    materializer_version: str | None = None,
    created_at_ms: int,
    updated_at_ms: int | None = None,
    debt_id: str | None = None,
    manage_transaction: bool = True,
) -> str:
    """Add or refresh one convergence-debt row and return its ``debt_id``."""
    require_literal(status, ConvergenceDebtStatus, name="convergence debt status")
    if debt_id is None:
        debt_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO convergence_debt (
            debt_id,
            stage,
            target_type,
            target_id,
            status,
            priority,
            attempts,
            last_error,
            next_retry_at,
            materializer_version,
            created_at_ms,
            updated_at_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (stage, target_type, target_id) DO UPDATE SET
            debt_id = convergence_debt.debt_id,
            status = excluded.status,
            priority = excluded.priority,
            attempts = convergence_debt.attempts + excluded.attempts,
            last_error = excluded.last_error,
            next_retry_at = excluded.next_retry_at,
            materializer_version = excluded.materializer_version,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            debt_id,
            stage,
            target_type,
            target_id,
            status,
            priority,
            attempts,
            last_error,
            next_retry_at,
            materializer_version,
            created_at_ms,
            updated_at_ms if updated_at_ms is not None else created_at_ms,
        ),
    )
    if manage_transaction:
        conn.commit()
    return debt_id


def record_cursor_lag_sample(
    conn: sqlite3.Connection,
    *,
    family: str,
    source_path: str | None,
    lag_ms: int,
    severity: str,
    sampled_at_ms: int,
    stuck_file_count: int = 1,
    p50_lag_ms: int | None = None,
    p95_lag_ms: int | None = None,
    sample_id: str | None = None,
) -> str:
    """Record one cursor lag observation and return its sample id."""
    require_literal(severity, CursorLagSeverity, name="cursor lag severity")
    if sample_id is None:
        sample_id = str(uuid.uuid4())
    resolved_p50_lag_ms = lag_ms if p50_lag_ms is None else p50_lag_ms
    resolved_p95_lag_ms = lag_ms if p95_lag_ms is None else p95_lag_ms
    conn.execute(
        """
        INSERT INTO cursor_lag_samples (
            sample_id,
            family,
            source_path,
            lag_ms,
            stuck_file_count,
            p50_lag_ms,
            p95_lag_ms,
            severity,
            sampled_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sample_id) DO UPDATE SET
            family = excluded.family,
            source_path = excluded.source_path,
            lag_ms = excluded.lag_ms,
            stuck_file_count = excluded.stuck_file_count,
            p50_lag_ms = excluded.p50_lag_ms,
            p95_lag_ms = excluded.p95_lag_ms,
            severity = excluded.severity,
            sampled_at_ms = excluded.sampled_at_ms
        """,
        (
            sample_id,
            family,
            source_path,
            lag_ms,
            stuck_file_count,
            resolved_p50_lag_ms,
            resolved_p95_lag_ms,
            severity,
            sampled_at_ms,
        ),
    )
    conn.commit()
    return sample_id


def read_cursor_lag_sample(conn: sqlite3.Connection, sample_id: str) -> ArchiveCursorLagSample:
    """Read one cursor lag sample by id."""
    row = conn.execute(
        """
        SELECT sample_id, family, source_path, lag_ms, stuck_file_count, p50_lag_ms, p95_lag_ms, severity, sampled_at_ms
        FROM cursor_lag_samples
        WHERE sample_id = ?
        """,
        (sample_id,),
    ).fetchone()
    if row is None:
        raise KeyError(sample_id)
    return ArchiveCursorLagSample(*row)


def list_cursor_lag_samples(
    conn: sqlite3.Connection,
    *,
    family: str | None = None,
    source_path: str | None = None,
) -> tuple[ArchiveCursorLagSample, ...]:
    """Return cursor lag samples ordered by newest sample first."""
    query = """
        SELECT sample_id, family, source_path, lag_ms, stuck_file_count, p50_lag_ms, p95_lag_ms, severity, sampled_at_ms
        FROM cursor_lag_samples
    """
    clauses: list[str] = []
    params: list[object] = []
    if family is not None:
        clauses.append("family = ?")
        params.append(family)
    if source_path is not None:
        clauses.append("source_path = ?")
        params.append(source_path)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY sampled_at_ms DESC, sample_id DESC"
    return tuple(ArchiveCursorLagSample(*row) for row in conn.execute(query, tuple(params)).fetchall())


def record_daemon_stage_event(
    conn: sqlite3.Connection,
    *,
    stage: str,
    status: str,
    observed_at_ms: int,
    attempt_id: str | None = None,
    payload: dict[str, object] | None = None,
    event_id: str | None = None,
    commit: bool = True,
) -> str:
    """Record one daemon stage event and return its event id.

    ``commit=False`` leaves the row in the caller's open transaction, so a
    caller batching several events (or pairing one with the attempt-row update
    it describes) takes a single commit instead of one per row. The row itself
    is written in full either way.
    """
    if event_id is None:
        event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO daemon_stage_events (
            event_id, attempt_id, stage, status, observed_at_ms, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            attempt_id = excluded.attempt_id,
            stage = excluded.stage,
            status = excluded.status,
            observed_at_ms = excluded.observed_at_ms,
            payload_json = excluded.payload_json
        """,
        (event_id, attempt_id, stage, status, observed_at_ms, _json_dumps(payload or {})),
    )
    if commit:
        conn.commit()
    return event_id


def record_daemon_lifecycle_start(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    started_at_ms: int,
    details: dict[str, object] | None = None,
) -> None:
    """Create the authoritative lifecycle row for one daemon process."""
    conn.execute(
        """
        INSERT INTO daemon_lifecycle (
            run_id, started_at_ms, last_heartbeat_at_ms, details_json
        ) VALUES (?, ?, ?, ?)
        """,
        (run_id, started_at_ms, started_at_ms, _json_dumps(details or {})),
    )
    conn.commit()


def record_daemon_lifecycle_heartbeat(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    heartbeat_at_ms: int,
) -> None:
    """Advance a daemon process's durable heartbeat."""
    conn.execute(
        """
        UPDATE daemon_lifecycle
        SET last_heartbeat_at_ms = ?
        WHERE run_id = ?
        """,
        (heartbeat_at_ms, run_id),
    )
    conn.commit()


def record_daemon_lifecycle_signal(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    signal_name: str,
    observed_at_ms: int,
) -> None:
    """Persist the terminating signal before control leaves the process."""
    conn.execute(
        """
        UPDATE daemon_lifecycle
        SET signal = ?, last_heartbeat_at_ms = ?
        WHERE run_id = ?
        """,
        (signal_name, observed_at_ms, run_id),
    )
    conn.commit()


def record_daemon_lifecycle_stop(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stopped_at_ms: int,
    exit_kind: str,
) -> None:
    """Mark a daemon lifecycle row stopped without erasing its signal."""
    conn.execute(
        """
        UPDATE daemon_lifecycle
        SET stopped_at_ms = ?, last_heartbeat_at_ms = ?, exit_kind = ?
        WHERE run_id = ?
        """,
        (stopped_at_ms, stopped_at_ms, exit_kind, run_id),
    )
    conn.commit()


def latest_daemon_lifecycle(conn: sqlite3.Connection) -> ArchiveDaemonLifecycle | None:
    """Read the newest daemon lifecycle row, if the ops tier has one."""
    row = conn.execute(
        """
        SELECT run_id, started_at_ms, stopped_at_ms, last_heartbeat_at_ms,
               signal, exit_kind, details_json
        FROM daemon_lifecycle
        ORDER BY started_at_ms DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return _daemon_lifecycle_from_row(row)


def _daemon_lifecycle_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveDaemonLifecycle:
    return ArchiveDaemonLifecycle(
        run_id=str(row[0]),
        started_at_ms=_int_value(row[1]),
        stopped_at_ms=None if row[2] is None else _int_value(row[2]),
        last_heartbeat_at_ms=_int_value(row[3]),
        signal=None if row[4] is None else str(row[4]),
        exit_kind=None if row[5] is None else str(row[5]),
        details=_json_loads(row[6] if isinstance(row[6], str) else None),
    )


@dataclass(frozen=True, slots=True)
class UnreconciledDaemonRun:
    """A daemon run with no termination receipt, and when the next run began."""

    lifecycle: ArchiveDaemonLifecycle
    next_started_at_ms: int | None
    """Start of the run after this one: the latest instant this run can have
    ended. ``None`` when no later run is recorded."""


def unreconciled_daemon_runs(conn: sqlite3.Connection, *, current_run_id: str) -> tuple[UnreconciledDaemonRun, ...]:
    """Return every ended-or-vanished run other than ``current_run_id`` that has no receipt, oldest first."""
    rows = conn.execute(
        """
        SELECT l.run_id, l.started_at_ms, l.stopped_at_ms, l.last_heartbeat_at_ms,
               l.signal, l.exit_kind, l.details_json, l.next_started_at_ms
        FROM (
            SELECT daemon_lifecycle.*,
                   LEAD(started_at_ms) OVER (ORDER BY started_at_ms, run_id) AS next_started_at_ms
            FROM daemon_lifecycle
        ) AS l
        LEFT JOIN daemon_termination_receipts AS r ON r.run_id = l.run_id
        WHERE r.run_id IS NULL AND l.run_id != ?
        ORDER BY l.started_at_ms, l.run_id
        """,
        (current_run_id,),
    ).fetchall()
    return tuple(
        UnreconciledDaemonRun(
            lifecycle=_daemon_lifecycle_from_row(row),
            next_started_at_ms=None if row[7] is None else _int_value(row[7]),
        )
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class ArchiveDaemonTerminationReceipt:
    """One persisted termination receipt (``receipt`` is its JSON document)."""

    run_id: str
    classification: str
    reconciled_at_ms: int
    reconciled_by_run_id: str
    receipt: dict[str, object]


def record_daemon_termination_receipt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    classification: str,
    reconciled_at_ms: int,
    reconciled_by_run_id: str,
    receipt: dict[str, object],
) -> bool:
    """Persist a run's termination receipt once; return whether this call wrote it.

    A run is reconciled at most once: a second reconciliation (a restart that
    raced the first, a retried startup) is a no-op, never a rewrite, so the
    receipt a reader saw cannot change under it.
    """
    require_literal(classification, DaemonTerminationClass, name="daemon termination classification")
    cursor = conn.execute(
        """
        INSERT INTO daemon_termination_receipts (
            run_id, classification, reconciled_at_ms, reconciled_by_run_id, receipt_json
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO NOTHING
        """,
        (run_id, classification, reconciled_at_ms, reconciled_by_run_id, _json_dumps(receipt)),
    )
    conn.commit()
    return cursor.rowcount == 1


def latest_daemon_termination_receipt(conn: sqlite3.Connection) -> ArchiveDaemonTerminationReceipt | None:
    """Return the receipt of the most recently started run that has one."""
    row = conn.execute(
        """
        SELECT r.run_id, r.classification, r.reconciled_at_ms, r.reconciled_by_run_id, r.receipt_json
        FROM daemon_termination_receipts AS r
        LEFT JOIN daemon_lifecycle AS l ON l.run_id = r.run_id
        ORDER BY COALESCE(l.started_at_ms, r.reconciled_at_ms) DESC, r.run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return ArchiveDaemonTerminationReceipt(
        run_id=str(row[0]),
        classification=str(row[1]),
        reconciled_at_ms=_int_value(row[2]),
        reconciled_by_run_id=str(row[3]),
        receipt=_json_loads(row[4] if isinstance(row[4], str) else None),
    )


def read_daemon_stage_event(conn: sqlite3.Connection, event_id: str) -> ArchiveDaemonStageEvent:
    """Read one daemon stage event by id."""
    row = conn.execute(
        """
        SELECT event_id, attempt_id, stage, status, observed_at_ms, payload_json
        FROM daemon_stage_events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if row is None:
        raise KeyError(event_id)
    return _stage_event_from_row(row)


def list_daemon_stage_events(
    conn: sqlite3.Connection,
    *,
    attempt_id: str | None = None,
    stage: str | None = None,
) -> tuple[ArchiveDaemonStageEvent, ...]:
    """Return daemon stage events ordered by newest observation first."""
    query = """
        SELECT event_id, attempt_id, stage, status, observed_at_ms, payload_json
        FROM daemon_stage_events
    """
    clauses: list[str] = []
    params: list[object] = []
    if attempt_id is not None:
        clauses.append("attempt_id = ?")
        params.append(attempt_id)
    if stage is not None:
        clauses.append("stage = ?")
        params.append(stage)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY observed_at_ms DESC, event_id DESC"
    return tuple(_stage_event_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def upsert_embedding_catchup_run(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    started_at_ms: int,
    finished_at_ms: int | None = None,
    status: OperationStatus | OperationRunStatus | str,
    origin: Origin | str | None = None,
    scanned_sessions: int = 0,
    embedded_sessions: int = 0,
    skipped_sessions: int = 0,
    error_count: int = 0,
    embedded_messages: int = 0,
    estimated_cost_usd: float | None = None,
    error_message: str | None = None,
) -> str:
    """Create or replace one ``embedding_catchup_runs`` row and return ``run_id``."""
    if run_id is None:
        run_id = str(uuid.uuid4())
    status_value = require_literal(status, OperationRunStatus, name="embedding catchup status")
    conn.execute(
        """
        INSERT INTO embedding_catchup_runs (
            run_id,
            started_at_ms,
            finished_at_ms,
            status,
            origin,
            scanned_sessions,
            embedded_sessions,
            skipped_sessions,
            error_count,
            embedded_messages,
            estimated_cost_usd,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            started_at_ms = excluded.started_at_ms,
            finished_at_ms = excluded.finished_at_ms,
            status = excluded.status,
            origin = excluded.origin,
            scanned_sessions = excluded.scanned_sessions,
            embedded_sessions = excluded.embedded_sessions,
            skipped_sessions = excluded.skipped_sessions,
            error_count = excluded.error_count,
            embedded_messages = excluded.embedded_messages,
            estimated_cost_usd = excluded.estimated_cost_usd,
            error_message = excluded.error_message
        """,
        (
            run_id,
            started_at_ms,
            finished_at_ms,
            status_value,
            _origin_value(origin),
            scanned_sessions,
            embedded_sessions,
            skipped_sessions,
            error_count,
            embedded_messages,
            estimated_cost_usd,
            error_message,
        ),
    )
    conn.commit()
    return run_id


def list_embedding_catchup_runs(
    conn: sqlite3.Connection,
    *,
    status: OperationStatus | OperationRunStatus | str | None = None,
    schema: str = "main",
) -> tuple[ArchiveEmbeddingCatchupRun, ...]:
    """Return embedding catchup runs ordered by newest start first."""
    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported embedding catchup reader schema: {schema!r}")
    query = f"""
        SELECT
            run_id, started_at_ms, finished_at_ms, status, origin,
            scanned_sessions, embedded_sessions, skipped_sessions, error_count,
            embedded_messages, estimated_cost_usd, error_message
        FROM {schema}.embedding_catchup_runs
    """
    params: tuple[object, ...] = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (require_literal(status, OperationRunStatus, name="embedding catchup status"),)
    query += " ORDER BY started_at_ms DESC, run_id DESC"

    return tuple(ArchiveEmbeddingCatchupRun(*row) for row in conn.execute(query, params).fetchall())


def read_embedding_catchup_run(conn: sqlite3.Connection, run_id: str) -> ArchiveEmbeddingCatchupRun:
    """Read one embedding catchup run by ``run_id``."""
    row = conn.execute(
        """
        SELECT
            run_id, started_at_ms, finished_at_ms, status, origin,
            scanned_sessions, embedded_sessions, skipped_sessions, error_count,
            embedded_messages, estimated_cost_usd, error_message
        FROM embedding_catchup_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    return ArchiveEmbeddingCatchupRun(*row)


def record_mcp_call(
    conn: sqlite3.Connection,
    *,
    tool_name: str,
    started_at_ms: int,
    finished_at_ms: int,
    success: bool,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
    error_detail: str | None = None,
    call_id: str | None = None,
) -> str:
    """Record one durable MCP tool call-log entry and return its call id.

    ``ops.db`` is the disposable telemetry tier (#7s57): the filesystem outbox
    retries delivery across daemon/process outages, while this writer remains a
    freeform-additive table (``CREATE TABLE IF NOT EXISTS``, no migration
    chain) recording tool name, session id (when the caller knows one),
    timing, and success/failure per MCP tool invocation, so resume/context
    tool efficacy (``get_resume_brief``, ``compose_context_preamble``, ...)
    can be reconstructed per session.
    """
    if call_id is None:
        call_id = str(uuid.uuid4())
    duration_ms = max(0, finished_at_ms - started_at_ms)
    values = (
        call_id,
        tool_name,
        session_id,
        started_at_ms,
        finished_at_ms,
        duration_ms,
        1 if success else 0,
        error_detail,
    )
    desired_refs: dict[str, str] = {
        value: "member" for value in dict.fromkeys(session_ids) if value and value != session_id
    }
    if session_id is not None:
        desired_refs[session_id] = "primary"
    for relation in desired_refs.values():
        require_literal(relation, McpCallSessionRelation, name="MCP call session relation")
    with conn:
        existing = conn.execute(
            """
            SELECT
                call_id, tool_name, session_id, started_at_ms,
                finished_at_ms, duration_ms, success, error_detail
            FROM mcp_call_log
            WHERE call_id = ?
            """,
            (call_id,),
        ).fetchone()
        if existing is not None and tuple(existing) != values:
            raise ValueError(f"conflicting MCP call payload for call_id {call_id}")
        existing_refs = {
            str(row[0]): str(row[1])
            for row in conn.execute(
                "SELECT session_id, relation FROM mcp_call_session_refs WHERE call_id = ?",
                (call_id,),
            ).fetchall()
        }
        for relation in existing_refs.values():
            require_literal(relation, McpCallSessionRelation, name="stored MCP call session relation")
        if existing_refs and existing_refs != desired_refs:
            raise ValueError(f"conflicting MCP call session refs for call_id {call_id}")
        if existing is None:
            conn.execute(
                """
                INSERT INTO mcp_call_log (
                    call_id,
                    tool_name,
                    session_id,
                    started_at_ms,
                    finished_at_ms,
                    duration_ms,
                    success,
                    error_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        if not existing_refs:
            conn.executemany(
                "INSERT INTO mcp_call_session_refs (call_id, session_id, relation) VALUES (?, ?, ?)",
                ((call_id, ref, relation) for ref, relation in desired_refs.items()),
            )
        conn.execute(
            "DELETE FROM mcp_call_log WHERE started_at_ms < ?",
            (finished_at_ms - MCP_CALL_LOG_RETENTION_MS,),
        )
    return call_id


def list_mcp_calls(
    conn: sqlite3.Connection,
    *,
    session_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 100,
) -> tuple[ArchiveMcpCallLogEntry, ...]:
    """Return MCP call-log entries newest-first, optionally filtered."""
    query = """
        SELECT call_id, tool_name, session_id, started_at_ms, finished_at_ms, duration_ms, success, error_detail
        FROM mcp_call_log AS calls
    """
    clauses: list[str] = []
    params: list[object] = []
    if session_id is not None:
        clauses.append(
            "(calls.session_id = ? OR EXISTS ("
            "SELECT 1 FROM mcp_call_session_refs AS refs "
            "WHERE refs.call_id = calls.call_id AND refs.session_id = ?))"
        )
        params.extend((session_id, session_id))
    if tool_name is not None:
        clauses.append("tool_name = ?")
        params.append(tool_name)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY started_at_ms DESC, call_id DESC LIMIT ?"
    params.append(limit)
    return tuple(_mcp_call_log_entry_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def _mcp_call_log_entry_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveMcpCallLogEntry:
    return ArchiveMcpCallLogEntry(
        call_id=str(row[0]),
        tool_name=str(row[1]),
        session_id=None if row[2] is None else str(row[2]),
        started_at_ms=_int_value(row[3]),
        finished_at_ms=_int_value(row[4]),
        duration_ms=_int_value(row[5]),
        success=bool(row[6]),
        error_detail=None if row[7] is None else str(row[7]),
    )


def read_compact_state(conn: sqlite3.Connection) -> OpsCompactState:
    """Read a compact status snapshot across OPS-tier state tables."""
    cursor_count = int(conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0])

    status_rows = conn.execute("SELECT status, COUNT(*) AS count FROM ingest_attempts GROUP BY status").fetchall()
    attempt_counts = {str(row[0]): int(row[1]) for row in status_rows}

    total_attempts = sum(attempt_counts.values())
    running = attempt_counts.get("running", 0)
    completed = attempt_counts.get("completed", 0)
    failed = attempt_counts.get("failed", 0)

    debt_count = int(conn.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0])

    latest_attempt = conn.execute(
        """
        SELECT attempt_id, status
        FROM ingest_attempts
        ORDER BY COALESCE(heartbeat_at_ms, started_at_ms) DESC, started_at_ms DESC
        LIMIT 1
        """,
    ).fetchone()
    latest_attempt_id = latest_attempt[0] if latest_attempt is not None else None
    latest_attempt_status = latest_attempt[1] if latest_attempt is not None else None

    latest_cursor = conn.execute("SELECT source_path FROM ingest_cursor ORDER BY updated_at_ms DESC LIMIT 1").fetchone()
    latest_cursor_path = latest_cursor[0] if latest_cursor is not None else None

    latest_debt = conn.execute(
        "SELECT stage, priority FROM convergence_debt ORDER BY updated_at_ms DESC LIMIT 1"
    ).fetchone()
    latest_debt_stage = latest_debt[0] if latest_debt is not None else None
    latest_debt_priority = int(latest_debt[1]) if latest_debt is not None else 0

    return OpsCompactState(
        cursor_count=cursor_count,
        ingest_attempt_total=total_attempts,
        ingest_attempt_running=running,
        ingest_attempt_completed=completed,
        ingest_attempt_failed=failed,
        convergence_debt_count=debt_count,
        latest_attempt_id=latest_attempt_id,
        latest_attempt_status=latest_attempt_status,
        latest_cursor_path=latest_cursor_path,
        latest_debt_stage=latest_debt_stage,
        latest_debt_priority=latest_debt_priority,
    )


def _origin_value(origin: Origin | str | None) -> str | None:
    if isinstance(origin, Origin):
        return origin.value
    return origin


def _stage_event_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveDaemonStageEvent:
    return ArchiveDaemonStageEvent(
        event_id=str(row[0]),
        attempt_id=str(row[1]) if row[1] is not None else None,
        stage=str(row[2]),
        status=str(row[3]),
        observed_at_ms=_int_value(row[4]),
        payload=_json_loads(row[5] if isinstance(row[5], str) else None),
    )


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    return 0


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(raw_json: str | None) -> dict[str, object]:
    if not raw_json:
        return {}
    loaded = json.loads(raw_json)
    return loaded if isinstance(loaded, dict) else {}


__all__ = [
    "ArchiveCursorLagSample",
    "ArchiveDaemonLifecycle",
    "ArchiveDaemonTerminationReceipt",
    "ArchiveJudgmentSchedulerReceipt",
    "ArchiveDaemonStageEvent",
    "ArchiveEmbeddingCatchupRun",
    "ArchiveFtsDriftSample",
    "ArchiveSchemaDriftSample",
    "FTS_DRIFT_SAMPLE_RETENTION_MS",
    "FTS_DRIFT_SAMPLE_ROW_CAP",
    "SCHEMA_DRIFT_SAMPLE_RETENTION_MS",
    "SCHEMA_DRIFT_SAMPLE_ROW_CAP",
    "SchemaDriftOriginSummary",
    "OpsCompactState",
    "add_convergence_debt",
    "list_cursor_lag_samples",
    "list_fts_drift_samples",
    "list_schema_drift_samples",
    "iter_schema_drift_signature",
    "latest_daemon_lifecycle",
    "latest_daemon_termination_receipt",
    "list_daemon_stage_events",
    "list_embedding_catchup_runs",
    "read_cursor_lag_sample",
    "read_daemon_stage_event",
    "read_embedding_catchup_run",
    "read_compact_state",
    "record_cursor_lag_sample",
    "record_daemon_lifecycle_heartbeat",
    "record_daemon_lifecycle_signal",
    "record_daemon_lifecycle_start",
    "record_daemon_lifecycle_stop",
    "record_daemon_stage_event",
    "record_daemon_termination_receipt",
    "record_judgment_scheduler_receipt",
    "record_fts_drift_sample",
    "record_ingest_attempt",
    "UnreconciledDaemonRun",
    "unreconciled_daemon_runs",
    "record_schema_drift_sample",
    "summarize_schema_drift_since",
    "upsert_embedding_catchup_run",
    "upsert_ingest_cursor",
]
