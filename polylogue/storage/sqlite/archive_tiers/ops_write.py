"""Minimal ops-tier archive read/write helpers.

Writer module: ops.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

from polylogue.core.enums import Origin


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
class ArchiveOtlpSpan:
    """Compact read-back row for one OTLP span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    origin: str | None
    name: str
    kind: str | None
    started_at_ms: int | None
    ended_at_ms: int | None
    attributes_json: str
    events_json: str


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


def upsert_ingest_cursor(
    conn: sqlite3.Connection,
    *,
    source_path: str,
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
) -> None:
    """Create or refresh one cursor row in ``ingest_cursor``."""
    conn.execute(
        """
        INSERT INTO ingest_cursor (
            source_path,
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
            updated_at_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (source_path) DO UPDATE SET
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
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            source_path,
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
            updated_at_ms,
        ),
    )
    conn.commit()


def record_ingest_attempt(
    conn: sqlite3.Connection,
    *,
    status: str,
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
) -> str:
    """Create or replace one ``ingest_attempts`` row and return its ``attempt_id``."""
    if attempt_id is None:
        attempt_id = str(uuid.uuid4())
    has_storage_route = _table_has_column(conn, "ingest_attempts", "storage_route")
    route_column = "storage_route,\n            " if has_storage_route else ""
    route_value = "?, " if has_storage_route else ""
    route_update = (
        "storage_route = COALESCE(excluded.storage_route, ingest_attempts.storage_route),\n            "
        if has_storage_route
        else ""
    )
    route_params: tuple[object, ...] = (storage_route,) if has_storage_route else ()
    conn.execute(
        f"""
        INSERT INTO ingest_attempts (
            attempt_id,
            source_path,
            origin,
            status,
            phase,
            {route_column}
            started_at_ms,
            heartbeat_at_ms,
            finished_at_ms,
            parsed_raw_count,
            materialized_count,
            error_message,
            source_paths_json
        )
        VALUES (?, ?, ?, ?, ?, {route_value}?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (attempt_id) DO UPDATE SET
            source_path = excluded.source_path,
            origin = excluded.origin,
            status = excluded.status,
            phase = excluded.phase,
            {route_update}
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
            status,
            phase,
            *route_params,
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
) -> str:
    """Add or refresh one convergence-debt row and return its ``debt_id``."""
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
) -> str:
    """Record one daemon stage event and return its event id."""
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
    conn.commit()
    return event_id


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
    status: str,
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
    _ensure_embedding_catchup_run_outcome_columns(conn)
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
            status,
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
    status: str | None = None,
) -> tuple[ArchiveEmbeddingCatchupRun, ...]:
    """Return embedding catchup runs ordered by newest start first."""
    outcome_columns = _embedding_catchup_run_outcome_columns(conn)
    query = """
        SELECT
            run_id, started_at_ms, finished_at_ms, status, origin,
            scanned_sessions, {embedded_sessions}, {skipped_sessions}, {error_count},
            embedded_messages, estimated_cost_usd, error_message
        FROM embedding_catchup_runs
    """.format(**outcome_columns)
    params: tuple[object, ...] = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY started_at_ms DESC, run_id DESC"

    return tuple(ArchiveEmbeddingCatchupRun(*row) for row in conn.execute(query, params).fetchall())


def read_embedding_catchup_run(conn: sqlite3.Connection, run_id: str) -> ArchiveEmbeddingCatchupRun:
    """Read one embedding catchup run by ``run_id``."""
    outcome_columns = _embedding_catchup_run_outcome_columns(conn)
    row = conn.execute(
        """
        SELECT
            run_id, started_at_ms, finished_at_ms, status, origin,
            scanned_sessions, {embedded_sessions}, {skipped_sessions}, {error_count},
            embedded_messages, estimated_cost_usd, error_message
        FROM embedding_catchup_runs
        WHERE run_id = ?
        """.format(**outcome_columns),
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    return ArchiveEmbeddingCatchupRun(*row)


def _ensure_embedding_catchup_run_outcome_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_catchup_runs)")}
    additions = {
        "embedded_sessions": "INTEGER NOT NULL DEFAULT 0 CHECK(embedded_sessions >= 0)",
        "skipped_sessions": "INTEGER NOT NULL DEFAULT 0 CHECK(skipped_sessions >= 0)",
        "error_count": "INTEGER NOT NULL DEFAULT 0 CHECK(error_count >= 0)",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE embedding_catchup_runs ADD COLUMN {name} {definition}")


def _embedding_catchup_run_outcome_columns(conn: sqlite3.Connection) -> dict[str, str]:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_catchup_runs)")}
    return {
        "embedded_sessions": "embedded_sessions" if "embedded_sessions" in existing else "0 AS embedded_sessions",
        "skipped_sessions": "skipped_sessions" if "skipped_sessions" in existing else "0 AS skipped_sessions",
        "error_count": "error_count"
        if "error_count" in existing
        else "CASE WHEN error_message IS NULL THEN 0 ELSE 1 END AS error_count",
    }


def upsert_otlp_span(
    conn: sqlite3.Connection,
    *,
    trace_id: str,
    span_id: str,
    name: str,
    parent_span_id: str | None = None,
    origin: Origin | str | None = None,
    kind: str | None = None,
    started_at_ms: int | None = None,
    ended_at_ms: int | None = None,
    attributes_json: str = "{}",
    events_json: str = "[]",
) -> None:
    """Create or replace one ``otlp_spans`` row."""
    conn.execute(
        """
        INSERT INTO otlp_spans (
            trace_id,
            span_id,
            parent_span_id,
            origin,
            name,
            kind,
            started_at_ms,
            ended_at_ms,
            attributes_json,
            events_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (trace_id, span_id) DO UPDATE SET
            parent_span_id = excluded.parent_span_id,
            origin = excluded.origin,
            name = excluded.name,
            kind = excluded.kind,
            started_at_ms = excluded.started_at_ms,
            ended_at_ms = excluded.ended_at_ms,
            attributes_json = excluded.attributes_json,
            events_json = excluded.events_json
        """,
        (
            trace_id,
            span_id,
            parent_span_id,
            _origin_value(origin),
            name,
            kind,
            started_at_ms,
            ended_at_ms,
            attributes_json,
            events_json,
        ),
    )
    conn.commit()


def list_otlp_spans(
    conn: sqlite3.Connection,
    *,
    trace_id: str | None = None,
) -> tuple[ArchiveOtlpSpan, ...]:
    """Return OTLP spans ordered by start time descending."""
    query = """
        SELECT
            trace_id, span_id, parent_span_id, origin, name, kind,
            started_at_ms, ended_at_ms, attributes_json, events_json
        FROM otlp_spans
    """
    params: tuple[object, ...] = ()
    if trace_id is not None:
        query += " WHERE trace_id = ?"
        params = (trace_id,)
    query += " ORDER BY started_at_ms DESC, span_id DESC"

    return tuple(ArchiveOtlpSpan(*row) for row in conn.execute(query, params).fetchall())


def read_otlp_span(conn: sqlite3.Connection, trace_id: str, span_id: str) -> ArchiveOtlpSpan:
    """Read one OTLP span by composite primary key."""
    row = conn.execute(
        """
        SELECT
            trace_id, span_id, parent_span_id, origin, name, kind,
            started_at_ms, ended_at_ms, attributes_json, events_json
        FROM otlp_spans
        WHERE trace_id = ? AND span_id = ?
        """,
        (trace_id, span_id),
    ).fetchone()
    if row is None:
        raise KeyError((trace_id, span_id))
    return ArchiveOtlpSpan(*row)


def record_mcp_call(
    conn: sqlite3.Connection,
    *,
    tool_name: str,
    started_at_ms: int,
    finished_at_ms: int,
    success: bool,
    session_id: str | None = None,
    error_detail: str | None = None,
    call_id: str | None = None,
) -> str:
    """Record one durable MCP tool call-log entry and return its call id.

    ``ops.db`` is the disposable telemetry tier (#7s57): this is a best-effort,
    freeform-additive table (``CREATE TABLE IF NOT EXISTS``, no migration
    chain) recording tool name, session id (when the caller knows one),
    timing, and success/failure per MCP tool invocation, so resume/context
    tool efficacy (``get_resume_brief``, ``compose_context_preamble``, ...)
    can be reconstructed per session.
    """
    if call_id is None:
        call_id = str(uuid.uuid4())
    duration_ms = max(0, finished_at_ms - started_at_ms)
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
        (
            call_id,
            tool_name,
            session_id,
            started_at_ms,
            finished_at_ms,
            duration_ms,
            1 if success else 0,
            error_detail,
        ),
    )
    conn.commit()
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
        FROM mcp_call_log
    """
    clauses: list[str] = []
    params: list[object] = []
    if session_id is not None:
        clauses.append("session_id = ?")
        params.append(session_id)
    if tool_name is not None:
        clauses.append("tool_name = ?")
        params.append(tool_name)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY started_at_ms DESC, call_id DESC LIMIT ?"
    params.append(limit)
    return tuple(_mcp_call_log_entry_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def read_mcp_call(conn: sqlite3.Connection, call_id: str) -> ArchiveMcpCallLogEntry:
    """Read one MCP call-log entry by id."""
    row = conn.execute(
        """
        SELECT call_id, tool_name, session_id, started_at_ms, finished_at_ms, duration_ms, success, error_detail
        FROM mcp_call_log
        WHERE call_id = ?
        """,
        (call_id,),
    ).fetchone()
    if row is None:
        raise KeyError(call_id)
    return _mcp_call_log_entry_from_row(row)


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


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})"))


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
    "ArchiveDaemonStageEvent",
    "ArchiveEmbeddingCatchupRun",
    "ArchiveOtlpSpan",
    "OpsCompactState",
    "add_convergence_debt",
    "list_cursor_lag_samples",
    "list_daemon_stage_events",
    "list_embedding_catchup_runs",
    "list_otlp_spans",
    "read_cursor_lag_sample",
    "read_daemon_stage_event",
    "read_embedding_catchup_run",
    "read_otlp_span",
    "read_compact_state",
    "record_cursor_lag_sample",
    "record_daemon_stage_event",
    "record_ingest_attempt",
    "upsert_embedding_catchup_run",
    "upsert_ingest_cursor",
    "upsert_otlp_span",
]
