"""Session insight storage writes (profiles, timelines, aggregates)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel

from polylogue.core.timestamps import parse_timestamp
from polylogue.storage.runtime import (
    SessionLatencyProfileRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    ThreadRecord,
    _json_array_or_none,
    _json_or_none,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_SYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}
_DELETE_WHERE_IN_CHUNK_SIZE = 900
SqlValue = str | int | float | None
SqlBindings = tuple[SqlValue, ...]


_SessionRecordT = TypeVar("_SessionRecordT")


def _dedupe_records_by_session(records: Sequence[_SessionRecordT]) -> list[_SessionRecordT]:
    deduped: dict[str, _SessionRecordT] = {}
    for record in records:
        session_id = str(getattr(record, "session_id", ""))
        if session_id:
            deduped[session_id] = record
    return list(deduped.values())


_SESSION_PROFILE_BASE_COLUMNS = (
    "session_id",
    "logical_session_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "source_name",
    "title",
    "first_message_at",
    "last_message_at",
    "canonical_session_date",
    "repo_paths_json",
    "repo_names_json",
    "tags_json",
    "auto_tags_json",
    "message_count",
    "substantive_count",
    "attachment_count",
    "work_event_count",
    "phase_count",
    "word_count",
    "tool_use_count",
    "thinking_count",
    "total_cost_usd",
    "total_duration_ms",
    "engaged_duration_ms",
    "tool_active_duration_ms",
    "wall_duration_ms",
    "workflow_shape",
    "workflow_shape_confidence",
    "workflow_shape_features_json",
    "terminal_state",
    "terminal_state_confidence",
    "terminal_state_evidence_json",
    "cost_is_estimated",
    "thinking_duration_ms",
    "output_duration_ms",
    "tool_duration_ms",
    "latency_percentiles_ms_json",
    "tool_calls_per_minute",
    "timing_provenance",
    "total_input_tokens",
    "total_output_tokens",
    "total_cache_read_tokens",
    "total_cache_write_tokens",
    "total_credit_cost",
    "cost_provenance",
    "per_model_cost_json",
)
_SESSION_PROFILE_PAYLOAD_COLUMNS = (
    "evidence_payload_json",
    "inference_payload_json",
    "enrichment_payload_json",
    "search_text",
    "evidence_search_text",
    "inference_search_text",
    "enrichment_search_text",
    "enrichment_version",
    "enrichment_family",
    "inference_version",
    "inference_family",
)
_SESSION_LATENCY_PROFILE_COLUMNS = (
    "session_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "source_name",
    "title",
    "first_message_at",
    "last_message_at",
    "canonical_session_date",
    "median_tool_call_ms",
    "p90_tool_call_ms",
    "max_tool_call_ms",
    "stuck_tool_count",
    "median_agent_response_ms",
    "median_user_response_ms",
    "tool_call_count_by_category_json",
    "evidence_payload_json",
    "search_text",
)
_SESSION_WORK_EVENT_BASE_COLUMNS = (
    "session_id",
    "position",
    "work_event_type",
    "summary",
    "confidence",
    "start_index",
    "end_index",
    "started_at_ms",
    "ended_at_ms",
    "duration_ms",
    "file_paths_json",
    "tools_used_json",
    "input_high_water_mark",
    "input_high_water_mark_source",
)
_TIMELINE_PAYLOAD_COLUMNS = (
    "evidence_json",
    "inference_json",
    "search_text",
)
_SESSION_PHASE_BASE_COLUMNS = (
    "session_id",
    "position",
    "phase_type",
    "confidence",
    "start_index",
    "end_index",
    "started_at_ms",
    "ended_at_ms",
    "duration_ms",
    "tool_counts_json",
    "word_count",
    "input_high_water_mark",
    "input_high_water_mark_source",
)


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    key = (id(conn), f"{table}.{column}")
    cached = _SYNC_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    found = any(str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())
    _SYNC_COLUMN_CACHE[key] = found
    return found


def _placeholders(columns: Sequence[str]) -> str:
    return ", ".join("?" for _ in columns)


def build_insert_sql(table: str, columns: Sequence[str]) -> str:
    return f"""
        INSERT INTO {table} (
            {", ".join(columns)}
        ) VALUES ({_placeholders(columns)})
        """


def _delete_where_in(conn: sqlite3.Connection, table: str, column: str, values: Sequence[str]) -> None:
    normalized = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    for start in range(0, len(normalized), _DELETE_WHERE_IN_CHUNK_SIZE):
        chunk = normalized[start : start + _DELETE_WHERE_IN_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", chunk)


def _with_fallback_payload_column(
    base_columns: tuple[str, ...],
    payload_columns: tuple[str, ...],
    *,
    has_fallback_payload: bool,
) -> tuple[str, ...]:
    if has_fallback_payload:
        return base_columns + ("payload_json",) + payload_columns
    return base_columns + payload_columns


def _compose_bindings(
    base_values: Sequence[SqlValue],
    payload_values: Sequence[SqlValue],
    *,
    has_fallback_payload: bool,
    fallback_payload_json: str | None,
) -> SqlBindings:
    values = list(base_values)
    if has_fallback_payload:
        values.append(fallback_payload_json)
    values.extend(payload_values)
    return tuple(values)


# The denormalized native session_profiles columns (workflow_shape /
# terminal_state + their confidences) are the authoritative ranking signals.
# Both read paths (insights.archive.SessionProfileInsight.from_record and the
# live _session_profile_components_from_archive_row) reconcile the inference
# payload onto those columns, so persisting a second copy inside
# inference_payload_json would only re-open a write/read drift surface (#14).
# Strip them from the stored inference payload; the read paths repopulate them
# from the native columns.
_INFERENCE_NATIVE_MIRRORED_FIELDS: frozenset[str] = frozenset(
    {
        "workflow_shape",
        "workflow_shape_confidence",
        "terminal_state",
        "terminal_state_confidence",
    }
)


def _stored_inference_payload_json(record: SessionProfileRecord) -> str | None:
    return _json_or_none(
        record.inference_payload.model_dump(mode="json", exclude=set(_INFERENCE_NATIVE_MIRRORED_FIELDS))
    )


def _fallback_profile_payload_json(record: SessionProfileRecord) -> str | None:
    return _json_or_none(
        {
            **record.evidence_payload.model_dump(mode="json"),
            **record.inference_payload.model_dump(mode="json"),
            "session_id": str(record.session_id),
            "logical_session_id": str(record.logical_session_id),
            "provider": record.source_name,
            "title": record.title,
        }
    )


def session_profile_insert_columns(
    *,
    has_fallback_payload: bool,
) -> tuple[str, ...]:
    return _with_fallback_payload_column(
        _SESSION_PROFILE_BASE_COLUMNS,
        _SESSION_PROFILE_PAYLOAD_COLUMNS,
        has_fallback_payload=has_fallback_payload,
    )


def session_profile_insert_values(
    record: SessionProfileRecord,
    *,
    has_fallback_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.session_id,
        record.logical_session_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.source_name,
        record.title,
        record.first_message_at,
        record.last_message_at,
        record.canonical_session_date,
        _json_array_or_none(record.repo_paths),
        _json_array_or_none(record.repo_names),
        _json_array_or_none(record.tags),
        _json_array_or_none(record.auto_tags),
        record.message_count,
        record.substantive_count,
        record.attachment_count,
        record.work_event_count,
        record.phase_count,
        record.word_count,
        record.tool_use_count,
        record.thinking_count,
        record.total_cost_usd,
        record.total_duration_ms,
        record.engaged_duration_ms,
        record.tool_active_duration_ms,
        record.wall_duration_ms,
        record.workflow_shape,
        record.workflow_shape_confidence,
        record.workflow_shape_features_json,
        record.terminal_state,
        record.terminal_state_confidence,
        record.terminal_state_evidence_json,
        int(record.cost_is_estimated),
        record.thinking_duration_ms,
        record.output_duration_ms,
        record.tool_duration_ms,
        record.latency_percentiles_ms_json,
        record.tool_calls_per_minute,
        record.timing_provenance,
        record.total_input_tokens,
        record.total_output_tokens,
        record.total_cache_read_tokens,
        record.total_cache_write_tokens,
        record.total_credit_cost,
        record.cost_provenance,
        record.per_model_cost_json,
    ]
    payload_values: tuple[SqlValue, ...] = (
        _json_or_none(record.evidence_payload),
        _stored_inference_payload_json(record),
        _json_or_none(record.enrichment_payload),
        record.search_text,
        record.evidence_search_text,
        record.inference_search_text,
        record.enrichment_search_text,
        record.enrichment_version,
        record.enrichment_family,
        record.inference_version,
        record.inference_family,
    )
    return _compose_bindings(
        base_values,
        payload_values,
        has_fallback_payload=has_fallback_payload,
        fallback_payload_json=_fallback_profile_payload_json(record),
    )


def session_latency_profile_insert_values(record: SessionLatencyProfileRecord) -> SqlBindings:
    return (
        record.session_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.source_name,
        record.title,
        record.first_message_at,
        record.last_message_at,
        record.canonical_session_date,
        record.median_tool_call_ms,
        record.p90_tool_call_ms,
        record.max_tool_call_ms,
        record.stuck_tool_count,
        record.median_agent_response_ms,
        record.median_user_response_ms,
        record.tool_call_count_by_category_json,
        record.evidence_payload_json,
        record.search_text,
    )


def _fallback_timeline_payload_json(
    evidence_payload: BaseModel,
    inference_payload: BaseModel,
) -> str | None:
    return _json_or_none(
        {
            **evidence_payload.model_dump(mode="json"),
            **inference_payload.model_dump(mode="json"),
        }
    )


def _epoch_ms_or_none(value: str | None) -> int | None:
    if not value:
        return None
    parsed = parse_timestamp(value)
    return int(parsed.timestamp() * 1000) if parsed is not None else None


def session_work_event_insert_columns(
    *,
    has_fallback_payload: bool,
) -> tuple[str, ...]:
    return _with_fallback_payload_column(
        _SESSION_WORK_EVENT_BASE_COLUMNS,
        _TIMELINE_PAYLOAD_COLUMNS,
        has_fallback_payload=has_fallback_payload,
    )


def session_work_event_insert_values(
    record: SessionWorkEventRecord,
    *,
    has_fallback_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.session_id,
        record.event_index,
        record.heuristic_label,
        record.summary,
        record.confidence,
        record.start_index,
        record.end_index,
        _epoch_ms_or_none(record.start_time),
        _epoch_ms_or_none(record.end_time),
        record.duration_ms,
        _json_array_or_none(record.file_paths) or "[]",
        _json_array_or_none(record.tools_used) or "[]",
        record.input_high_water_mark,
        record.input_high_water_mark_source,
    ]
    payload_values: tuple[SqlValue, ...] = (
        _json_or_none(record.evidence_payload),
        _json_or_none(record.inference_payload),
        record.search_text,
    )
    return _compose_bindings(
        base_values,
        payload_values,
        has_fallback_payload=has_fallback_payload,
        fallback_payload_json=None,
    )


def session_phase_insert_columns(
    *,
    has_fallback_payload: bool,
) -> tuple[str, ...]:
    return _with_fallback_payload_column(
        _SESSION_PHASE_BASE_COLUMNS,
        _TIMELINE_PAYLOAD_COLUMNS,
        has_fallback_payload=has_fallback_payload,
    )


def session_phase_insert_values(
    record: SessionPhaseRecord,
    *,
    has_fallback_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.session_id,
        record.phase_index,
        record.kind,
        record.confidence,
        record.start_index,
        record.end_index,
        _epoch_ms_or_none(record.start_time),
        _epoch_ms_or_none(record.end_time),
        record.duration_ms,
        _json_or_none(record.tool_counts),
        record.word_count,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
    ]
    payload_values: tuple[SqlValue, ...] = (
        _json_or_none(record.evidence_payload),
        _json_or_none(record.inference_payload),
        record.search_text,
    )
    return _compose_bindings(
        base_values,
        payload_values,
        has_fallback_payload=has_fallback_payload,
        fallback_payload_json=None,
    )


# The ``threads`` row has two owners split by column (#1743 P13):
#
# * The *spine* — ``created_at_ms``, ``session_count``, ``depth`` — plus the
#   ``thread_sessions`` membership join is owned by the topology refresh
#   (``archive_tiers/write.py:_refresh_thread`` at ingest, and the equivalent
#   spine writer in the insight rebuild). It is derived from the parent/root
#   chain and the root session's timestamps.
# * The *analytics* columns below are owned by the session-insight materializer
#   (``build_thread_records_for_roots_sync``). They are written via an
#   ``ON CONFLICT(thread_id) DO UPDATE`` upsert that never deletes the row, so
#   it can neither zero ``created_at_ms`` nor cascade-wipe ``thread_sessions``.
_THREAD_ANALYTICS_COLUMNS: tuple[str, ...] = (
    "thread_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "start_time",
    "end_time",
    "dominant_repo",
    "session_ids_json",
    "branch_count",
    "total_messages",
    "total_cost_usd",
    "wall_duration_ms",
    "work_event_breakdown_json",
    "payload_json",
    "search_text",
)


def thread_insert_values(record: ThreadRecord) -> SqlBindings:
    """Bind the full thread column set (spine + analytics).

    Retained for the async thread query path
    (``session_insight_thread_queries.replace_thread``), which still rewrites
    the whole row. The sync materializer uses ``thread_analytics_insert_values``
    plus a dedicated spine writer instead (#1743 P13).
    """
    return (
        record.thread_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.start_time,
        record.end_time,
        record.dominant_repo,
        _json_array_or_none(record.session_ids),
        record.session_count,
        record.depth,
        record.branch_count,
        record.total_messages,
        record.total_cost_usd,
        record.wall_duration_ms,
        _json_or_none(record.work_event_breakdown or {}),
        _json_or_none(record.payload),
        record.search_text,
    )


def thread_analytics_insert_values(record: ThreadRecord) -> SqlBindings:
    """Bind only the analytics columns the insight materializer owns.

    ``session_count``/``depth``/``created_at_ms`` (the topology-owned spine) are
    intentionally excluded so the analytics upsert preserves them.
    """
    return (
        record.thread_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.start_time,
        record.end_time,
        record.dominant_repo,
        _json_array_or_none(record.session_ids),
        record.branch_count,
        record.total_messages,
        record.total_cost_usd,
        record.wall_duration_ms,
        _json_or_none(record.work_event_breakdown or {}),
        _json_or_none(record.payload),
        record.search_text,
    )


def _thread_analytics_upsert_sql() -> str:
    update_columns = _THREAD_ANALYTICS_COLUMNS[1:]
    return f"""
        INSERT INTO threads (
            {", ".join(_THREAD_ANALYTICS_COLUMNS)}
        ) VALUES ({_placeholders(_THREAD_ANALYTICS_COLUMNS)})
        ON CONFLICT(thread_id) DO UPDATE SET
            {", ".join(f"{column} = excluded.{column}" for column in update_columns)}
        """


def session_tag_rollup_insert_values(record: SessionTagRollupRecord) -> SqlBindings:
    return (
        record.tag,
        record.bucket_day,
        record.source_name,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.session_count,
        record.logical_session_count,
        _json_array_or_none(record.logical_session_ids),
        record.explicit_count,
        record.auto_count,
        _json_or_none(record.repo_breakdown),
        record.search_text,
    )


# ---------------------------------------------------------------------------
# Profile writes
# ---------------------------------------------------------------------------


def replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
    conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (record.session_id,))
    has_fallback_payload = table_has_column(conn, "session_profiles", "payload_json")
    columns = session_profile_insert_columns(has_fallback_payload=has_fallback_payload)
    conn.execute(
        build_insert_sql("session_profiles", columns),
        session_profile_insert_values(record, has_fallback_payload=has_fallback_payload),
    )


def replace_session_profiles_bulk_sync(
    conn: sqlite3.Connection,
    records: Sequence[SessionProfileRecord],
) -> None:
    if not records:
        return
    records = _dedupe_records_by_session(records)
    _delete_where_in(conn, "session_profiles", "session_id", [record.session_id for record in records])
    has_fallback_payload = table_has_column(conn, "session_profiles", "payload_json")
    columns = session_profile_insert_columns(has_fallback_payload=has_fallback_payload)
    conn.executemany(
        build_insert_sql("session_profiles", columns),
        [session_profile_insert_values(record, has_fallback_payload=has_fallback_payload) for record in records],
    )


def replace_session_latency_profiles_bulk_sync(
    conn: sqlite3.Connection,
    records: Sequence[SessionLatencyProfileRecord],
) -> None:
    if not records:
        return
    records = _dedupe_records_by_session(records)
    _delete_where_in(conn, "session_latency_profiles", "session_id", [record.session_id for record in records])
    conn.executemany(
        build_insert_sql("session_latency_profiles", _SESSION_LATENCY_PROFILE_COLUMNS),
        [session_latency_profile_insert_values(record) for record in records],
    )


# ---------------------------------------------------------------------------
# Timeline writes
# ---------------------------------------------------------------------------


def replace_session_work_events_sync(
    conn: sqlite3.Connection,
    session_id: str,
    records: Sequence[SessionWorkEventRecord],
) -> None:
    conn.execute("DELETE FROM session_work_events WHERE session_id = ?", (session_id,))
    if records:
        has_fallback_payload = table_has_column(conn, "session_work_events", "payload_json")
        columns = session_work_event_insert_columns(has_fallback_payload=has_fallback_payload)
        conn.executemany(
            build_insert_sql("session_work_events", columns),
            [session_work_event_insert_values(record, has_fallback_payload=has_fallback_payload) for record in records],
        )


def replace_session_work_events_bulk_sync(
    conn: sqlite3.Connection,
    records_by_session: Mapping[str, Sequence[SessionWorkEventRecord]],
) -> None:
    if not records_by_session:
        return
    _delete_where_in(conn, "session_work_events", "session_id", tuple(records_by_session))
    records = [record for session_records in records_by_session.values() for record in session_records]
    if records:
        has_fallback_payload = table_has_column(conn, "session_work_events", "payload_json")
        columns = session_work_event_insert_columns(has_fallback_payload=has_fallback_payload)
        conn.executemany(
            build_insert_sql("session_work_events", columns),
            [session_work_event_insert_values(record, has_fallback_payload=has_fallback_payload) for record in records],
        )


def replace_session_phases_sync(
    conn: sqlite3.Connection,
    session_id: str,
    records: Sequence[SessionPhaseRecord],
) -> None:
    conn.execute("DELETE FROM session_phases WHERE session_id = ?", (session_id,))
    if records:
        has_fallback_payload = table_has_column(conn, "session_phases", "payload_json")
        columns = session_phase_insert_columns(has_fallback_payload=has_fallback_payload)
        conn.executemany(
            build_insert_sql("session_phases", columns),
            [session_phase_insert_values(record, has_fallback_payload=has_fallback_payload) for record in records],
        )


def replace_session_phases_bulk_sync(
    conn: sqlite3.Connection,
    records_by_session: Mapping[str, Sequence[SessionPhaseRecord]],
) -> None:
    if not records_by_session:
        return
    _delete_where_in(conn, "session_phases", "session_id", tuple(records_by_session))
    records = [record for session_records in records_by_session.values() for record in session_records]
    if records:
        has_fallback_payload = table_has_column(conn, "session_phases", "payload_json")
        columns = session_phase_insert_columns(has_fallback_payload=has_fallback_payload)
        conn.executemany(
            build_insert_sql("session_phases", columns),
            [session_phase_insert_values(record, has_fallback_payload=has_fallback_payload) for record in records],
        )


# ---------------------------------------------------------------------------
# Aggregate writes
# ---------------------------------------------------------------------------


def replace_thread_sync(
    conn: sqlite3.Connection,
    thread_id: str,
    record: ThreadRecord | None,
) -> None:
    """Upsert the analytics columns of one thread row without touching the spine.

    A ``None`` record is a no-op: thread-row lifecycle (creation/deletion) is
    owned by the topology spine writer, not the analytics materializer.
    """
    del thread_id
    if record is not None:
        conn.execute(_thread_analytics_upsert_sql(), thread_analytics_insert_values(record))


def replace_threads_bulk_sync(
    conn: sqlite3.Connection,
    records_by_thread_id: Mapping[str, ThreadRecord | None],
) -> None:
    records = [record for record in records_by_thread_id.values() if record is not None]
    if records:
        conn.executemany(
            _thread_analytics_upsert_sql(),
            [thread_analytics_insert_values(record) for record in records],
        )


def replace_session_tag_rollup_rows_sync(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    bucket_day: str,
    records: Sequence[SessionTagRollupRecord],
) -> None:
    conn.execute(
        "DELETE FROM session_tag_rollups WHERE source_name = ? AND bucket_day = ?",
        (source_name, bucket_day),
    )
    if records:
        conn.executemany(
            build_insert_sql(
                "session_tag_rollups",
                (
                    "tag",
                    "bucket_day",
                    "source_name",
                    "materializer_version",
                    "materialized_at",
                    "source_updated_at",
                    "source_sort_key",
                    "input_high_water_mark",
                    "input_high_water_mark_source",
                    "input_row_count",
                    "session_count",
                    "logical_session_count",
                    "logical_session_ids_json",
                    "explicit_count",
                    "auto_count",
                    "repo_breakdown_json",
                    "search_text",
                ),
            ),
            [session_tag_rollup_insert_values(record) for record in records],
        )


__all__ = [
    "build_insert_sql",
    "replace_session_phases_bulk_sync",
    "replace_session_phases_sync",
    "replace_session_latency_profiles_bulk_sync",
    "replace_session_profiles_bulk_sync",
    "replace_session_profile_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_session_work_events_bulk_sync",
    "replace_session_work_events_sync",
    "replace_threads_bulk_sync",
    "replace_thread_sync",
    "session_phase_insert_columns",
    "session_phase_insert_values",
    "session_profile_insert_columns",
    "session_profile_insert_values",
    "session_latency_profile_insert_values",
    "session_tag_rollup_insert_values",
    "session_work_event_insert_columns",
    "session_work_event_insert_values",
    "table_has_column",
    "thread_analytics_insert_values",
    "thread_insert_values",
]
