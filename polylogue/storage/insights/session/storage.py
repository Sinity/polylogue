"""Session insight storage writes (profiles, timelines, aggregates)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from polylogue.storage.runtime import (
    DaySessionSummaryRecord,
    SessionLatencyProfileRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
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

_SESSION_PROFILE_BASE_COLUMNS = (
    "conversation_id",
    "logical_conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "provider_name",
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
    "conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "provider_name",
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
    "event_id",
    "conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "provider_name",
    "event_index",
    "heuristic_label",
    "confidence",
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "duration_ms",
    "canonical_session_date",
    "summary",
    "file_paths_json",
    "tools_used_json",
)
_TIMELINE_PAYLOAD_COLUMNS = (
    "evidence_payload_json",
    "inference_payload_json",
    "search_text",
    "inference_version",
    "inference_family",
)
_SESSION_PHASE_BASE_COLUMNS = (
    "phase_id",
    "conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "input_high_water_mark",
    "input_high_water_mark_source",
    "input_row_count",
    "provider_name",
    "phase_index",
    "kind",
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "duration_ms",
    "canonical_session_date",
    "confidence",
    "evidence_reasons_json",
    "tool_counts_json",
    "word_count",
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


def _with_legacy_payload_column(
    base_columns: tuple[str, ...],
    payload_columns: tuple[str, ...],
    *,
    has_legacy_payload: bool,
) -> tuple[str, ...]:
    if has_legacy_payload:
        return base_columns + ("payload_json",) + payload_columns
    return base_columns + payload_columns


def _compose_bindings(
    base_values: Sequence[SqlValue],
    payload_values: Sequence[SqlValue],
    *,
    has_legacy_payload: bool,
    legacy_payload_json: str | None,
) -> SqlBindings:
    values = list(base_values)
    if has_legacy_payload:
        values.append(legacy_payload_json)
    values.extend(payload_values)
    return tuple(values)


def _legacy_profile_payload_json(record: SessionProfileRecord) -> str | None:
    return _json_or_none(
        {
            **record.evidence_payload.model_dump(mode="json"),
            **record.inference_payload.model_dump(mode="json"),
            "conversation_id": str(record.conversation_id),
            "logical_conversation_id": str(record.logical_conversation_id),
            "provider": record.provider_name,
            "title": record.title,
        }
    )


def session_profile_insert_columns(
    *,
    has_legacy_payload: bool,
) -> tuple[str, ...]:
    return _with_legacy_payload_column(
        _SESSION_PROFILE_BASE_COLUMNS,
        _SESSION_PROFILE_PAYLOAD_COLUMNS,
        has_legacy_payload=has_legacy_payload,
    )


def session_profile_insert_values(
    record: SessionProfileRecord,
    *,
    has_legacy_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.conversation_id,
        record.logical_conversation_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.provider_name,
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
        _json_or_none(record.inference_payload),
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
        has_legacy_payload=has_legacy_payload,
        legacy_payload_json=_legacy_profile_payload_json(record),
    )


def session_latency_profile_insert_values(record: SessionLatencyProfileRecord) -> SqlBindings:
    return (
        record.conversation_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.provider_name,
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


def _legacy_timeline_payload_json(
    evidence_payload: BaseModel,
    inference_payload: BaseModel,
) -> str | None:
    return _json_or_none(
        {
            **evidence_payload.model_dump(mode="json"),
            **inference_payload.model_dump(mode="json"),
        }
    )


def session_work_event_insert_columns(
    *,
    has_legacy_payload: bool,
) -> tuple[str, ...]:
    return _with_legacy_payload_column(
        _SESSION_WORK_EVENT_BASE_COLUMNS,
        _TIMELINE_PAYLOAD_COLUMNS,
        has_legacy_payload=has_legacy_payload,
    )


def session_work_event_insert_values(
    record: SessionWorkEventRecord,
    *,
    has_legacy_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.event_id,
        record.conversation_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.provider_name,
        record.event_index,
        record.heuristic_label,
        record.confidence,
        record.start_index,
        record.end_index,
        record.start_time,
        record.end_time,
        record.duration_ms,
        record.canonical_session_date,
        record.summary,
        _json_array_or_none(record.file_paths),
        _json_array_or_none(record.tools_used),
    ]
    payload_values: tuple[SqlValue, ...] = (
        _json_or_none(record.evidence_payload),
        _json_or_none(record.inference_payload),
        record.search_text,
        record.inference_version,
        record.inference_family,
    )
    return _compose_bindings(
        base_values,
        payload_values,
        has_legacy_payload=has_legacy_payload,
        legacy_payload_json=_legacy_timeline_payload_json(record.evidence_payload, record.inference_payload),
    )


def session_phase_insert_columns(
    *,
    has_legacy_payload: bool,
) -> tuple[str, ...]:
    return _with_legacy_payload_column(
        _SESSION_PHASE_BASE_COLUMNS,
        _TIMELINE_PAYLOAD_COLUMNS,
        has_legacy_payload=has_legacy_payload,
    )


def session_phase_insert_values(
    record: SessionPhaseRecord,
    *,
    has_legacy_payload: bool,
) -> SqlBindings:
    base_values: list[SqlValue] = [
        record.phase_id,
        record.conversation_id,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.provider_name,
        record.phase_index,
        record.kind,
        record.start_index,
        record.end_index,
        record.start_time,
        record.end_time,
        record.duration_ms,
        record.canonical_session_date,
        record.confidence,
        _json_array_or_none(record.evidence_reasons) or "[]",
        _json_or_none(record.tool_counts),
        record.word_count,
    ]
    payload_values: tuple[SqlValue, ...] = (
        _json_or_none(record.evidence_payload),
        _json_or_none(record.inference_payload),
        record.search_text,
        record.inference_version,
        record.inference_family,
    )
    return _compose_bindings(
        base_values,
        payload_values,
        has_legacy_payload=has_legacy_payload,
        legacy_payload_json=_legacy_timeline_payload_json(record.evidence_payload, record.inference_payload),
    )


def work_thread_insert_values(record: WorkThreadRecord) -> SqlBindings:
    return (
        record.thread_id,
        record.root_id,
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


def session_tag_rollup_insert_values(record: SessionTagRollupRecord) -> SqlBindings:
    return (
        record.tag,
        record.bucket_day,
        record.provider_name,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.conversation_count,
        record.logical_session_count,
        _json_array_or_none(record.logical_conversation_ids),
        record.explicit_count,
        record.auto_count,
        _json_or_none(record.repo_breakdown),
        record.search_text,
    )


def day_session_summary_insert_values(record: DaySessionSummaryRecord) -> SqlBindings:
    return (
        record.day,
        record.provider_name,
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
        record.input_high_water_mark,
        record.input_high_water_mark_source,
        record.input_row_count,
        record.conversation_count,
        record.logical_session_count,
        _json_array_or_none(record.logical_conversation_ids),
        record.total_cost_usd,
        record.total_duration_ms,
        record.total_tool_active_duration_ms,
        record.total_wall_duration_ms,
        record.total_messages,
        record.total_words,
        _json_or_none(record.work_event_breakdown),
        _json_array_or_none(record.repos_active),
        _json_or_none(record.payload),
        record.search_text,
    )


# ---------------------------------------------------------------------------
# Profile writes
# ---------------------------------------------------------------------------


def replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
    conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (record.conversation_id,))
    has_legacy_payload = table_has_column(conn, "session_profiles", "payload_json")
    columns = session_profile_insert_columns(has_legacy_payload=has_legacy_payload)
    conn.execute(
        build_insert_sql("session_profiles", columns),
        session_profile_insert_values(record, has_legacy_payload=has_legacy_payload),
    )


def replace_session_profiles_bulk_sync(
    conn: sqlite3.Connection,
    records: Sequence[SessionProfileRecord],
) -> None:
    if not records:
        return
    _delete_where_in(conn, "session_profiles", "conversation_id", [record.conversation_id for record in records])
    has_legacy_payload = table_has_column(conn, "session_profiles", "payload_json")
    columns = session_profile_insert_columns(has_legacy_payload=has_legacy_payload)
    conn.executemany(
        build_insert_sql("session_profiles", columns),
        [session_profile_insert_values(record, has_legacy_payload=has_legacy_payload) for record in records],
    )


def replace_session_latency_profiles_bulk_sync(
    conn: sqlite3.Connection,
    records: Sequence[SessionLatencyProfileRecord],
) -> None:
    if not records:
        return
    _delete_where_in(
        conn, "session_latency_profiles", "conversation_id", [record.conversation_id for record in records]
    )
    conn.executemany(
        build_insert_sql("session_latency_profiles", _SESSION_LATENCY_PROFILE_COLUMNS),
        [session_latency_profile_insert_values(record) for record in records],
    )


# ---------------------------------------------------------------------------
# Timeline writes
# ---------------------------------------------------------------------------


def replace_session_work_events_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[SessionWorkEventRecord],
) -> None:
    conn.execute("DELETE FROM session_work_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        has_legacy_payload = table_has_column(conn, "session_work_events", "payload_json")
        columns = session_work_event_insert_columns(has_legacy_payload=has_legacy_payload)
        conn.executemany(
            build_insert_sql("session_work_events", columns),
            [session_work_event_insert_values(record, has_legacy_payload=has_legacy_payload) for record in records],
        )


def replace_session_work_events_bulk_sync(
    conn: sqlite3.Connection,
    records_by_conversation: Mapping[str, Sequence[SessionWorkEventRecord]],
) -> None:
    if not records_by_conversation:
        return
    _delete_where_in(conn, "session_work_events", "conversation_id", tuple(records_by_conversation))
    records = [record for conversation_records in records_by_conversation.values() for record in conversation_records]
    if records:
        has_legacy_payload = table_has_column(conn, "session_work_events", "payload_json")
        columns = session_work_event_insert_columns(has_legacy_payload=has_legacy_payload)
        conn.executemany(
            build_insert_sql("session_work_events", columns),
            [session_work_event_insert_values(record, has_legacy_payload=has_legacy_payload) for record in records],
        )


def replace_session_phases_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[SessionPhaseRecord],
) -> None:
    conn.execute("DELETE FROM session_phases WHERE conversation_id = ?", (conversation_id,))
    if records:
        has_legacy_payload = table_has_column(conn, "session_phases", "payload_json")
        columns = session_phase_insert_columns(has_legacy_payload=has_legacy_payload)
        conn.executemany(
            build_insert_sql("session_phases", columns),
            [session_phase_insert_values(record, has_legacy_payload=has_legacy_payload) for record in records],
        )


def replace_session_phases_bulk_sync(
    conn: sqlite3.Connection,
    records_by_conversation: Mapping[str, Sequence[SessionPhaseRecord]],
) -> None:
    if not records_by_conversation:
        return
    _delete_where_in(conn, "session_phases", "conversation_id", tuple(records_by_conversation))
    records = [record for conversation_records in records_by_conversation.values() for record in conversation_records]
    if records:
        has_legacy_payload = table_has_column(conn, "session_phases", "payload_json")
        columns = session_phase_insert_columns(has_legacy_payload=has_legacy_payload)
        conn.executemany(
            build_insert_sql("session_phases", columns),
            [session_phase_insert_values(record, has_legacy_payload=has_legacy_payload) for record in records],
        )


# ---------------------------------------------------------------------------
# Aggregate writes
# ---------------------------------------------------------------------------


def replace_work_thread_sync(
    conn: sqlite3.Connection,
    thread_id: str,
    record: WorkThreadRecord | None,
) -> None:
    conn.execute("DELETE FROM work_threads WHERE thread_id = ?", (thread_id,))
    if record is not None:
        conn.execute(
            build_insert_sql(
                "work_threads",
                (
                    "thread_id",
                    "root_id",
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
                    "session_count",
                    "depth",
                    "branch_count",
                    "total_messages",
                    "total_cost_usd",
                    "wall_duration_ms",
                    "work_event_breakdown_json",
                    "payload_json",
                    "search_text",
                ),
            ),
            work_thread_insert_values(record),
        )


def replace_work_threads_bulk_sync(
    conn: sqlite3.Connection,
    records_by_thread_id: Mapping[str, WorkThreadRecord | None],
) -> None:
    if not records_by_thread_id:
        return
    _delete_where_in(conn, "work_threads", "thread_id", tuple(records_by_thread_id))
    records = [record for record in records_by_thread_id.values() if record is not None]
    if records:
        conn.executemany(
            build_insert_sql(
                "work_threads",
                (
                    "thread_id",
                    "root_id",
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
                    "session_count",
                    "depth",
                    "branch_count",
                    "total_messages",
                    "total_cost_usd",
                    "wall_duration_ms",
                    "work_event_breakdown_json",
                    "payload_json",
                    "search_text",
                ),
            ),
            [work_thread_insert_values(record) for record in records],
        )


def replace_session_tag_rollup_rows_sync(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    bucket_day: str,
    records: Sequence[SessionTagRollupRecord],
) -> None:
    conn.execute(
        "DELETE FROM session_tag_rollups WHERE provider_name = ? AND bucket_day = ?",
        (provider_name, bucket_day),
    )
    if records:
        conn.executemany(
            build_insert_sql(
                "session_tag_rollups",
                (
                    "tag",
                    "bucket_day",
                    "provider_name",
                    "materializer_version",
                    "materialized_at",
                    "source_updated_at",
                    "source_sort_key",
                    "input_high_water_mark",
                    "input_high_water_mark_source",
                    "input_row_count",
                    "conversation_count",
                    "logical_session_count",
                    "logical_conversation_ids_json",
                    "explicit_count",
                    "auto_count",
                    "repo_breakdown_json",
                    "search_text",
                ),
            ),
            [session_tag_rollup_insert_values(record) for record in records],
        )


def replace_day_session_summaries_sync(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    day: str,
    records: Sequence[DaySessionSummaryRecord],
) -> None:
    conn.execute(
        "DELETE FROM day_session_summaries WHERE provider_name = ? AND day = ?",
        (provider_name, day),
    )
    if records:
        conn.executemany(
            build_insert_sql(
                "day_session_summaries",
                (
                    "day",
                    "provider_name",
                    "materializer_version",
                    "materialized_at",
                    "source_updated_at",
                    "source_sort_key",
                    "input_high_water_mark",
                    "input_high_water_mark_source",
                    "input_row_count",
                    "conversation_count",
                    "logical_session_count",
                    "logical_conversation_ids_json",
                    "total_cost_usd",
                    "total_duration_ms",
                    "total_tool_active_duration_ms",
                    "total_wall_duration_ms",
                    "total_messages",
                    "total_words",
                    "work_event_breakdown_json",
                    "repos_active_json",
                    "payload_json",
                    "search_text",
                ),
            ),
            [day_session_summary_insert_values(record) for record in records],
        )


__all__ = [
    "build_insert_sql",
    "day_session_summary_insert_values",
    "replace_day_session_summaries_sync",
    "replace_session_phases_bulk_sync",
    "replace_session_phases_sync",
    "replace_session_latency_profiles_bulk_sync",
    "replace_session_profiles_bulk_sync",
    "replace_session_profile_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_session_work_events_bulk_sync",
    "replace_session_work_events_sync",
    "replace_work_threads_bulk_sync",
    "replace_work_thread_sync",
    "session_phase_insert_columns",
    "session_phase_insert_values",
    "session_profile_insert_columns",
    "session_profile_insert_values",
    "session_latency_profile_insert_values",
    "session_tag_rollup_insert_values",
    "session_work_event_insert_columns",
    "session_work_event_insert_values",
    "table_has_column",
    "work_thread_insert_values",
]
