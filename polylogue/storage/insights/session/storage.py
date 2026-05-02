"""Session insight storage writes (profiles, timelines, aggregates)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from pydantic import BaseModel

from polylogue.storage.runtime import (
    DaySessionSummaryRecord,
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
SqlValue = str | int | float | None
SqlBindings = tuple[SqlValue, ...]

_SESSION_PROFILE_BASE_COLUMNS = (
    "conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
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
    "wall_duration_ms",
    "cost_is_estimated",
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
_SESSION_WORK_EVENT_BASE_COLUMNS = (
    "event_id",
    "conversation_id",
    "materializer_version",
    "materialized_at",
    "source_updated_at",
    "source_sort_key",
    "provider_name",
    "event_index",
    "kind",
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
        record.materializer_version,
        record.materialized_at,
        record.source_updated_at,
        record.source_sort_key,
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
        record.wall_duration_ms,
        int(record.cost_is_estimated),
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
        record.provider_name,
        record.event_index,
        record.kind,
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
        record.conversation_count,
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
        record.conversation_count,
        record.total_cost_usd,
        record.total_duration_ms,
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
                    "conversation_count",
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
                    "conversation_count",
                    "total_cost_usd",
                    "total_duration_ms",
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
    "replace_session_phases_sync",
    "replace_session_profile_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_session_work_events_sync",
    "replace_work_thread_sync",
    "session_phase_insert_columns",
    "session_phase_insert_values",
    "session_profile_insert_columns",
    "session_profile_insert_values",
    "session_tag_rollup_insert_values",
    "session_work_event_insert_columns",
    "session_work_event_insert_values",
    "table_has_column",
    "work_thread_insert_values",
]
