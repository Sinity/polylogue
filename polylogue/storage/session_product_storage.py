"""Session-product storage writes (profiles, timelines, aggregates)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from polylogue.storage.store import (
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


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    key = (id(conn), f"{table}.{column}")
    cached = _SYNC_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    found = any(str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())
    _SYNC_COLUMN_CACHE[key] = found
    return found


# ---------------------------------------------------------------------------
# Profile writes
# ---------------------------------------------------------------------------


def replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
    conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (record.conversation_id,))
    payload_json = _json_or_none(
        {
            **record.evidence_payload,
            **record.inference_payload,
            "conversation_id": str(record.conversation_id),
            "provider": record.provider_name,
            "title": record.title,
        }
    )
    columns = [
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
    ]
    values: list[object] = [
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
    if table_has_column(conn, "session_profiles", "payload_json"):
        columns.append("payload_json")
        values.append(payload_json)
    columns.extend(
        [
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
        ]
    )
    values.extend(
        [
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
        ]
    )
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"""
        INSERT INTO session_profiles (
            {", ".join(columns)}
        ) VALUES ({placeholders})
        """,
        tuple(values),
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
        columns = [
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
        ]
        if has_legacy_payload:
            columns.append("payload_json")
        columns.extend(
            [
                "evidence_payload_json",
                "inference_payload_json",
                "search_text",
                "inference_version",
                "inference_family",
            ]
        )
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"""
            INSERT INTO session_work_events (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                tuple(
                    [
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
                    + (
                        [
                            _json_or_none(
                                {
                                    **record.evidence_payload,
                                    **record.inference_payload,
                                }
                            )
                        ]
                        if has_legacy_payload
                        else []
                    )
                    + [
                        _json_or_none(record.evidence_payload),
                        _json_or_none(record.inference_payload),
                        record.search_text,
                        record.inference_version,
                        record.inference_family,
                    ]
                )
                for record in records
            ],
        )


def replace_session_phases_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[SessionPhaseRecord],
) -> None:
    conn.execute("DELETE FROM session_phases WHERE conversation_id = ?", (conversation_id,))
    if records:
        has_legacy_payload = table_has_column(conn, "session_phases", "payload_json")
        columns = [
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
        ]
        if has_legacy_payload:
            columns.append("payload_json")
        columns.extend(
            [
                "evidence_payload_json",
                "inference_payload_json",
                "search_text",
                "inference_version",
                "inference_family",
            ]
        )
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"""
            INSERT INTO session_phases (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                tuple(
                    [
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
                        _json_or_none(dict[str, object](record.tool_counts)),
                        record.word_count,
                    ]
                    + (
                        [
                            _json_or_none(
                                {
                                    **record.evidence_payload,
                                    **record.inference_payload,
                                }
                            )
                        ]
                        if has_legacy_payload
                        else []
                    )
                    + [
                        _json_or_none(record.evidence_payload),
                        _json_or_none(record.inference_payload),
                        record.search_text,
                        record.inference_version,
                        record.inference_family,
                    ]
                )
                for record in records
            ],
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
            """
            INSERT INTO work_threads (
                thread_id,
                root_id,
                materializer_version,
                materialized_at,
                start_time,
                end_time,
                dominant_repo,
                session_ids_json,
                session_count,
                depth,
                branch_count,
                total_messages,
                total_cost_usd,
                wall_duration_ms,
                work_event_breakdown_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                _json_or_none(dict[str, object](record.work_event_breakdown or {})),
                _json_or_none(record.payload),
                record.search_text,
            ),
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
            """
            INSERT INTO session_tag_rollups (
                tag,
                bucket_day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                explicit_count,
                auto_count,
                repo_breakdown_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
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
                    _json_or_none(dict[str, object](record.repo_breakdown)),
                    record.search_text,
                )
                for record in records
            ],
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
            """
            INSERT INTO day_session_summaries (
                day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                total_cost_usd,
                total_duration_ms,
                total_wall_duration_ms,
                total_messages,
                total_words,
                work_event_breakdown_json,
                repos_active_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
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
                    _json_or_none(dict[str, object](record.work_event_breakdown)),
                    _json_array_or_none(record.repos_active),
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )


__all__ = [
    "replace_day_session_summaries_sync",
    "replace_session_phases_sync",
    "replace_session_profile_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_session_work_events_sync",
    "replace_work_thread_sync",
    "table_has_column",
]
