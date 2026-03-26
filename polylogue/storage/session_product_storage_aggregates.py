"""Aggregate-row storage writes for session-product lifecycle flows."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from polylogue.storage.store import DaySessionSummaryRecord, SessionTagRollupRecord, _json_array_or_none, _json_or_none


def replace_work_thread_sync(conn: sqlite3.Connection, thread_id: str, record: object | None) -> None:
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
                dominant_project,
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
                record.dominant_project,
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
                project_breakdown_json,
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
                    _json_or_none(record.project_breakdown),
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
                projects_active_json,
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
                    _json_or_none(record.work_event_breakdown),
                    _json_array_or_none(record.projects_active),
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )


__all__ = [
    "replace_day_session_summaries_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_work_thread_sync",
]
