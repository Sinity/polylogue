"""Summary/rollup durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_day_session_summary_record,
    _row_to_session_tag_rollup_record,
)
from polylogue.storage.store import (
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
    _json_array_or_none,
    _json_or_none,
)

__all__ = [
    "list_day_session_summaries",
    "list_session_tag_rollup_rows",
    "replace_day_session_summaries",
    "replace_session_tag_rollup_rows",
]


async def list_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    query: str | None = None,
) -> list[SessionTagRollupRecord]:
    params: list[object] = []
    where: list[str] = []
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if since:
        where.append("bucket_day >= date(?)")
        params.append(since)
    if until:
        where.append("bucket_day <= date(?)")
        params.append(until)
    if query:
        where.append("LOWER(tag) LIKE ?")
        params.append(f"%{query.strip().lower()}%")

    sql = """
        SELECT *
        FROM session_tag_rollups
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY bucket_day DESC, provider_name, tag"

    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_tag_rollup_record(row) for row in rows]


async def list_day_session_summaries(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[DaySessionSummaryRecord]:
    params: list[object] = []
    where: list[str] = []
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if since:
        where.append("day >= date(?)")
        params.append(since)
    if until:
        where.append("day <= date(?)")
        params.append(until)

    sql = """
        SELECT *
        FROM day_session_summaries
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY day DESC, provider_name"

    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_day_session_summary_record(row) for row in rows]


async def replace_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
    records: list[SessionTagRollupRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_tag_rollups WHERE provider_name = ? AND bucket_day = ?",
        (provider_name, bucket_day),
    )
    if records:
        await conn.executemany(
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
    if transaction_depth == 0:
        await conn.commit()


async def replace_day_session_summaries(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    day: str,
    records: list[DaySessionSummaryRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM day_session_summaries WHERE provider_name = ? AND day = ?",
        (provider_name, day),
    )
    if records:
        await conn.executemany(
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
    if transaction_depth == 0:
        await conn.commit()
