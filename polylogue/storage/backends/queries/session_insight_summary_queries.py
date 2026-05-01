"""Summary/rollup durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_day_session_summary_record,
    _row_to_session_tag_rollup_record,
)
from polylogue.storage.insights.session.storage import (
    day_session_summary_insert_values,
    session_tag_rollup_insert_values,
)
from polylogue.storage.query_models import (
    DaySessionSummaryListQuery,
    SessionTagRollupListQuery,
)
from polylogue.storage.runtime import DaySessionSummaryRecord, SessionTagRollupRecord

__all__ = [
    "list_day_session_summaries",
    "list_session_tag_rollup_rows",
    "replace_day_session_summaries",
    "replace_session_tag_rollup_rows",
]


async def list_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    query: SessionTagRollupListQuery,
) -> list[SessionTagRollupRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.provider:
        where.append("provider_name = ?")
        params.append(query.provider)
    if query.since:
        where.append("bucket_day >= date(?)")
        params.append(query.since)
    if query.until:
        where.append("bucket_day <= date(?)")
        params.append(query.until)
    if query.query:
        where.append("LOWER(tag) LIKE ?")
        params.append(f"%{query.query.strip().lower()}%")

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
    query: DaySessionSummaryListQuery,
) -> list[DaySessionSummaryRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.provider:
        where.append("provider_name = ?")
        params.append(query.provider)
    if query.since:
        where.append("day >= date(?)")
        params.append(query.since)
    if query.until:
        where.append("day <= date(?)")
        params.append(query.until)

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
                repo_breakdown_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [session_tag_rollup_insert_values(record) for record in records],
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
                repos_active_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [day_session_summary_insert_values(record) for record in records],
        )
    if transaction_depth == 0:
        await conn.commit()
