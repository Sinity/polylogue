"""Thread durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.insights.session.storage import thread_insert_values
from polylogue.storage.query_models import ThreadListQuery
from polylogue.storage.runtime import ThreadRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_thread_record

__all__ = [
    "get_thread",
    "list_threads",
    "replace_thread",
]


async def get_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord | None:
    cursor = await conn.execute(
        "SELECT *, thread_id AS root_id FROM threads WHERE thread_id = ?",
        (thread_id,),
    )
    row = await cursor.fetchone()
    return _row_to_thread_record(row) if row else None


async def list_threads(
    conn: aiosqlite.Connection,
    query: ThreadListQuery,
) -> list[ThreadRecord]:
    params: list[object] = []
    if query.query:
        # polylogue-eizc: threads_fts (the FTS5 MATCH index this used to
        # query) was dropped -- zero production callers ever reached this
        # branch, and the live "analyze threads" search path
        # (archive_tiers/archive.py:list_thread_insights) already does a
        # manual LIKE substring scan rather than using threads_fts. Mirror
        # that same LIKE shape here instead of an FTS5 MATCH.
        from_clause = "FROM threads wt"
        where = ["lower(wt.search_text) LIKE ?"]
        params.append(f"%{query.query.strip().lower()}%")
        order_by = "ORDER BY COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    else:
        from_clause = "FROM threads wt"
        where = []
        order_by = "ORDER BY COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    if query.since:
        where.append("COALESCE(wt.end_time, wt.start_time, wt.materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(wt.start_time, wt.end_time, wt.materialized_at) <= ?")
        params.append(query.until)
    sql = "SELECT wt.*, wt.thread_id AS root_id " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_thread_record(row) for row in rows]


async def replace_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
    record: ThreadRecord | None,
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    if record is not None:
        await conn.execute(
            """
            INSERT INTO threads (
                thread_id,
                materializer_version,
                materialized_at,
                source_updated_at,
                input_high_water_mark,
                input_high_water_mark_source,
                input_row_count,
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            thread_insert_values(record),
        )
    if transaction_depth == 0:
        await conn.commit()
