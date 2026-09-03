"""Thread durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.query_models import ThreadListQuery
from polylogue.storage.runtime import ThreadRecord
from polylogue.storage.sqlite.queries.mappers_insight_timelines import _row_to_thread_record

__all__ = [
    "get_thread",
    "list_threads",
]


async def get_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord | None:
    from polylogue.storage.insights.session.threads import build_thread_records_for_roots_async

    records = await build_thread_records_for_roots_async(conn, [thread_id])
    return records.get(thread_id)


async def list_threads(
    conn: aiosqlite.Connection,
    query: ThreadListQuery,
) -> list[ThreadRecord]:
    from polylogue.storage.insights.session.threads import build_thread_records_for_roots_async

    params: list[object] = []
    where: list[str] = []
    if query.query:
        where.append("lower(wt.search_text) LIKE ?")
        params.append(f"%{query.query.strip().lower()}%")
    order_by = "ORDER BY COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    if query.since:
        where.append("COALESCE(wt.end_time, wt.start_time, wt.materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(wt.start_time, wt.end_time, wt.materialized_at) <= ?")
        params.append(query.until)
    sql = "SELECT wt.thread_id FROM threads wt"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    has_sessions = await (
        await conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sessions' AND type = 'table'")
    ).fetchone()
    if has_sessions is None:
        cursor = await conn.execute(
            "SELECT wt.*, wt.thread_id AS root_id FROM threads wt"
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" {order_by}"
            + (" LIMIT ? OFFSET ?" if query.limit is not None else ""),
            tuple(params),
        )
        return [_row_to_thread_record(row) for row in await cursor.fetchall()]
    root_ids = [str(row["thread_id"]) for row in rows]
    records = await build_thread_records_for_roots_async(conn, root_ids)
    return [records[root_id] for root_id in root_ids if root_id in records]
