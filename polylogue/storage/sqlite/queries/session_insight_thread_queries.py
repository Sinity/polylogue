"""Thread durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.derived.session.threads import build_thread_records_for_roots_async
from polylogue.storage.query_models import ThreadListQuery
from polylogue.storage.runtime import ThreadRecord
from polylogue.storage.search.query_support import normalize_fts5_query

__all__ = [
    "get_thread",
    "list_threads",
]


async def get_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord | None:
    records = await build_thread_records_for_roots_async(conn, [thread_id])
    return records.get(thread_id)


async def list_threads(
    conn: aiosqlite.Connection,
    query: ThreadListQuery,
) -> list[ThreadRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.query:
        match = normalize_fts5_query(query.query)
        like = f"%{query.query.strip().lower()}%"
        where.append(
            "("
            "lower(wt.thread_id) LIKE ? "
            "OR EXISTS ("
            "SELECT 1 FROM thread_sessions qts "
            "JOIN sessions qs ON qs.session_id = qts.session_id "
            "WHERE qts.thread_id = wt.thread_id AND ("
            "lower(qs.session_id) LIKE ? OR lower(COALESCE(qs.title, '')) LIKE ? "
            "OR lower(COALESCE(qs.git_repository_url, '')) LIKE ? "
            "OR lower(COALESCE(qs.git_branch, '')) LIKE ?"
            "))"
            + (
                " OR EXISTS ("
                "SELECT 1 FROM messages_fts mf "
                "JOIN blocks mb ON mb.rowid = mf.rowid "
                "JOIN thread_sessions fts_ts ON fts_ts.session_id = mb.session_id "
                "WHERE fts_ts.thread_id = wt.thread_id AND messages_fts MATCH ?"
                ")"
                if match
                else ""
            )
            + ")"
        )
        params.extend([like, like, like, like, like])
        if match:
            params.append(match)
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
    root_ids = [str(row["thread_id"]) for row in rows]
    records = await build_thread_records_for_roots_async(conn, root_ids)
    return [records[root_id] for root_id in root_ids if root_id in records]
