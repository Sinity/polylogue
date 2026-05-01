"""Work-thread durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_work_thread_record
from polylogue.storage.insights.session.storage import work_thread_insert_values
from polylogue.storage.query_models import WorkThreadListQuery
from polylogue.storage.runtime import WorkThreadRecord

__all__ = [
    "get_work_thread",
    "list_work_threads",
    "replace_work_thread",
]


async def get_work_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> WorkThreadRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM work_threads WHERE thread_id = ?",
        (thread_id,),
    )
    row = await cursor.fetchone()
    return _row_to_work_thread_record(row) if row else None


async def list_work_threads(
    conn: aiosqlite.Connection,
    query: WorkThreadListQuery,
) -> list[WorkThreadRecord]:
    params: list[object] = []
    if query.query:
        from_clause = """
            FROM work_threads wt
            JOIN work_threads_fts
              ON work_threads_fts.thread_id = wt.thread_id
        """
        where = ["work_threads_fts MATCH ?"]
        params.append(query.query)
        order_by = "ORDER BY bm25(work_threads_fts), COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    else:
        from_clause = "FROM work_threads wt"
        where = []
        order_by = "ORDER BY COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    if query.since:
        where.append("COALESCE(wt.end_time, wt.start_time, wt.materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(wt.start_time, wt.end_time, wt.materialized_at) <= ?")
        params.append(query.until)
    sql = "SELECT wt.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_work_thread_record(row) for row in rows]


async def replace_work_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
    record: WorkThreadRecord | None,
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM work_threads WHERE thread_id = ?", (thread_id,))
    if record is not None:
        await conn.execute(
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
            work_thread_insert_values(record),
        )
    if transaction_depth == 0:
        await conn.commit()
