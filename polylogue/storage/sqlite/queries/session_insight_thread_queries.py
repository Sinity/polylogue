"""Work-thread durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.storage.insights.session.storage import work_thread_insert_values
from polylogue.storage.query_models import WorkThreadListQuery
from polylogue.storage.runtime import WorkThreadRecord
from polylogue.storage.search.query_support import escape_fts5_query
from polylogue.storage.sqlite.queries.mappers import _row_to_work_thread_record

__all__ = [
    "get_work_thread",
    "list_work_threads",
    "replace_work_thread",
]


async def _require_work_threads_fts_ready(conn: aiosqlite.Connection) -> None:
    exists = bool(
        await (
            await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='work_threads_fts'")
        ).fetchone()
    )
    if not exists:
        raise DatabaseError("Work-thread search index is missing.")
    trigger_row = await (
        await conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='trigger'
              AND name IN ('work_threads_fts_ai', 'work_threads_fts_ad', 'work_threads_fts_au')
            """
        )
    ).fetchone()
    if trigger_row is None or int(trigger_row[0] or 0) != 3:
        raise DatabaseError("Work-thread search index triggers are missing.")
    missing_row = await (
        await conn.execute(
            """
            SELECT COUNT(*)
            FROM work_threads AS wt
            LEFT JOIN work_threads_fts AS f ON f.thread_id = wt.thread_id
            WHERE f.thread_id IS NULL
            """
        )
    ).fetchone()
    excess_row = await (
        await conn.execute(
            """
            SELECT COUNT(DISTINCT f.thread_id)
            FROM work_threads_fts AS f
            LEFT JOIN work_threads AS wt ON wt.thread_id = f.thread_id
            WHERE wt.thread_id IS NULL
            """
        )
    ).fetchone()
    duplicate_row = await (
        await conn.execute("SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM work_threads_fts")
    ).fetchone()
    missing = int(missing_row[0] or 0) if missing_row else 0
    excess = int(excess_row[0] or 0) if excess_row else 0
    duplicate = int(duplicate_row[0] or 0) if duplicate_row else 0
    if missing or excess or duplicate:
        raise DatabaseError(
            f"Work-thread search index is incomplete (missing={missing}, stale={excess}, duplicate={duplicate})."
        )


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
        await _require_work_threads_fts_ready(conn)
        from_clause = """
            FROM work_threads wt
            JOIN work_threads_fts
              ON work_threads_fts.thread_id = wt.thread_id
        """
        where = ["work_threads_fts MATCH ?"]
        params.append(escape_fts5_query(query.query))
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            work_thread_insert_values(record),
        )
    if transaction_depth == 0:
        await conn.commit()
