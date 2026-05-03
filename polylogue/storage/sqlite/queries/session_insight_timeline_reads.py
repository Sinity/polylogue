"""Timeline-oriented durable session-insight read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.query_models import SessionTimelineListQuery
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.sqlite.queries.mappers import (
    _row_to_session_phase_record,
    _row_to_session_work_event_record,
)

__all__ = [
    "get_session_phases",
    "get_work_events",
    "list_session_phases",
    "list_work_events",
]


async def get_work_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[SessionWorkEventRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_work_events
        WHERE conversation_id = ?
        ORDER BY event_index
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def get_session_phases(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[SessionPhaseRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_phases
        WHERE conversation_id = ?
        ORDER BY phase_index
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]


async def list_work_events(
    conn: aiosqlite.Connection,
    query: SessionTimelineListQuery,
) -> list[SessionWorkEventRecord]:
    params: list[object] = []
    if query.query:
        from_clause = """
            FROM session_work_events swe
            JOIN session_work_events_fts
              ON session_work_events_fts.event_id = swe.event_id
        """
        where = ["session_work_events_fts MATCH ?"]
        params.append(query.query)
        order_by = "ORDER BY bm25(session_work_events_fts), COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"
    else:
        from_clause = "FROM session_work_events swe"
        where = []
        order_by = "ORDER BY COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"

    if query.conversation_id:
        where.append("swe.conversation_id = ?")
        params.append(query.conversation_id)
    if query.provider:
        where.append("swe.provider_name = ?")
        params.append(query.provider)
    if query.kind:
        where.append("swe.kind = ?")
        params.append(query.kind)
    if query.since:
        where.append("COALESCE(swe.end_time, swe.start_time, swe.source_updated_at, swe.materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) <= ?")
        params.append(query.until)

    sql = "SELECT swe.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def list_session_phases(
    conn: aiosqlite.Connection,
    query: SessionTimelineListQuery,
) -> list[SessionPhaseRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.conversation_id:
        where.append("conversation_id = ?")
        params.append(query.conversation_id)
    if query.provider:
        where.append("provider_name = ?")
        params.append(query.provider)
    if query.kind:
        where.append("kind = ?")
        params.append(query.kind)
    if query.since:
        where.append("COALESCE(end_time, start_time, source_updated_at, materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(start_time, end_time, source_updated_at, materialized_at) <= ?")
        params.append(query.until)

    sql = """
        SELECT *
        FROM session_phases
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(start_time, end_time, materialized_at) DESC, phase_index"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]
