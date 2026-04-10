"""Timeline-oriented durable session-product read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_session_phase_record,
    _row_to_session_work_event_record,
)
from polylogue.storage.store import SessionPhaseRecord, SessionWorkEventRecord

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
    *,
    conversation_id: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionWorkEventRecord]:
    params: list[object] = []
    if query:
        from_clause = """
            FROM session_work_events swe
            JOIN session_work_events_fts
              ON session_work_events_fts.event_id = swe.event_id
        """
        where = ["session_work_events_fts MATCH ?"]
        params.append(query)
        order_by = "ORDER BY bm25(session_work_events_fts), COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"
    else:
        from_clause = "FROM session_work_events swe"
        where = []
        order_by = "ORDER BY COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"

    if conversation_id:
        where.append("swe.conversation_id = ?")
        params.append(conversation_id)
    if provider:
        where.append("swe.provider_name = ?")
        params.append(provider)
    if kind:
        where.append("swe.kind = ?")
        params.append(kind)
    if since:
        where.append("COALESCE(swe.end_time, swe.start_time, swe.source_updated_at, swe.materialized_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) <= ?")
        params.append(until)

    sql = "SELECT swe.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def list_session_phases(
    conn: aiosqlite.Connection,
    *,
    conversation_id: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
) -> list[SessionPhaseRecord]:
    params: list[object] = []
    where: list[str] = []
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if since:
        where.append("COALESCE(end_time, start_time, source_updated_at, materialized_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(start_time, end_time, source_updated_at, materialized_at) <= ?")
        params.append(until)

    sql = """
        SELECT *
        FROM session_phases
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(start_time, end_time, materialized_at) DESC, phase_index"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]
