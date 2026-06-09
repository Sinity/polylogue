"""Timeline-oriented durable session-insight read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.storage.query_models import SessionTimelineListQuery
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.search.query_support import escape_fts5_query
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


async def _require_session_work_events_fts_ready(conn: aiosqlite.Connection) -> None:
    exists = bool(
        await (
            await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_work_events_fts'")
        ).fetchone()
    )
    if not exists:
        raise DatabaseError("Session work-event search index is missing.")
    trigger_row = await (
        await conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='trigger'
              AND name IN ('session_work_events_fts_ai', 'session_work_events_fts_ad', 'session_work_events_fts_au')
            """
        )
    ).fetchone()
    if trigger_row is None or int(trigger_row[0] or 0) != 3:
        raise DatabaseError("Session work-event search index triggers are missing.")
    missing_row = await (
        await conn.execute(
            """
            SELECT COUNT(*)
            FROM session_work_events AS swe
            LEFT JOIN session_work_events_fts AS f ON f.event_id = swe.event_id
            WHERE f.event_id IS NULL
            """
        )
    ).fetchone()
    excess_row = await (
        await conn.execute(
            """
            SELECT COUNT(DISTINCT f.event_id)
            FROM session_work_events_fts AS f
            LEFT JOIN session_work_events AS swe ON swe.event_id = f.event_id
            WHERE swe.event_id IS NULL
            """
        )
    ).fetchone()
    duplicate_row = await (
        await conn.execute("SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM session_work_events_fts")
    ).fetchone()
    missing = int(missing_row[0] or 0) if missing_row else 0
    excess = int(excess_row[0] or 0) if excess_row else 0
    duplicate = int(duplicate_row[0] or 0) if duplicate_row else 0
    if missing or excess or duplicate:
        raise DatabaseError(
            f"Session work-event search index is incomplete (missing={missing}, stale={excess}, duplicate={duplicate})."
        )


async def get_work_events(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionWorkEventRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_work_events
        WHERE session_id = ?
        ORDER BY event_index
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def get_session_phases(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionPhaseRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_phases
        WHERE session_id = ?
        ORDER BY phase_index
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]


async def list_work_events(
    conn: aiosqlite.Connection,
    query: SessionTimelineListQuery,
) -> list[SessionWorkEventRecord]:
    params: list[object] = []
    if query.query:
        await _require_session_work_events_fts_ready(conn)
        from_clause = """
            FROM session_work_events swe
            JOIN session_work_events_fts
              ON session_work_events_fts.event_id = swe.event_id
        """
        where = ["session_work_events_fts MATCH ?"]
        params.append(escape_fts5_query(query.query))
        order_by = (
            "ORDER BY bm25(session_work_events_fts), "
            "COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) DESC, swe.event_index"
        )
    else:
        from_clause = "FROM session_work_events swe"
        where = []
        order_by = "ORDER BY COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) DESC, swe.event_index"

    if query.session_id:
        where.append("swe.session_id = ?")
        params.append(query.session_id)
    if query.provider:
        where.append("swe.source_name = ?")
        params.append(query.provider)
    if query.heuristic_label:
        where.append("swe.heuristic_label = ?")
        params.append(query.heuristic_label)
    if query.since:
        where.append("COALESCE(swe.end_time, swe.start_time, swe.source_updated_at, swe.materialized_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) <= ?")
        params.append(query.until)
    if query.session_date_since:
        where.append("swe.canonical_session_date >= date(?)")
        params.append(query.session_date_since)
    if query.session_date_until:
        where.append("swe.canonical_session_date <= date(?)")
        params.append(query.session_date_until)

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
    if query.session_id:
        where.append("session_id = ?")
        params.append(query.session_id)
    if query.provider:
        where.append("source_name = ?")
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
    if query.session_date_since:
        where.append("canonical_session_date >= date(?)")
        params.append(query.session_date_since)
    if query.session_date_until:
        where.append("canonical_session_date <= date(?)")
        params.append(query.session_date_until)

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
