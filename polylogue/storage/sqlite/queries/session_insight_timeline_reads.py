"""Timeline-oriented durable session-insight read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.core.errors import DatabaseError
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

_ISO_MS = "'%Y-%m-%dT%H:%M:%fZ'"

_WORK_EVENT_SELECT = f"""
    swe.event_id,
    swe.session_id,
    COALESCE(im.materializer_version, 1) AS materializer_version,
    COALESCE(strftime({_ISO_MS}, im.materialized_at_ms / 1000.0, 'unixepoch'), '1970-01-01T00:00:00.000Z')
        AS materialized_at,
    strftime({_ISO_MS}, im.source_updated_at_ms / 1000.0, 'unixepoch') AS source_updated_at,
    im.source_sort_key_ms / 1000.0 AS source_sort_key,
    swe.input_high_water_mark,
    swe.input_high_water_mark_source,
    COALESCE(im.input_row_count, 0) AS input_row_count,
    s.origin AS source_name,
    swe.position AS event_index,
    swe.work_event_type AS heuristic_label,
    swe.confidence,
    swe.start_index,
    swe.end_index,
    strftime({_ISO_MS}, swe.started_at_ms / 1000.0, 'unixepoch') AS start_time,
    strftime({_ISO_MS}, swe.ended_at_ms / 1000.0, 'unixepoch') AS end_time,
    swe.duration_ms,
    date(COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch')
        AS canonical_session_date,
    swe.summary,
    swe.file_paths_json,
    swe.tools_used_json,
    swe.evidence_json AS evidence_payload_json,
    swe.inference_json AS inference_payload_json,
    swe.search_text,
    1 AS inference_version,
    'heuristic_session_semantics' AS inference_family
"""

_PHASE_SELECT = f"""
    sph.phase_id,
    sph.session_id,
    COALESCE(im.materializer_version, 1) AS materializer_version,
    COALESCE(strftime({_ISO_MS}, im.materialized_at_ms / 1000.0, 'unixepoch'), '1970-01-01T00:00:00.000Z')
        AS materialized_at,
    strftime({_ISO_MS}, im.source_updated_at_ms / 1000.0, 'unixepoch') AS source_updated_at,
    im.source_sort_key_ms / 1000.0 AS source_sort_key,
    sph.input_high_water_mark,
    sph.input_high_water_mark_source,
    COALESCE(im.input_row_count, 0) AS input_row_count,
    s.origin AS source_name,
    sph.position AS phase_index,
    'phase' AS kind,
    sph.start_index,
    sph.end_index,
    strftime({_ISO_MS}, sph.started_at_ms / 1000.0, 'unixepoch') AS start_time,
    strftime({_ISO_MS}, sph.ended_at_ms / 1000.0, 'unixepoch') AS end_time,
    sph.duration_ms,
    date(COALESCE(sph.started_at_ms, sph.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch')
        AS canonical_session_date,
    0.0 AS confidence,
    '[]' AS evidence_reasons_json,
    sph.tool_counts_json,
    sph.word_count,
    sph.evidence_json AS evidence_payload_json,
    sph.inference_json AS inference_payload_json,
    sph.search_text,
    1 AS inference_version,
    'heuristic_session_semantics' AS inference_family
"""

_WORK_EVENT_FROM = """
    FROM session_work_events swe
    JOIN sessions s ON s.session_id = swe.session_id
    LEFT JOIN insight_materialization im
      ON im.session_id = swe.session_id AND im.insight_type = 'work_events'
"""

_PHASE_FROM = """
    FROM session_phases sph
    JOIN sessions s ON s.session_id = sph.session_id
    LEFT JOIN insight_materialization im
      ON im.session_id = sph.session_id AND im.insight_type = 'phases'
"""


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
        f"""
        SELECT {_WORK_EVENT_SELECT}
        {_WORK_EVENT_FROM}
        WHERE swe.session_id = ?
        ORDER BY swe.position
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
        f"""
        SELECT {_PHASE_SELECT}
        {_PHASE_FROM}
        WHERE sph.session_id = ?
        ORDER BY sph.position
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
            JOIN sessions s ON s.session_id = swe.session_id
            LEFT JOIN insight_materialization im
              ON im.session_id = swe.session_id AND im.insight_type = 'work_events'
            JOIN session_work_events_fts
              ON session_work_events_fts.event_id = swe.event_id
        """
        where = ["session_work_events_fts MATCH ?"]
        params.append(escape_fts5_query(query.query))
        order_by = (
            "ORDER BY bm25(session_work_events_fts), "
            "COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms, im.materialized_at_ms) DESC, swe.position"
        )
    else:
        from_clause = _WORK_EVENT_FROM
        where = []
        order_by = (
            "ORDER BY COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms, im.materialized_at_ms) DESC, "
            "swe.position"
        )

    if query.session_id:
        where.append("swe.session_id = ?")
        params.append(query.session_id)
    if query.origin:
        where.append("s.origin = ?")
        params.append(query.origin)
    if query.heuristic_label:
        where.append("swe.work_event_type = ?")
        params.append(query.heuristic_label)
    if query.since:
        where.append(
            "strftime('%Y-%m-%dT%H:%M:%fZ', COALESCE(swe.ended_at_ms, swe.started_at_ms, im.source_sort_key_ms, im.materialized_at_ms) / 1000.0, 'unixepoch') >= ?"
        )
        params.append(query.since)
    if query.until:
        where.append(
            "strftime('%Y-%m-%dT%H:%M:%fZ', COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms, im.materialized_at_ms) / 1000.0, 'unixepoch') <= ?"
        )
        params.append(query.until)
    if query.session_date_since:
        where.append(
            "date(COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch') >= date(?)"
        )
        params.append(query.session_date_since)
    if query.session_date_until:
        where.append(
            "date(COALESCE(swe.started_at_ms, swe.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch') <= date(?)"
        )
        params.append(query.session_date_until)

    sql = f"SELECT {_WORK_EVENT_SELECT} " + from_clause
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
        where.append("sph.session_id = ?")
        params.append(query.session_id)
    if query.origin:
        where.append("s.origin = ?")
        params.append(query.origin)
    if query.kind:
        where.append("? = 'phase'")
        params.append(query.kind)
    if query.since:
        where.append(
            "strftime('%Y-%m-%dT%H:%M:%fZ', COALESCE(sph.ended_at_ms, sph.started_at_ms, im.source_sort_key_ms, im.materialized_at_ms) / 1000.0, 'unixepoch') >= ?"
        )
        params.append(query.since)
    if query.until:
        where.append(
            "strftime('%Y-%m-%dT%H:%M:%fZ', COALESCE(sph.started_at_ms, sph.ended_at_ms, im.source_sort_key_ms, im.materialized_at_ms) / 1000.0, 'unixepoch') <= ?"
        )
        params.append(query.until)
    if query.session_date_since:
        where.append(
            "date(COALESCE(sph.started_at_ms, sph.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch') >= date(?)"
        )
        params.append(query.session_date_since)
    if query.session_date_until:
        where.append(
            "date(COALESCE(sph.started_at_ms, sph.ended_at_ms, im.source_sort_key_ms) / 1000.0, 'unixepoch') <= date(?)"
        )
        params.append(query.session_date_until)

    sql = f"SELECT {_PHASE_SELECT} {_PHASE_FROM}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(sph.started_at_ms, sph.ended_at_ms, im.source_sort_key_ms, im.materialized_at_ms) DESC, sph.position"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]
