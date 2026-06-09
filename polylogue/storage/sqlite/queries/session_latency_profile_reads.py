"""Session latency profile read queries."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.runtime import SessionLatencyProfileRecord
from polylogue.storage.sqlite.queries.mappers_support import _row_float, _row_int, _row_text
from polylogue.types import SessionId


def _row_to_session_latency_profile_record(row: sqlite3.Row) -> SessionLatencyProfileRecord:
    return SessionLatencyProfileRecord(
        session_id=SessionId(row["session_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        source_sort_key=_row_float(row, "source_sort_key"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        source_name=row["source_name"],
        title=_row_text(row, "title"),
        first_message_at=_row_text(row, "first_message_at"),
        last_message_at=_row_text(row, "last_message_at"),
        canonical_session_date=_row_text(row, "canonical_session_date"),
        median_tool_call_ms=int(_row_int(row, "median_tool_call_ms", 0) or 0),
        p90_tool_call_ms=int(_row_int(row, "p90_tool_call_ms", 0) or 0),
        max_tool_call_ms=int(_row_int(row, "max_tool_call_ms", 0) or 0),
        stuck_tool_count=int(_row_int(row, "stuck_tool_count", 0) or 0),
        median_agent_response_ms=int(_row_int(row, "median_agent_response_ms", 0) or 0),
        median_user_response_ms=int(_row_int(row, "median_user_response_ms", 0) or 0),
        tool_call_count_by_category_json=_row_text(row, "tool_call_count_by_category_json") or "{}",
        evidence_payload_json=_row_text(row, "evidence_payload_json") or "{}",
        search_text=_row_text(row, "search_text") or str(row["session_id"]),
    )


async def get_session_latency_profile(
    conn: aiosqlite.Connection,
    session_id: str,
) -> SessionLatencyProfileRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM session_latency_profiles WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    return _row_to_session_latency_profile_record(row) if row else None


async def find_stuck_session_latency_profiles(
    conn: aiosqlite.Connection,
    *,
    since: str | None = None,
    limit: int = 50,
) -> list[SessionLatencyProfileRecord]:
    where = ["stuck_tool_count > 0"]
    params: list[object] = []
    if since:
        where.append("COALESCE(last_message_at, source_updated_at, first_message_at) >= ?")
        params.append(since)
    params.append(limit)
    cursor = await conn.execute(
        f"""
        SELECT * FROM session_latency_profiles
        WHERE {" AND ".join(where)}
        ORDER BY stuck_tool_count DESC, COALESCE(last_message_at, source_updated_at, first_message_at) DESC
        LIMIT ?
        """,
        tuple(params),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_latency_profile_record(row) for row in rows]


async def list_session_latency_profiles(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = 500,
) -> list[SessionLatencyProfileRecord]:
    where: list[str] = []
    params: list[object] = []
    if provider:
        where.append("source_name = ?")
        params.append(provider)
    if since:
        where.append("COALESCE(last_message_at, source_updated_at, first_message_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(first_message_at, source_updated_at, last_message_at) <= ?")
        params.append(until)
    sql = "SELECT * FROM session_latency_profiles"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(last_message_at, source_updated_at, first_message_at) DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_latency_profile_record(row) for row in rows]


__all__ = [
    "_row_to_session_latency_profile_record",
    "find_stuck_session_latency_profiles",
    "get_session_latency_profile",
    "list_session_latency_profiles",
]
