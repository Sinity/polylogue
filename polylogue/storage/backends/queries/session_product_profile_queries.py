"""Profile-oriented durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.store import (
    SessionProfileRecord,
    _json_array_or_none,
    _json_or_none,
)

__all__ = [
    "get_session_profile",
    "get_session_profiles_batch",
    "list_session_profiles",
    "replace_session_profile",
]


async def get_session_profile(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> SessionProfileRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM session_profiles WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    return _row_to_session_profile_record(row) if row else None


async def get_session_profiles_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
) -> dict[str, SessionProfileRecord]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    )
    rows = await cursor.fetchall()
    return {str(row["conversation_id"]): _row_to_session_profile_record(row) for row in rows}


async def list_session_profiles(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    first_message_since: str | None = None,
    first_message_until: str | None = None,
    session_date_since: str | None = None,
    session_date_until: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionProfileRecord]:
    params: list[object] = []
    if query:
        from_clause = """
            FROM session_profiles sp
            JOIN session_profiles_fts
              ON session_profiles_fts.conversation_id = sp.conversation_id
        """
        where = ["session_profiles_fts MATCH ?"]
        params.append(query)
        order_by = "ORDER BY bm25(session_profiles_fts), COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"
    else:
        from_clause = "FROM session_profiles sp"
        where = []
        order_by = "ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"

    if provider:
        where.append("sp.provider_name = ?")
        params.append(provider)
    if since:
        where.append(
            "COALESCE(sp.last_message_at, sp.source_updated_at, sp.first_message_at) >= ?"
        )
        params.append(since)
    if until:
        where.append(
            "COALESCE(sp.first_message_at, sp.source_updated_at, sp.last_message_at) <= ?"
        )
        params.append(until)
    if first_message_since:
        where.append("sp.first_message_at >= ?")
        params.append(first_message_since)
    if first_message_until:
        where.append("sp.first_message_at <= ?")
        params.append(first_message_until)
    if session_date_since:
        where.append("sp.canonical_session_date >= date(?)")
        params.append(session_date_since)
    if session_date_until:
        where.append("sp.canonical_session_date <= date(?)")
        params.append(session_date_until)

    sql = "SELECT sp.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def replace_session_profile(
    conn: aiosqlite.Connection,
    record: SessionProfileRecord,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_profiles WHERE conversation_id = ?",
        (record.conversation_id,),
    )
    await conn.execute(
        """
        INSERT INTO session_profiles (
            conversation_id,
            materializer_version,
            materialized_at,
            source_updated_at,
            source_sort_key,
            provider_name,
            title,
            first_message_at,
            last_message_at,
            canonical_session_date,
            primary_work_kind,
            repo_paths_json,
            canonical_projects_json,
            tags_json,
            auto_tags_json,
            message_count,
            work_event_count,
            phase_count,
            word_count,
            tool_use_count,
            thinking_count,
            total_cost_usd,
            total_duration_ms,
            engaged_duration_ms,
            wall_duration_ms,
            payload_json,
            search_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.conversation_id,
            record.materializer_version,
            record.materialized_at,
            record.source_updated_at,
            record.source_sort_key,
            record.provider_name,
            record.title,
            record.first_message_at,
            record.last_message_at,
            record.canonical_session_date,
            record.primary_work_kind,
            _json_array_or_none(record.repo_paths),
            _json_array_or_none(record.canonical_projects),
            _json_array_or_none(record.tags),
            _json_array_or_none(record.auto_tags),
            record.message_count,
            record.work_event_count,
            record.phase_count,
            record.word_count,
            record.tool_use_count,
            record.thinking_count,
            record.total_cost_usd,
            record.total_duration_ms,
            record.engaged_duration_ms,
            record.wall_duration_ms,
            _json_or_none(record.payload),
            record.search_text,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()
