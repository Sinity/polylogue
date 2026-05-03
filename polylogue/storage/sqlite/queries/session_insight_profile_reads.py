"""Profile-oriented durable session-insight read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.query_models import SessionProfileListQuery
from polylogue.storage.runtime import SessionProfileRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_session_profile_record

__all__ = [
    "get_session_profile",
    "get_session_profiles_batch",
    "list_session_profiles",
]


def _session_profile_order_by(sort: str) -> str:
    if sort == "first-message":
        return "ORDER BY COALESCE(unixepoch(sp.first_message_at), unixepoch(sp.source_updated_at), unixepoch(sp.last_message_at)) DESC, sp.conversation_id"
    if sort == "last-message":
        return "ORDER BY COALESCE(unixepoch(sp.last_message_at), unixepoch(sp.source_updated_at), unixepoch(sp.first_message_at)) DESC, sp.conversation_id"
    if sort == "wallclock":
        return "ORDER BY sp.wall_duration_ms DESC, COALESCE(unixepoch(sp.last_message_at), unixepoch(sp.source_updated_at), unixepoch(sp.first_message_at)) DESC, sp.conversation_id"
    return "ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"


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
    query: SessionProfileListQuery,
) -> list[SessionProfileRecord]:
    params: list[object] = []
    secondary_order_by = _session_profile_order_by(query.sort).removeprefix("ORDER BY ")
    if query.query:
        fts_table = {
            "evidence": "session_profile_evidence_fts",
            "inference": "session_profile_inference_fts",
            "enrichment": "session_profile_enrichment_fts",
            "merged": "session_profiles_fts",
        }.get(query.tier, "session_profiles_fts")
        from_clause = f"""
            FROM session_profiles sp
            JOIN {fts_table}
              ON {fts_table}.conversation_id = sp.conversation_id
        """
        where = [f"{fts_table} MATCH ?"]
        params.append(query.query)
        order_by = f"ORDER BY bm25({fts_table}), {secondary_order_by}"
    else:
        from_clause = "FROM session_profiles sp"
        where = []
        order_by = _session_profile_order_by(query.sort)

    if query.provider:
        where.append("sp.provider_name = ?")
        params.append(query.provider)
    if query.since:
        where.append("COALESCE(sp.last_message_at, sp.source_updated_at, sp.first_message_at) >= ?")
        params.append(query.since)
    if query.until:
        where.append("COALESCE(sp.first_message_at, sp.source_updated_at, sp.last_message_at) <= ?")
        params.append(query.until)
    if query.first_message_since:
        where.append("sp.first_message_at >= ?")
        params.append(query.first_message_since)
    if query.first_message_until:
        where.append("sp.first_message_at <= ?")
        params.append(query.first_message_until)
    if query.session_date_since:
        where.append("sp.canonical_session_date >= date(?)")
        params.append(query.session_date_since)
    if query.session_date_until:
        where.append("sp.canonical_session_date <= date(?)")
        params.append(query.session_date_until)
    if query.min_wallclock_seconds is not None:
        where.append("sp.wall_duration_ms >= ?")
        params.append(query.min_wallclock_seconds * 1000)
    if query.max_wallclock_seconds is not None:
        where.append("sp.wall_duration_ms <= ?")
        params.append(query.max_wallclock_seconds * 1000)
    sql = "SELECT sp.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if query.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_profile_record(row) for row in rows]
