"""Profile-oriented durable session-product read queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.store import SessionProfileRecord

__all__ = [
    "get_session_profile",
    "get_session_profiles_batch",
    "list_session_profiles",
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
    tier: str = "merged",
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionProfileRecord]:
    params: list[object] = []
    if query:
        fts_table = {
            "evidence": "session_profile_evidence_fts",
            "inference": "session_profile_inference_fts",
            "enrichment": "session_profile_enrichment_fts",
            "merged": "session_profiles_fts",
        }.get(tier, "session_profiles_fts")
        from_clause = f"""
            FROM session_profiles sp
            JOIN {fts_table}
              ON {fts_table}.conversation_id = sp.conversation_id
        """
        where = [f"{fts_table} MATCH ?"]
        params.append(query)
        order_by = f"ORDER BY bm25({fts_table}), COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"
    else:
        from_clause = "FROM session_profiles sp"
        where = []
        order_by = "ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"

    if provider:
        where.append("sp.provider_name = ?")
        params.append(provider)
    if since:
        where.append("COALESCE(sp.last_message_at, sp.source_updated_at, sp.first_message_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(sp.first_message_at, sp.source_updated_at, sp.last_message_at) <= ?")
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
