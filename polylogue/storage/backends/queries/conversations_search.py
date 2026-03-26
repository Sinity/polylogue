"""Conversation search helpers."""

from __future__ import annotations

import aiosqlite


async def search_conversations(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    from polylogue.storage.search import build_ranked_conversation_search_query

    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    )
    exists = await cursor.fetchone()
    if not exists:
        from polylogue.errors import DatabaseError

        raise DatabaseError("Search index not built. Run indexing first or use a different backend.")

    query_spec = build_ranked_conversation_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
    )
    if query_spec is None:
        return []

    sql, params = query_spec
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def search_action_conversations(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    from polylogue.storage.search import build_ranked_action_search_query

    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events_fts'"
    )
    exists = await cursor.fetchone()
    if not exists:
        from polylogue.errors import DatabaseError

        raise DatabaseError("Action search index not built. Rebuild the search index first.")

    query_spec = build_ranked_action_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
    )
    if query_spec is None:
        return []

    sql, params = query_spec
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [str(row["conversation_id"]) for row in rows]


__all__ = ["search_action_conversations", "search_conversations"]
