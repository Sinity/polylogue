"""Conversation search helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.maintenance_targets import build_maintenance_target_catalog

_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_MESSAGE_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("dangling_fts",), include_run_all=True)
_ACTION_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(
    ("action_event_read_model",),
    include_run_all=True,
)


async def search_conversations(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    from polylogue.errors import DatabaseError
    from polylogue.storage.fts_lifecycle import message_fts_readiness_async

    readiness = await message_fts_readiness_async(conn)
    if not bool(readiness["exists"]):
        raise DatabaseError(f"Search index not built. {_MESSAGE_SEARCH_REPAIR_HINT}")
    if not bool(readiness["ready"]):
        raise DatabaseError(f"Search index is incomplete. {_MESSAGE_SEARCH_REPAIR_HINT}")

    from polylogue.storage.search import build_ranked_conversation_search_query

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
    from polylogue.errors import DatabaseError
    from polylogue.storage.action_event_status import action_event_read_model_status_async
    from polylogue.storage.search import build_ranked_action_search_query

    status = await action_event_read_model_status_async(conn)
    if not bool(status["action_fts_exists"]):
        raise DatabaseError(f"Action search index not built. {_ACTION_SEARCH_REPAIR_HINT}")
    if not bool(status["action_fts_ready"]):
        raise DatabaseError(f"Action search index is incomplete. {_ACTION_SEARCH_REPAIR_HINT}")

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
