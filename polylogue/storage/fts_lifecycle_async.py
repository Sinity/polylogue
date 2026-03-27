"""Asynchronous FTS lifecycle operations."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.fts_lifecycle_sql import (
    ACTION_FTS_INDEX_DOC_COUNT_SQL,
    ACTION_FTS_INDEX_EXISTS_SQL,
    ACTION_FTS_REBUILD_SQL,
    FTS_ACTIONS_TABLE_SQL,
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_EXISTS_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_REBUILD_SQL,
    chunked,
    delete_action_rows_sql,
    delete_conversation_rows_sql,
    insert_action_rows_sql,
    insert_conversation_rows_sql,
)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 tables exist on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)
    await conn.execute(FTS_ACTIONS_TABLE_SQL)


async def rebuild_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Rebuild the full FTS index from persisted message rows."""
    await ensure_fts_index_async(conn)
    await conn.execute("DELETE FROM messages_fts")
    await conn.execute("DELETE FROM action_events_fts")
    await conn.execute(FTS_REBUILD_SQL)
    await conn.execute(ACTION_FTS_REBUILD_SQL)


async def repair_fts_index_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> None:
    """Repair FTS rows for the supplied conversations from persisted rows."""
    await ensure_fts_index_async(conn)
    if not conversation_ids:
        return

    for chunk in chunked(list(conversation_ids), size=500):
        params = tuple(chunk)
        await conn.execute(delete_conversation_rows_sql(len(chunk)), params)
        await conn.execute(insert_conversation_rows_sql(len(chunk)), params)
        await conn.execute(delete_action_rows_sql(len(chunk)), params)
        await conn.execute(insert_action_rows_sql(len(chunk)), params)


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document counts for the async FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    action_count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
        action_exists_row = await (await conn.execute(ACTION_FTS_INDEX_EXISTS_SQL)).fetchone()
        if action_exists_row:
            action_count_row = await (await conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL)).fetchone()
            action_count = action_count_row[0] if action_count_row else 0
    return {"exists": exists, "count": int(count), "action_count": int(action_count)}


__all__ = [
    "ensure_fts_index_async",
    "fts_index_status_async",
    "rebuild_fts_index_async",
    "repair_fts_index_async",
]
