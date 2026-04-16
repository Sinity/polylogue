"""Status and readiness helpers for the action-event read model."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION

_ACTION_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events'"
_ACTION_EVENT_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events"
_ACTION_EVENT_MISMATCH_COUNT_SQL = """
    SELECT COUNT(*)
    FROM action_events
    WHERE materializer_version != ?
"""
_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT conversation_id)
    FROM content_blocks
    WHERE type = 'tool_use'
"""
_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT conversation_id)
    FROM action_events
"""
_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
"""
_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""
_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL = """
    SELECT COUNT(*)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""


def action_event_read_model_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    from polylogue.storage.fts_lifecycle import fts_index_status_sync

    exists = bool(conn.execute(_ACTION_EVENTS_EXISTS_SQL).fetchone())
    count = 0
    stale_count = 0
    source_conversation_count = int(conn.execute(_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0)
    valid_source_conversation_count = int(
        conn.execute(_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
    )
    orphan_source_conversation_count = int(
        conn.execute(_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
    )
    orphan_tool_block_count = int(conn.execute(_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL).fetchone()[0] or 0)
    materialized_conversation_count = 0
    if exists:
        count = int(conn.execute(_ACTION_EVENT_DOC_COUNT_SQL).fetchone()[0] or 0)
        stale_count = int(
            conn.execute(
                _ACTION_EVENT_MISMATCH_COUNT_SQL,
                (ACTION_EVENT_MATERIALIZER_VERSION,),
            ).fetchone()[0]
            or 0
        )
        materialized_conversation_count = int(
            conn.execute(_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
        )
    fts_status = fts_index_status_sync(conn)
    action_fts_count = int(fts_status.get("action_count", 0))
    action_fts_ready = count == action_fts_count
    rows_ready = valid_source_conversation_count == 0 or (
        valid_source_conversation_count == materialized_conversation_count and stale_count == 0
    )
    ready = rows_ready and action_fts_ready
    return {
        "exists": exists,
        "count": count,
        "stale_count": stale_count,
        "matches_version": stale_count == 0,
        "source_conversation_count": source_conversation_count,
        "valid_source_conversation_count": valid_source_conversation_count,
        "orphan_source_conversation_count": orphan_source_conversation_count,
        "orphan_tool_block_count": orphan_tool_block_count,
        "materialized_conversation_count": materialized_conversation_count,
        "rows_ready": rows_ready,
        "action_fts_exists": bool(fts_status.get("exists", False)),
        "action_fts_count": action_fts_count,
        "action_fts_ready": action_fts_ready,
        "ready": ready,
    }


async def action_event_read_model_status_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    from polylogue.storage.fts_lifecycle import fts_index_status_async

    exists = bool(await (await conn.execute(_ACTION_EVENTS_EXISTS_SQL)).fetchone())
    count = 0
    stale_count = 0
    source_row = await (await conn.execute(_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL)).fetchone()
    source_conversation_count = int(source_row[0] or 0) if source_row else 0
    valid_source_row = await (await conn.execute(_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL)).fetchone()
    valid_source_conversation_count = int(valid_source_row[0] or 0) if valid_source_row else 0
    orphan_source_row = await (await conn.execute(_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL)).fetchone()
    orphan_source_conversation_count = int(orphan_source_row[0] or 0) if orphan_source_row else 0
    orphan_block_row = await (await conn.execute(_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL)).fetchone()
    orphan_tool_block_count = int(orphan_block_row[0] or 0) if orphan_block_row else 0
    materialized_conversation_count = 0
    if exists:
        row = await (await conn.execute(_ACTION_EVENT_DOC_COUNT_SQL)).fetchone()
        count = int(row[0] or 0) if row else 0
        mismatch_row = await (
            await conn.execute(
                _ACTION_EVENT_MISMATCH_COUNT_SQL,
                (ACTION_EVENT_MATERIALIZER_VERSION,),
            )
        ).fetchone()
        stale_count = int(mismatch_row[0] or 0) if mismatch_row else 0
        materialized_row = await (await conn.execute(_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL)).fetchone()
        materialized_conversation_count = int(materialized_row[0] or 0) if materialized_row else 0
    fts_status = await fts_index_status_async(conn)
    action_fts_count = int(fts_status.get("action_count", 0))
    action_fts_ready = count == action_fts_count
    rows_ready = valid_source_conversation_count == 0 or (
        valid_source_conversation_count == materialized_conversation_count and stale_count == 0
    )
    ready = rows_ready and action_fts_ready
    return {
        "exists": exists,
        "count": count,
        "stale_count": stale_count,
        "matches_version": stale_count == 0,
        "source_conversation_count": source_conversation_count,
        "valid_source_conversation_count": valid_source_conversation_count,
        "orphan_source_conversation_count": orphan_source_conversation_count,
        "orphan_tool_block_count": orphan_tool_block_count,
        "materialized_conversation_count": materialized_conversation_count,
        "rows_ready": rows_ready,
        "action_fts_exists": bool(fts_status.get("exists", False)),
        "action_fts_count": action_fts_count,
        "action_fts_ready": action_fts_ready,
        "ready": ready,
    }


__all__ = [
    "action_event_read_model_status_async",
    "action_event_read_model_status_sync",
]
