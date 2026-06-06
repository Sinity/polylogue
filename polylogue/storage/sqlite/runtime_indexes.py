"""Runtime index extensions that are safe to ensure on existing archives."""

from __future__ import annotations

import sqlite3

import aiosqlite

ACTION_EVENT_RUNTIME_INDEX_DDL: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_action_events_conv_kind
    ON action_events(session_id, action_kind)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_action_events_conv_tool
    ON action_events(session_id, normalized_tool_name)
    """,
)


def ensure_runtime_indexes_sync(conn: sqlite3.Connection) -> None:
    for ddl in ACTION_EVENT_RUNTIME_INDEX_DDL:
        conn.execute(ddl)


async def ensure_runtime_indexes_async(conn: aiosqlite.Connection) -> None:
    for ddl in ACTION_EVENT_RUNTIME_INDEX_DDL:
        await conn.execute(ddl)


__all__ = [
    "ACTION_EVENT_RUNTIME_INDEX_DDL",
    "ensure_runtime_indexes_async",
    "ensure_runtime_indexes_sync",
]
