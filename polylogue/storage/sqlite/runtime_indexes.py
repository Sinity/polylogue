"""Runtime index extensions that are safe to ensure on existing archives."""

from __future__ import annotations

import sqlite3

import aiosqlite

_RUNTIME_INDEX_SQL: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_session_events_source_message
    ON session_events(source_message_id)
    WHERE source_message_id IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_agent_policies_source_message
    ON session_agent_policies(source_message_id)
    WHERE source_message_id IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_source_message
    ON session_provider_usage_events(source_message_id)
    WHERE source_message_id IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_message_type
    ON messages(message_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_material_origin
    ON messages(material_origin)
    """,
)


def ensure_runtime_indexes_sync(conn: sqlite3.Connection) -> None:
    for sql in _RUNTIME_INDEX_SQL:
        conn.execute(sql)


async def ensure_runtime_indexes_async(conn: aiosqlite.Connection) -> None:
    for sql in _RUNTIME_INDEX_SQL:
        await conn.execute(sql)


__all__ = [
    "ensure_runtime_indexes_async",
    "ensure_runtime_indexes_sync",
]
