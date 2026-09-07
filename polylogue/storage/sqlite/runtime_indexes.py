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
    CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_time_model
    ON session_provider_usage_events(occurred_at_ms, model_name, session_id)
    WHERE occurred_at_ms IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_message_type
    ON messages(message_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_material_origin
    ON messages(material_origin)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_active_leaf
    ON messages(session_id, is_active_leaf)
    WHERE is_active_leaf = 1
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_paste_spans_session
    ON paste_spans(session_id)
    """,
)

# These secondary indexes are useful to readers but are not required to
# construct a fresh generation.  Cold-build callers may drop them for the
# write phase and recreate them once at the quiescence boundary.  Keep this
# list explicit: the primary-key and foreign-key indexes remain present, and
# an index is never removed merely because it is currently unused by a test.
DEFERRED_SECONDARY_INDEX_NAMES: tuple[str, ...] = (
    "idx_messages_parent",
    "idx_messages_role",
    "idx_messages_message_type",
    "idx_messages_material_origin",
    "idx_blocks_content_hash",
    "idx_blocks_type",
    "idx_blocks_tool_result_outcome",
    "idx_blocks_type_tool",
    "idx_blocks_tool_id",
    "idx_blocks_search_text_populated",
)

_DEFERRED_SECONDARY_INDEX_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)",
    "CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type)",
    "CREATE INDEX IF NOT EXISTS idx_messages_material_origin ON messages(material_origin)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_content_hash ON blocks(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_type ON blocks(block_type)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_tool_result_outcome ON blocks(block_type, tool_result_is_error, tool_result_exit_code, session_id, tool_id, message_id) WHERE block_type = 'tool_result'",
    "CREATE INDEX IF NOT EXISTS idx_blocks_type_tool ON blocks(block_type, COALESCE(NULLIF(LOWER(tool_name), ''), 'unknown'))",
    "CREATE INDEX IF NOT EXISTS idx_blocks_tool_id ON blocks(tool_id) WHERE tool_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_blocks_search_text_populated ON blocks(message_id, position) WHERE search_text != ''",
)


def runtime_index_ddl() -> str:
    """Return the canonical runtime-index DDL used by the index identity."""
    return "\n".join(statement.strip() for statement in _RUNTIME_INDEX_SQL)


def ensure_runtime_indexes_sync(conn: sqlite3.Connection) -> None:
    for sql in _RUNTIME_INDEX_SQL:
        conn.execute(sql)


def defer_secondary_indexes_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Drop only the explicitly deferrable reader indexes.

    The return value records which indexes were present so a caller can make
    restoration part of its boundary receipt.  This helper is safe on an
    empty or partially initialized schema and never touches canonical FTS
    tables or primary-key indexes.
    """
    dropped: list[str] = []
    for name in DEFERRED_SECONDARY_INDEX_NAMES:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()
        if present is None:
            continue
        conn.execute(f'DROP INDEX IF EXISTS "{name}"')
        dropped.append(name)
    return tuple(dropped)


def restore_deferred_secondary_indexes_sync(conn: sqlite3.Connection) -> None:
    """Recreate the complete runtime index set at the build boundary."""
    for sql in _DEFERRED_SECONDARY_INDEX_SQL:
        conn.execute(sql)
    ensure_runtime_indexes_sync(conn)


async def ensure_runtime_indexes_async(conn: aiosqlite.Connection) -> None:
    for sql in _RUNTIME_INDEX_SQL:
        await conn.execute(sql)


__all__ = [
    "DEFERRED_SECONDARY_INDEX_NAMES",
    "defer_secondary_indexes_sync",
    "ensure_runtime_indexes_async",
    "ensure_runtime_indexes_sync",
    "restore_deferred_secondary_indexes_sync",
    "runtime_index_ddl",
]
