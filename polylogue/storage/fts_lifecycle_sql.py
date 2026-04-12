"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        text,
        tokenize='unicode61'
    );
"""

FTS_ACTIONS_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS action_events_fts USING fts5(
        event_id UNINDEXED,
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        action_kind UNINDEXED,
        tool_name UNINDEXED,
        text,
        tokenize='unicode61'
    );
"""

FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM messages_fts_docsize"
FTS_INDEXABLE_MESSAGE_COUNT_SQL = "SELECT COUNT(*) FROM messages WHERE text IS NOT NULL"
ACTION_FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events_fts'"
ACTION_FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events_fts_docsize"
FTS_REBUILD_SQL = """
    INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
    SELECT messages.rowid, messages.message_id, messages.conversation_id, messages.text
    FROM messages
    WHERE messages.text IS NOT NULL
"""

ACTION_FTS_REBUILD_SQL = """
    INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
    SELECT
        ae.event_id,
        ae.message_id,
        ae.conversation_id,
        ae.action_kind,
        ae.normalized_tool_name,
        ae.search_text
    FROM action_events ae
"""


class IndexedMessage(Protocol):
    message_id: str
    conversation_id: str
    text: str | None


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def delete_conversation_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})"


def insert_conversation_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
        SELECT messages.rowid, messages.message_id, messages.conversation_id, messages.text
        FROM messages
        WHERE messages.text IS NOT NULL AND messages.conversation_id IN ({placeholders})
    """


def delete_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"DELETE FROM action_events_fts WHERE conversation_id IN ({placeholders})"


def insert_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
        SELECT
            ae.event_id,
            ae.message_id,
            ae.conversation_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        WHERE ae.conversation_id IN ({placeholders})
    """


__all__ = [
    "ACTION_FTS_INDEX_DOC_COUNT_SQL",
    "ACTION_FTS_INDEX_EXISTS_SQL",
    "ACTION_FTS_REBUILD_SQL",
    "FTS_ACTIONS_TABLE_SQL",
    "FTS_INDEXABLE_MESSAGE_COUNT_SQL",
    "FTS_INDEX_DOC_COUNT_SQL",
    "FTS_INDEX_EXISTS_SQL",
    "FTS_MESSAGES_TABLE_SQL",
    "FTS_REBUILD_SQL",
    "IndexedMessage",
    "chunked",
    "delete_action_rows_sql",
    "delete_conversation_rows_sql",
    "insert_action_rows_sql",
    "insert_conversation_rows_sql",
]
