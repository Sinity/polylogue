"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from typing import Protocol

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked

FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        text,
        content='messages',
        content_rowid='rowid',
        tokenize='unicode61'
    );
"""

FTS_ACTIONS_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS action_events_fts USING fts5(
        event_id UNINDEXED,
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        action_kind UNINDEXED,
        normalized_tool_name UNINDEXED,
        search_text,
        content='action_events',
        content_rowid='rowid',
        tokenize='unicode61'
    );
"""

FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM messages_fts_docsize"
FTS_INDEXABLE_MESSAGE_COUNT_SQL = "SELECT COUNT(*) FROM messages WHERE text IS NOT NULL"
ACTION_FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events_fts'"
ACTION_FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events_fts_docsize"
FTS_REBUILD_SQL = """
    INSERT INTO messages_fts(messages_fts) VALUES('rebuild')
"""

ACTION_FTS_REBUILD_SQL = """
    INSERT INTO action_events_fts(action_events_fts) VALUES('rebuild')
"""


class IndexedMessage(Protocol):
    message_id: str
    conversation_id: str
    text: str | None


def delete_conversation_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    # DELETE on external-content FTS5 raises ``database disk image is malformed``
    # when a rowid is not present in the FTS index.  Filter the deletion to
    # rowids that actually have a docsize shadow entry (i.e., are indexed).
    return f"""
        DELETE FROM messages_fts
        WHERE rowid IN (
            SELECT messages.rowid
            FROM messages
            WHERE messages.conversation_id IN ({placeholders})
        )
        AND rowid IN (SELECT id FROM messages_fts_docsize)
    """


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
    # External-content FTS5: DELETE only works when both the base-table row
    # is reachable (for re-tokenization) AND the FTS rowid is actually
    # indexed.  Filter against docsize so the statement is idempotent.
    return f"""
        DELETE FROM action_events_fts
        WHERE rowid IN (
            SELECT ae.rowid
            FROM action_events ae
            WHERE ae.conversation_id IN ({placeholders})
        )
        AND rowid IN (SELECT id FROM action_events_fts_docsize)
    """


def insert_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO action_events_fts (rowid, event_id, message_id, conversation_id, action_kind, normalized_tool_name, search_text)
        SELECT
            ae.rowid,
            ae.event_id,
            ae.message_id,
            ae.conversation_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        WHERE ae.conversation_id IN ({placeholders})
    """


def insert_missing_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO action_events_fts (rowid, event_id, message_id, conversation_id, action_kind, normalized_tool_name, search_text)
        SELECT
            ae.rowid,
            ae.event_id,
            ae.message_id,
            ae.conversation_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        LEFT JOIN action_events_fts_docsize d ON d.id = ae.rowid
        WHERE ae.conversation_id IN ({placeholders})
          AND d.id IS NULL
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
    "insert_missing_action_rows_sql",
]
