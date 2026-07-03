"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from typing import Protocol

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked

FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        block_id UNINDEXED,
        message_id UNINDEXED,
        session_id UNINDEXED,
        block_type UNINDEXED,
        text,
        content='',
        contentless_delete=1,
        tokenize='unicode61'
    );
"""

FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM messages_fts_docsize"
FTS_INDEXABLE_MESSAGE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM blocks
    WHERE search_text != ''
"""
FTS_REBUILD_SQL = """
    DELETE FROM messages_fts
"""


class IndexedMessage(Protocol):
    message_id: str
    session_id: str
    text: str | None


def delete_session_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    # DELETE on external-content FTS5 raises ``database disk image is malformed``
    # when a rowid is not present in the FTS index.  Filter the deletion to
    # rowids that actually have a docsize shadow entry (i.e., are indexed).
    return f"""
        DELETE FROM messages_fts
        WHERE rowid IN (
            SELECT blocks.rowid
            FROM blocks
            WHERE blocks.session_id IN ({placeholders})
        )
        AND rowid IN (SELECT id FROM messages_fts_docsize)
    """


def insert_session_rows_sql(chunk_size: int) -> str:
    values = ", ".join("(?)" for _ in range(chunk_size))
    return f"""
        WITH raw_target_sessions(session_id) AS (
            VALUES {values}
        ),
        target_sessions AS (
            SELECT DISTINCT session_id
            FROM raw_target_sessions
        )
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type, b.search_text
        FROM blocks AS b
        JOIN target_sessions AS target
          ON target.session_id = b.session_id
        WHERE b.search_text != ''
    """


def insert_all_message_rows_sql() -> str:
    return """
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, search_text
        FROM blocks
        WHERE search_text != ''
    """


def insert_missing_message_rows_sql() -> str:
    return """
        WITH missing(rowid, block_id, message_id, session_id, block_type, search_text) AS (
            SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type, b.search_text
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL AND b.search_text != ''
        )
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, search_text
        FROM missing
    """


def insert_missing_message_rows_range_sql() -> str:
    return """
        WITH missing(rowid, block_id, message_id, session_id, block_type, search_text) AS (
            SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type, b.search_text
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL
              AND b.search_text != ''
              AND b.rowid > ?
              AND b.rowid <= ?
        )
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, search_text
        FROM missing
    """


__all__ = [
    "FTS_INDEXABLE_MESSAGE_COUNT_SQL",
    "FTS_INDEX_DOC_COUNT_SQL",
    "FTS_INDEX_EXISTS_SQL",
    "FTS_MESSAGES_TABLE_SQL",
    "FTS_REBUILD_SQL",
    "IndexedMessage",
    "chunked",
    "delete_session_rows_sql",
    "insert_all_message_rows_sql",
    "insert_session_rows_sql",
    "insert_missing_message_rows_sql",
    "insert_missing_message_rows_range_sql",
]
