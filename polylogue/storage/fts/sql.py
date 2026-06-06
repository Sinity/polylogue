"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from typing import Protocol

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked

FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message_id UNINDEXED,
        session_id UNINDEXED,
        text,
        content='',
        contentless_delete=1,
        tokenize='unicode61'
    );
"""

FTS_ACTIONS_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS action_events_fts USING fts5(
        event_id UNINDEXED,
        message_id UNINDEXED,
        session_id UNINDEXED,
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
FTS_INDEXABLE_MESSAGE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM messages AS m
    WHERE NULLIF(m.text, '') IS NOT NULL
       OR EXISTS (
           SELECT 1
           FROM content_blocks AS cb
           WHERE cb.message_id = m.message_id
             AND (
                 NULLIF(cb.text, '') IS NOT NULL
                 OR NULLIF(cb.tool_input, '') IS NOT NULL
                 OR NULLIF(cb.metadata, '') IS NOT NULL
             )
       )
"""
ACTION_FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events_fts'"
ACTION_FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events_fts_docsize"
FTS_REBUILD_SQL = """
    INSERT INTO messages_fts(messages_fts) VALUES('delete-all')
"""

ACTION_FTS_REBUILD_SQL = """
    INSERT INTO action_events_fts(action_events_fts) VALUES('rebuild')
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
            SELECT messages.rowid
            FROM messages
            WHERE messages.session_id IN ({placeholders})
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
        ),
        message_parts AS (
            SELECT m.message_id, m.text AS part, -1 AS block_index, 0 AS part_index
            FROM messages AS m
            JOIN target_sessions AS target
              ON target.session_id = m.session_id
            WHERE m.text IS NOT NULL AND m.text != ''
            UNION ALL
            SELECT cb.message_id, cb.text AS part, cb.block_index, 1 AS part_index
            FROM content_blocks AS cb
            JOIN target_sessions AS target
              ON target.session_id = cb.session_id
            WHERE cb.text IS NOT NULL AND cb.text != ''
            UNION ALL
            SELECT cb.message_id, cb.tool_input AS part, cb.block_index, 2 AS part_index
            FROM content_blocks AS cb
            JOIN target_sessions AS target
              ON target.session_id = cb.session_id
            WHERE cb.tool_input IS NOT NULL AND cb.tool_input != ''
            UNION ALL
            SELECT cb.message_id, cb.metadata AS part, cb.block_index, 3 AS part_index
            FROM content_blocks AS cb
            JOIN target_sessions AS target
              ON target.session_id = cb.session_id
            WHERE cb.metadata IS NOT NULL AND cb.metadata != ''
        ),
        grouped_parts AS (
            SELECT
                message_parts.message_id,
                group_concat(message_parts.part, char(10)) AS search_text
            FROM (
                SELECT message_id, part, block_index, part_index
                FROM message_parts
                ORDER BY message_id, block_index, part_index
            ) AS message_parts
            GROUP BY message_parts.message_id
        )
        INSERT INTO messages_fts (rowid, message_id, session_id, text)
        SELECT m.rowid, m.message_id, m.session_id, grouped_parts.search_text
        FROM messages AS m
        JOIN target_sessions AS target
          ON target.session_id = m.session_id
        JOIN grouped_parts ON grouped_parts.message_id = m.message_id
    """


def insert_all_message_rows_sql() -> str:
    return """
        INSERT INTO messages_fts (rowid, message_id, session_id, text)
        SELECT m.rowid, m.message_id, m.session_id, parts.search_text
        FROM messages AS m
        JOIN (
            SELECT
                source.message_id,
                group_concat(source.part, char(10)) AS search_text
            FROM (
                SELECT message_id, text AS part, -1 AS block_index, 0 AS part_index
                FROM messages
                WHERE text IS NOT NULL AND text != ''
                UNION ALL
                SELECT message_id, text AS part, block_index, 1 AS part_index
                FROM content_blocks
                WHERE text IS NOT NULL AND text != ''
                UNION ALL
                SELECT message_id, tool_input AS part, block_index, 2 AS part_index
                FROM content_blocks
                WHERE tool_input IS NOT NULL AND tool_input != ''
                UNION ALL
                SELECT message_id, metadata AS part, block_index, 3 AS part_index
                FROM content_blocks
                WHERE metadata IS NOT NULL AND metadata != ''
                ORDER BY 1, 3, 4
            ) AS source
            GROUP BY source.message_id
        ) AS parts ON parts.message_id = m.message_id
    """


def insert_missing_message_rows_sql() -> str:
    return """
        WITH missing(rowid, message_id, session_id) AS (
            SELECT m.rowid, m.message_id, m.session_id
            FROM messages AS m
            LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
            WHERE d.id IS NULL
              AND (
                  NULLIF(m.text, '') IS NOT NULL
                  OR EXISTS (
                      SELECT 1
                      FROM content_blocks AS cb
                      WHERE cb.message_id = m.message_id
                        AND (
                            NULLIF(cb.text, '') IS NOT NULL
                            OR NULLIF(cb.tool_input, '') IS NOT NULL
                            OR NULLIF(cb.metadata, '') IS NOT NULL
                        )
                  )
              )
        )
        INSERT INTO messages_fts (rowid, message_id, session_id, text)
        SELECT missing.rowid, missing.message_id, missing.session_id, parts.search_text
        FROM missing
        JOIN (
            SELECT
                source.message_id,
                group_concat(source.part, char(10)) AS search_text
            FROM (
                SELECT *
                FROM (
                    SELECT m.message_id, m.text AS part, -1 AS block_index, 0 AS part_index
                    FROM messages AS m
                    JOIN missing ON missing.message_id = m.message_id
                    WHERE m.text IS NOT NULL AND m.text != ''
                    UNION ALL
                    SELECT cb.message_id, cb.text AS part, cb.block_index, 1 AS part_index
                    FROM content_blocks AS cb
                    JOIN missing ON missing.message_id = cb.message_id
                    WHERE cb.text IS NOT NULL AND cb.text != ''
                    UNION ALL
                    SELECT cb.message_id, cb.tool_input AS part, cb.block_index, 2 AS part_index
                    FROM content_blocks AS cb
                    JOIN missing ON missing.message_id = cb.message_id
                    WHERE cb.tool_input IS NOT NULL AND cb.tool_input != ''
                    UNION ALL
                    SELECT cb.message_id, cb.metadata AS part, cb.block_index, 3 AS part_index
                    FROM content_blocks AS cb
                    JOIN missing ON missing.message_id = cb.message_id
                    WHERE cb.metadata IS NOT NULL AND cb.metadata != ''
                ) AS ordered_source
                ORDER BY ordered_source.message_id, ordered_source.block_index, ordered_source.part_index
            ) AS source
            GROUP BY source.message_id
        ) AS parts ON parts.message_id = missing.message_id
    """


def insert_missing_plain_message_rows_sql() -> str:
    return """
        INSERT INTO messages_fts (rowid, message_id, session_id, text)
        SELECT m.rowid, m.message_id, m.session_id, m.text
        FROM messages AS m
        LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
        WHERE d.id IS NULL
          AND NULLIF(m.text, '') IS NOT NULL
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
            WHERE ae.session_id IN ({placeholders})
        )
        AND rowid IN (SELECT id FROM action_events_fts_docsize)
    """


def insert_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO action_events_fts (rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
        SELECT
            ae.rowid,
            ae.event_id,
            ae.message_id,
            ae.session_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        WHERE ae.session_id IN ({placeholders})
    """


def insert_missing_action_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO action_events_fts (rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
        SELECT
            ae.rowid,
            ae.event_id,
            ae.message_id,
            ae.session_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        LEFT JOIN action_events_fts_docsize d ON d.id = ae.rowid
        WHERE ae.session_id IN ({placeholders})
          AND d.id IS NULL
    """


def insert_all_missing_action_rows_sql() -> str:
    return """
        INSERT INTO action_events_fts (rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
        SELECT
            ae.rowid,
            ae.event_id,
            ae.message_id,
            ae.session_id,
            ae.action_kind,
            ae.normalized_tool_name,
            ae.search_text
        FROM action_events ae
        LEFT JOIN action_events_fts_docsize d ON d.id = ae.rowid
        WHERE d.id IS NULL
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
    "delete_session_rows_sql",
    "insert_action_rows_sql",
    "insert_all_message_rows_sql",
    "insert_all_missing_action_rows_sql",
    "insert_session_rows_sql",
    "insert_missing_action_rows_sql",
    "insert_missing_message_rows_sql",
    "insert_missing_plain_message_rows_sql",
]
