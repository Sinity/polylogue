"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from typing import Protocol

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked
from polylogue.storage.fts.pl_fold import pl_fold_sql_expr

# polylogue-9jsi: "unicode61 remove_diacritics 2" folds combining-mark
# diacritics (o accent, a ogonek, z dot, ...) symmetrically for indexed text
# and MATCH query text -- see polylogue/storage/fts/pl_fold.py for the full
# rationale. This tokenizer string is one of two canonical DDL sites (the
# other is the CREATE VIRTUAL TABLE messages_fts definition embedded in
# polylogue/storage/sqlite/archive_tiers/index.py); a drift-lock test keeps
# them identical.
FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        block_id UNINDEXED,
        message_id UNINDEXED,
        session_id UNINDEXED,
        block_type UNINDEXED,
        text,
        content='',
        contentless_delete=1,
        tokenize='unicode61 remove_diacritics 2'
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

# FTS trigger DDL for message/block FTS maintenance.
BLOCKS_FTS_TRIGGER_DDL = [
    f"""CREATE TRIGGER IF NOT EXISTS messages_fts_ai
       AFTER INSERT ON blocks WHEN new.search_text != '' BEGIN
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, {pl_fold_sql_expr("new.search_text")});
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ad
       AFTER DELETE ON blocks WHEN old.search_text != '' BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
       END""",
    f"""CREATE TRIGGER IF NOT EXISTS messages_fts_au
       AFTER UPDATE ON blocks BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, {pl_fold_sql_expr("new.search_text")}
           WHERE new.search_text != '';
       END""",
]

# FTS trigger DDL for session_work_events FTS maintenance.
SESSION_WORK_EVENT_FTS_TRIGGER_DDL = [
    f"""CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
       AFTER INSERT ON session_work_events BEGIN
           INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
           VALUES (new.event_id, new.session_id, new.work_event_type, {pl_fold_sql_expr("new.search_text")});
       END""",
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
       AFTER DELETE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
       END""",
    f"""CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
       AFTER UPDATE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
           INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
           VALUES (new.event_id, new.session_id, new.work_event_type, {pl_fold_sql_expr("new.search_text")});
       END""",
]

# FTS trigger DDL for threads FTS maintenance.
THREAD_FTS_TRIGGER_DDL = [
    f"""CREATE TRIGGER IF NOT EXISTS threads_fts_ai
       AFTER INSERT ON threads BEGIN
           INSERT INTO threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.thread_id, {pl_fold_sql_expr("new.search_text")});
       END""",
    """CREATE TRIGGER IF NOT EXISTS threads_fts_ad
       AFTER DELETE ON threads BEGIN
           DELETE FROM threads_fts WHERE thread_id = old.thread_id;
       END""",
    f"""CREATE TRIGGER IF NOT EXISTS threads_fts_au
       AFTER UPDATE ON threads BEGIN
           DELETE FROM threads_fts WHERE thread_id = old.thread_id;
           INSERT INTO threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.thread_id, {pl_fold_sql_expr("new.search_text")});
       END""",
]

# Combined trigger DDL for all FTS surfaces.
FTS_TRIGGER_DDL = BLOCKS_FTS_TRIGGER_DDL + SESSION_WORK_EVENT_FTS_TRIGGER_DDL + THREAD_FTS_TRIGGER_DDL


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
        SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type, {pl_fold_sql_expr("b.search_text")}
        FROM blocks AS b
        JOIN target_sessions AS target
          ON target.session_id = b.session_id
        WHERE b.search_text != ''
    """


def insert_all_message_rows_sql() -> str:
    return f"""
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, {pl_fold_sql_expr("search_text")}
        FROM blocks
        WHERE search_text != ''
    """


def insert_missing_message_rows_sql() -> str:
    return f"""
        WITH missing(rowid, block_id, message_id, session_id, block_type, search_text) AS (
            SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type, b.search_text
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL AND b.search_text != ''
        )
        INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, {pl_fold_sql_expr("search_text")}
        FROM missing
    """


def insert_missing_message_rows_range_sql() -> str:
    return f"""
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
        SELECT rowid, block_id, message_id, session_id, block_type, {pl_fold_sql_expr("search_text")}
        FROM missing
    """


def excess_message_rows_sql(limit: int) -> str:
    return f"""
        SELECT d.id
        FROM messages_fts_docsize AS d
        LEFT JOIN blocks AS b
          ON b.rowid = d.id
         AND b.search_text != ''
        WHERE b.rowid IS NULL
        LIMIT {max(1, int(limit))}
    """


__all__ = [
    "BLOCKS_FTS_TRIGGER_DDL",
    "FTS_INDEXABLE_MESSAGE_COUNT_SQL",
    "FTS_INDEX_DOC_COUNT_SQL",
    "FTS_INDEX_EXISTS_SQL",
    "FTS_MESSAGES_TABLE_SQL",
    "FTS_REBUILD_SQL",
    "FTS_TRIGGER_DDL",
    "IndexedMessage",
    "SESSION_WORK_EVENT_FTS_TRIGGER_DDL",
    "THREAD_FTS_TRIGGER_DDL",
    "chunked",
    "delete_session_rows_sql",
    "excess_message_rows_sql",
    "insert_all_message_rows_sql",
    "insert_missing_message_rows_range_sql",
    "insert_missing_message_rows_sql",
    "insert_session_rows_sql",
]
