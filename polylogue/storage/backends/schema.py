"""SQLite schema management: DDL and version control."""

from __future__ import annotations

import sqlite3
from contextlib import suppress

from polylogue.logging import get_logger

logger = get_logger(__name__)
SCHEMA_VERSION = 1


_VEC0_DDL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id TEXT PRIMARY KEY,
        embedding float[1024],
        +provider_name TEXT,
        +conversation_id TEXT
    )
"""


# Complete target schema applied to fresh databases.
SCHEMA_DDL = """
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT CHECK (validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL),
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_conv_provider
        ON raw_conversations(provider_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_payload_provider
        ON raw_conversations(payload_provider)
        WHERE payload_provider IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source
        ON raw_conversations(source_path);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime
        ON raw_conversations(source_path, file_mtime);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_parse_ready
        ON raw_conversations(raw_id)
        WHERE parsed_at IS NULL
          AND validated_at IS NOT NULL
          AND (validation_status IS NULL OR validation_status != 'failed');

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL,
            parent_conversation_id TEXT REFERENCES conversations(conversation_id),
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork', 'subagent') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_conversations(raw_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX IF NOT EXISTS idx_conversations_source_name
        ON conversations(source_name) WHERE source_name IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_parent
        ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_content_hash
        ON conversations(content_hash);

        CREATE INDEX IF NOT EXISTS idx_conversations_sortkey
        ON conversations(sort_key);

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            parent_message_id TEXT REFERENCES messages(message_id),
            branch_index INTEGER NOT NULL DEFAULT 0,
            provider_name TEXT NOT NULL DEFAULT '',
            word_count INTEGER NOT NULL DEFAULT 0,
            has_tool_use INTEGER NOT NULL DEFAULT 0,
            has_thinking INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_sortkey
        ON messages(conversation_id, sort_key);

        CREATE INDEX IF NOT EXISTS idx_messages_parent
        ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_messages_provider_role
        ON messages(provider_name, role);

        CREATE INDEX IF NOT EXISTS idx_messages_provider_stats
        ON messages(provider_name, role, has_tool_use, has_thinking, word_count, conversation_id);

        CREATE TABLE IF NOT EXISTS content_blocks (
            block_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            block_index INTEGER NOT NULL,
            type TEXT NOT NULL,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            media_type TEXT,
            metadata TEXT,
            semantic_type TEXT,
            UNIQUE (message_id, block_index)
        );

        CREATE INDEX IF NOT EXISTS idx_content_blocks_message
        ON content_blocks(message_id);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_conversation
        ON content_blocks(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_type
        ON content_blocks(type);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_conv_type
        ON content_blocks(conversation_id, type);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_semantic_type
        ON content_blocks(semantic_type);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_conv_semantic
        ON content_blocks(conversation_id, semantic_type);

        CREATE TABLE IF NOT EXISTS conversation_stats (
            conversation_id TEXT PRIMARY KEY
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            provider_name   TEXT NOT NULL DEFAULT '',
            message_count   INTEGER NOT NULL DEFAULT 0,
            word_count      INTEGER NOT NULL DEFAULT 0,
            tool_use_count  INTEGER NOT NULL DEFAULT 0,
            thinking_count  INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_conv_stats_provider
        ON conversation_stats(provider_name);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_messages
        ON conversation_stats(message_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_words
        ON conversation_stats(word_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_tool_use
        ON conversation_stats(tool_use_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_thinking
        ON conversation_stats(thinking_count);

        CREATE TABLE IF NOT EXISTS attachments (
            attachment_id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            ref_count INTEGER NOT NULL DEFAULT 0,
            provider_meta TEXT,
            UNIQUE (attachment_id)
        );

        CREATE TABLE IF NOT EXISTS attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            provider_meta TEXT,
            FOREIGN KEY (attachment_id)
                REFERENCES attachments(attachment_id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (message_id)
                REFERENCES messages(message_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_conversation
        ON attachment_refs(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_attachment
        ON attachment_refs(attachment_id);

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_runs_timestamp
        ON runs(timestamp DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS messages_fts_ai
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_ad
        AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_au
        AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _ensure_vec0_table(conn: sqlite3.Connection) -> None:
    with suppress(Exception):
        conn.execute("SELECT vec_version()")
        conn.execute(_VEC0_DDL)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    The schema is defined entirely by ``SCHEMA_DDL``. Fresh databases get that
    DDL applied and are stamped with version 1. Existing databases must already
    match version 1 exactly; there is no migration ladder.
    """
    cursor = conn.execute("PRAGMA user_version")
    current_version = cursor.fetchone()[0]

    if current_version == 0:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        _ensure_vec0_table(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        logger.debug("Created fresh schema v%s", SCHEMA_VERSION)
        return

    if current_version == SCHEMA_VERSION:
        _ensure_vec0_table(conn)
        return

    from polylogue.errors import DatabaseError

    raise DatabaseError(
        f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
        "This database was created with a different schema. Recreate the database "
        f"or update its user_version to match v{SCHEMA_VERSION} after verifying the schema."
    )
