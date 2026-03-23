"""Schema DDL declarations grouped by storage domain bands."""

from __future__ import annotations

from polylogue.storage.backends.schema_ddl_actions import (
    ACTION_EVENT_DDL as _ACTION_EVENT_DDL,
)
from polylogue.storage.backends.schema_ddl_actions import (
    ACTION_FTS_DDL as _ACTION_FTS_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    ARTIFACT_OBSERVATION_DDL as _ARTIFACT_OBSERVATION_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    MAINTENANCE_RUN_DDL as _MAINTENANCE_RUN_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    PUBLICATION_DDL as _PUBLICATION_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    VEC0_DDL as _VEC0_DDL,
)
from polylogue.storage.backends.schema_ddl_products import (
    SESSION_PRODUCT_DDL as _SESSION_PRODUCT_DDL,
)

SCHEMA_VERSION = 1


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

""" + _ARTIFACT_OBSERVATION_DDL + """

""" + _PUBLICATION_DDL + """

""" + _MAINTENANCE_RUN_DDL + """

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

SCHEMA_DDL += _ACTION_EVENT_DDL
SCHEMA_DDL += _ACTION_FTS_DDL
SCHEMA_DDL += _SESSION_PRODUCT_DDL

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_ACTION_EVENT_DDL",
    "_ACTION_FTS_DDL",
    "_ARTIFACT_OBSERVATION_DDL",
    "_MAINTENANCE_RUN_DDL",
    "_PUBLICATION_DDL",
    "_SESSION_PRODUCT_DDL",
    "_VEC0_DDL",
]
