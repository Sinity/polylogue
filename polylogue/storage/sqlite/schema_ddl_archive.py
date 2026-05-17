"""Core raw/archive storage DDL fragments."""

from __future__ import annotations

RAW_ARCHIVE_DDL = """
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            blob_size INTEGER NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT CHECK (validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL),
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT,
            detection_warnings TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_conv_provider
        ON raw_conversations(provider_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_payload_provider
        ON raw_conversations(payload_provider)
        WHERE payload_provider IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime
        ON raw_conversations(source_path, file_mtime)
        WHERE file_mtime IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_path_raw_id
        ON raw_conversations(source_path, raw_id);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_parse_ready
        ON raw_conversations(raw_id)
        WHERE parsed_at IS NULL
          AND validated_at IS NOT NULL
          AND (validation_status IS NULL OR validation_status != 'failed');
"""

ARCHIVE_STORAGE_DDL = """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL DEFAULT '',
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT NOT NULL DEFAULT '',
            working_directories_json TEXT,
            git_branch TEXT,
            git_repository_url TEXT,
            version INTEGER NOT NULL,
            parent_conversation_id TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork', 'subagent') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_conversations(raw_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX IF NOT EXISTS idx_conversations_source_name
        ON conversations(source_name);

        CREATE INDEX IF NOT EXISTS idx_conversations_parent
        ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_content_hash
        ON conversations(content_hash);

        CREATE INDEX IF NOT EXISTS idx_conversations_sortkey
        ON conversations(sort_key);

        CREATE INDEX IF NOT EXISTS idx_conversations_raw_id
        ON conversations(raw_id)
        WHERE raw_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL,
            parent_message_id TEXT REFERENCES messages(message_id) ON DELETE NO ACTION,
            branch_index INTEGER NOT NULL DEFAULT 0,
            provider_name TEXT NOT NULL DEFAULT '',
            word_count INTEGER NOT NULL DEFAULT 0,
            has_tool_use INTEGER NOT NULL DEFAULT 0,
            has_thinking INTEGER NOT NULL DEFAULT 0,
            has_paste INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            model_name TEXT,
            message_type TEXT NOT NULL DEFAULT 'message',
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_sortkey
        ON messages(conversation_id, sort_key);

        CREATE INDEX IF NOT EXISTS idx_messages_parent
        ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_message_type
        ON messages(conversation_id, message_type);

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

        CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_conversation
        ON content_blocks(conversation_id)
        WHERE type = 'tool_use';

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
            thinking_count  INTEGER NOT NULL DEFAULT 0,
            paste_count     INTEGER NOT NULL DEFAULT 0
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
            provider_attachment_id TEXT,
            provider_file_id TEXT,
            provider_drive_id TEXT,
            UNIQUE (attachment_id)
        );

        CREATE TABLE IF NOT EXISTS attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            provider_meta TEXT,
            provider_attachment_id TEXT,
            provider_file_id TEXT,
            provider_drive_id TEXT,
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

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
        ON attachment_refs(message_id)
        WHERE message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_attachments_provider_attachment_id
        ON attachments(provider_attachment_id);

        CREATE INDEX IF NOT EXISTS idx_attachments_provider_file_id
        ON attachments(provider_file_id);

        CREATE INDEX IF NOT EXISTS idx_attachments_provider_drive_id
        ON attachments(provider_drive_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_attachment_id
        ON attachment_refs(provider_attachment_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_file_id
        ON attachment_refs(provider_file_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_drive_id
        ON attachment_refs(provider_drive_id);

"""


MESSAGE_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            text,
            content='messages',
            content_rowid='rowid',
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS messages_fts_ai
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            SELECT new.rowid, new.message_id, new.conversation_id, new.text
            WHERE new.text IS NOT NULL;
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_ad
        AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, message_id, conversation_id, text)
            VALUES('delete', old.rowid, old.message_id, old.conversation_id, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_au
        AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, message_id, conversation_id, text)
            VALUES('delete', old.rowid, old.message_id, old.conversation_id, old.text);
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            SELECT new.rowid, new.message_id, new.conversation_id, new.text
            WHERE new.text IS NOT NULL;
        END;
"""


# ---------------------------------------------------------------------------
# Tags many-to-many (replaces JSON metadata['tags'] field)
# MIGRATION NOTE: When schema version is bumped, migrate old JSON tag field
# on conversations.metadata -> conversation_tags + tags rows.
# ---------------------------------------------------------------------------

TAGS_M2M_DDL = """
        CREATE TABLE IF NOT EXISTS tags (
            id   INTEGER PRIMARY KEY,
            name TEXT    UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_tags (
            conversation_id TEXT    NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            tag_id          INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (conversation_id, tag_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conversation_tags_tag
            ON conversation_tags(tag_id);
"""

# ---------------------------------------------------------------------------
# Blob reference leases (prevents GC races with concurrent ingest)
# ---------------------------------------------------------------------------

BLOB_LEASE_DDL = """
        CREATE TABLE IF NOT EXISTS pending_blob_refs (
            blob_hash    TEXT    NOT NULL,
            operation_id TEXT    NOT NULL,
            acquired_at  INTEGER NOT NULL,
            PRIMARY KEY (blob_hash, operation_id)
        );

        CREATE TABLE IF NOT EXISTS gc_generations (
            generation   INTEGER PRIMARY KEY,
            completed_at INTEGER
        );
"""

# ---------------------------------------------------------------------------
# User state — target-aware marks and annotations
# ---------------------------------------------------------------------------

# NOTE (#1114): ``user_marks`` and ``user_annotations`` survive the conversation
# row's lifecycle.  ``identity_key`` is the stable surface identifier
# (``conversation:{cid}`` or ``message:{cid}:{mid}``) used to repoint the
# resolved ``conversation_id``/``message_id`` columns when a logically identical
# conversation is re-imported after delete/reset.  Conversation row FKs use
# ``ON DELETE SET NULL`` so marks/annotations persist across hard delete and are
# rebound by ``repoint_user_state_by_identity`` once the conversation reappears.
USER_MARKS_DDL = """
        CREATE TABLE IF NOT EXISTS user_marks (
            target_type     TEXT NOT NULL CHECK (target_type IN (
                'conversation', 'message', 'session', 'work_event',
                'thread', 'content_block', 'attachment', 'paste_span'
            )),
            target_id       TEXT NOT NULL,
            identity_key    TEXT NOT NULL,
            conversation_id TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
            message_id      TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            mark_type       TEXT NOT NULL CHECK (mark_type IN ('star', 'pin', 'archive')),
            created_at      TEXT NOT NULL,
            PRIMARY KEY (target_type, target_id, mark_type)
        );

        CREATE INDEX IF NOT EXISTS idx_user_marks_type
            ON user_marks(mark_type);

        CREATE INDEX IF NOT EXISTS idx_user_marks_conversation
            ON user_marks(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_user_marks_message
            ON user_marks(message_id);

        CREATE INDEX IF NOT EXISTS idx_user_marks_identity_key
            ON user_marks(identity_key);
"""

USER_ANNOTATIONS_DDL = """
        CREATE TABLE IF NOT EXISTS user_annotations (
            annotation_id   TEXT PRIMARY KEY,
            target_type     TEXT NOT NULL CHECK (target_type IN (
                'conversation', 'message', 'session', 'work_event',
                'thread', 'content_block', 'attachment', 'paste_span'
            )),
            target_id       TEXT NOT NULL,
            identity_key    TEXT NOT NULL,
            conversation_id TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
            message_id      TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            note_text       TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_user_annotations_target
            ON user_annotations(target_type, target_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_conversation
            ON user_annotations(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_message
            ON user_annotations(message_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_identity_key
            ON user_annotations(identity_key);
"""

# ---------------------------------------------------------------------------
# Saved views — named query presets storing ConversationQuerySpec as JSON
# ---------------------------------------------------------------------------

SAVED_VIEWS_DDL = """
        CREATE TABLE IF NOT EXISTS saved_views (
            view_id    TEXT PRIMARY KEY,
            name       TEXT NOT NULL UNIQUE,
            query_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
"""

# ---------------------------------------------------------------------------
# Recall packs — saved conversation selections with context-pack payloads
# ---------------------------------------------------------------------------

RECALL_PACKS_DDL = """
        CREATE TABLE IF NOT EXISTS recall_packs (
            pack_id               TEXT PRIMARY KEY,
            label                 TEXT NOT NULL,
            conversation_ids_json TEXT NOT NULL DEFAULT '[]',
            payload_json           TEXT NOT NULL DEFAULT '{}',
            created_at            TEXT NOT NULL
        );
"""

# ---------------------------------------------------------------------------
# Reader workspaces — durable multi-target reader layouts
# ---------------------------------------------------------------------------

READER_WORKSPACES_DDL = """
        CREATE TABLE IF NOT EXISTS reader_workspaces (
            workspace_id       TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            mode               TEXT NOT NULL DEFAULT 'tabs'
                CHECK (mode IN ('tabs', 'stack', 'compare', 'timeline')),
            open_targets_json  TEXT NOT NULL DEFAULT '[]',
            layout_json        TEXT NOT NULL DEFAULT '{}',
            active_target_json TEXT NOT NULL DEFAULT '{}',
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reader_workspaces_updated
            ON reader_workspaces(updated_at DESC);

        CREATE INDEX IF NOT EXISTS idx_reader_workspaces_mode
            ON reader_workspaces(mode, updated_at DESC);
"""


# ---------------------------------------------------------------------------
# User corrections — learning-feedback loop (#1131)
# ---------------------------------------------------------------------------
# Corrections live outside the content-hashed payload by design: the table is
# keyed by ``(conversation_id, insight_kind)`` and is consulted by insight
# materialization paths after they compute their base suggestion. Applying or
# removing a correction never touches the conversation's ``content_hash`` — the
# archive's idempotency invariant is preserved (AC #1131).
#
# Schema:
#   correction_id   surrogate identifier (uuid4 hex, stable across rebuilds).
#   conversation_id target session.
#   insight_kind    one of the closed enum values in
#                   ``polylogue.insights.feedback.CorrectionKind`` (the
#                   application layer enforces the closed set; storage is
#                   permissive to keep the table compatible across versions).
#   payload_json    correction-specific data (override label, accept/reject
#                   verdict, replacement summary text, etc.). Always JSON.
#   note            optional free-form reason text from the user.
#   created_at      ISO-8601 timestamp of the latest write (upsert refreshes).
#
# The ``(conversation_id, insight_kind)`` uniqueness contract is what makes
# rebuilds deterministic: at most one correction of each kind per session,
# and it always wins over the heuristic suggestion.

USER_CORRECTIONS_DDL = """
        CREATE TABLE IF NOT EXISTS user_corrections (
            correction_id   TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            insight_kind    TEXT NOT NULL,
            payload_json    TEXT NOT NULL,
            note            TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE (conversation_id, insight_kind)
        );

        CREATE INDEX IF NOT EXISTS idx_user_corrections_conversation
            ON user_corrections(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_user_corrections_kind
            ON user_corrections(insight_kind);
"""


__all__ = [
    "ARCHIVE_STORAGE_DDL",
    "BLOB_LEASE_DDL",
    "MESSAGE_FTS_DDL",
    "RAW_ARCHIVE_DDL",
    "READER_WORKSPACES_DDL",
    "RECALL_PACKS_DDL",
    "SAVED_VIEWS_DDL",
    "TAGS_M2M_DDL",
    "USER_ANNOTATIONS_DDL",
    "USER_CORRECTIONS_DDL",
    "USER_MARKS_DDL",
]
