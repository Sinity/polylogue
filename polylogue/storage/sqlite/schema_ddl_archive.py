"""Core raw/archive storage DDL fragments."""

from __future__ import annotations

RAW_ARCHIVE_DDL = """
        CREATE TABLE IF NOT EXISTS raw_sessions (
            raw_id TEXT PRIMARY KEY,
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

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source
        ON raw_sessions(source_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_payload_provider
        ON raw_sessions(payload_provider)
        WHERE payload_provider IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime
        ON raw_sessions(source_path, file_mtime)
        WHERE file_mtime IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_path_raw_id
        ON raw_sessions(source_path, raw_id);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_parse_ready
        ON raw_sessions(raw_id)
        WHERE parsed_at IS NULL
          AND validated_at IS NOT NULL
          AND (validation_status IS NULL OR validation_status != 'failed');
"""

ARCHIVE_STORAGE_DDL = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            provider_session_id TEXT NOT NULL,
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
            parent_session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork', 'subagent') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_sessions(raw_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_source_name
        ON sessions(source_name);

        CREATE INDEX IF NOT EXISTS idx_sessions_source_provider_id
        ON sessions(source_name, provider_session_id);

        CREATE INDEX IF NOT EXISTS idx_sessions_parent
        ON sessions(parent_session_id) WHERE parent_session_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_sessions_content_hash
        ON sessions(content_hash);

        CREATE INDEX IF NOT EXISTS idx_sessions_sortkey
        ON sessions(sort_key);

        CREATE INDEX IF NOT EXISTS idx_sessions_raw_id
        ON sessions(raw_id)
        WHERE raw_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL,
            parent_message_id TEXT REFERENCES messages(message_id) ON DELETE NO ACTION,
            branch_index INTEGER NOT NULL DEFAULT 0,
            source_name TEXT NOT NULL DEFAULT '',
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
            paste_boundary_state TEXT CHECK (paste_boundary_state IN
                ('exact', 'projected', 'whole_message_fallback', 'hash_only')
                OR paste_boundary_state IS NULL),
            FOREIGN KEY (session_id)
                REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id);

        CREATE INDEX IF NOT EXISTS idx_messages_session_sortkey
        ON messages(session_id, sort_key);

        CREATE INDEX IF NOT EXISTS idx_messages_parent
        ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_messages_session_message_type
        ON messages(session_id, message_type);

        CREATE INDEX IF NOT EXISTS idx_messages_source_role
        ON messages(source_name, role);

        CREATE INDEX IF NOT EXISTS idx_messages_source_stats
        ON messages(source_name, role, has_tool_use, has_thinking, word_count, session_id);

        CREATE TABLE IF NOT EXISTS content_blocks (
            block_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            block_index INTEGER NOT NULL,
            type TEXT NOT NULL,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            metadata TEXT,
            semantic_type TEXT,
            UNIQUE (message_id, block_index)
        );

        CREATE INDEX IF NOT EXISTS idx_content_blocks_message
        ON content_blocks(message_id);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_session
        ON content_blocks(session_id);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_type
        ON content_blocks(type);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_conv_type
        ON content_blocks(session_id, type);

        CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_session
        ON content_blocks(session_id)
        WHERE type = 'tool_use';

        CREATE TABLE IF NOT EXISTS session_stats (
            session_id       TEXT PRIMARY KEY
                REFERENCES sessions(session_id) ON DELETE CASCADE,
            source_name           TEXT NOT NULL DEFAULT '',
            message_count         INTEGER NOT NULL DEFAULT 0,
            word_count            INTEGER NOT NULL DEFAULT 0,
            tool_use_count        INTEGER NOT NULL DEFAULT 0,
            thinking_count        INTEGER NOT NULL DEFAULT 0,
            paste_count           INTEGER NOT NULL DEFAULT 0,
            user_msg_count        INTEGER NOT NULL DEFAULT 0,
            assistant_msg_count   INTEGER NOT NULL DEFAULT 0,
            system_msg_count      INTEGER NOT NULL DEFAULT 0,
            tool_msg_count        INTEGER NOT NULL DEFAULT 0,
            user_word_count       INTEGER NOT NULL DEFAULT 0,
            assistant_word_count  INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_conv_stats_source
        ON session_stats(source_name);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_messages
        ON session_stats(message_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_words
        ON session_stats(word_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_tool_use
        ON session_stats(tool_use_count);

        CREATE INDEX IF NOT EXISTS idx_conv_stats_thinking
        ON session_stats(thinking_count);

        -- Attachments carry first-class native identifier columns (#1252).
        -- Lookups by `provider_attachment_id` / `provider_file_id` /
        -- `provider_drive_id` resolve against stored TEXT columns; no
        -- json_extract on the hot path. `upload_origin` is a closed
        -- vocabulary ({"drive","paste","url","oauth"} or NULL) that lets the
        -- attachment-library UI (#1199) group attachments by where they
        -- entered the archive without scanning `provider_meta`.
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
            upload_origin TEXT,
            UNIQUE (attachment_id)
        );

        CREATE TABLE IF NOT EXISTS attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            message_id TEXT,
            provider_meta TEXT,
            provider_attachment_id TEXT,
            provider_file_id TEXT,
            provider_drive_id TEXT,
            upload_origin TEXT,
            FOREIGN KEY (attachment_id)
                REFERENCES attachments(attachment_id) ON DELETE CASCADE,
            FOREIGN KEY (session_id)
                REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (message_id)
                REFERENCES messages(message_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_session
        ON attachment_refs(session_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_attachment
        ON attachment_refs(attachment_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
        ON attachment_refs(message_id)
        WHERE message_id IS NOT NULL;

        -- #1199 attachment-library: composite index over the session
        -- provider plus the attachment origin so the UI's grouping query
        -- "all attachments uploaded via drive in chatgpt sessions" answers
        -- from index without scanning attachments.
        CREATE INDEX IF NOT EXISTS idx_attachment_refs_upload_origin
        ON attachment_refs(upload_origin, session_id)
        WHERE upload_origin IS NOT NULL;

"""


MESSAGE_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            session_id UNINDEXED,
            text,
            content='',
            contentless_delete=1,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS messages_fts_ai
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT new.rowid, new.message_id, new.session_id, new.text
            WHERE new.text IS NOT NULL AND new.text != '';
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_ad
        AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_au
        AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT m.rowid, m.message_id, m.session_id, group_concat(source.part, char(10))
            FROM messages AS m
            JOIN (
                SELECT m2.message_id, m2.text AS part, -1 AS block_index, 0 AS part_index
                FROM messages AS m2
                WHERE m2.message_id = new.message_id AND m2.text IS NOT NULL AND m2.text != ''
                UNION ALL
                SELECT cb.message_id, cb.text AS part, cb.block_index, 1 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.text IS NOT NULL AND cb.text != ''
                UNION ALL
                SELECT cb.message_id, cb.tool_input AS part, cb.block_index, 2 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.tool_input IS NOT NULL AND cb.tool_input != ''
                UNION ALL
                SELECT cb.message_id, cb.metadata AS part, cb.block_index, 3 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.metadata IS NOT NULL AND cb.metadata != ''
                ORDER BY message_id, block_index, part_index
            ) AS source ON source.message_id = m.message_id
            WHERE m.message_id = new.message_id
            HAVING group_concat(source.part, char(10)) IS NOT NULL;
        END;

        -- #1606: INSERT trigger is O(1) per block instead of O(N).
        -- Instead of DELETE + full re-projection scanning ALL blocks
        -- for the message (O(N²) cumulative for N blocks), we do a
        -- single INSERT OR REPLACE that reads the existing FTS text,
        -- appends the new block's fields, and writes back. The DELETE
        -- and UPDATE triggers still do full rebuilds — those are rare.
        CREATE TRIGGER IF NOT EXISTS content_blocks_fts_ai
        AFTER INSERT ON content_blocks BEGIN
            INSERT OR REPLACE INTO messages_fts(rowid, message_id, session_id, text)
            SELECT
                msg.rowid,
                new.message_id,
                msg.session_id,
                -- Preserve existing FTS text (if any), falling back to
                -- message body text, falling back to empty string. Then
                -- append the new block's text, tool_input, and metadata
                -- separated by newlines.
                COALESCE(
                    (SELECT mf.text FROM messages_fts AS mf
                     WHERE mf.rowid = msg.rowid),
                    CASE WHEN msg.text IS NOT NULL AND msg.text != ''
                         THEN msg.text ELSE '' END
                )
                || CASE WHEN new.text IS NOT NULL AND new.text != ''
                      THEN char(10) || new.text ELSE '' END
                || CASE WHEN new.tool_input IS NOT NULL AND new.tool_input != ''
                      THEN char(10) || new.tool_input ELSE '' END
                || CASE WHEN new.metadata IS NOT NULL AND new.metadata != ''
                      THEN char(10) || new.metadata ELSE '' END
            FROM messages AS msg
            WHERE msg.message_id = new.message_id;
        END;

        CREATE TRIGGER IF NOT EXISTS content_blocks_fts_ad
        AFTER DELETE ON content_blocks BEGIN
            DELETE FROM messages_fts
            WHERE rowid = (SELECT rowid FROM messages WHERE message_id = old.message_id);
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT m.rowid, m.message_id, m.session_id, group_concat(source.part, char(10))
            FROM messages AS m
            JOIN (
                SELECT m2.message_id, m2.text AS part, -1 AS block_index, 0 AS part_index
                FROM messages AS m2
                WHERE m2.message_id = old.message_id AND m2.text IS NOT NULL AND m2.text != ''
                UNION ALL
                SELECT cb.message_id, cb.text AS part, cb.block_index, 1 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = old.message_id AND cb.text IS NOT NULL AND cb.text != ''
                UNION ALL
                SELECT cb.message_id, cb.tool_input AS part, cb.block_index, 2 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = old.message_id AND cb.tool_input IS NOT NULL AND cb.tool_input != ''
                UNION ALL
                SELECT cb.message_id, cb.metadata AS part, cb.block_index, 3 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = old.message_id AND cb.metadata IS NOT NULL AND cb.metadata != ''
                ORDER BY message_id, block_index, part_index
            ) AS source ON source.message_id = m.message_id
            WHERE m.message_id = old.message_id
            HAVING group_concat(source.part, char(10)) IS NOT NULL;
        END;

        CREATE TRIGGER IF NOT EXISTS content_blocks_fts_au
        AFTER UPDATE ON content_blocks BEGIN
            DELETE FROM messages_fts
            WHERE rowid = (SELECT rowid FROM messages WHERE message_id = new.message_id);
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT m.rowid, m.message_id, m.session_id, group_concat(source.part, char(10))
            FROM messages AS m
            JOIN (
                SELECT m2.message_id, m2.text AS part, -1 AS block_index, 0 AS part_index
                FROM messages AS m2
                WHERE m2.message_id = new.message_id AND m2.text IS NOT NULL AND m2.text != ''
                UNION ALL
                SELECT cb.message_id, cb.text AS part, cb.block_index, 1 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.text IS NOT NULL AND cb.text != ''
                UNION ALL
                SELECT cb.message_id, cb.tool_input AS part, cb.block_index, 2 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.tool_input IS NOT NULL AND cb.tool_input != ''
                UNION ALL
                SELECT cb.message_id, cb.metadata AS part, cb.block_index, 3 AS part_index
                FROM content_blocks AS cb
                WHERE cb.message_id = new.message_id AND cb.metadata IS NOT NULL AND cb.metadata != ''
                ORDER BY message_id, block_index, part_index
            ) AS source ON source.message_id = m.message_id
            WHERE m.message_id = new.message_id
            HAVING group_concat(source.part, char(10)) IS NOT NULL;
        END;
"""


# ---------------------------------------------------------------------------
# Tags many-to-many (replaces JSON metadata['tags'] field)
# Schema bumps update the canonical DDL directly; historical tag JSON is
# re-derived from source/user tiers rather than patched in place.
# ---------------------------------------------------------------------------

TAGS_M2M_DDL = """
        CREATE TABLE IF NOT EXISTS tags (
            id   INTEGER PRIMARY KEY,
            name TEXT    UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS session_tags (
            session_id TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            tag_id          INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (session_id, tag_id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_tags_tag
            ON session_tags(tag_id);
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
            completed_at INTEGER,
            evidence     TEXT
        );
"""

# ---------------------------------------------------------------------------
# User state — target-aware marks and annotations
# ---------------------------------------------------------------------------

# NOTE (#1114): ``user_marks`` and ``user_annotations`` survive the session
# row's lifecycle.  ``identity_key`` is the stable surface identifier
# (``session:{cid}`` or ``message:{cid}:{mid}``) used to repoint the
# resolved ``session_id``/``message_id`` columns when a logically identical
# session is re-imported after delete/reset.  Session row FKs use
# ``ON DELETE SET NULL`` so marks/annotations persist across hard delete and are
# rebound by ``repoint_user_state_by_identity`` once the session reappears.
USER_MARKS_DDL = """
        CREATE TABLE IF NOT EXISTS user_marks (
            target_type     TEXT NOT NULL CHECK (target_type IN (
                'session', 'message', 'session', 'work_event',
                'thread', 'content_block', 'attachment', 'paste_span'
            )),
            target_id       TEXT NOT NULL,
            identity_key    TEXT NOT NULL,
            session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
            message_id      TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            mark_type       TEXT NOT NULL CHECK (mark_type IN ('star', 'pin', 'archive')),
            created_at      TEXT NOT NULL,
            PRIMARY KEY (target_type, target_id, mark_type)
        );

        CREATE INDEX IF NOT EXISTS idx_user_marks_type
            ON user_marks(mark_type);

        CREATE INDEX IF NOT EXISTS idx_user_marks_session
            ON user_marks(session_id);

        CREATE INDEX IF NOT EXISTS idx_user_marks_message
            ON user_marks(message_id);

        CREATE INDEX IF NOT EXISTS idx_user_marks_identity_key
            ON user_marks(identity_key);
"""

USER_ANNOTATIONS_DDL = """
        CREATE TABLE IF NOT EXISTS user_annotations (
            annotation_id   TEXT PRIMARY KEY,
            target_type     TEXT NOT NULL CHECK (target_type IN (
                'session', 'message', 'session', 'work_event',
                'thread', 'content_block', 'attachment', 'paste_span'
            )),
            target_id       TEXT NOT NULL,
            identity_key    TEXT NOT NULL,
            session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
            message_id      TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            note_text       TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_user_annotations_target
            ON user_annotations(target_type, target_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_session
            ON user_annotations(session_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_message
            ON user_annotations(message_id);

        CREATE INDEX IF NOT EXISTS idx_user_annotations_identity_key
            ON user_annotations(identity_key);
"""

# ---------------------------------------------------------------------------
# Saved views — named query presets storing SessionQuerySpec as JSON
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
# Recall packs — saved session selections with context-pack payloads
# ---------------------------------------------------------------------------

RECALL_PACKS_DDL = """
        CREATE TABLE IF NOT EXISTS recall_packs (
            pack_id               TEXT PRIMARY KEY,
            label                 TEXT NOT NULL,
            session_ids_json TEXT NOT NULL DEFAULT '[]',
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
# keyed by ``(session_id, insight_kind)`` and is consulted by insight
# materialization paths after they compute their base suggestion. Applying or
# removing a correction never touches the session's ``content_hash`` — the
# archive's idempotency invariant is preserved (AC #1131).
#
# Schema:
#   correction_id   surrogate identifier (uuid4 hex, stable across rebuilds).
#   session_id target session.
#   insight_kind    one of the closed enum values in
#                   ``polylogue.insights.feedback.CorrectionKind`` (the
#                   application layer enforces the closed set; storage is
#                   permissive to keep the table compatible across versions).
#   payload_json    correction-specific data (override label, accept/reject
#                   verdict, replacement summary text, etc.). Always JSON.
#   note            optional free-form reason text from the user.
#   created_at      ISO-8601 timestamp of the latest write (upsert refreshes).
#
# The ``(session_id, insight_kind)`` uniqueness contract is what makes
# rebuilds deterministic: at most one correction of each kind per session,
# and it always wins over the heuristic suggestion.

USER_CORRECTIONS_DDL = """
        CREATE TABLE IF NOT EXISTS user_corrections (
            correction_id   TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            insight_kind    TEXT NOT NULL,
            payload_json    TEXT NOT NULL,
            note            TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE (session_id, insight_kind)
        );

        CREATE INDEX IF NOT EXISTS idx_user_corrections_session
            ON user_corrections(session_id);

        CREATE INDEX IF NOT EXISTS idx_user_corrections_kind
            ON user_corrections(insight_kind);
"""

# ---------------------------------------------------------------------------
# topology_edges (#1258 / #866 slice A)
#
# Persists every parent reference emitted by a parser as a typed row, including
# references whose parent has not yet been ingested (out-of-order ingestion) or
# never will be. The pre-existing fast path
# (``sessions.parent_session_id``) is preserved unchanged; this table
# is the durable record that always carries the original provider-native parent
# id alongside the edge kind.
#
# Columns
#   edge_id                       Synthetic primary key.
#   src_session_id           The child session that asserted the
#                                 parent reference. FK CASCADE: deleting the
#                                 child drops the edge.
#   dst_provider_native_id        The parent identifier as emitted by the
#                                 provider (provider's archive session id,
#                                 not a polylogue session_id). Required.
#   dst_provider_name             The provider name on which the resolver
#                                 should look up dst_provider_native_id.
#   edge_type                     Closed enum, see polylogue/archive/topology/
#                                 edge.py::TopologyEdgeType.
#   resolved_dst_session_id  When the parent has been ingested, the
#                                 polylogue session_id. FK SET NULL so a
#                                 parent hard-delete demotes the edge to
#                                 unresolved rather than dropping it.
#   raw_evidence                  JSON blob of parsing-time evidence
#                                 (record uuid, sidechain/subagent flag, ...).
#   confidence                    [0, 1]. Slice A always writes 1.0.
#   status                        Closed enum: unresolved | resolved | repaired.
#   observed_at                   ISO-8601 timestamp of the edge's first ingest.
#   resolved_at                   ISO-8601 timestamp set on the unresolved →
#                                 resolved transition.
#
# Uniqueness: (src_session_id, dst_provider_native_id, edge_type).
# Re-ingesting the same child twice produces exactly one row.

TOPOLOGY_EDGES_DDL = """
        CREATE TABLE IF NOT EXISTS topology_edges (
            edge_id                      TEXT PRIMARY KEY,
            src_session_id          TEXT NOT NULL
                REFERENCES sessions(session_id) ON DELETE CASCADE,
            dst_provider_native_id       TEXT NOT NULL,
            dst_provider_name            TEXT NOT NULL,
            edge_type                    TEXT NOT NULL
                CHECK (edge_type IN (
                    'continuation', 'sidechain', 'subagent',
                    'branch', 'fork', 'resume', 'repaired'
                )),
            resolved_dst_session_id TEXT
                REFERENCES sessions(session_id) ON DELETE SET NULL,
            raw_evidence                 TEXT,
            confidence                   REAL NOT NULL DEFAULT 1.0,
            status                       TEXT NOT NULL
                CHECK (status IN ('unresolved', 'resolved', 'repaired', 'quarantined')),
            observed_at                  TEXT NOT NULL,
            resolved_at                  TEXT,
            UNIQUE (src_session_id, dst_provider_native_id, edge_type)
        );

        CREATE INDEX IF NOT EXISTS idx_topology_edges_resolver
            ON topology_edges(status, dst_provider_name, dst_provider_native_id);

        CREATE INDEX IF NOT EXISTS idx_topology_edges_resolved_dst
            ON topology_edges(resolved_dst_session_id)
            WHERE resolved_dst_session_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_topology_edges_src
            ON topology_edges(src_session_id);
"""

OTLP_SPANS_DDL = """
        CREATE TABLE IF NOT EXISTS otlp_spans (
            span_id         TEXT PRIMARY KEY,
            trace_id        TEXT NOT NULL,
            parent_span_id  TEXT,
            agent_id        TEXT,
            parent_agent_id TEXT,
            session_id      TEXT,
            operation_name  TEXT NOT NULL,
            start_time_unix_ns INTEGER NOT NULL,
            end_time_unix_ns   INTEGER NOT NULL,
            duration_ms     INTEGER NOT NULL,
            status_code     INTEGER NOT NULL DEFAULT 0,
            status_message  TEXT,
            attributes_json TEXT NOT NULL DEFAULT '{}',
            ingested_at     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_otlp_spans_trace
            ON otlp_spans(trace_id);

        CREATE INDEX IF NOT EXISTS idx_otlp_spans_session
            ON otlp_spans(session_id)
            WHERE session_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_otlp_spans_agent
            ON otlp_spans(agent_id)
            WHERE agent_id IS NOT NULL;
"""

BLACKBOARD_NOTES_DDL = """
        CREATE TABLE IF NOT EXISTS blackboard_notes (
            note_id         TEXT PRIMARY KEY,
            kind            TEXT NOT NULL CHECK (kind IN
                ('finding', 'blocker', 'decision', 'handoff', 'question', 'observation')),
            title           TEXT NOT NULL,
            content         TEXT NOT NULL DEFAULT '',
            scope_repo      TEXT,
            scope_session   TEXT,
            scope_issue     INTEGER,
            scope_path      TEXT,
            related_session_ids_json TEXT DEFAULT '[]',
            created_at       TEXT NOT NULL,
            resolved_at      TEXT,
            resolution      TEXT,
            materialized_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_blackboard_kind
            ON blackboard_notes(kind);

        CREATE INDEX IF NOT EXISTS idx_blackboard_scope_repo
            ON blackboard_notes(scope_repo)
            WHERE scope_repo IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_blackboard_resolved
            ON blackboard_notes(resolved_at)
            WHERE resolved_at IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_blackboard_created
            ON blackboard_notes(created_at);
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
    "TOPOLOGY_EDGES_DDL",
    "USER_ANNOTATIONS_DDL",
    "USER_CORRECTIONS_DDL",
    "USER_MARKS_DDL",
]
