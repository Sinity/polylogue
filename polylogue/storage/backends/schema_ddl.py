"""Schema DDL declarations grouped by storage domain bands."""

from __future__ import annotations

SCHEMA_VERSION = 1


_VEC0_DDL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id TEXT PRIMARY KEY,
        embedding float[1024],
        +provider_name TEXT,
        +conversation_id TEXT
    )
"""


_ARTIFACT_OBSERVATION_DDL = """
        CREATE TABLE IF NOT EXISTS artifact_observations (
            observation_id TEXT PRIMARY KEY,
            raw_id TEXT NOT NULL REFERENCES raw_conversations(raw_id) ON DELETE CASCADE,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            file_mtime TEXT,
            wire_format TEXT,
            artifact_kind TEXT NOT NULL,
            classification_reason TEXT NOT NULL,
            parse_as_conversation INTEGER NOT NULL DEFAULT 0,
            schema_eligible INTEGER NOT NULL DEFAULT 0,
            support_status TEXT NOT NULL
                CHECK (support_status IN (
                    'supported_parseable',
                    'recognized_unparsed',
                    'unsupported_parseable',
                    'decode_failed',
                    'unknown'
                )),
            malformed_jsonl_lines INTEGER NOT NULL DEFAULT 0,
            decode_error TEXT,
            bundle_scope TEXT,
            cohort_id TEXT,
            resolved_package_version TEXT,
            resolved_element_kind TEXT,
            resolution_reason TEXT,
            link_group_key TEXT,
            sidecar_agent_type TEXT,
            first_observed_at TEXT NOT NULL,
            last_observed_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_obs_source_identity
        ON artifact_observations(
            COALESCE(source_name, ''),
            source_path,
            COALESCE(source_index, -1)
        );

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_raw_id
        ON artifact_observations(raw_id);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_provider_status
        ON artifact_observations(payload_provider, support_status);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_kind_status
        ON artifact_observations(artifact_kind, support_status);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_cohort
        ON artifact_observations(cohort_id)
        WHERE cohort_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_link_group
        ON artifact_observations(link_group_key)
        WHERE link_group_key IS NOT NULL;
"""


_PUBLICATION_DDL = """
        CREATE TABLE IF NOT EXISTS publications (
            publication_id TEXT PRIMARY KEY,
            publication_kind TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            duration_ms INTEGER,
            manifest_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_publications_kind_generated
        ON publications(publication_kind, generated_at DESC);
"""


_MAINTENANCE_RUN_DDL = """
        CREATE TABLE IF NOT EXISTS maintenance_runs (
            maintenance_run_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL DEFAULT 1,
            executed_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            preview INTEGER NOT NULL DEFAULT 0,
            repair_selected INTEGER NOT NULL DEFAULT 0,
            cleanup_selected INTEGER NOT NULL DEFAULT 0,
            vacuum_requested INTEGER NOT NULL DEFAULT 0,
            target_names_json TEXT,
            success INTEGER NOT NULL DEFAULT 1,
            manifest_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_maintenance_runs_executed
        ON maintenance_runs(executed_at DESC);
"""


_ACTION_EVENT_DDL = """
        CREATE TABLE IF NOT EXISTS action_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            source_block_id TEXT,
            timestamp TEXT,
            sort_key REAL,
            sequence_index INTEGER NOT NULL,
            provider_name TEXT,
            action_kind TEXT NOT NULL,
            tool_name TEXT,
            normalized_tool_name TEXT NOT NULL,
            tool_id TEXT,
            affected_paths_json TEXT,
            cwd_path TEXT,
            branch_names_json TEXT,
            command TEXT,
            query_text TEXT,
            url TEXT,
            output_text TEXT,
            search_text TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_action_events_conversation
        ON action_events(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_action_events_message
        ON action_events(message_id);

        CREATE INDEX IF NOT EXISTS idx_action_events_kind
        ON action_events(action_kind);

        CREATE INDEX IF NOT EXISTS idx_action_events_tool
        ON action_events(normalized_tool_name);

        CREATE INDEX IF NOT EXISTS idx_action_events_sort
        ON action_events(conversation_id, sort_key, sequence_index);
"""


_ACTION_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS action_events_fts USING fts5(
            event_id UNINDEXED,
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            action_kind UNINDEXED,
            tool_name UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS action_events_fts_ai
        AFTER INSERT ON action_events BEGIN
            INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
            VALUES (
                new.event_id,
                new.message_id,
                new.conversation_id,
                new.action_kind,
                new.normalized_tool_name,
                new.search_text
            );
        END;

        CREATE TRIGGER IF NOT EXISTS action_events_fts_ad
        AFTER DELETE ON action_events BEGIN
            DELETE FROM action_events_fts WHERE event_id = old.event_id;
        END;

        CREATE TRIGGER IF NOT EXISTS action_events_fts_au
        AFTER UPDATE ON action_events BEGIN
            DELETE FROM action_events_fts WHERE event_id = old.event_id;
            INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
            VALUES (
                new.event_id,
                new.message_id,
                new.conversation_id,
                new.action_kind,
                new.normalized_tool_name,
                new.search_text
            );
        END;
"""


_SESSION_PRODUCT_DDL = """
        CREATE TABLE IF NOT EXISTS session_profiles (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            provider_name TEXT NOT NULL,
            title TEXT,
            first_message_at TEXT,
            last_message_at TEXT,
            canonical_session_date TEXT,
            primary_work_kind TEXT,
            repo_paths_json TEXT,
            canonical_projects_json TEXT,
            tags_json TEXT,
            auto_tags_json TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            work_event_count INTEGER NOT NULL DEFAULT 0,
            phase_count INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER NOT NULL DEFAULT 0,
            tool_use_count INTEGER NOT NULL DEFAULT 0,
            thinking_count INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            total_duration_ms INTEGER NOT NULL DEFAULT 0,
            engaged_duration_ms INTEGER NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            search_text TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_session_profiles_provider
        ON session_profiles(provider_name);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_sort
        ON session_profiles(source_sort_key DESC);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_first_message
        ON session_profiles(first_message_at DESC);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_canonical_date
        ON session_profiles(canonical_session_date DESC);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_work_kind
        ON session_profiles(primary_work_kind)
        WHERE primary_work_kind IS NOT NULL;

        CREATE VIRTUAL TABLE IF NOT EXISTS session_profiles_fts USING fts5(
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_profiles_fts_ai
        AFTER INSERT ON session_profiles BEGIN
            INSERT INTO session_profiles_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.search_text);
        END;

        CREATE TRIGGER IF NOT EXISTS session_profiles_fts_ad
        AFTER DELETE ON session_profiles BEGIN
            DELETE FROM session_profiles_fts WHERE conversation_id = old.conversation_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_profiles_fts_au
        AFTER UPDATE ON session_profiles BEGIN
            DELETE FROM session_profiles_fts WHERE conversation_id = old.conversation_id;
            INSERT INTO session_profiles_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.search_text);
        END;

        CREATE TABLE IF NOT EXISTS session_work_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            provider_name TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            kind TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            start_index INTEGER NOT NULL DEFAULT 0,
            end_index INTEGER NOT NULL DEFAULT 0,
            start_time TEXT,
            end_time TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            canonical_session_date TEXT,
            summary TEXT NOT NULL,
            file_paths_json TEXT,
            tools_used_json TEXT,
            payload_json TEXT NOT NULL,
            search_text TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_session_work_events_conversation
        ON session_work_events(conversation_id, event_index);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_kind
        ON session_work_events(kind);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_provider
        ON session_work_events(provider_name);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_time
        ON session_work_events(start_time DESC, end_time DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS session_work_events_fts USING fts5(
            event_id UNINDEXED,
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            kind UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
        AFTER INSERT ON session_work_events BEGIN
            INSERT INTO session_work_events_fts (event_id, conversation_id, provider_name, kind, text)
            VALUES (new.event_id, new.conversation_id, new.provider_name, new.kind, new.search_text);
        END;

        CREATE TABLE IF NOT EXISTS session_phases (
            phase_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            provider_name TEXT NOT NULL,
            phase_index INTEGER NOT NULL,
            kind TEXT NOT NULL,
            start_index INTEGER NOT NULL DEFAULT 0,
            end_index INTEGER NOT NULL DEFAULT 0,
            start_time TEXT,
            end_time TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            canonical_session_date TEXT,
            tool_counts_json TEXT NOT NULL DEFAULT '{}',
            word_count INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            search_text TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_session_phases_conversation
        ON session_phases(conversation_id, phase_index);

        CREATE INDEX IF NOT EXISTS idx_session_phases_kind
        ON session_phases(kind);

        CREATE INDEX IF NOT EXISTS idx_session_phases_provider
        ON session_phases(provider_name);

        CREATE INDEX IF NOT EXISTS idx_session_phases_time
        ON session_phases(start_time DESC, end_time DESC);

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
        AFTER DELETE ON session_work_events BEGIN
            DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
        AFTER UPDATE ON session_work_events BEGIN
            DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
            INSERT INTO session_work_events_fts (event_id, conversation_id, provider_name, kind, text)
            VALUES (new.event_id, new.conversation_id, new.provider_name, new.kind, new.search_text);
        END;

        CREATE TABLE IF NOT EXISTS work_threads (
            thread_id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            dominant_project TEXT,
            session_ids_json TEXT,
            session_count INTEGER NOT NULL DEFAULT 0,
            depth INTEGER NOT NULL DEFAULT 0,
            branch_count INTEGER NOT NULL DEFAULT 0,
            total_messages INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            work_event_breakdown_json TEXT,
            payload_json TEXT NOT NULL,
            search_text TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_work_threads_root
        ON work_threads(root_id);

        CREATE INDEX IF NOT EXISTS idx_work_threads_time
        ON work_threads(end_time DESC, start_time DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS work_threads_fts USING fts5(
            thread_id UNINDEXED,
            root_id UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS work_threads_fts_ai
        AFTER INSERT ON work_threads BEGIN
            INSERT INTO work_threads_fts (thread_id, root_id, text)
            VALUES (new.thread_id, new.root_id, new.search_text);
        END;

        CREATE TRIGGER IF NOT EXISTS work_threads_fts_ad
        AFTER DELETE ON work_threads BEGIN
            DELETE FROM work_threads_fts WHERE thread_id = old.thread_id;
        END;

        CREATE TRIGGER IF NOT EXISTS work_threads_fts_au
        AFTER UPDATE ON work_threads BEGIN
            DELETE FROM work_threads_fts WHERE thread_id = old.thread_id;
            INSERT INTO work_threads_fts (thread_id, root_id, text)
            VALUES (new.thread_id, new.root_id, new.search_text);
        END;

        CREATE TABLE IF NOT EXISTS session_tag_rollups (
            tag TEXT NOT NULL,
            bucket_day TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            conversation_count INTEGER NOT NULL DEFAULT 0,
            explicit_count INTEGER NOT NULL DEFAULT 0,
            auto_count INTEGER NOT NULL DEFAULT 0,
            project_breakdown_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            PRIMARY KEY (tag, bucket_day, provider_name)
        );

        CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_day
        ON session_tag_rollups(bucket_day DESC, provider_name, tag);

        CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_provider
        ON session_tag_rollups(provider_name, tag);

        CREATE TABLE IF NOT EXISTS day_session_summaries (
            day TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            conversation_count INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            total_duration_ms INTEGER NOT NULL DEFAULT 0,
            total_wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            total_messages INTEGER NOT NULL DEFAULT 0,
            total_words INTEGER NOT NULL DEFAULT 0,
            work_event_breakdown_json TEXT NOT NULL DEFAULT '{}',
            projects_active_json TEXT,
            payload_json TEXT NOT NULL,
            search_text TEXT NOT NULL,
            PRIMARY KEY (day, provider_name)
        );

        CREATE INDEX IF NOT EXISTS idx_day_session_summaries_day
        ON day_session_summaries(day DESC, provider_name);
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
