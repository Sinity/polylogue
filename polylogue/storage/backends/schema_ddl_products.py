"""Session-product schema DDL fragments."""

from __future__ import annotations

SESSION_PRODUCT_DDL = """
        CREATE TABLE IF NOT EXISTS session_profiles (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 3,
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
            substantive_count INTEGER NOT NULL DEFAULT 0,
            attachment_count INTEGER NOT NULL DEFAULT 0,
            work_event_count INTEGER NOT NULL DEFAULT 0,
            phase_count INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER NOT NULL DEFAULT 0,
            tool_use_count INTEGER NOT NULL DEFAULT 0,
            thinking_count INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            total_duration_ms INTEGER NOT NULL DEFAULT 0,
            engaged_duration_ms INTEGER NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            cost_is_estimated INTEGER NOT NULL DEFAULT 0,
            evidence_payload_json TEXT NOT NULL DEFAULT '{}',
            inference_payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            evidence_search_text TEXT NOT NULL DEFAULT '',
            inference_search_text TEXT NOT NULL DEFAULT '',
            enrichment_payload_json TEXT NOT NULL DEFAULT '{}',
            enrichment_search_text TEXT NOT NULL DEFAULT '',
            enrichment_version INTEGER NOT NULL DEFAULT 1,
            enrichment_family TEXT NOT NULL DEFAULT 'scored_session_enrichment',
            inference_version INTEGER NOT NULL DEFAULT 1,
            inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'
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

        CREATE VIRTUAL TABLE IF NOT EXISTS session_profile_evidence_fts USING fts5(
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_profile_evidence_fts_ai
        AFTER INSERT ON session_profiles BEGIN
            INSERT INTO session_profile_evidence_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.evidence_search_text);
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_evidence_fts_ad
        AFTER DELETE ON session_profiles BEGIN
            DELETE FROM session_profile_evidence_fts WHERE conversation_id = old.conversation_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_evidence_fts_au
        AFTER UPDATE ON session_profiles BEGIN
            DELETE FROM session_profile_evidence_fts WHERE conversation_id = old.conversation_id;
            INSERT INTO session_profile_evidence_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.evidence_search_text);
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS session_profile_inference_fts USING fts5(
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_profile_inference_fts_ai
        AFTER INSERT ON session_profiles BEGIN
            INSERT INTO session_profile_inference_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.inference_search_text);
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_inference_fts_ad
        AFTER DELETE ON session_profiles BEGIN
            DELETE FROM session_profile_inference_fts WHERE conversation_id = old.conversation_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_inference_fts_au
        AFTER UPDATE ON session_profiles BEGIN
            DELETE FROM session_profile_inference_fts WHERE conversation_id = old.conversation_id;
            INSERT INTO session_profile_inference_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.inference_search_text);
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS session_profile_enrichment_fts USING fts5(
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_profile_enrichment_fts_ai
        AFTER INSERT ON session_profiles BEGIN
            INSERT INTO session_profile_enrichment_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.enrichment_search_text);
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_enrichment_fts_ad
        AFTER DELETE ON session_profiles BEGIN
            DELETE FROM session_profile_enrichment_fts WHERE conversation_id = old.conversation_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_profile_enrichment_fts_au
        AFTER UPDATE ON session_profiles BEGIN
            DELETE FROM session_profile_enrichment_fts WHERE conversation_id = old.conversation_id;
            INSERT INTO session_profile_enrichment_fts (conversation_id, provider_name, text)
            VALUES (new.conversation_id, new.provider_name, new.enrichment_search_text);
        END;

        CREATE TABLE IF NOT EXISTS session_work_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 3,
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
            evidence_payload_json TEXT NOT NULL DEFAULT '{}',
            inference_payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            inference_version INTEGER NOT NULL DEFAULT 1,
            inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'
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
            materializer_version INTEGER NOT NULL DEFAULT 3,
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
            confidence REAL NOT NULL DEFAULT 0,
            evidence_reasons_json TEXT NOT NULL DEFAULT '[]',
            tool_counts_json TEXT NOT NULL DEFAULT '{}',
            word_count INTEGER NOT NULL DEFAULT 0,
            evidence_payload_json TEXT NOT NULL DEFAULT '{}',
            inference_payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            inference_version INTEGER NOT NULL DEFAULT 1,
            inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'
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


__all__ = ["SESSION_PRODUCT_DDL"]
