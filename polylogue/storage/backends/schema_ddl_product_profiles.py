"""DDL fragments for profile and timeline session products."""

from __future__ import annotations

SESSION_PRODUCT_PROFILE_DDL = """
        CREATE TABLE IF NOT EXISTS session_profiles (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 5,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            provider_name TEXT NOT NULL,
            title TEXT,
            first_message_at TEXT,
            last_message_at TEXT,
            canonical_session_date TEXT,
            repo_paths_json TEXT,
            repo_names_json TEXT,
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
"""
