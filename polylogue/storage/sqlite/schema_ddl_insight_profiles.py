"""DDL fragments for profile and timeline session insights."""

from __future__ import annotations

from polylogue.storage.sqlite.schema_ddl_insight_common import (
    MATERIALIZATION_COLUMNS_SQL,
)

SESSION_INSIGHT_PROFILE_DDL = (
    """
        CREATE TABLE IF NOT EXISTS session_profiles (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
            logical_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,"""
    + MATERIALIZATION_COLUMNS_SQL
    + """
            source_name TEXT NOT NULL,
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
            tool_active_duration_ms INTEGER NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            workflow_shape TEXT NOT NULL DEFAULT 'unknown',
            workflow_shape_confidence REAL NOT NULL DEFAULT 0.0,
            workflow_shape_features_json TEXT NOT NULL DEFAULT '{}',
            terminal_state TEXT NOT NULL DEFAULT 'unknown',
            terminal_state_confidence REAL NOT NULL DEFAULT 0.0,
            terminal_state_evidence_json TEXT NOT NULL DEFAULT '{}',
            cost_is_estimated INTEGER NOT NULL DEFAULT 0,
            thinking_duration_ms INTEGER NOT NULL DEFAULT 0,
            output_duration_ms INTEGER NOT NULL DEFAULT 0,
            tool_duration_ms INTEGER NOT NULL DEFAULT 0,
            latency_percentiles_ms_json TEXT NOT NULL DEFAULT '{}',
            tool_calls_per_minute REAL NOT NULL DEFAULT 0.0,
            timing_provenance TEXT NOT NULL DEFAULT 'sort_key_estimated',
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
            inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics',
            total_input_tokens INTEGER NOT NULL DEFAULT 0,
            total_output_tokens INTEGER NOT NULL DEFAULT 0,
            total_cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            total_cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            total_credit_cost REAL NOT NULL DEFAULT 0.0,
            cost_provenance TEXT NOT NULL DEFAULT 'unknown',
            per_model_cost_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_session_profiles_provider
        ON session_profiles(source_name);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_logical_session
        ON session_profiles(logical_session_id);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_sort
        ON session_profiles(source_sort_key DESC);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_first_message
        ON session_profiles(first_message_at DESC);

        CREATE INDEX IF NOT EXISTS idx_session_profiles_canonical_date
        ON session_profiles(canonical_session_date DESC);
"""
)
