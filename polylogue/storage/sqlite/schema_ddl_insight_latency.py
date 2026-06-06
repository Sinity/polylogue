"""DDL for session latency insight products."""

from __future__ import annotations

from polylogue.storage.sqlite.schema_ddl_insight_common import (
    MATERIALIZATION_COLUMNS_SQL,
)

SESSION_INSIGHT_LATENCY_DDL = (
    """
        CREATE TABLE IF NOT EXISTS session_latency_profiles (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,"""
    + MATERIALIZATION_COLUMNS_SQL
    + """
            source_name TEXT NOT NULL,
            title TEXT,
            first_message_at TEXT,
            last_message_at TEXT,
            canonical_session_date TEXT,
            median_tool_call_ms INTEGER NOT NULL DEFAULT 0,
            p90_tool_call_ms INTEGER NOT NULL DEFAULT 0,
            max_tool_call_ms INTEGER NOT NULL DEFAULT 0,
            stuck_tool_count INTEGER NOT NULL DEFAULT 0,
            median_agent_response_ms INTEGER NOT NULL DEFAULT 0,
            median_user_response_ms INTEGER NOT NULL DEFAULT 0,
            tool_call_count_by_category_json TEXT NOT NULL DEFAULT '{}',
            evidence_payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_provider
        ON session_latency_profiles(source_name);

        CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_date
        ON session_latency_profiles(canonical_session_date DESC);

        CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_stuck
        ON session_latency_profiles(stuck_tool_count DESC, canonical_session_date DESC);
"""
)

__all__ = ["SESSION_INSIGHT_LATENCY_DDL"]
