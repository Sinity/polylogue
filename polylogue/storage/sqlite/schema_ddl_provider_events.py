"""Provider-event archive DDL."""

from __future__ import annotations

PROVIDER_EVENT_DDL = """
        CREATE TABLE IF NOT EXISTS provider_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL
                REFERENCES sessions(session_id) ON DELETE CASCADE,
            source_name TEXT NOT NULL DEFAULT 'unknown',
            event_index INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            normalized_kind TEXT NOT NULL DEFAULT 'provider_native',
            timestamp TEXT,
            sort_key REAL,
            source_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            raw_id TEXT REFERENCES raw_sessions(raw_id) ON DELETE SET NULL,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE (session_id, event_index)
        );

        CREATE INDEX IF NOT EXISTS idx_provider_events_session
        ON provider_events(session_id);

        CREATE INDEX IF NOT EXISTS idx_provider_events_source_message
        ON provider_events(source_message_id)
        WHERE source_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_provider_events_raw_id
        ON provider_events(raw_id)
        WHERE raw_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_provider_events_type
        ON provider_events(event_type);

        CREATE INDEX IF NOT EXISTS idx_provider_events_session_type
        ON provider_events(session_id, event_type);

        CREATE INDEX IF NOT EXISTS idx_provider_events_session_sort
        ON provider_events(session_id, sort_key, event_index);

        CREATE TABLE IF NOT EXISTS provider_event_compactions (
            event_id TEXT PRIMARY KEY REFERENCES provider_events(event_id) ON DELETE CASCADE,
            summary TEXT NOT NULL DEFAULT '',
            trigger TEXT,
            pre_tokens INTEGER,
            preserved_segment_id TEXT,
            is_modern INTEGER NOT NULL DEFAULT 0,
            replacement_history_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS provider_event_turn_contexts (
            event_id TEXT PRIMARY KEY REFERENCES provider_events(event_id) ON DELETE CASCADE,
            cwd TEXT,
            model TEXT,
            effort TEXT,
            approval_policy TEXT,
            sandbox_policy TEXT,
            summary TEXT
        );

        CREATE TABLE IF NOT EXISTS provider_event_tool_calls (
            event_id TEXT PRIMARY KEY REFERENCES provider_events(event_id) ON DELETE CASCADE,
            call_id TEXT,
            tool_name TEXT,
            status TEXT,
            input_chars INTEGER NOT NULL DEFAULT 0,
            output_chars INTEGER NOT NULL DEFAULT 0,
            has_input_body INTEGER NOT NULL DEFAULT 0,
            has_output_body INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_provider_event_tool_calls_call_id
        ON provider_event_tool_calls(call_id)
        WHERE call_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS provider_event_reasoning (
            event_id TEXT PRIMARY KEY REFERENCES provider_events(event_id) ON DELETE CASCADE,
            summary TEXT,
            encrypted_content_hash TEXT,
            encrypted_content_bytes INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS provider_event_ghost_snapshots (
            event_id TEXT PRIMARY KEY REFERENCES provider_events(event_id) ON DELETE CASCADE,
            ghost_commit TEXT
        );
"""


__all__ = ["PROVIDER_EVENT_DDL"]
