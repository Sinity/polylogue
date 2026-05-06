"""Provider-event archive DDL."""

from __future__ import annotations

PROVIDER_EVENT_DDL = """
        CREATE TABLE IF NOT EXISTS provider_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            provider_name TEXT NOT NULL DEFAULT 'unknown',
            event_index INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT,
            sort_key REAL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            source_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
            raw_id TEXT REFERENCES raw_conversations(raw_id) ON DELETE SET NULL,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE (conversation_id, event_index)
        );

        CREATE INDEX IF NOT EXISTS idx_provider_events_conversation
        ON provider_events(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_provider_events_source_message
        ON provider_events(source_message_id)
        WHERE source_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_provider_events_type
        ON provider_events(event_type);

        CREATE INDEX IF NOT EXISTS idx_provider_events_conversation_type
        ON provider_events(conversation_id, event_type);

        CREATE INDEX IF NOT EXISTS idx_provider_events_conversation_sort
        ON provider_events(conversation_id, sort_key, event_index);
"""


__all__ = ["PROVIDER_EVENT_DDL"]
