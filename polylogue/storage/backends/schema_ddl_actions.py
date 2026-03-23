"""Action-event schema DDL fragments."""

from __future__ import annotations

ACTION_EVENT_DDL = """
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


ACTION_FTS_DDL = """
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


__all__ = ["ACTION_EVENT_DDL", "ACTION_FTS_DDL"]
