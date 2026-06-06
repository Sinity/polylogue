"""Action-event schema DDL fragments."""

from __future__ import annotations

ACTION_EVENT_DDL = """
        CREATE TABLE IF NOT EXISTS action_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            source_block_id TEXT,
            timestamp TEXT,
            sort_key REAL,
            sequence_index INTEGER NOT NULL,
            source_name TEXT,
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

        CREATE INDEX IF NOT EXISTS idx_action_events_session
        ON action_events(session_id);

        CREATE INDEX IF NOT EXISTS idx_action_events_message
        ON action_events(message_id);

        CREATE INDEX IF NOT EXISTS idx_action_events_kind
        ON action_events(action_kind);

        CREATE INDEX IF NOT EXISTS idx_action_events_tool
        ON action_events(normalized_tool_name);

        CREATE INDEX IF NOT EXISTS idx_action_events_conv_kind
        ON action_events(session_id, action_kind);

        CREATE INDEX IF NOT EXISTS idx_action_events_conv_tool
        ON action_events(session_id, normalized_tool_name);

        CREATE INDEX IF NOT EXISTS idx_action_events_sort
        ON action_events(session_id, sort_key, sequence_index);
"""


ACTION_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS action_events_fts USING fts5(
            event_id UNINDEXED,
            message_id UNINDEXED,
            session_id UNINDEXED,
            action_kind UNINDEXED,
            normalized_tool_name UNINDEXED,
            search_text,
            content='action_events',
            content_rowid='rowid',
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS action_events_fts_ai
        AFTER INSERT ON action_events BEGIN
            INSERT INTO action_events_fts (
                rowid, event_id, message_id, session_id,
                action_kind, normalized_tool_name, search_text
            ) VALUES (
                new.rowid,
                new.event_id,
                new.message_id,
                new.session_id,
                new.action_kind,
                new.normalized_tool_name,
                new.search_text
            );
        END;

        CREATE TRIGGER IF NOT EXISTS action_events_fts_ad
        AFTER DELETE ON action_events BEGIN
            INSERT INTO action_events_fts (
                action_events_fts, rowid, event_id, message_id, session_id,
                action_kind, normalized_tool_name, search_text
            ) VALUES (
                'delete',
                old.rowid,
                old.event_id,
                old.message_id,
                old.session_id,
                old.action_kind,
                old.normalized_tool_name,
                old.search_text
            );
        END;

        CREATE TRIGGER IF NOT EXISTS action_events_fts_au
        AFTER UPDATE ON action_events BEGIN
            INSERT INTO action_events_fts (
                action_events_fts, rowid, event_id, message_id, session_id,
                action_kind, normalized_tool_name, search_text
            ) VALUES (
                'delete',
                old.rowid,
                old.event_id,
                old.message_id,
                old.session_id,
                old.action_kind,
                old.normalized_tool_name,
                old.search_text
            );
            INSERT INTO action_events_fts (
                rowid, event_id, message_id, session_id,
                action_kind, normalized_tool_name, search_text
            ) VALUES (
                new.rowid,
                new.event_id,
                new.message_id,
                new.session_id,
                new.action_kind,
                new.normalized_tool_name,
                new.search_text
            );
        END;
"""


__all__ = ["ACTION_EVENT_DDL", "ACTION_FTS_DDL"]
