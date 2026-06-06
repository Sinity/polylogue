"""DDL fragments for timeline-oriented session insights."""

from __future__ import annotations

from polylogue.storage.sqlite.schema_ddl_insight_common import (
    MATERIALIZATION_COLUMNS_SQL,
)

SESSION_INSIGHT_TIMELINE_DDL = (
    """
        CREATE TABLE IF NOT EXISTS session_work_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,"""
    + MATERIALIZATION_COLUMNS_SQL
    + """
            source_name TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            heuristic_label TEXT NOT NULL,
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

        CREATE INDEX IF NOT EXISTS idx_session_work_events_session
        ON session_work_events(session_id, event_index);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_heuristic_label
        ON session_work_events(heuristic_label);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_provider
        ON session_work_events(source_name);

        CREATE INDEX IF NOT EXISTS idx_session_work_events_time
        ON session_work_events(start_time DESC, end_time DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS session_work_events_fts USING fts5(
            event_id UNINDEXED,
            session_id UNINDEXED,
            source_name UNINDEXED,
            heuristic_label UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
        AFTER INSERT ON session_work_events BEGIN
            INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
            VALUES (new.event_id, new.session_id, new.source_name, new.heuristic_label, new.search_text);
        END;

        CREATE TABLE IF NOT EXISTS session_phases (
            phase_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,"""
    + MATERIALIZATION_COLUMNS_SQL
    + """
            source_name TEXT NOT NULL,
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

        CREATE INDEX IF NOT EXISTS idx_session_phases_session
        ON session_phases(session_id, phase_index);

        CREATE INDEX IF NOT EXISTS idx_session_phases_kind
        ON session_phases(kind);

        CREATE INDEX IF NOT EXISTS idx_session_phases_provider
        ON session_phases(source_name);

        CREATE INDEX IF NOT EXISTS idx_session_phases_time
        ON session_phases(start_time DESC, end_time DESC);

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
        AFTER DELETE ON session_work_events BEGIN
            DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
        END;

        CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
        AFTER UPDATE ON session_work_events BEGIN
            DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
            INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
            VALUES (new.event_id, new.session_id, new.source_name, new.heuristic_label, new.search_text);
        END;
"""
)
