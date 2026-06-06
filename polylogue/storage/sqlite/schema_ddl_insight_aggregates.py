"""DDL fragments for aggregate and thread session insights."""

from __future__ import annotations

from polylogue.storage.sqlite.schema_ddl_insight_common import (
    MATERIALIZATION_COLUMNS_SQL,
    MATERIALIZATION_COLUMNS_SQL_NO_SORT_KEY,
)

SESSION_INSIGHT_AGGREGATE_DDL = (
    """
        CREATE TABLE IF NOT EXISTS work_threads (
            thread_id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,"""
    + MATERIALIZATION_COLUMNS_SQL_NO_SORT_KEY
    + """
            start_time TEXT,
            end_time TEXT,
            dominant_repo TEXT,
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
            source_name TEXT NOT NULL,"""
    + MATERIALIZATION_COLUMNS_SQL
    + """
            session_count INTEGER NOT NULL DEFAULT 0,
            logical_session_count INTEGER NOT NULL DEFAULT 0,
            logical_session_ids_json TEXT NOT NULL DEFAULT '[]',
            explicit_count INTEGER NOT NULL DEFAULT 0,
            auto_count INTEGER NOT NULL DEFAULT 0,
            repo_breakdown_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            PRIMARY KEY (tag, bucket_day, source_name)
        );

        CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_day
        ON session_tag_rollups(bucket_day DESC, source_name, tag);

        CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_provider
        ON session_tag_rollups(source_name, tag);

"""
)
