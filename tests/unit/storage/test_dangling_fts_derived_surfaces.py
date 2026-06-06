"""Bounded FTS repair covers derived insight search surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.fts.dangling_repair import repair_missing_fts_rows, repair_stale_fts_rows
from polylogue.storage.fts.freshness import message_fts_recorded_state_sync
from polylogue.storage.fts.fts_lifecycle import message_fts_search_readiness_sync, restore_fts_triggers_sync


def test_repair_missing_fts_rows_marks_derived_surfaces_ready(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, session_id TEXT, text TEXT);
            CREATE VIRTUAL TABLE messages_fts USING fts5(message_id UNINDEXED, session_id UNINDEXED, text);
            CREATE TABLE action_events (event_id TEXT, message_id TEXT, session_id TEXT, action_kind TEXT,
                normalized_tool_name TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE action_events_fts USING fts5(event_id UNINDEXED, message_id UNINDEXED,
                session_id UNINDEXED, action_kind UNINDEXED, normalized_tool_name UNINDEXED, search_text);
            CREATE TABLE session_work_events (event_id TEXT PRIMARY KEY, session_id TEXT, source_name TEXT,
                heuristic_label TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE session_work_events_fts USING fts5(event_id UNINDEXED, session_id UNINDEXED,
                source_name UNINDEXED, heuristic_label UNINDEXED, text);
            CREATE TABLE work_threads (thread_id TEXT PRIMARY KEY, root_id TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE work_threads_fts USING fts5(thread_id UNINDEXED, root_id UNINDEXED, text);
            INSERT INTO session_work_events VALUES ('event-1', 'conv-1', 'codex', 'decision', 'ship it');
            INSERT INTO work_threads VALUES ('thread-1', 'conv-1', 'startup fts repair');
            """
        )
        restore_fts_triggers_sync(conn)
        outcome = repair_missing_fts_rows(conn)
        states = dict(conn.execute("SELECT surface, state FROM fts_freshness_state").fetchall())
        work_events = conn.execute("SELECT COUNT(*) FROM session_work_events_fts_docsize").fetchone()[0]
        threads = conn.execute("SELECT COUNT(*) FROM work_threads_fts_docsize").fetchone()[0]
    finally:
        conn.close()

    assert outcome.success is True
    assert work_events == 1
    assert threads == 1
    assert states["session_work_events_fts"] == "ready"
    assert states["work_threads_fts"] == "ready"


def test_repair_stale_fts_rows_skips_ready_archive_surfaces(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, session_id TEXT, text TEXT);
            CREATE VIRTUAL TABLE messages_fts USING fts5(message_id UNINDEXED, session_id UNINDEXED, text);
            CREATE TABLE action_events (event_id TEXT, message_id TEXT, session_id TEXT, action_kind TEXT,
                normalized_tool_name TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE action_events_fts USING fts5(event_id UNINDEXED, message_id UNINDEXED,
                session_id UNINDEXED, action_kind UNINDEXED, normalized_tool_name UNINDEXED, search_text);
            CREATE TABLE session_work_events (event_id TEXT PRIMARY KEY, session_id TEXT, source_name TEXT,
                heuristic_label TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE session_work_events_fts USING fts5(event_id UNINDEXED, session_id UNINDEXED,
                source_name UNINDEXED, heuristic_label UNINDEXED, text);
            CREATE TABLE work_threads (thread_id TEXT PRIMARY KEY, root_id TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE work_threads_fts USING fts5(thread_id UNINDEXED, root_id UNINDEXED, text);
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('action_events_fts', 'ready', 'now', 0, 0, NULL);
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('session_work_events_fts', 'stale', 'now', 0, 0, 'old');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('work_threads_fts', 'stale', 'now', 0, 0, 'old');
            INSERT INTO session_work_events VALUES ('event-1', 'conv-1', 'codex', 'decision', 'ship it');
            INSERT INTO work_threads VALUES ('thread-1', 'conv-1', 'startup fts repair');
            """
        )
        restore_fts_triggers_sync(conn)
        conn.execute(
            "UPDATE fts_freshness_state SET state='ready' WHERE surface IN ('messages_fts', 'action_events_fts')"
        )
        outcome = repair_stale_fts_rows(conn)
        states = dict(conn.execute("SELECT surface, state FROM fts_freshness_state").fetchall())
    finally:
        conn.close()

    assert outcome.success is True
    assert states == {
        "messages_fts": "ready",
        "action_events_fts": "ready",
        "session_work_events_fts": "ready",
        "work_threads_fts": "ready",
    }


def test_search_readiness_rejects_poisoned_zero_count_ready_marker(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, session_id TEXT, text TEXT);
            CREATE TABLE content_blocks (
                message_id TEXT, session_id TEXT, text TEXT, tool_input TEXT, metadata TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(message_id UNINDEXED, session_id UNINDEXED, text);
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO messages VALUES ('msg-1', 'conv-1', 'needle freshness');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            """
        )

        readiness = message_fts_search_readiness_sync(conn)
        state = message_fts_recorded_state_sync(conn)
        recorded = conn.execute(
            "SELECT source_rows, indexed_rows, detail FROM fts_freshness_state WHERE surface='messages_fts'"
        ).fetchone()
    finally:
        conn.close()

    assert readiness["ready"] is False
    assert readiness["total_rows"] == 1
    assert readiness["indexed_rows"] == 0
    assert state == "stale"
    assert recorded == (1, 0, "exact message readiness failed")


def test_repair_stale_fts_rows_recomputes_poisoned_archive_counts(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, session_id TEXT, text TEXT);
            CREATE VIRTUAL TABLE messages_fts USING fts5(message_id UNINDEXED, session_id UNINDEXED, text);
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO messages VALUES ('msg-1', 'conv-1', 'repair freshness');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            """
        )
        restore_fts_triggers_sync(conn)

        outcome = repair_stale_fts_rows(conn)
        messages = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows
            FROM fts_freshness_state
            WHERE surface='messages_fts'
            """
        ).fetchone()
        indexed = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
    finally:
        conn.close()

    assert outcome.success is True
    assert indexed == 1
    assert messages == ("ready", 1, 1, 0, 0, 0)
