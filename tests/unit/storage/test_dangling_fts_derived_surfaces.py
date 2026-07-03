"""Bounded FTS repair covers derived insight search surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.storage.fts.dangling_repair import (
    repair_missing_fts_rows,
    repair_stale_fts_rows,
    reset_and_repair_fts_rows,
)
from polylogue.storage.fts.freshness import (
    message_fts_recorded_state_async,
    message_fts_recorded_state_sync,
    record_fts_surface_state_async,
    record_fts_surface_state_sync,
)
from polylogue.storage.fts.fts_lifecycle import (
    message_fts_search_readiness_async,
    message_fts_search_readiness_sync,
    restore_fts_triggers_sync,
)


def test_repair_missing_fts_rows_marks_derived_surfaces_ready(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE session_work_events (event_id TEXT PRIMARY KEY, session_id TEXT,
                work_event_type TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE session_work_events_fts USING fts5(event_id UNINDEXED, session_id UNINDEXED,
                work_event_type UNINDEXED, text);
            CREATE TABLE threads (thread_id TEXT PRIMARY KEY, search_text TEXT);
            CREATE VIRTUAL TABLE threads_fts USING fts5(thread_id UNINDEXED, root_id UNINDEXED, text);
            INSERT INTO session_work_events VALUES ('event-1', 'conv-1', 'decision', 'ship it');
            INSERT INTO threads VALUES ('thread-1', 'startup fts repair');
            """
        )
        restore_fts_triggers_sync(conn)
        outcome = repair_missing_fts_rows(conn)
        states = dict(conn.execute("SELECT surface, state FROM fts_freshness_state").fetchall())
        work_events = conn.execute("SELECT COUNT(*) FROM session_work_events_fts_docsize").fetchone()[0]
        threads = conn.execute("SELECT COUNT(*) FROM threads_fts_docsize").fetchone()[0]
    finally:
        conn.close()

    assert outcome.success is True
    assert work_events == 1
    assert threads == 1
    assert states["session_work_events_fts"] == "ready"
    assert states["threads_fts"] == "ready"


def test_reset_and_repair_fts_rows_recovers_excess_message_rows(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE session_work_events (event_id TEXT PRIMARY KEY, session_id TEXT,
                work_event_type TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE session_work_events_fts USING fts5(event_id UNINDEXED, session_id UNINDEXED,
                work_event_type UNINDEXED, text);
            CREATE TABLE threads (thread_id TEXT PRIMARY KEY, search_text TEXT);
            CREATE VIRTUAL TABLE threads_fts USING fts5(thread_id UNINDEXED, root_id UNINDEXED, text);
            """
        )
        restore_fts_triggers_sync(conn)
        conn.execute(
            "INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'orphan needle', 'orphan needle')"
        )
        conn.execute(
            "INSERT INTO blocks VALUES ('block-2', 'msg-2', 'conv-1', 'text', 'survivor needle', 'survivor needle')"
        )
        conn.execute("INSERT INTO session_work_events VALUES ('event-1', 'conv-1', 'decision', 'ship it')")
        conn.execute("INSERT INTO threads VALUES ('thread-1', 'startup fts repair')")
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DELETE FROM blocks WHERE block_id = 'block-1'")

        assert repair_missing_fts_rows(conn).success is False

        outcome = reset_and_repair_fts_rows(conn)
        message_state = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows
            FROM fts_freshness_state
            WHERE surface='messages_fts'
            """
        ).fetchone()
        trigger_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='trigger'
              AND name IN ('messages_fts_ai', 'messages_fts_ad', 'messages_fts_au')
            """
        ).fetchone()[0]
        survivor_hits = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'survivor'"
        ).fetchone()[0]
        orphan_hits = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'orphan'").fetchone()[0]
    finally:
        conn.close()

    assert outcome.success is True
    assert message_state == ("ready", 1, 1, 0, 0, 0)
    assert trigger_count == 3
    assert survivor_hits == 1
    assert orphan_hits == 0


def test_repair_stale_fts_rows_skips_ready_archive_surfaces(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE session_work_events (event_id TEXT PRIMARY KEY, session_id TEXT,
                work_event_type TEXT, search_text TEXT);
            CREATE VIRTUAL TABLE session_work_events_fts USING fts5(event_id UNINDEXED, session_id UNINDEXED,
                work_event_type UNINDEXED, text);
            CREATE TABLE threads (thread_id TEXT PRIMARY KEY, search_text TEXT);
            CREATE VIRTUAL TABLE threads_fts USING fts5(thread_id UNINDEXED, root_id UNINDEXED, text);
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('session_work_events_fts', 'stale', 'now', 0, 0, 'old');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('threads_fts', 'stale', 'now', 0, 0, 'old');
            INSERT INTO session_work_events VALUES ('event-1', 'conv-1', 'decision', 'ship it');
            INSERT INTO threads VALUES ('thread-1', 'startup fts repair');
            """
        )
        restore_fts_triggers_sync(conn)
        conn.execute("UPDATE fts_freshness_state SET state='ready' WHERE surface = 'messages_fts'")
        outcome = repair_stale_fts_rows(conn)
        states = dict(conn.execute("SELECT surface, state FROM fts_freshness_state").fetchall())
    finally:
        conn.close()

    assert outcome.success is True
    assert states == {
        "messages_fts": "ready",
        "session_work_events_fts": "ready",
        "threads_fts": "ready",
    }


def test_search_readiness_rejects_poisoned_zero_count_ready_marker(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle freshness', 'needle freshness');
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


def test_search_readiness_trusts_stale_ledger_without_exact_recount(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle one', 'needle one');
            INSERT INTO blocks VALUES ('block-2', 'msg-2', 'conv-1', 'text', 'needle two', 'needle two');
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            SELECT rowid, block_id, message_id, session_id, block_type, search_text
            FROM blocks
            WHERE block_id = 'block-1';
            """
        )
        restore_fts_triggers_sync(conn)
        record_fts_surface_state_sync(
            conn,
            surface="messages_fts",
            state="stale",
            source_rows=2,
            indexed_rows=1,
            missing_rows=1,
            excess_rows=0,
            duplicate_rows=0,
            detail="cached stale verdict",
        )

        traced: list[str] = []
        conn.set_trace_callback(traced.append)
        try:
            readiness = message_fts_search_readiness_sync(conn)
        finally:
            conn.set_trace_callback(None)
    finally:
        conn.close()

    assert readiness == {
        "exists": True,
        "indexed_rows": 1,
        "total_rows": 2,
        "ready": False,
        "triggers_present": True,
    }
    traced_sql = "\n".join(traced)
    assert "FROM blocks WHERE search_text != ''" not in traced_sql
    assert "messages_fts_docsize" not in traced_sql


@pytest.mark.asyncio
async def test_async_search_readiness_trusts_stale_ledger_without_exact_recount(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    async with aiosqlite.connect(db) as conn:
        await conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle one', 'needle one');
            INSERT INTO blocks VALUES ('block-2', 'msg-2', 'conv-1', 'text', 'needle two', 'needle two');
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            SELECT rowid, block_id, message_id, session_id, block_type, search_text
            FROM blocks
            WHERE block_id = 'block-1';
            """
        )
        await record_fts_surface_state_async(
            conn,
            surface="messages_fts",
            state="stale",
            source_rows=2,
            indexed_rows=1,
            missing_rows=1,
            excess_rows=0,
            duplicate_rows=0,
            detail="cached stale verdict",
        )

        traced: list[str] = []
        await conn.set_trace_callback(traced.append)
        try:
            readiness = await message_fts_search_readiness_async(conn)
        finally:
            await conn.set_trace_callback(lambda _statement: None)

    assert readiness == {
        "exists": True,
        "indexed_rows": 1,
        "total_rows": 2,
        "ready": False,
        "triggers_present": True,
    }
    traced_sql = "\n".join(traced)
    assert "FROM blocks WHERE search_text != ''" not in traced_sql
    assert "messages_fts_docsize" not in traced_sql


def test_search_readiness_rejects_ready_marker_when_triggers_are_missing(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle freshness', 'needle freshness');
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            SELECT rowid, block_id, message_id, session_id, block_type, search_text FROM blocks;
            INSERT INTO fts_freshness_state (
                surface, state, checked_at, source_rows, indexed_rows,
                missing_rows, excess_rows, duplicate_rows, detail
            )
            VALUES ('messages_fts', 'ready', 'now', 1, 1, 0, 0, 0, NULL);
            """
        )

        readiness = message_fts_search_readiness_sync(conn)
        state = message_fts_recorded_state_sync(conn)
        recorded = conn.execute(
            "SELECT state, source_rows, indexed_rows, detail FROM fts_freshness_state WHERE surface='messages_fts'"
        ).fetchone()
    finally:
        conn.close()

    assert readiness["ready"] is False
    assert readiness["triggers_present"] is False
    assert readiness["total_rows"] == 1
    assert readiness["indexed_rows"] == 1
    assert state == "stale"
    assert recorded == ("stale", 1, 1, "exact message readiness failed")


def test_search_readiness_uses_blocks_as_zero_count_trust_source(tmp_path: Path) -> None:
    """A ready|0|0 ledger is untrusted when the actual FTS source has rows."""
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT PRIMARY KEY);
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle freshness', 'needle freshness');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            """
        )

        assert message_fts_recorded_state_sync(conn) == "unknown"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_async_recorded_readiness_uses_blocks_as_zero_count_trust_source(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    async with aiosqlite.connect(db) as conn:
        await conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT PRIMARY KEY);
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'needle freshness', 'needle freshness');
            INSERT INTO fts_freshness_state (surface, state, checked_at, source_rows, indexed_rows, detail)
            VALUES ('messages_fts', 'ready', 'now', 0, 0, NULL);
            """
        )

        assert await message_fts_recorded_state_async(conn) == "unknown"


def test_repair_stale_fts_rows_recomputes_poisoned_archive_counts(tmp_path: Path) -> None:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE blocks (
                block_id TEXT, message_id TEXT, session_id TEXT, block_type TEXT, text TEXT, search_text TEXT
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED, block_type UNINDEXED, text,
                content='', contentless_delete=1
            );
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY, state TEXT NOT NULL, checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0, indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0, excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0, detail TEXT
            );
            INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'text', 'repair freshness', 'repair freshness');
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
