"""Regression tests for FTS5 escaping on insight read paths.

User-supplied query strings flow into `MATCH ?` clauses on the durable
session-insight read paths. Without escaping, FTS5 operator characters,
unbalanced quotes, and bare boolean operators (`OR`, `AND`, `NEAR`) cause
sqlite3.OperationalError. These tests exercise each affected read path with
malicious inputs and assert the queries return cleanly.

Issue: #814.
"""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.query_models import SessionTimelineListQuery, ThreadListQuery
from polylogue.storage.sqlite.queries.session_insight_thread_queries import list_threads
from polylogue.storage.sqlite.queries.session_insight_timeline_reads import list_work_events

# Inputs that would crash a bare FTS5 MATCH if passed unescaped.
MALICIOUS_QUERIES = [
    "foo OR",
    "AND",
    'unclosed "quote',
    'foo "bar',
    "NEAR",
    "* * *",
    "a AND OR b",
    "(((",
    "col:value",
]


async def _make_work_events_db() -> aiosqlite.Connection:
    """Create an in-memory db with the FTS5 table the timeline read path joins."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.executescript(
        """
        CREATE TABLE session_work_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            materializer_version INTEGER NOT NULL DEFAULT 5,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
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
            inference_family TEXT NOT NULL DEFAULT 'heuristic'
        );
        CREATE VIRTUAL TABLE session_work_events_fts USING fts5(
            event_id UNINDEXED,
            session_id UNINDEXED,
            source_name UNINDEXED,
            heuristic_label UNINDEXED,
            text,
            tokenize='unicode61'
        );
        CREATE TRIGGER session_work_events_fts_ai AFTER INSERT ON session_work_events BEGIN SELECT 1; END;
        CREATE TRIGGER session_work_events_fts_ad AFTER DELETE ON session_work_events BEGIN SELECT 1; END;
        CREATE TRIGGER session_work_events_fts_au AFTER UPDATE ON session_work_events BEGIN SELECT 1; END;
        INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
        VALUES ('e1', 'c1', 'claude-code', 'edit', 'hello world');
        INSERT INTO session_work_events (
            event_id, session_id, materialized_at, source_name, event_index,
            heuristic_label, summary, search_text
        ) VALUES ('e1', 'c1', '2026-01-01T00:00:00Z', 'claude-code', 0, 'edit', 'hello', 'hello world');
        """
    )
    await conn.commit()
    return conn


async def _make_threads_db() -> aiosqlite.Connection:
    """Create an in-memory db with the FTS5 table the thread read path joins."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.executescript(
        """
        CREATE TABLE threads (
            thread_id TEXT PRIMARY KEY,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            materialized_at TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            dominant_repo TEXT,
            session_ids_json TEXT NOT NULL DEFAULT '[]',
            session_count INTEGER NOT NULL DEFAULT 0,
            depth INTEGER NOT NULL DEFAULT 0,
            branch_count INTEGER NOT NULL DEFAULT 0,
            total_messages INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            work_event_breakdown_json TEXT NOT NULL DEFAULT '{}',
            payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE threads_fts USING fts5(
            thread_id UNINDEXED,
            root_id UNINDEXED,
            text,
            tokenize='unicode61'
        );
        CREATE TRIGGER threads_fts_ai AFTER INSERT ON threads BEGIN SELECT 1; END;
        CREATE TRIGGER threads_fts_ad AFTER DELETE ON threads BEGIN SELECT 1; END;
        CREATE TRIGGER threads_fts_au AFTER UPDATE ON threads BEGIN SELECT 1; END;
        INSERT INTO threads_fts (thread_id, root_id, text)
        VALUES ('t1', 't1', 'hello world');
        INSERT INTO threads (
            thread_id, materialized_at, search_text
        ) VALUES ('t1', '2026-01-01T00:00:00Z', 'hello world');
        """
    )
    await conn.commit()
    return conn


@pytest.mark.asyncio
@pytest.mark.parametrize("malicious", MALICIOUS_QUERIES)
async def test_list_work_events_escapes_fts5_query(malicious: str) -> None:
    conn = await _make_work_events_db()
    try:
        # Must not raise sqlite3.OperationalError. Returns empty or literal-phrase matches.
        result = await list_work_events(conn, SessionTimelineListQuery(query=malicious, limit=5))
        assert isinstance(result, list)
    except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
        pytest.fail(f"unescaped FTS5 query {malicious!r} raised: {exc}")
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malicious", MALICIOUS_QUERIES)
async def test_list_threads_escapes_fts5_query(malicious: str) -> None:
    conn = await _make_threads_db()
    try:
        result = await list_threads(conn, ThreadListQuery(query=malicious, limit=5))
        assert isinstance(result, list)
    except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
        pytest.fail(f"unescaped FTS5 query {malicious!r} raised: {exc}")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_list_work_events_escaped_query_matches_literal_phrase() -> None:
    """A query that looks like an operator should match the literal token, not crash."""
    conn = await _make_work_events_db()
    try:
        # 'hello' exists in indexed text; 'OR' alone would crash without escaping.
        # After escaping, the OR is wrapped as a phrase and returns no rows.
        result = await list_work_events(conn, SessionTimelineListQuery(query="OR", limit=5))
        assert result == []
        # 'hello' as a normal token still works (escape leaves bare alphanumerics alone).
        result_hit = await list_work_events(conn, SessionTimelineListQuery(query="hello", limit=5))
        assert len(result_hit) == 1
    finally:
        await conn.close()
