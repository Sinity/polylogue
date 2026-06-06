"""Descriptor contracts for session-insight status queries."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from polylogue.storage.insights.session.status import (
    SessionInsightCountDescriptor,
    SessionInsightFtsDescriptor,
    SessionInsightReadyDescriptor,
    session_insight_status_async,
    session_insight_status_sync,
    session_profile_repair_candidate_ids_sync,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION


def test_fts_descriptor_marks_duplicates_unready() -> None:
    descriptor = SessionInsightFtsDescriptor(
        table_key="demo_fts",
        table_name="demo_fts",
        count_key="indexed_rows",
        duplicate_count_key="duplicate_rows",
        source_count_key="source_rows",
        distinct_sql="SELECT COUNT(DISTINCT id) FROM demo_fts",
        duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT id) FROM demo_fts",
        ready_key="profile_merged_fts_ready",
    )

    tables = {"demo_fts": True}

    assert descriptor.ready(tables, {"source_rows": 2, "indexed_rows": 3, "duplicate_rows": 1}) is False
    assert descriptor.ready(tables, {"source_rows": 2, "indexed_rows": 2, "duplicate_rows": 0}) is True
    assert descriptor.ready({"demo_fts": False}, {"source_rows": 0, "indexed_rows": 0, "duplicate_rows": 0}) is False


async def test_fts_descriptor_async_can_skip_distinct_freshness_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "fts-status.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            """
            CREATE TABLE demo_fts (id TEXT NOT NULL);
            INSERT INTO demo_fts (id) VALUES ('a'), ('a'), ('b');
            """
        )
        await conn.commit()

        descriptor = SessionInsightFtsDescriptor(
            table_key="demo_fts",
            table_name="demo_fts",
            count_key="indexed_rows",
            duplicate_count_key="duplicate_rows",
            source_count_key="source_rows",
            distinct_sql="SELECT COUNT(DISTINCT id) FROM demo_fts",
            duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT id) FROM demo_fts",
            ready_key="profile_merged_fts_ready",
        )

        fresh = await descriptor.counts_async(
            conn,
            {"demo_fts": True},
            {"source_rows": 2},
            verify_freshness=True,
        )
        lightweight = await descriptor.counts_async(
            conn,
            {"demo_fts": True},
            {"source_rows": 2},
            verify_freshness=False,
        )

    assert fresh == {"indexed_rows": 2, "duplicate_rows": 1}
    assert lightweight == {"indexed_rows": 3, "duplicate_rows": 1}


def test_count_descriptor_uses_fallback_when_freshness_is_disabled() -> None:
    descriptor = SessionInsightCountDescriptor(
        count_key="expected_rows",
        table_key="source_table",
        sql="SELECT 99",
        requires_freshness=True,
        fallback_count_key="materialized_rows",
    )

    with sqlite3.connect(":memory:") as conn:
        assert descriptor.count_sync(
            conn,
            {"source_table": True},
            {"materialized_rows": 7},
            verify_freshness=False,
        ) == ("expected_rows", 7)


def test_ready_descriptor_combines_table_equalities_and_zero_counts() -> None:
    descriptor = SessionInsightReadyDescriptor(
        ready_key="threads_ready",
        table_key="work_threads",
        equal_counts=(("thread_count", "root_threads"),),
        zero_counts=("stale_thread_count", "orphan_thread_count"),
    )

    assert descriptor.ready(
        {"work_threads": True},
        {"thread_count": 3, "root_threads": 3, "stale_thread_count": 0, "orphan_thread_count": 0},
    )
    assert not descriptor.ready(
        {"work_threads": True},
        {"thread_count": 3, "root_threads": 3, "stale_thread_count": 1, "orphan_thread_count": 0},
    )


def test_profile_repair_candidates_match_sort_key_freshness() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('ready-even-if-updated-at-differs', 1.0, '2026-05-01T12:00:00Z');

            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('stale-sort-key', 2.0, '2026-05-01T12:00:00Z');
            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('missing-profile', 3.0, '2026-05-01T12:00:00Z');
            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('hot-missing-profile', 4102444800.0, '2100-01-01T00:00:00Z');
            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('hot-stale-sort-key', 4102444800.0, '2100-01-01T00:00:00Z');
            """
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, materializer_version, source_sort_key, source_updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "ready-even-if-updated-at-differs",
                SESSION_INSIGHT_MATERIALIZER_VERSION,
                1.0,
                "2026-04-30T12:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, materializer_version, source_sort_key, source_updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "stale-sort-key",
                SESSION_INSIGHT_MATERIALIZER_VERSION,
                1.5,
                "2026-05-01T12:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, materializer_version, source_sort_key, source_updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "hot-stale-sort-key",
                SESSION_INSIGHT_MATERIALIZER_VERSION,
                1.5,
                "2100-01-01T00:00:00Z",
            ),
        )

        candidates = session_profile_repair_candidate_ids_sync(conn)

    assert candidates == ["missing-profile", "stale-sort-key"]


def test_profile_repair_candidates_do_not_require_row_factory() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('missing-profile', 3.0, '2026-05-01T12:00:00Z');
            """
        )

        candidates = session_profile_repair_candidate_ids_sync(conn)

    assert candidates == ["missing-profile"]


def test_profile_repair_candidates_ignore_hot_recent_sources() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('cold-missing-profile', 3.0, '2026-05-01T12:00:00Z');
            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('hot-missing-profile', strftime('%s', 'now'), '2026-05-24T07:00:00Z');
            """
        )

        candidates = session_profile_repair_candidate_ids_sync(conn)

    assert candidates == ["cold-missing-profile"]


def test_session_insight_status_requires_latency_rows_for_ready_profiles() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            f"""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                canonical_session_date TEXT,
                first_message_at TEXT,
                last_message_at TEXT,
                source_updated_at TEXT,
                evidence_payload_json TEXT,
                tags_json TEXT,
                auto_tags_json TEXT,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                work_event_count INTEGER NOT NULL,
                phase_count INTEGER NOT NULL
            );
            CREATE TABLE session_latency_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL
            );

            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('ready', NULL, 1.0, '2026-05-01T12:00:00Z');
            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('missing-latency', NULL, 2.0, '2026-05-01T12:00:00Z');
            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('stale-latency', NULL, 3.0, '2026-05-01T12:00:00Z');

            INSERT INTO session_profiles (
                session_id, source_name, canonical_session_date, tags_json, auto_tags_json,
                first_message_at, last_message_at, source_updated_at, evidence_payload_json,
                materializer_version, source_sort_key, work_event_count, phase_count
            ) VALUES
                (
                    'ready', 'codex', '2026-05-01', '[]', '[]',
                    NULL, NULL, '2026-05-01T12:00:00Z', '{{}}',
                    {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1.0, 0, 0
                ),
                (
                    'missing-latency', 'codex', '2026-05-01', '[]', '[]',
                    NULL, NULL, '2026-05-01T12:00:00Z', '{{}}',
                    {SESSION_INSIGHT_MATERIALIZER_VERSION}, 2.0, 0, 0
                ),
                (
                    'stale-latency', 'codex', '2026-05-01', '[]', '[]',
                    NULL, NULL, '2026-05-01T12:00:00Z', '{{}}',
                    {SESSION_INSIGHT_MATERIALIZER_VERSION}, 3.0, 0, 0
                );

            INSERT INTO session_latency_profiles (session_id, materializer_version, source_sort_key)
            VALUES
                ('ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1.0),
                ('stale-latency', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 2.5);
            """
        )

        status = session_insight_status_sync(conn)

    assert status.profile_rows_ready is True
    assert status.latency_profile_rows_ready is False
    assert status.latency_profile_row_count == 2
    assert status.missing_latency_profile_row_count == 1
    assert status.stale_latency_profile_row_count == 1


async def test_status_sync_and_async_match_when_product_tables_are_absent(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('root', NULL, 1.0, '2026-04-01T00:00:00Z');
            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('child', 'root', 2.0, '2026-04-01T00:01:00Z');
            """
        )
        sync_status = session_insight_status_sync(conn)

    async with aiosqlite.connect(db_path) as conn:
        async_status = await session_insight_status_async(conn)

    assert asdict(sync_status) == asdict(async_status)
    assert sync_status.total_sessions == 2
    assert sync_status.root_threads == 1
    assert sync_status.missing_profile_row_count == 2
    assert sync_status.stale_profile_row_count == 0
    assert sync_status.profile_rows_ready is False
    assert sync_status.threads_ready is False


async def test_lightweight_status_sync_and_async_match_with_freshness_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "status-lightweight.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                work_event_count INTEGER NOT NULL,
                phase_count INTEGER NOT NULL
            );
            CREATE TABLE session_profiles_fts (session_id TEXT NOT NULL);
            CREATE TABLE work_threads (thread_id TEXT PRIMARY KEY);

            INSERT INTO sessions (session_id, parent_session_id, sort_key, updated_at)
            VALUES ('root', NULL, 1.0, '2026-04-01T00:00:00Z');
            INSERT INTO session_profiles (session_id, work_event_count, phase_count)
            VALUES ('root', 0, 0);
            INSERT INTO session_profiles_fts (session_id) VALUES ('root'), ('root');
            INSERT INTO work_threads (thread_id) VALUES ('root');
            """
        )
        sync_status = session_insight_status_sync(conn, verify_freshness=False)

    async with aiosqlite.connect(db_path) as conn:
        async_status = await session_insight_status_async(conn, verify_freshness=False)

    assert asdict(sync_status) == asdict(async_status)
    assert sync_status.root_threads == sync_status.thread_count == 1
    assert sync_status.stale_profile_row_count == 0
    # profile_merged_fts_* fields are present on the struct but not yet
    # populated by any readiness descriptor (the merged-fts index is now
    # tracked via session_work_event_fts). #944 follow-up wires the descriptor.
    assert sync_status.profile_merged_fts_duplicate_count == 0  # not yet populated
