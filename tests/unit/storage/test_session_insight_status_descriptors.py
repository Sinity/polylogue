"""Descriptor contracts for session-insight status queries."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from polylogue.insights.readiness import InsightReadinessQuery, build_insight_readiness_report
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
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
        table_key="threads",
        equal_counts=(("thread_count", "root_threads"),),
        zero_counts=("stale_thread_count", "orphan_thread_count"),
    )

    assert descriptor.ready(
        {"threads": True},
        {"thread_count": 3, "root_threads": 3, "stale_thread_count": 0, "orphan_thread_count": 0},
    )
    assert not descriptor.ready(
        {"threads": True},
        {"thread_count": 3, "root_threads": 3, "stale_thread_count": 1, "orphan_thread_count": 0},
    )


def test_profile_repair_candidates_match_sort_key_freshness() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('ready-even-if-updated-at-differs', 1000, 1777636800000);

            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('stale-sort-key', 2000, 1777636800000);
            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('missing-profile', 3000, 1777636800000);
            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('hot-missing-profile', 4102444800000, 4102444800000);
            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('hot-stale-sort-key', 4102444800000, 4102444800000);
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
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('missing-profile', 3000, 1777636800000);
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
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                source_updated_at TEXT
            );

            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('cold-missing-profile', 3000, 1777636800000);
            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('hot-missing-profile', strftime('%s', 'now') * 1000, strftime('%s', 'now') * 1000);
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
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
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
                source_sort_key REAL,
                source_updated_at TEXT
            );
            CREATE TABLE insight_materialization (
                insight_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                materializer_version INTEGER NOT NULL,
                source_sort_key_ms INTEGER
            );

            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('ready', NULL, 1000, 1777636800000);
            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('missing-latency', NULL, 2000, 1777636800000);
            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('stale-latency', NULL, 3000, 1777636800000);

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

            INSERT INTO insight_materialization (
                insight_type, session_id, materializer_version, source_sort_key_ms
            )
            VALUES
                ('session_profile', 'ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1000),
                ('session_profile', 'missing-latency', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 2000),
                ('session_profile', 'stale-latency', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 3000),
                ('latency', 'ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1000),
                ('latency', 'missing-latency', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 2000),
                ('latency', 'stale-latency', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 3000);
            """
        )

        status = session_insight_status_sync(conn)

    assert status.profile_rows_ready is True
    assert status.latency_profile_rows_ready is False
    assert status.latency_profile_row_count == 2
    assert status.missing_latency_profile_row_count == 1
    assert status.stale_latency_profile_row_count == 1


def test_status_treats_run_projection_materialization_as_optional_cache() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            f"""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            CREATE TABLE session_runs (session_id TEXT NOT NULL);
            CREATE TABLE session_observed_events (session_id TEXT NOT NULL);
            CREATE TABLE session_context_snapshots (session_id TEXT NOT NULL);
            CREATE TABLE insight_materialization (
                insight_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                materializer_version INTEGER NOT NULL,
                source_sort_key_ms INTEGER
            );

            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('ready', NULL, 1000, 1777636800000);
            INSERT INTO session_runs (session_id) VALUES ('ready');
            INSERT INTO session_observed_events (session_id) VALUES ('ready');
            INSERT INTO session_context_snapshots (session_id) VALUES ('ready');
            INSERT INTO insight_materialization (
                insight_type, session_id, materializer_version, source_sort_key_ms
            )
            VALUES
                ('runs', 'ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1000),
                ('observed_events', 'ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1000),
                ('context_snapshots', 'ready', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1000);
            """
        )

        ready = session_insight_status_sync(conn)
        conn.execute("DELETE FROM insight_materialization WHERE insight_type = 'runs'")
        missing = session_insight_status_sync(conn)
        conn.execute(
            "INSERT INTO insight_materialization (insight_type, session_id, materializer_version, source_sort_key_ms) "
            "VALUES ('runs', 'ready', ?, 1000)",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1,),
        )
        stale = session_insight_status_sync(conn)

    assert ready.run_count == 1
    assert ready.observed_event_count == 1
    assert ready.context_snapshot_count == 1
    assert ready.run_rows_ready is True
    assert ready.observed_event_rows_ready is True
    assert ready.context_snapshot_rows_ready is True
    assert missing.missing_run_materialization_count == 1
    assert missing.run_rows_ready is True
    assert stale.missing_run_materialization_count == 1
    assert stale.run_rows_ready is True


async def test_readiness_report_treats_empty_run_projection_cache_as_ready(tmp_path: Path) -> None:
    db_path = tmp_path / "insights.db"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        await conn.executescript(
            """
            CREATE TABLE session_runs (session_id TEXT NOT NULL);
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                sort_key_ms INTEGER
            );
            """
        )
        await conn.commit()
        report = await build_insight_readiness_report(
            conn,
            SessionInsightStatusSnapshot(
                total_sessions=1,
                run_count=0,
                missing_run_materialization_count=1,
                run_rows_ready=True,
            ),
            InsightReadinessQuery(insights=("session_runs",)),
        )

    entry = report.insights[0]
    assert entry.insight_name == "session_runs"
    assert entry.row_count == 0
    assert entry.missing_count == 0
    assert entry.ready_flags == {"run_rows_ready": True}
    assert entry.verdict == "ready"


def test_status_tracks_work_and_phase_staleness_from_materialization_ledger() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            f"""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                source_name TEXT,
                canonical_session_date TEXT,
                first_message_at TEXT,
                last_message_at TEXT,
                source_updated_at TEXT,
                evidence_payload_json TEXT,
                tags_json TEXT,
                auto_tags_json TEXT,
                materialized_at TEXT,
                materializer_version INTEGER NOT NULL,
                source_sort_key REAL,
                work_event_count INTEGER NOT NULL,
                phase_count INTEGER NOT NULL
            );
            CREATE TABLE session_work_events (session_id TEXT NOT NULL);
            CREATE TABLE session_phases (session_id TEXT NOT NULL);
            CREATE TABLE insight_materialization (
                insight_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                materializer_version INTEGER NOT NULL,
                source_sort_key_ms INTEGER
            );

            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('stale-work', NULL, 1000, 1777636800000),
                   ('stale-phase', NULL, 2000, 1777636800000);
            INSERT INTO session_profiles (
                session_id, source_name, canonical_session_date, first_message_at, last_message_at,
                source_updated_at, evidence_payload_json, tags_json, auto_tags_json, materialized_at,
                materializer_version, source_sort_key, work_event_count, phase_count
            ) VALUES
                (
                    'stale-work', 'codex', '2026-05-01', NULL, NULL,
                    '2026-05-01T12:00:00Z', '{{}}', '[]', '[]', '2026-05-01T12:00:00Z',
                    {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1.0, 1, 0
                ),
                (
                    'stale-phase', 'codex', '2026-05-01', NULL, NULL,
                    '2026-05-01T12:00:00Z', '{{}}', '[]', '[]', '2026-05-01T12:00:00Z',
                    {SESSION_INSIGHT_MATERIALIZER_VERSION}, 2.0, 0, 1
                );
            INSERT INTO session_work_events (session_id) VALUES ('stale-work');
            INSERT INTO session_phases (session_id) VALUES ('stale-phase');
            INSERT INTO insight_materialization (
                insight_type, session_id, materializer_version, source_sort_key_ms
            ) VALUES
                ('work_events', 'stale-work', {SESSION_INSIGHT_MATERIALIZER_VERSION - 1}, 1000),
                ('phases', 'stale-phase', {SESSION_INSIGHT_MATERIALIZER_VERSION}, 1001);
            """
        )

        status = session_insight_status_sync(conn)

    assert status.stale_work_event_inference_count == 1
    assert status.work_event_inference_rows_ready is False
    assert status.stale_phase_inference_count == 1
    assert status.phase_inference_rows_ready is False


async def test_status_sync_and_async_match_when_product_tables_are_absent(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('root', NULL, 1000, 1775001600000);
            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('child', 'root', 2000, 1775001660000);
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
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                sort_key_ms INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                work_event_count INTEGER NOT NULL,
                phase_count INTEGER NOT NULL
            );
            CREATE TABLE session_profiles_fts (session_id TEXT NOT NULL);
            CREATE TABLE threads (thread_id TEXT PRIMARY KEY);

            INSERT INTO sessions (session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('root', NULL, 1000, 1775001600000);
            INSERT INTO session_profiles (session_id, work_event_count, phase_count)
            VALUES ('root', 0, 0);
            INSERT INTO session_profiles_fts (session_id) VALUES ('root'), ('root');
            INSERT INTO threads (thread_id) VALUES ('root');
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
