"""Descriptor contracts for session-insight status queries."""

from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.derived.session.status import (
    _COUNT_DESCRIPTORS,
    _FTS_DESCRIPTORS,
    _TABLE_DESCRIPTORS,
    SessionInsightCountDescriptor,
    SessionInsightFtsDescriptor,
    session_insight_status_async,
    session_insight_status_sync,
    session_profile_repair_candidate_ids_sync,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_fts_descriptor_reports_duplicate_counts() -> None:
    descriptor = SessionInsightFtsDescriptor(
        table_key="demo_fts",
        table_name="demo_fts",
        count_key="indexed_rows",
        duplicate_count_key="duplicate_rows",
        source_count_key="source_rows",
        distinct_sql="SELECT COUNT(DISTINCT id) FROM demo_fts",
        duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT id) FROM demo_fts",
    )

    tables = {"demo_fts": True}

    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE demo_fts (id TEXT)")
        assert descriptor.counts_sync(
            conn,
            tables,
            {"source_rows": 2},
            verify_freshness=False,
        ) == {"indexed_rows": 0, "duplicate_rows": 0}


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
        table_keys=("source_table",),
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
                tool_outcome TEXT,
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
                tool_outcome TEXT,
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
    assert sync_status.profile_row_count == 0
    assert sync_status.thread_count == 0


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
                tool_outcome TEXT,
                search_text TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                work_event_count INTEGER NOT NULL,
                phase_count INTEGER NOT NULL
            );
            CREATE TABLE session_profiles_fts (session_id TEXT NOT NULL);
            CREATE TABLE session_work_events (session_id TEXT NOT NULL);
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


def test_status_descriptors_resolve_against_declared_tables_and_snapshot_fields() -> None:
    """Descriptor keys are resolved by subscript, so a descriptor naming a
    relation or a count that no longer exists fails every status call rather
    than import. Anti-vacuity: reinstating a descriptor keyed on a retired
    table, or emitting a count the snapshot does not declare, turns this red.
    """
    declared_tables = {descriptor.key for descriptor in _TABLE_DESCRIPTORS}
    snapshot_fields = {field.name for field in dataclasses.fields(SessionInsightStatusSnapshot)}

    emitted = {descriptor.count_key for descriptor in _TABLE_DESCRIPTORS if descriptor.count_key is not None}
    emitted |= {descriptor.count_key for descriptor in _COUNT_DESCRIPTORS}
    emitted |= {descriptor.count_key for descriptor in _FTS_DESCRIPTORS}
    emitted |= {descriptor.duplicate_count_key for descriptor in _FTS_DESCRIPTORS}

    referenced = {descriptor.table_key for descriptor in _FTS_DESCRIPTORS}
    referenced |= {table_key for descriptor in _COUNT_DESCRIPTORS for table_key in descriptor.table_keys}

    assert referenced <= declared_tables
    assert emitted <= snapshot_fields


def test_status_snapshot_reads_a_fresh_index_tier(tmp_path: Path) -> None:
    """The status route runs against real index DDL. Anti-vacuity: a descriptor
    that queries a relation the index tier does not create raises
    ``sqlite3.OperationalError`` here.
    """
    with sqlite3.connect(tmp_path / "index.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        status = session_insight_status_sync(conn)

    assert status.total_sessions == 0
    assert status.missing_profile_row_count == 0
