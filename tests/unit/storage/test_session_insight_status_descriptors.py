"""Descriptor contracts for session-product status queries."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from polylogue.storage.insights.session.status import (
    SessionInsightCountDescriptor,
    SessionInsightFtsDescriptor,
    SessionInsightReadyDescriptor,
    session_product_status_async,
    session_product_status_sync,
)


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


async def test_status_sync_and_async_match_when_product_tables_are_absent(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                parent_conversation_id TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            INSERT INTO conversations (conversation_id, parent_conversation_id, sort_key, updated_at)
            VALUES ('root', NULL, 1.0, '2026-04-01T00:00:00Z');
            INSERT INTO conversations (conversation_id, parent_conversation_id, sort_key, updated_at)
            VALUES ('child', 'root', 2.0, '2026-04-01T00:01:00Z');
            """
        )
        sync_status = session_product_status_sync(conn)

    async with aiosqlite.connect(db_path) as conn:
        async_status = await session_product_status_async(conn)

    assert asdict(sync_status) == asdict(async_status)
    assert sync_status.total_conversations == 2
    assert sync_status.root_threads == 1
    assert sync_status.missing_profile_row_count == 2
    assert sync_status.stale_profile_row_count == 0
    assert sync_status.profile_rows_ready is False
    assert sync_status.threads_ready is False
