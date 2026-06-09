"""Tests for the cursor-lag status projection (#1232)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polylogue.daemon.cursor_lag_status import cursor_lag_summary_info
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import record_cursor_lag_sample, upsert_ingest_cursor
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_cursor(
    db_path: Path,
    *,
    rows: list[tuple[str, int, int, int, int, str]],
) -> None:
    """Insert ``live_cursor`` rows: ``(source_path, byte_size, byte_offset,
    failure_count, excluded, updated_at_iso)``.

    Uses the minimal column set the projection reads; other columns default.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE live_cursor (
                source_path TEXT PRIMARY KEY,
                byte_size INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL DEFAULT 0,
                last_complete_newline INTEGER NOT NULL DEFAULT 0,
                record_count INTEGER NOT NULL DEFAULT 0,
                last_record_ts TEXT,
                parser_fingerprint TEXT,
                content_fingerprint TEXT,
                tail_hash TEXT,
                source_name TEXT,
                st_dev INTEGER,
                st_ino INTEGER,
                mtime_ns INTEGER,
                source_generation INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                excluded INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO live_cursor (source_path, byte_size, byte_offset,
                                     failure_count, excluded, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_cursor_lag_summary_returns_empty_when_db_missing(tmp_path: Path) -> None:
    summary = cursor_lag_summary_info(tmp_path / "nonexistent.db")
    assert summary.tracked_file_count == 0
    assert summary.stuck_file_count == 0
    assert summary.family_summaries == []


def test_cursor_lag_summary_reads_ops_tier_from_archive_tiers(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    ops_db = db.with_name("ops.db")
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_ingest_cursor(
            conn,
            source_path="/x/behind.jsonl",
            stat_size=2_000,
            byte_offset=500,
            failure_count=0,
            excluded=False,
            updated_at_ms=int((now - timedelta(seconds=120)).timestamp() * 1000),
        )
        upsert_ingest_cursor(
            conn,
            source_path="/x/idle.jsonl",
            stat_size=1_000,
            byte_offset=1_000,
            failure_count=0,
            excluded=False,
            updated_at_ms=int((now - timedelta(seconds=10)).timestamp() * 1000),
        )

    summary = cursor_lag_summary_info(db, now=now)

    assert summary.tracked_file_count == 2
    assert summary.stuck_file_count == 1
    assert summary.idle_file_count == 1
    assert summary.max_lag_s == 120.0
    assert [item.source_path for item in summary.stuck] == ["/x/behind.jsonl"]


def test_cursor_lag_summary_decorates_ops_tier_baseline_from_archive_tiers(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    ops_db = db.with_name("ops.db")
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_ingest_cursor(
            conn,
            source_path="/x/behind.jsonl",
            stat_size=2_000,
            byte_offset=500,
            failure_count=0,
            excluded=False,
            updated_at_ms=int((now - timedelta(seconds=120)).timestamp() * 1000),
        )
        for index, lag_ms in enumerate((90_000, 100_000, 110_000), start=1):
            record_cursor_lag_sample(
                conn,
                sample_id=f"sample-{index}",
                family="unknown",
                source_path="/x/behind.jsonl",
                lag_ms=lag_ms,
                severity="warning",
                sampled_at_ms=int((now - timedelta(minutes=index)).timestamp() * 1000),
            )
        conn.commit()

    summary = cursor_lag_summary_info(db, now=now)

    assert not db.exists()
    assert summary.family_summaries[0].baseline.sample_count == 3
    assert summary.family_summaries[0].baseline.rolling_median_lag_s == 100.0


def test_cursor_lag_summary_prefers_archive_ops_when_both_exist(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            ("/legacy/stuck.jsonl", 2_000, 500, 0, 0, (now - timedelta(seconds=300)).isoformat()),
        ],
    )
    ops_db = db.with_name("ops.db")
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_ingest_cursor(
            conn,
            source_path="/v1/idle.jsonl",
            stat_size=1_000,
            byte_offset=1_000,
            failure_count=0,
            excluded=False,
            updated_at_ms=int((now - timedelta(seconds=10)).timestamp() * 1000),
        )

    summary = cursor_lag_summary_info(db, now=now)

    assert summary.tracked_file_count == 1
    assert summary.stuck_file_count == 0
    assert summary.idle_file_count == 1
    assert summary.stuck == []


def test_cursor_lag_summary_returns_empty_when_table_missing(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE other_table (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    summary = cursor_lag_summary_info(db)
    assert summary.tracked_file_count == 0


def test_cursor_lag_summary_classifies_caught_up_cursor_as_idle(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            # byte_offset == byte_size, no failures: idle
            ("/a/file.jsonl", 1000, 1000, 0, 0, (now - timedelta(seconds=10)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    assert summary.tracked_file_count == 1
    assert summary.stuck_file_count == 0
    assert summary.idle_file_count == 1
    assert summary.stuck == []


def test_cursor_lag_summary_classifies_behind_cursor_as_stuck(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            # byte_offset < byte_size: stuck
            ("/a/behind.jsonl", 2000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    assert summary.stuck_file_count == 1
    assert summary.idle_file_count == 0
    assert summary.max_lag_s == 120.0
    assert len(summary.stuck) == 1
    assert summary.stuck[0].source_path == "/a/behind.jsonl"


def test_cursor_lag_summary_classifies_failing_cursor_as_stuck(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            # byte_offset == byte_size but failure_count > 0: still stuck
            ("/a/failing.jsonl", 1000, 1000, 3, 0, (now - timedelta(seconds=60)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    assert summary.stuck_file_count == 1
    assert summary.max_lag_s == 60.0


def test_cursor_lag_summary_treats_excluded_cursor_as_idle(tmp_path: Path) -> None:
    # Quarantined cursors live under the raw-failures alert surface, not the
    # SLO alert surface. They must not raise cursor-lag alerts.
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            ("/a/quarantined.jsonl", 2000, 500, 5, 1, (now - timedelta(hours=2)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    assert summary.stuck_file_count == 0
    assert summary.idle_file_count == 1


def test_cursor_lag_summary_buckets_by_family_unknown_for_arbitrary_path(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=300)).isoformat()),
            ("/x/b.jsonl", 1000, 0, 0, 0, (now - timedelta(seconds=120)).isoformat()),
            ("/x/c.jsonl", 1000, 1000, 0, 0, (now - timedelta(seconds=10)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    # Arbitrary paths don't match any configured watch root → "unknown"
    assert summary.tracked_file_count == 3
    assert summary.stuck_file_count == 2
    assert summary.idle_file_count == 1
    assert len(summary.family_summaries) == 1
    family = summary.family_summaries[0]
    assert family.family == "unknown"
    assert family.stuck_file_count == 2
    assert family.idle_file_count == 1
    assert family.max_lag_s == 300.0


def test_cursor_lag_summary_sorts_stuck_items_by_lag_desc(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    _seed_cursor(
        db,
        rows=[
            ("/x/small.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=10)).isoformat()),
            ("/x/big.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=999)).isoformat()),
            ("/x/mid.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=100)).isoformat()),
        ],
    )
    summary = cursor_lag_summary_info(db, now=now)
    paths = [item.source_path for item in summary.stuck]
    assert paths == ["/x/big.jsonl", "/x/mid.jsonl", "/x/small.jsonl"]
