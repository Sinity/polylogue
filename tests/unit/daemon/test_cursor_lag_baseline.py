"""Tests for the cursor-lag rolling-baseline substrate (#1349)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polylogue.daemon.cursor_lag_baseline import (
    FamilyBaseline,
    ensure_lag_sample_table,
    gc_cursor_lag_samples,
    load_family_baseline,
    load_family_baselines,
    record_cursor_lag_sample,
)
from polylogue.daemon.cursor_lag_status import (
    CursorLagFamilySummary,
    CursorLagItem,
    CursorLagSummary,
)


def _summary(
    family: str,
    *,
    stuck_count: int,
    max_lag_s: float,
    item_lags: list[float] | None = None,
) -> CursorLagSummary:
    items: list[CursorLagItem] = []
    for i, lag in enumerate(item_lags or [max_lag_s]):
        items.append(
            CursorLagItem(
                family=family,
                source_path=f"/x/{family}/{i}.jsonl",
                byte_offset=0,
                byte_size=1,
                failure_count=0,
                updated_at="2026-05-18T00:00:00+00:00",
                lag_s=lag,
            )
        )
    return CursorLagSummary(
        tracked_file_count=stuck_count,
        stuck_file_count=stuck_count,
        idle_file_count=0,
        max_lag_s=max_lag_s,
        family_summaries=[
            CursorLagFamilySummary(
                family=family,
                tracked_file_count=stuck_count,
                stuck_file_count=stuck_count,
                idle_file_count=0,
                max_lag_s=max_lag_s,
            )
        ],
        stuck=items,
    )


# ---------------------------------------------------------------------------
# Sample table lifecycle
# ---------------------------------------------------------------------------


def test_ensure_lag_sample_table_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = sqlite3.connect(str(db))
    try:
        ensure_lag_sample_table(conn)
        ensure_lag_sample_table(conn)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='live_cursor_lag_sample'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sample writes
# ---------------------------------------------------------------------------


def test_record_sample_no_stuck_files_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    summary = CursorLagSummary()  # all-default = no stuck
    assert record_cursor_lag_sample(db, summary) == 0
    assert not db.exists()
    assert not db.with_name("ops.db").exists()


def test_record_sample_writes_one_row_per_stuck_family(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    summary = _summary("claude-code-session", stuck_count=2, max_lag_s=120.0, item_lags=[60.0, 120.0])
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)

    written = record_cursor_lag_sample(db, summary, now=now)

    assert written == 1
    assert not db.exists()
    with sqlite3.connect(str(db.with_name("ops.db"))) as conn:
        archive_row = conn.execute(
            """
            SELECT family, source_path, lag_ms, stuck_file_count, p50_lag_ms, p95_lag_ms, severity, sampled_at_ms
            FROM cursor_lag_samples
            """
        ).fetchone()
    assert archive_row == (
        "claude-code-session",
        "/x/claude-code-session/1.jsonl",
        120_000,
        2,
        90_000,
        117_000,
        "warning",
        int(now.timestamp() * 1000),
    )


def test_record_sample_falls_back_when_stuck_list_does_not_include_family(tmp_path: Path) -> None:
    # The CursorLagSummary.stuck list is bounded; if a family's items did
    # not make the top-10, the row still records the family's max_lag_s
    # with p50/p95 collapsed to that single value.
    db = tmp_path / "index.db"
    summary = CursorLagSummary(
        tracked_file_count=1,
        stuck_file_count=1,
        idle_file_count=0,
        max_lag_s=42.0,
        family_summaries=[CursorLagFamilySummary(family="ghost", stuck_file_count=1, max_lag_s=42.0)],
        stuck=[],
    )
    written = record_cursor_lag_sample(db, summary)
    assert written == 1
    assert not db.exists()
    with sqlite3.connect(str(db.with_name("ops.db"))) as conn:
        row = conn.execute("SELECT p50_lag_ms, p95_lag_ms FROM cursor_lag_samples").fetchone()
    assert row[0] == 42_000
    assert row[1] == 42_000


# ---------------------------------------------------------------------------
# GC
# ---------------------------------------------------------------------------


def test_gc_drops_samples_older_than_retention(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    summary = _summary("f", stuck_count=1, max_lag_s=10.0)
    # Old sample
    record_cursor_lag_sample(db, summary, now=now - timedelta(days=30))
    # Fresh sample
    record_cursor_lag_sample(db, summary, now=now)

    removed = gc_cursor_lag_samples(db, retention_days=14, now=now)
    assert removed == 1
    assert not db.exists()
    with sqlite3.connect(str(db.with_name("ops.db"))) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM cursor_lag_samples").fetchone()[0]
    assert remaining == 1


def test_gc_is_noop_when_retention_zero(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    assert gc_cursor_lag_samples(db, retention_days=0) == 0


def test_gc_is_noop_when_db_missing(tmp_path: Path) -> None:
    assert gc_cursor_lag_samples(tmp_path / "nonexistent.db", retention_days=14) == 0


# ---------------------------------------------------------------------------
# Baseline reads
# ---------------------------------------------------------------------------


def test_load_baseline_returns_unconfident_when_db_missing(tmp_path: Path) -> None:
    baseline = load_family_baseline(tmp_path / "nope.db", "f", window_days=7, min_samples=50)
    assert isinstance(baseline, FamilyBaseline)
    assert baseline.sample_count == 0
    assert baseline.confident is False


def test_load_baseline_returns_unconfident_when_table_missing(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    sqlite3.connect(str(db)).close()
    baseline = load_family_baseline(db, "f", window_days=7, min_samples=50)
    assert baseline.sample_count == 0
    assert baseline.confident is False


def test_load_baseline_computes_p50_p95_over_window(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    # 100 samples ranging 1..100; expected p50=50.5, p95=95.05
    for i in range(1, 101):
        summary = _summary("f", stuck_count=1, max_lag_s=float(i))
        record_cursor_lag_sample(db, summary, now=now - timedelta(minutes=i))

    baseline = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)

    assert baseline.sample_count == 100
    assert baseline.confident is True
    assert 50.0 <= baseline.rolling_median_lag_s <= 51.0
    assert 94.0 <= baseline.rolling_p95_lag_s <= 96.0


def test_load_baseline_excludes_samples_outside_window(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    # 50 fresh samples + 50 ancient ones
    for i in range(50):
        record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=10.0), now=now - timedelta(minutes=i))
    for i in range(50):
        record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=1000.0), now=now - timedelta(days=30 + i))

    baseline = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)
    # Only the fresh 50 should count, p95 should reflect them, not the
    # ancient outliers.
    assert baseline.sample_count == 50
    assert baseline.rolling_p95_lag_s == 10.0


def test_load_baseline_unconfident_below_min_samples(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    for i in range(10):
        record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=10.0), now=now - timedelta(minutes=i))

    baseline = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)
    assert baseline.sample_count == 10
    assert baseline.confident is False


def test_load_family_baselines_batches_reads(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    for family in ("a", "b"):
        for i in range(60):
            record_cursor_lag_sample(
                db,
                _summary(family, stuck_count=1, max_lag_s=10.0 if family == "a" else 100.0),
                now=now - timedelta(minutes=i),
            )

    baselines = load_family_baselines(db, ["a", "b", "missing"], window_days=7, min_samples=50, now=now)
    assert baselines["a"].rolling_p95_lag_s == 10.0
    assert baselines["b"].rolling_p95_lag_s == 100.0
    assert baselines["missing"].confident is False


def test_load_baseline_handles_single_sample_gracefully(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=42.0))
    baseline = load_family_baseline(db, "f", window_days=7, min_samples=1)
    assert baseline.sample_count == 1
    assert baseline.rolling_median_lag_s == 42.0
    assert baseline.rolling_p95_lag_s == 42.0
    assert baseline.confident is True


def test_load_baseline_reads_ops_tier_from_archive_tiers(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    for i in range(60):
        record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=10.0 + i), now=now - timedelta(minutes=i))

    baseline = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)

    assert baseline.sample_count == 60
    assert baseline.confident is True
    assert 39.0 <= baseline.rolling_median_lag_s <= 41.0
    assert 65.0 <= baseline.rolling_p95_lag_s <= 67.0


def test_gc_drops_ops_tier_samples_from_archive_tiers(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    summary = _summary("f", stuck_count=1, max_lag_s=10.0)
    record_cursor_lag_sample(db, summary, now=now - timedelta(days=30))
    record_cursor_lag_sample(db, summary, now=now)

    removed = gc_cursor_lag_samples(db, retention_days=14, now=now)

    assert removed == 1
    with sqlite3.connect(str(db.with_name("ops.db"))) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM cursor_lag_samples").fetchone()[0]
    assert remaining == 1


def test_baseline_restart_safe_persists_history(tmp_path: Path) -> None:
    # AC #6: stopping the daemon for N hours, restarting, and observing
    # one health-loop tick reproduces the same baseline.
    db = tmp_path / "index.db"
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
    # Pre-restart: 60 samples accumulated.
    for i in range(60):
        record_cursor_lag_sample(db, _summary("f", stuck_count=1, max_lag_s=10.0), now=now - timedelta(hours=i))
    pre_restart = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)

    # "Restart" = re-open the read without any in-memory state — the
    # baseline is unchanged because the substrate persisted.
    post_restart = load_family_baseline(db, "f", window_days=7, min_samples=50, now=now)
    assert pre_restart == post_restart
    assert post_restart.confident is True
