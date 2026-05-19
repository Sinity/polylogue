"""Slow-vs-stuck classification helper coverage (#1246).

These tests pin the boundary behaviour of
``polylogue.daemon.live_ingest_attempt_progress`` independently of the
larger ``daemon_status_payload`` surface so a regression in the
heuristic surfaces directly.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polylogue.daemon.live_ingest_attempt_progress import (
    SLOW_MIN_SAMPLES,
    STUCK_AFTER_S,
    classify_attempt_progress,
    completed_total_time_samples,
    compute_slow_threshold_s,
)


class TestClassifyAttemptProgress:
    def test_non_running_status_is_always_healthy(self) -> None:
        # Completed/failed attempts are not classified as slow or stuck;
        # the heuristic is meaningless once the attempt has terminated.
        for status in ("completed", "failed", "cancelled"):
            assert (
                classify_attempt_progress(
                    status=status,
                    updated_age_s=STUCK_AFTER_S * 10,
                    total_time_s=10_000.0,
                    slow_threshold_s=1.0,
                )
                == "healthy"
            )

    def test_running_below_stuck_threshold_within_slow_budget_is_healthy(self) -> None:
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=5.0,
                total_time_s=10.0,
                slow_threshold_s=60.0,
            )
            == "healthy"
        )

    def test_running_above_stuck_threshold_is_stuck(self) -> None:
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=STUCK_AFTER_S,
                total_time_s=5.0,
                slow_threshold_s=60.0,
            )
            == "stuck"
        )

    def test_stuck_wins_over_slow_when_both_conditions_meet(self) -> None:
        # Stuck is the operator-actionable signal; do not downgrade it to
        # "slow" just because the wall-clock also exceeds the p95.
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=STUCK_AFTER_S + 1,
                total_time_s=10_000.0,
                slow_threshold_s=60.0,
            )
            == "stuck"
        )

    def test_running_below_stuck_above_slow_is_slow(self) -> None:
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=2.0,
                total_time_s=90.0,
                slow_threshold_s=60.0,
            )
            == "slow"
        )

    def test_slow_threshold_none_disables_slow_classification(self) -> None:
        # When the archive has too few completed attempts the heuristic
        # returns ``None`` and the rollup must not flag running attempts
        # as slow on rumour.
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=2.0,
                total_time_s=10_000.0,
                slow_threshold_s=None,
            )
            == "healthy"
        )

    def test_unparseable_updated_age_does_not_flip_to_stuck(self) -> None:
        # The status query may legitimately return ``None`` for
        # ``updated_age_s`` (e.g. when the timestamp could not be parsed).
        # In that case we report "healthy" rather than guessing.
        assert (
            classify_attempt_progress(
                status="running",
                updated_age_s=None,
                total_time_s=10.0,
                slow_threshold_s=60.0,
            )
            == "healthy"
        )


class TestComputeSlowThreshold:
    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        c = sqlite3.connect(tmp_path / "attempts.db")
        c.execute(
            """
            CREATE TABLE live_ingest_attempt (
                attempt_id TEXT PRIMARY KEY,
                started_at TEXT,
                completed_at TEXT,
                status TEXT
            )
            """
        )
        return c

    def _insert(
        self,
        conn: sqlite3.Connection,
        *,
        attempt_id: str,
        duration_s: float,
        status: str = "completed",
    ) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = start + timedelta(seconds=duration_s)
        conn.execute(
            "INSERT INTO live_ingest_attempt VALUES (?, ?, ?, ?)",
            (attempt_id, start.isoformat(), end.isoformat(), status),
        )

    def test_missing_table_returns_none(self, tmp_path: Path) -> None:
        empty = sqlite3.connect(tmp_path / "empty.db")
        assert compute_slow_threshold_s(empty) is None
        assert completed_total_time_samples(empty) == []

    def test_below_min_samples_returns_none(self, conn: sqlite3.Connection) -> None:
        for i in range(SLOW_MIN_SAMPLES - 1):
            self._insert(conn, attempt_id=f"a{i}", duration_s=float(i + 1))
        conn.commit()
        assert compute_slow_threshold_s(conn) is None

    def test_at_min_samples_returns_p95(self, conn: sqlite3.Connection) -> None:
        for i in range(SLOW_MIN_SAMPLES):
            self._insert(conn, attempt_id=f"a{i}", duration_s=float((i + 1) * 10))
        conn.commit()
        threshold = compute_slow_threshold_s(conn)
        assert threshold is not None
        # Durations are 10, 20, 30, 40, 50. p95 of [10..50] interpolated
        # is at position 0.95 * 4 = 3.8 → between idx 3 (40) and 4 (50)
        # → 40 + 0.8 * 10 = 48.
        assert threshold == pytest.approx(48.0, rel=1e-3)

    def test_failed_attempts_are_excluded_from_baseline(self, conn: sqlite3.Connection) -> None:
        # Failed attempts often have very small durations (early-abort
        # paths). Including them in the p95 would bias the slow cutoff
        # downward and over-classify healthy long-running attempts as
        # slow.
        for i in range(SLOW_MIN_SAMPLES):
            self._insert(conn, attempt_id=f"good{i}", duration_s=100.0)
        for i in range(SLOW_MIN_SAMPLES):
            self._insert(conn, attempt_id=f"bad{i}", duration_s=0.1, status="failed")
        conn.commit()
        threshold = compute_slow_threshold_s(conn)
        assert threshold == pytest.approx(100.0, rel=1e-3)
