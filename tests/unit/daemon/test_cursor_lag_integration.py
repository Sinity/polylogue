"""End-to-end integration tests for the cursor-lag stack (#1232 + #1349).

These tests drive ``_check_cursor_lag_medium`` against real SQLite cursor
state plus a synthetic archive ops sample history so the static
ladder, the sample writer, the GC, and the anomaly band can be
verified together rather than in isolation. The hardcoded ACs from
issue #1349 are pinned here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from polylogue.daemon.cursor_lag_alert import reset_default_dedup_state as reset_static_dedup
from polylogue.daemon.cursor_lag_anomaly import (
    reset_default_dedup_state as reset_anomaly_dedup,
)
from polylogue.daemon.cursor_lag_baseline import gc_cursor_lag_samples, record_cursor_lag_sample
from polylogue.daemon.cursor_lag_status import (
    CursorLagFamilySummary,
    CursorLagSummary,
    cursor_lag_summary_info,
)
from polylogue.daemon.health import HealthSeverity, _check_cursor_lag_medium
from tests.infra.frozen_clock import FrozenClock

# Pin ``datetime.now`` everywhere the cursor-lag stack reads it so the
# test's "now" anchor and the production code's "now" anchor coincide.
# Without this, sub-second drift between test setup and the health check
# could shift lag calculations near threshold edges (#1300).
pytestmark = pytest.mark.frozen_clock_modules(
    "polylogue.daemon.health",
    "polylogue.daemon.cursor_lag_status",
    "polylogue.daemon.cursor_lag_baseline",
    "polylogue.daemon.cursor_lag_alert",
    "polylogue.daemon.cursor_lag_anomaly",
)


@pytest.fixture(autouse=True)
def _reset_dedup_state() -> Iterator[None]:
    """Independence: every test starts AND ends with empty dedup state.

    These tests drive ``_check_cursor_lag_medium`` / the anomaly layer
    against the process-wide ``_DEDUP_STATE`` singletons in
    ``cursor_lag_alert`` and ``cursor_lag_anomaly``. Resetting only on
    setup leaks the last test's stuck-family entries into whatever test
    runs next in the randomized session order: a later ``_run_medium_checks``
    then emits spurious ``cursor_lag[...]`` / ``cursor_lag_anomaly[...]``
    resolution alerts for the leaked family, breaking the daemon
    health-inventory and HTTP-contract pins. Reset on teardown too so the
    singletons never escape this module.
    """
    reset_static_dedup()
    reset_anomaly_dedup()
    yield
    reset_static_dedup()
    reset_anomaly_dedup()


def _seed_live_cursor(
    db: Path,
    rows: list[tuple[str, int, int, int, int, str]],
) -> None:
    """Create + populate the native ``ingest_cursor`` table in the sibling
    ``ops.db`` with the minimal column set the projection reads.

    ``rows`` = ``(source_path, byte_size, byte_offset, failure_count,
    excluded, updated_at_iso)``. The archive cursor-lag projection
    (``polylogue/daemon/cursor_lag_status.py``) reads ``ingest_cursor``
    columns ``source_path, stat_size, byte_offset, failure_count,
    excluded, updated_at_ms`` from ops.db.
    """
    from datetime import datetime as _dt

    ops_db = db.with_name("ops.db")
    ops_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ops_db))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_cursor (
                source_path TEXT PRIMARY KEY,
                stat_size INTEGER,
                byte_offset INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                excluded INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL
            )
            """
        )
        native_rows = [
            (
                source_path,
                byte_size,
                byte_offset,
                failure_count,
                excluded,
                int(_dt.fromisoformat(updated_at_iso).timestamp() * 1000),
            )
            for (source_path, byte_size, byte_offset, failure_count, excluded, updated_at_iso) in rows
        ]
        conn.executemany(
            """
            INSERT INTO ingest_cursor (source_path, stat_size, byte_offset,
                                       failure_count, excluded, updated_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            native_rows,
        )
        conn.commit()
    finally:
        conn.close()


def _isolated_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg_text: str = "",
) -> Path:
    """Build an isolated archive_root + polylogue.toml + XDG_DATA_HOME so
    the periodic health check (which calls ``db_path()``) reads only this
    test's state."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    data_home = tmp_path / "data"
    (data_home / "polylogue").mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    # The periodic health check resolves the active index db to
    # ``archive_root/index.db`` and reads cursor state from the sibling
    # ``ops.db``; return that base path so seeding and reads agree.
    return archive_root / "index.db"


def _sample_history(
    db: Path,
    family: str,
    *,
    samples: int,
    max_lag_s: float,
    now: datetime,
    spacing: timedelta = timedelta(minutes=5),
) -> None:
    """Backfill N archive ops sample rows for one family."""
    summary = CursorLagSummary(
        tracked_file_count=1,
        stuck_file_count=1,
        idle_file_count=0,
        max_lag_s=max_lag_s,
        family_summaries=[CursorLagFamilySummary(family=family, stuck_file_count=1, max_lag_s=max_lag_s)],
    )
    for i in range(samples):
        record_cursor_lag_sample(db, summary, now=now - spacing * (i + 1))


def test_lag_sample_writes_do_not_block_health_loop_on_database_lock(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    db = tmp_path / "locked.db"
    ops_db = db.with_name("ops.db")
    summary = CursorLagSummary(
        tracked_file_count=1,
        stuck_file_count=1,
        idle_file_count=0,
        max_lag_s=120,
        family_summaries=[CursorLagFamilySummary(family="codex-session", stuck_file_count=1, max_lag_s=120)],
    )
    blocker = sqlite3.connect(str(ops_db))
    try:
        blocker.execute("CREATE TABLE lock_holder (id INTEGER PRIMARY KEY)")
        blocker.execute("BEGIN EXCLUSIVE")

        assert record_cursor_lag_sample(db, summary, now=frozen_clock.now()) == 0
        assert gc_cursor_lag_samples(db, retention_days=14, now=frozen_clock.now()) == 0
    finally:
        blocker.rollback()
        blocker.close()


# ---------------------------------------------------------------------------
# AC #4: anomaly is additive — disabling it leaves static behavior intact
# ---------------------------------------------------------------------------


def test_anomaly_disabled_leaves_static_ladder_behavior_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600
default_critical_s = 7200
anomaly_enabled = false
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            # File is behind (byte_offset < byte_size) and updated 120s ago →
            # stuck with lag 120s → above default_warning_s=60.
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )

    alerts = _check_cursor_lag_medium()
    check_names = sorted(a.check_name for a in alerts)
    # The static cursor_lag alert fires; no anomaly alert ever fires.
    assert "cursor_lag[unknown]" in check_names
    assert not any(name.startswith("cursor_lag_anomaly") for name in check_names)


# ---------------------------------------------------------------------------
# AC #1: each tick writes one sample row per stuck family + GC bounds the table
# ---------------------------------------------------------------------------


def test_periodic_tick_records_sample_for_each_stuck_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600
default_critical_s = 7200
anomaly_enabled = true
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
            ("/x/b.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=300)).isoformat()),
        ],
    )

    _check_cursor_lag_medium()

    with sqlite3.connect(str(db.with_name("ops.db"))) as conn:
        row = conn.execute("SELECT family, COUNT(*) FROM cursor_lag_samples GROUP BY family").fetchall()
    # Both files bucket to the same family ("unknown") so one row per
    # family per tick.
    assert len(row) == 1
    assert row[0][1] == 1


# ---------------------------------------------------------------------------
# AC #2 + #3: confidence + absolute-floor gates hold end-to-end
# ---------------------------------------------------------------------------


def test_unconfident_baseline_does_not_emit_anomaly_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
# Static defaults relaxed so the static layer doesn't fire on the same lag
default_warning_s = 10000
default_error_s = 20000
default_critical_s = 30000

anomaly_enabled = true
anomaly_baseline_window_days = 7
anomaly_baseline_min_samples = 50
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 5.0
anomaly_min_lag_s = 10
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )
    # Only 10 prior samples → not confident at min_samples=50.
    _sample_history(db, "unknown", samples=10, max_lag_s=10.0, now=now)

    alerts = _check_cursor_lag_medium()
    assert not any(a.check_name.startswith("cursor_lag_anomaly") for a in alerts)


def test_anomaly_fires_when_baseline_confident_and_ratio_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
default_warning_s = 10000  # static layer silent for this test
default_error_s = 20000
default_critical_s = 30000

anomaly_enabled = true
anomaly_baseline_window_days = 7
anomaly_baseline_min_samples = 5
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 5.0
anomaly_min_lag_s = 10
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            # 120s lag against a 10s baseline = 12x → ERROR
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )
    _sample_history(db, "unknown", samples=10, max_lag_s=10.0, now=now)

    alerts = _check_cursor_lag_medium()
    anomaly = [a for a in alerts if a.check_name.startswith("cursor_lag_anomaly")]
    assert len(anomaly) == 1
    assert anomaly[0].severity == HealthSeverity.ERROR


def test_below_absolute_floor_does_not_emit_anomaly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
default_warning_s = 10000
default_error_s = 20000
default_critical_s = 30000

anomaly_enabled = true
anomaly_baseline_window_days = 7
anomaly_baseline_min_samples = 5
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 5.0
anomaly_min_lag_s = 30
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            # 5s lag with 0.1s baseline = 50x — but 5s < min_lag_s=30 → silent
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=5)).isoformat()),
        ],
    )
    _sample_history(db, "unknown", samples=10, max_lag_s=0.1, now=now)

    alerts = _check_cursor_lag_medium()
    assert not any(a.check_name.startswith("cursor_lag_anomaly") for a in alerts)


# ---------------------------------------------------------------------------
# AC #5: dedup is independent — static and anomaly fire together
# ---------------------------------------------------------------------------


def test_static_and_anomaly_alerts_can_fire_simultaneously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600
default_critical_s = 7200

anomaly_enabled = true
anomaly_baseline_window_days = 7
anomaly_baseline_min_samples = 5
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 5.0
anomaly_min_lag_s = 30
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            # 120s lag → static WARN (above 60) AND anomaly ERROR (12x of 10s)
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )
    _sample_history(db, "unknown", samples=10, max_lag_s=10.0, now=now)

    alerts = _check_cursor_lag_medium()
    severities_by_check = {a.check_name: a.severity for a in alerts}
    assert severities_by_check.get("cursor_lag[unknown]") == HealthSeverity.WARNING
    assert severities_by_check.get("cursor_lag_anomaly[unknown]") == HealthSeverity.ERROR


# ---------------------------------------------------------------------------
# Status decoration — baseline state surfaces on the projection
# ---------------------------------------------------------------------------


def test_status_projection_decorates_family_summaries_with_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    cfg_text = """
[health.cursor_lag]
anomaly_enabled = true
anomaly_baseline_window_days = 7
anomaly_baseline_min_samples = 5
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 5.0
anomaly_min_lag_s = 30
"""
    db = _isolated_archive(tmp_path, monkeypatch, cfg_text)
    now = frozen_clock.now()
    _seed_live_cursor(
        db,
        rows=[
            ("/x/a.jsonl", 1000, 500, 0, 0, (now - timedelta(seconds=120)).isoformat()),
        ],
    )
    _sample_history(db, "unknown", samples=10, max_lag_s=10.0, now=now)

    summary = cursor_lag_summary_info(db, now=now)
    assert summary.family_summaries
    fs = summary.family_summaries[0]
    assert fs.baseline.sample_count == 10
    assert fs.baseline.confident is True
    assert fs.baseline.rolling_p95_lag_s == 10.0
    assert fs.baseline.current_multiplier >= 11.0  # 120 / 10 = 12.0
    assert fs.baseline.anomaly_severity == "error"
