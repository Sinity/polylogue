"""Hermetic steady-state SLO projection tests (polylogue-8jg9.3).

The mutation-proof cases exercise the real projection over existing
``catch_up_cycle`` event envelopes and cursor-lag models. Removing the event
source reuse or changing the idle/stalled predicate makes these tests fail.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import polylogue.operations.slo as slo_storage
from polylogue.core.enums import SloSampleLabel
from polylogue.daemon.cursor_lag_status import CursorLagSummary
from polylogue.daemon.events import emit_catch_up_cycle, query_daemon_events
from polylogue.daemon.slo import (
    SloVerdict,
    backlog_window_from_event,
    project_backlog_windows,
    record_backlog_window_sample,
    reduce_slo_samples,
    slo_status_info,
)
from polylogue.operations.slo import (
    ArchiveSloSample,
    list_slo_samples,
    record_slo_sample,
)


def _cycle(
    *,
    backlog_start: int,
    backlog_end: int,
    discovered: int,
    attempted: int = 0,
    ingested: int = 0,
    skipped: int = 0,
    quarantine_count: int = 0,
    duration_ms: float = 1000.0,
    operation_id: str = "cycle-1",
    phase: str = "end",
    timestamp_ms: int = 1_000,
) -> dict[str, object]:
    return {
        "kind": "catch_up_cycle",
        "operation_id": operation_id,
        "ts": datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(),
        "payload": {
            "phase": phase,
            "backlog_start": backlog_start,
            "backlog_end": backlog_end,
            "discovered": discovered,
            "attempted": attempted,
            "ingested": ingested,
            "skipped": skipped,
            "quarantine_count": quarantine_count,
            "duration_ms": duration_ms,
        },
    }


def test_idle_backlog_is_not_reported_as_stalled() -> None:
    windows = project_backlog_windows([_cycle(backlog_start=7, backlog_end=7, discovered=0)])

    assert len(windows) == 1
    assert windows[0].verdict is SloVerdict.IDLE
    assert windows[0].offered_work == 0
    assert windows[0].drain_rate_per_s == 0.0


def test_offered_work_without_drain_is_stalled() -> None:
    windows = project_backlog_windows([_cycle(backlog_start=7, backlog_end=7, discovered=3, attempted=3)])

    assert windows[0].verdict is SloVerdict.STALLED
    assert windows[0].offered_work == 3
    assert windows[0].drained_work == 0


def test_status_reuses_event_and_cursor_lag_sources() -> None:
    status = slo_status_info(
        cursor_lag=CursorLagSummary(stuck_file_count=2, degraded_file_count=1),
        events=[_cycle(backlog_start=5, backlog_end=3, discovered=2, attempted=2, ingested=2)],
    )

    assert status.verdict is SloVerdict.DRAINING
    assert status.backlog == 3
    assert status.offered_work == 2
    assert status.drained_work == 2
    assert status.cursor_lag_stuck_file_count == 2
    assert status.cursor_lag_degraded_file_count == 1


def test_bulk_import_marker_suppresses_ingest_slo() -> None:
    events: list[dict[str, object]] = [
        {"kind": "bulk_import_started", "payload": {}},
        _cycle(backlog_start=4, backlog_end=4, discovered=4, attempted=4),
    ]

    windows = project_backlog_windows(events)
    assert windows[0].verdict is SloVerdict.SUPPRESSED
    assert windows[0].bulk_import_suppressed is True


def test_newer_start_cycle_overrides_stale_completed_verdict() -> None:
    events = [
        _cycle(backlog_start=3, backlog_end=0, discovered=3, ingested=3, operation_id="old", timestamp_ms=1_000),
        _cycle(
            backlog_start=5,
            backlog_end=5,
            discovered=5,
            operation_id="current",
            phase="start",
            timestamp_ms=2_000,
        ),
    ]

    status = slo_status_info(cursor_lag=CursorLagSummary(), events=list(reversed(events)))

    assert status.verdict is SloVerdict.ACTIVE
    assert status.latest_window is not None
    assert status.latest_window.operation_id == "current"
    assert status.latest_window.in_progress is True
    assert status.latest_window.observed_at_ms == 2_000


def test_newer_end_cycle_closes_its_matching_start_before_status_projection() -> None:
    events = [
        _cycle(
            backlog_start=4,
            backlog_end=4,
            discovered=4,
            operation_id="completed",
            phase="start",
            timestamp_ms=1_000,
        ),
        _cycle(
            backlog_start=4,
            backlog_end=0,
            discovered=4,
            ingested=4,
            operation_id="completed",
            timestamp_ms=2_000,
        ),
    ]

    status = slo_status_info(cursor_lag=CursorLagSummary(), events=list(reversed(events)))

    assert status.verdict is SloVerdict.HEALTHY
    assert status.latest_window is not None
    assert status.latest_window.operation_id == "completed"
    assert status.latest_window.in_progress is False
    assert status.latest_window.observed_at_ms == 2_000


def test_event_reader_timestamp_is_preserved_without_wall_clock_fallback(workspace_env: dict[str, Path]) -> None:
    del workspace_env
    emit_catch_up_cycle(
        operation_id="timestamped",
        phase="end",
        backlog_start=2,
        backlog_end=1,
        discovered=1,
        attempted=1,
        skipped=0,
        ingested=1,
        quarantine_count=0,
        errors_by_kind={},
        cursor_before=None,
        cursor_after=None,
        duration_ms=10.0,
        stage_timings_s={},
        repair=None,
    )

    event = query_daemon_events(kind="catch_up_cycle", limit=1)[0]
    window = backlog_window_from_event(event)

    assert window is not None
    assert isinstance(event["ts"], str)
    expected_ms = int(datetime.fromisoformat(event["ts"].replace("Z", "+00:00")).timestamp() * 1000)
    assert window.observed_at_ms == expected_ms


def test_live_tail_sampling_adds_latency_but_bulk_sampling_does_not(tmp_path: Path) -> None:
    db = tmp_path / "ops.db"
    live = backlog_window_from_event(_cycle(backlog_start=3, backlog_end=2, discovered=1, ingested=1))
    suppressed = backlog_window_from_event(
        _cycle(backlog_start=3, backlog_end=2, discovered=1, ingested=1),
        bulk_import_suppressed=True,
    )
    assert live is not None and suppressed is not None

    live_ids = record_backlog_window_sample(db, live)
    suppressed_ids = record_backlog_window_sample(db, suppressed)

    with sqlite3.connect(db) as conn:
        labels = [row[0] for row in conn.execute("SELECT label FROM slo_samples ORDER BY rowid")]
    assert len(live_ids) == 4
    assert len(suppressed_ids) == 3
    assert labels.count(SloSampleLabel.INGEST_LATENCY.value) == 1


def test_reducer_is_level_only_on_cold_start_and_derives_trend_fields() -> None:
    cold = reduce_slo_samples(
        [ArchiveSloSample("cold", SloSampleLabel.BACKLOG.value, "archive", 8.0, 1_000, None, None, {})]
    )
    assert cold.level == 8.0
    assert cold.p95 == 8.0
    assert cold.confident is False
    assert cold.slope_per_s is None
    assert cold.eta_s is None

    samples = tuple(
        ArchiveSloSample(
            f"sample-{index}",
            SloSampleLabel.BACKLOG.value,
            "archive",
            float(value),
            1_000 + index * 1_000,
            None,
            None,
            {},
        )
        for index, value in enumerate((10, 6, 2))
    )
    reduction = reduce_slo_samples(samples, target_rate=2.0)
    assert reduction.confident is True
    assert reduction.level == 2.0
    assert reduction.slope_per_s == -4.0
    assert reduction.eta_s == 0.5
    assert reduction.burn_rate == 1.0
    assert reduction.p50 == 6.0
    assert reduction.p95 == 2.4


def test_slo_samples_self_heal_and_retention_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ops.db"
    monkeypatch.setattr(slo_storage, "SLO_SAMPLE_ROW_CAP", 2)
    record_slo_sample(
        db,
        label=SloSampleLabel.BACKLOG,
        value=9,
        observed_at_ms=1_000,
        sample_id="old",
    )
    record_slo_sample(
        db,
        label=SloSampleLabel.BACKLOG,
        value=3,
        observed_at_ms=30 * 24 * 60 * 60 * 1000 + 2_000,
        sample_id="new-1",
    )
    record_slo_sample(
        db,
        label=SloSampleLabel.BACKLOG,
        value=2,
        observed_at_ms=30 * 24 * 60 * 60 * 1000 + 3_000,
        sample_id="new-2",
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'slo_samples'").fetchone() is not None
    rows = list_slo_samples(db, label=SloSampleLabel.BACKLOG)

    assert [row.sample_id for row in rows] == ["new-2", "new-1"]
