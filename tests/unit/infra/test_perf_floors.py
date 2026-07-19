"""Tests for the nightly perf-floors runner (polylogue-196x).

Covers the pure comparison/serialization logic exhaustively (fast, no I/O)
plus one end-to-end smoke run of the full curated measurement pipeline in
``--quick`` shape, so a broken import or signature drift in any of the four
underlying production surfaces (revision-backfill census/replay,
``refresh_action_pairs``, ``compute_latency_percentiles``) fails loudly here
rather than silently in the nightly workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.benchmarks.perf_floors import (
    FloorComparison,
    compare_to_floors,
    format_delta_table,
    load_floors,
    run_perf_floor_set,
    save_floors,
)


def _sample_report(*, throughput_value: float = 100.0, latency_value: float = 10.0) -> dict[str, Any]:
    return {
        "generated_at": "2026-07-19T00:00:00+00:00",
        "git_sha": "abc123",
        "machine": {"hostname": "test-host"},
        "metrics": {
            "throughput_metric": {
                "value": throughput_value,
                "unit": "raws/s",
                "direction": "higher_is_better",
                "detail": {},
            },
            "latency_metric": {
                "value": latency_value,
                "unit": "ms",
                "direction": "lower_is_better",
                "detail": {},
            },
        },
    }


def _floors_with(*, latency_tolerance_pct: float = 20.0) -> dict[str, Any]:
    return {
        "default_tolerance_pct": 20.0,
        "metrics": {
            "throughput_metric": {
                "baseline": 100.0,
                "unit": "raws/s",
                "direction": "higher_is_better",
                "tolerance_pct": 20.0,
            },
            "latency_metric": {
                "baseline": 10.0,
                "unit": "ms",
                "direction": "lower_is_better",
                "tolerance_pct": latency_tolerance_pct,
            },
        },
    }


def test_higher_is_better_regresses_only_on_a_drop_beyond_tolerance() -> None:
    floors = _floors_with()

    within = {c.name: c for c in compare_to_floors(_sample_report(throughput_value=85.0), floors)}  # -15%
    assert within["throughput_metric"].regressed is False

    beyond = {c.name: c for c in compare_to_floors(_sample_report(throughput_value=70.0), floors)}  # -30%
    assert beyond["throughput_metric"].regressed is True

    # A rise in a higher_is_better metric is never a regression.
    improved = {c.name: c for c in compare_to_floors(_sample_report(throughput_value=500.0), floors)}
    assert improved["throughput_metric"].regressed is False


def test_lower_is_better_regresses_only_on_a_rise_beyond_tolerance() -> None:
    floors = _floors_with()

    within = {c.name: c for c in compare_to_floors(_sample_report(latency_value=11.5), floors)}  # +15%
    assert within["latency_metric"].regressed is False

    beyond = {c.name: c for c in compare_to_floors(_sample_report(latency_value=14.0), floors)}  # +40%
    assert beyond["latency_metric"].regressed is True

    # A drop in a lower_is_better metric (faster) is never a regression.
    improved = {c.name: c for c in compare_to_floors(_sample_report(latency_value=0.5), floors)}
    assert improved["latency_metric"].regressed is False


def test_metric_without_a_recorded_floor_is_flagged_new_not_regressed() -> None:
    report = _sample_report()
    report["metrics"]["brand_new_metric"] = {
        "value": 1.0,
        "unit": "ms",
        "direction": "lower_is_better",
        "detail": {},
    }
    floors = _floors_with()

    comparisons = {c.name: c for c in compare_to_floors(report, floors)}
    assert comparisons["brand_new_metric"].is_new is True
    assert comparisons["brand_new_metric"].regressed is False


def test_per_metric_tolerance_overrides_the_file_default() -> None:
    floors = _floors_with(latency_tolerance_pct=5.0)
    report = _sample_report(latency_value=10.8)  # +8%

    comparisons = {c.name: c for c in compare_to_floors(report, floors)}
    # +8% breaches the metric's own 5% tolerance even though the file default is 20%.
    assert comparisons["latency_metric"].regressed is True
    assert comparisons["latency_metric"].tolerance_pct == 5.0


def test_save_and_load_floors_round_trips(tmp_path: Path) -> None:
    floors_path = tmp_path / "floors.json"
    save_floors(
        floors_path,
        _sample_report(),
        default_tolerance_pct=25.0,
        measured_under_load=True,
        note="test note",
    )

    loaded = load_floors(floors_path)
    assert loaded["default_tolerance_pct"] == 25.0
    assert loaded["measured_under_load"] is True
    assert loaded["note"] == "test note"
    assert loaded["metrics"]["throughput_metric"]["baseline"] == 100.0
    assert loaded["metrics"]["throughput_metric"]["tolerance_pct"] == 25.0
    assert loaded["metrics"]["latency_metric"]["baseline"] == 10.0

    # File is valid, stable JSON.
    json.loads(floors_path.read_text())


def test_load_floors_missing_file_returns_empty_metrics(tmp_path: Path) -> None:
    loaded = load_floors(tmp_path / "does-not-exist.json")
    assert loaded == {"metrics": {}}


def test_format_delta_table_marks_regressions_and_new_metrics() -> None:
    comparisons = [
        FloorComparison(
            name="a_metric",
            baseline=100.0,
            candidate=50.0,
            unit="raws/s",
            direction="higher_is_better",
            delta_pct=-50.0,
            tolerance_pct=20.0,
            regressed=True,
        ),
        FloorComparison(
            name="b_metric",
            baseline=0.0,
            candidate=5.0,
            unit="ms",
            direction="lower_is_better",
            delta_pct=0.0,
            tolerance_pct=0.0,
            regressed=False,
            is_new=True,
        ),
    ]
    table = format_delta_table(comparisons)
    assert "a_metric" in table
    assert "REGRESSED" in table
    assert "b_metric" in table
    assert "NEW" in table


def test_run_perf_floor_set_quick_measures_every_curated_metric(tmp_path: Path) -> None:
    """End-to-end smoke run in --quick shape over every measurement group.

    Exercises the real production surfaces (revision_backfill census/replay,
    ``refresh_action_pairs``, ``compute_latency_percentiles``) so a signature
    or import drift in any of them fails here instead of silently in CI.
    Only shape/type/non-negativity is asserted -- wall-clock values are
    host-variable and covered by the floors mechanism, not this test.
    """
    report = run_perf_floor_set(workdir=tmp_path, quick=True)

    assert report["report_version"] == 1
    assert report["tool"] == "bench perf-floors"
    assert report["quick"] is True
    assert isinstance(report["machine"], dict)
    assert report["machine"]["cpu_count"]

    expected_metrics = {
        "census_small_raws_per_s",
        "census_chain_revisions_per_s",
        "replay_sessions_per_min",
        "action_pairs_refresh_mean_ms",
        "action_pairs_refresh_p95_ms",
        "query_search_summaries_p50_ms",
        "query_search_summaries_p95_ms",
        "query_list_summaries_p50_ms",
        "query_list_summaries_p95_ms",
    }
    assert expected_metrics.issubset(report["metrics"].keys())
    for name, entry in report["metrics"].items():
        assert entry["value"] >= 0.0, name
        assert entry["direction"] in ("higher_is_better", "lower_is_better")
        assert entry["unit"]
