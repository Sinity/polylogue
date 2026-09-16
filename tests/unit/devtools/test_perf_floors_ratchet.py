"""The perf-floor ratchet must refuse floors that can never regress.

A ``lower_is_better`` metric recorded as ``0.0`` is a permanent pass; a floor
whose metric stopped being measured (a route rename munged into a new metric
name) is a permanent pass too. Both were silent before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmarks import perf_floors


def _report(metrics: dict[str, float], *, direction: str = "lower_is_better") -> dict[str, object]:
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "git_sha": "deadbeef",
        "machine": {},
        "metrics": {
            name: {"value": value, "unit": "ms", "direction": direction, "detail": {}}
            for name, value in metrics.items()
        },
    }


def _floors(metrics: dict[str, float]) -> dict[str, object]:
    return {
        "default_tolerance_pct": 10.0,
        "metrics": {
            name: {"baseline": value, "unit": "ms", "direction": "lower_is_better", "tolerance_pct": 10.0}
            for name, value in metrics.items()
        },
    }


def test_a_zero_baseline_is_reported_as_unmeasured_not_as_a_pass() -> None:
    """Anti-vacuity: drop the ``baseline <= 0`` arm and this metric reports ok forever."""
    comparisons = perf_floors.compare_to_floors(_report({"parse_ms": 9000.0}), _floors({"parse_ms": 0.0}))

    (comparison,) = comparisons
    assert comparison.unmeasured_baseline is True
    assert comparison.regressed is False
    assert "UNMEASURED" in perf_floors.format_delta_table(comparisons)


def test_a_recorded_floor_with_no_measured_metric_is_reported_missing() -> None:
    """Anti-vacuity: remove the missing-candidate sweep and a renamed metric retires its floor silently."""
    comparisons = perf_floors.compare_to_floors(_report({"parse_ms_renamed": 10.0}), _floors({"parse_ms": 100.0}))

    missing = [comparison for comparison in comparisons if comparison.missing_candidate]
    assert [comparison.name for comparison in missing] == ["parse_ms"]
    assert "MISSING" in perf_floors.format_delta_table(comparisons)


def test_saving_floors_refuses_a_non_positive_measurement(tmp_path: Path) -> None:
    """Anti-vacuity: delete the guard and ``--update-floors`` ratchets a zero in as the baseline."""
    path = tmp_path / "floors.json"

    with pytest.raises(perf_floors.EmptySampleError):
        perf_floors.save_floors(
            path,
            _report({"parse_ms": 0.0}),
            default_tolerance_pct=10.0,
            measured_under_load=False,
            note=None,
        )

    assert not path.exists()


def test_saving_floors_accepts_a_real_measurement(tmp_path: Path) -> None:
    path = tmp_path / "floors.json"

    perf_floors.save_floors(
        path,
        _report({"parse_ms": 12.5}),
        default_tolerance_pct=10.0,
        measured_under_load=False,
        note=None,
    )

    assert json.loads(path.read_text())["metrics"]["parse_ms"]["baseline"] == 12.5
