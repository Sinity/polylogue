"""Stratified rebuild-cost model benchmark (polylogue-623q follow-up).

This is not a micro-benchmark of one hot function -- it is the "predict a
full-corpus rebuild's wall-clock without running one" tool the operator asked
for. ``tests/infra/rebuild_cost_model.py`` carries the real logic (stratify
the raw population by origin x byte-size decile, synthesize a representative
sample per stratum, drive it through the REAL rebuild engine, extrapolate).

The CI-safe test below exercises the model end to end against a small,
deterministic subset of the bundled population snapshot (fast: a handful of
tiny synthetic raws per stratum, not the full 41k-raw population) and asserts
the model machinery itself is sound (every stratum measured, no zero
denominators, wall-clock aggregates without error).

To reproduce the full population projection and its calibration against the
one real measured rebuild (4h20m / 41,363 raws / 92.4 GiB), run:

    pytest tests/benchmarks/test_rebuild_cost_model.py::test_full_population_projection \\
      --benchmark-enable -p no:xdist -o "addopts=" -v -s --run-cost-model-full

That variant is opt-in (skipped by default: it makes ~40 real rebuild passes
-- two per stratum, for the fixed/marginal regression -- tens of seconds
each) and prints a stratum-by-stratum report plus the predicted/actual
calibration ratio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.rebuild_cost_model import (
    CALIBRATION_WALL_S,
    POPULATION_SNAPSHOT,
    Stratum,
    run_cost_model,
)


def test_default_sample_n_scales_inversely_with_size() -> None:
    from polylogue.core.enums import Provider
    from tests.infra.rebuild_cost_model import default_sample_n

    whale = Stratum("whale", Provider.CODEX, count=10_000, total_bytes=10_000 * 5_000_000)
    tiny = Stratum("tiny", Provider.CODEX, count=10_000, total_bytes=10_000 * 500)
    assert default_sample_n(whale) < default_sample_n(tiny)
    assert default_sample_n(whale) >= 1
    assert default_sample_n(tiny) <= tiny.count


@pytest.mark.benchmark
def test_stratified_model_end_to_end_small(tmp_path: Path) -> None:
    """CI-fast smoke test: the model machinery works on a tiny synthetic subset."""
    from polylogue.core.enums import Provider

    strata = [
        Stratum("codex-small", Provider.CODEX, count=6, total_bytes=6 * 20_000),
        Stratum("claude-code-small", Provider.CLAUDE_CODE, count=6, total_bytes=6 * 20_000),
    ]
    predicted = run_cost_model(tmp_path, strata=strata, sample_sizes_override=(2, 4))
    assert len(predicted.measurements) == 2
    for measurement in predicted.measurements:
        assert measurement.n1 == 2
        assert measurement.n2 == 4
        assert measurement.regression_valid
        assert measurement.wall_s1 > 0.0
        assert measurement.wall_s2 > 0.0
        assert measurement.predicted_wall_s >= 0.0
    assert predicted.total_raws == 12
    assert predicted.total_predicted_wall_s >= 0.0


@pytest.mark.benchmark
def test_full_population_projection(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--run-cost-model-full", default=False):
        pytest.skip("opt-in: pass --run-cost-model-full to measure ~20 real rebuild passes")
    predicted = run_cost_model(tmp_path, strata=POPULATION_SNAPSHOT)
    report = predicted.to_report()
    print("\n" + report)
    assert predicted.total_raws == sum(s.count for s in POPULATION_SNAPSHOT)
    # Sanity floor: the model should land within an order of magnitude of the
    # one real measured rebuild, not merely "some positive number". A wider
    # miss means the model's assumptions (synthesized-payload shape, or the
    # seconds-per-raw scaling itself) don't hold and should not be trusted
    # for evaluating future changes -- see the module docstring's acceptance
    # criterion.
    ratio = predicted.calibration_ratio()
    assert 0.1 < ratio < 10.0, (
        f"predicted/actual={ratio:.2f} is outside a sane order-of-magnitude band; "
        f"predicted={predicted.total_predicted_wall_s / 60:.1f}min actual={CALIBRATION_WALL_S / 60:.1f}min"
    )
