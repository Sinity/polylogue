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
from typing import cast

import pytest

from tests.infra.rebuild_cost_model import (
    CALIBRATION_WALL_S,
    POPULATION_SNAPSHOT,
    REAL_RUN_STAGE_PROPORTIONS,
    Stratum,
    bucket_stage_seconds,
    build_stratum_sample_corpus,
    run_cost_model,
    stage_proportions,
)
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def test_default_sample_n_scales_inversely_with_size() -> None:
    from polylogue.core.enums import Provider
    from tests.infra.rebuild_cost_model import default_sample_n

    whale = Stratum("whale", Provider.CODEX, count=10_000, total_bytes=10_000 * 5_000_000)
    tiny = Stratum("tiny", Provider.CODEX, count=10_000, total_bytes=10_000 * 500)
    assert default_sample_n(whale) < default_sample_n(tiny)
    assert default_sample_n(whale) >= 1
    assert default_sample_n(tiny) <= tiny.count


def test_real_run_stage_proportions_sum_to_one() -> None:
    """Sanity floor on the ground-truth constant the fidelity table compares against."""
    assert sum(REAL_RUN_STAGE_PROPORTIONS.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.benchmark
def test_structural_corpus_exercises_cohort_arbitration(tmp_path: Path) -> None:
    """The fidelity fix's core claim: a stratum with chain/ambiguous_fraction
    set actually drives the REAL engine's cohort-arbitration machinery
    (``replay.classify_cohort``/``membership.candidates`` etc.) -- dormant
    with the pre-follow-up singleton-only corpus (polylogue-o56w).
    """
    from polylogue.core.enums import Provider
    from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

    stratum = Stratum(
        "claude-code-small",
        Provider.CLAUDE_CODE,
        count=1000,
        total_bytes=1000 * 4000,
        chain_fraction=0.25,
        ambiguous_fraction=0.25,
    )
    archive_root = tmp_path / "structural-corpus"
    raw_ids = build_stratum_sample_corpus(archive_root, stratum, sample_n=12)
    # 12 * 0.25 = 3 -> 1 ambiguous pair (2 raws); pool=10, 10*0.25=2.5 -> round
    # to 1 chain pair (2 raws); remaining 8 singles. Exact allocation is an
    # implementation detail -- the structural assertion below is what matters.
    assert len(raw_ids) == 12

    import os

    prior = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        receipt_path = write_valid_rebuild_receipt(
            archive_root, archive_root.parent / "schema-inference-gate-receipt.json"
        )
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=archive_root,
                promote=True,
                raw_batch_size=12,
                schema_inference_receipt_path=receipt_path,
            )
        )
    finally:
        if prior is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = prior

    stage_timings_s = receipt.replay.get("stage_timings_s", {})
    assert isinstance(stage_timings_s, dict)
    # The chain triggers the revision-replay cohort path; the ambiguous pair
    # triggers membership arbitration. Both were structurally impossible to
    # reach with a singleton-only corpus (every raw its own unambiguous
    # first-time session).
    assert "replay.classify_cohort" in stage_timings_s
    assert any(key.startswith("membership.") for key in stage_timings_s)
    # The ambiguous pair must genuinely fail to arbitrate a winner.
    quarantined_raw_count = cast("int", receipt.replay.get("quarantined_raw_count", 0))
    assert quarantined_raw_count >= 2

    bucketed = bucket_stage_seconds(receipt.replay, dict(receipt.timings_s))
    proportions = stage_proportions(bucketed)
    # untimed_apply (replay.*/membership.* + everything not directly an
    # index_parsed_write insert) must be non-zero once cohort arbitration
    # actually ran -- this was structurally impossible (always exactly 0)
    # before this follow-up.
    assert proportions["untimed_apply"] > 0.0


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
    # Fidelity floor (polylogue-o56w follow-up): the apply-phase mix must at
    # least SHOW cohort-arbitration cost, not be silently zero the way the
    # pre-follow-up singleton-only corpus always was. This is deliberately a
    # loose floor, not a tight tolerance -- see compare_to_real_run's
    # docstring for why an exact percentage match isn't the bar.
    apply_mix = predicted.population_stage_proportions
    assert apply_mix["untimed_apply"] > 0.0, (
        "harness corpus produced zero untimed-apply (cohort classification/"
        "membership arbitration) work -- it is not structurally representative "
        "of the real rebuild's apply phase; see compare_to_real_run()"
    )
