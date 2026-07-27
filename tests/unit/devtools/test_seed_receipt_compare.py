"""Contract tests for the like-for-like seed/incident receipt comparison (polylogue-b054.1.1.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.seed_receipt_compare import (
    ComparisonTargetKind,
    ReceiptComparisonTarget,
    compare_receipts,
    default_seed_comparison_targets,
    identity_mismatches,
    load_receipt,
    main,
)
from polylogue.scenarios.workload import (
    BudgetMeasure,
    BudgetVerdict,
    MeasurementScope,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)

_GIB = 1024**3


def _receipt(
    *,
    wall_ms: float,
    peak_pss_bytes: int | None,
    status: WorkloadRunStatus = WorkloadRunStatus.SUCCEEDED,
    workload_id: str = "verify:pytest focused",
    input_id: str = "pytest-selection:sha256:same-selection",
) -> dict[str, object]:
    """Build a real WorkloadReceipt payload (the exact shape devtools verify emits)."""
    spec = WorkloadEnvelopeSpec(
        workload_id=workload_id,
        family_id="verify-pytest",
        version=1,
        inputs=(WorkloadInputRef(input_id=input_id),),
        phases=("execute", "quiescent"),
        measurement_scope=MeasurementScope.PROCESS_TREE,
    )
    execute_unavailable = () if peak_pss_bytes is not None else ("peak_pss_bytes",)
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=status,
        build_id="git:deadbeef",
        runtime_id="python:3.12.0",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(
                name="execute",
                wall_ms=wall_ms,
                peak_pss_bytes=peak_pss_bytes,
                unavailable=execute_unavailable,
            ),
            WorkloadPhaseObservation(name="quiescent", quiescent=True),
        ),
    )
    return dict(receipt.to_payload())


def test_identical_receipts_are_like_for_like_and_meet_default_targets() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)
    candidate = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is True
    assert comparison["identity_mismatches"] == []
    # Same wall time as baseline is NOT a 2x speedup, so the default target is unmet.
    wall_result = next(r for r in comparison["results"] if r["measure"] == "wall_ms")
    assert wall_result["verdict"] == "exceeded"
    assert wall_result["ratio"] == pytest.approx(1.0)
    assert wall_result["blocker"] is not None
    assert wall_result["follow_up_ref"] == "polylogue-b054.1.1.2"
    assert comparison["targets_met"] is False


def test_real_two_x_speedup_and_low_memory_meets_default_targets() -> None:
    baseline = _receipt(wall_ms=20_000, peak_pss_bytes=2 * _GIB)
    candidate = _receipt(wall_ms=9_000, peak_pss_bytes=1 * _GIB)

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is True
    assert comparison["targets_met"] is True
    wall_result = next(r for r in comparison["results"] if r["measure"] == "wall_ms")
    assert wall_result["verdict"] == "pass"
    assert wall_result["ratio"] == pytest.approx(0.45)
    pss_result = next(r for r in comparison["results"] if r["measure"] == "peak_pss_bytes")
    assert pss_result["verdict"] == "pass"


def test_peak_pss_regression_is_flagged_with_blocker_and_follow_up() -> None:
    baseline = _receipt(wall_ms=20_000, peak_pss_bytes=2 * _GIB)
    candidate = _receipt(wall_ms=9_000, peak_pss_bytes=4 * _GIB)

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is True
    assert comparison["targets_met"] is False
    pss_result = next(r for r in comparison["results"] if r["measure"] == "peak_pss_bytes")
    assert pss_result["verdict"] == "exceeded"
    assert "peak_pss_bytes" in pss_result["blocker"]
    assert pss_result["follow_up_ref"] == "polylogue-b054.1.1.2"


def test_missing_measure_is_measurement_unavailable_not_a_fabricated_pass() -> None:
    baseline = _receipt(wall_ms=20_000, peak_pss_bytes=2 * _GIB)
    candidate = _receipt(wall_ms=9_000, peak_pss_bytes=None)

    comparison = compare_receipts(baseline, candidate)

    pss_result = next(r for r in comparison["results"] if r["measure"] == "peak_pss_bytes")
    assert pss_result["verdict"] == "measurement-unavailable"
    assert pss_result["candidate_value"] is None
    assert pss_result["blocker"] is None
    # A measurement gap does not, by itself, block the overall comparison.
    assert comparison["targets_met"] is True


def test_mismatched_workload_identity_is_not_like_for_like() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, workload_id="verify:pytest focused")
    candidate = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, workload_id="verify:pytest seed-testmon")

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is False
    assert any("workload_id" in mismatch for mismatch in comparison["identity_mismatches"])
    # A mismatched pair cannot license a target verdict.
    assert comparison["targets_met"] is False


def test_mismatched_input_selection_is_not_like_for_like() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, input_id="pytest-selection:sha256:aaa")
    candidate = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, input_id="pytest-selection:sha256:bbb")

    mismatches = identity_mismatches(baseline, candidate)

    assert any("inputs" in mismatch for mismatch in mismatches)


def test_unclean_baseline_run_is_not_like_for_like_even_with_matching_identity() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, status=WorkloadRunStatus.FAILED)
    candidate = _receipt(wall_ms=5_000, peak_pss_bytes=1 * _GIB)

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is False
    assert comparison["baseline_status"] == "failed"
    assert comparison["targets_met"] is False


def test_unclean_candidate_run_is_not_like_for_like() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)
    candidate = _receipt(wall_ms=5_000, peak_pss_bytes=1 * _GIB, status=WorkloadRunStatus.INTERRUPTED)

    comparison = compare_receipts(baseline, candidate)

    assert comparison["like_for_like"] is False
    assert comparison["candidate_status"] == "interrupted"


def test_custom_targets_override_the_default_seed_comparison_contract() -> None:
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)
    candidate = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)
    lenient_targets = (
        ReceiptComparisonTarget(
            measure=BudgetMeasure.WALL_MS,
            kind=ComparisonTargetKind.MAX_RATIO_OF_BASELINE,
            limit=1.5,
            follow_up_ref="polylogue-example-follow-up",
        ),
    )

    comparison = compare_receipts(baseline, candidate, lenient_targets)

    assert comparison["targets_met"] is True
    assert len(comparison["results"]) == 1


def test_default_seed_comparison_targets_are_2x_wall_and_3gib_peak_pss() -> None:
    targets = default_seed_comparison_targets()

    wall_target = next(t for t in targets if t.measure is BudgetMeasure.WALL_MS)
    pss_target = next(t for t in targets if t.measure is BudgetMeasure.PEAK_PSS_BYTES)
    assert wall_target.kind is ComparisonTargetKind.MAX_RATIO_OF_BASELINE
    assert wall_target.limit == pytest.approx(0.5)
    assert pss_target.kind is ComparisonTargetKind.MAX_ABSOLUTE
    assert pss_target.limit == pytest.approx(3 * _GIB)


def test_load_receipt_unwraps_a_postmortem_json_payload(tmp_path: Path) -> None:
    receipt = _receipt(wall_ms=1_000, peak_pss_bytes=1 * _GIB)
    postmortem = tmp_path / "postmortem.json"
    postmortem.write_text(json.dumps({"diagnosis": None, "workload_receipt": receipt}), encoding="utf-8")

    loaded = load_receipt(postmortem)

    assert loaded["status"] == "succeeded"
    assert loaded == receipt


def test_load_receipt_accepts_a_bare_workload_receipt_payload(tmp_path: Path) -> None:
    receipt = _receipt(wall_ms=1_000, peak_pss_bytes=1 * _GIB)
    bare = tmp_path / "receipt.json"
    bare.write_text(json.dumps(receipt), encoding="utf-8")

    assert load_receipt(bare) == receipt


def test_cli_exits_zero_when_like_for_like_and_targets_met(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_receipt(wall_ms=20_000, peak_pss_bytes=2 * _GIB)), encoding="utf-8")
    candidate.write_text(json.dumps(_receipt(wall_ms=9_000, peak_pss_bytes=1 * _GIB)), encoding="utf-8")

    exit_code = main(["--baseline", str(baseline), "--candidate", str(candidate), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["targets_met"] is True


def test_cli_exits_one_when_a_target_is_unmet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)), encoding="utf-8")
    candidate.write_text(json.dumps(_receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)), encoding="utf-8")

    exit_code = main(["--baseline", str(baseline), "--candidate", str(candidate)])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "EXCEEDED" in out
    assert "blocker" in out
    assert "polylogue-b054.1.1.2" in out


def test_cli_exits_two_when_receipts_are_not_like_for_like(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, workload_id="verify:pytest focused")),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB, workload_id="verify:pytest seed-testmon")),
        encoding="utf-8",
    )

    exit_code = main(["--baseline", str(baseline), "--candidate", str(candidate)])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "like-for-like: NO" in out


def test_evaluate_budgets_verdict_vocabulary_is_reused_verbatim() -> None:
    """The comparison must reuse BudgetVerdict, not invent a parallel vocabulary."""
    baseline = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)
    candidate = _receipt(wall_ms=10_000, peak_pss_bytes=1 * _GIB)

    comparison = compare_receipts(baseline, candidate)

    verdicts = {result["verdict"] for result in comparison["results"]}
    assert verdicts <= {verdict.value for verdict in BudgetVerdict}
