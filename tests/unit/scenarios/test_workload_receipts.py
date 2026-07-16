"""Contracts for shared workload declarations and resource receipts."""

from __future__ import annotations

import pytest

from polylogue.scenarios.workload import (
    BudgetAggregation,
    BudgetMeasure,
    BudgetSemantics,
    BudgetVerdict,
    MeasurementScope,
    WorkloadBudget,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
    evaluate_budgets,
    exact_session_actions_canary_spec,
)
from polylogue.schemas.workload_tiers import WorkloadScaleTier, WorkloadSelectivityTier


def _spec() -> WorkloadEnvelopeSpec:
    return WorkloadEnvelopeSpec(
        workload_id="schema-profile:chatgpt:v1:scale-10x",
        family_id="schema-profile-generation",
        version=1,
        inputs=(
            WorkloadInputRef(
                input_id="chatgpt:v1:scale-10x",
                corpus_id="corpus:sha256:abc",
                profile_id="workload-profile:sha256:def",
                package_ref="chatgpt:v1",
                scale_tier="10x",
                seed=42,
                distribution_refs=("field-distributions-v1", "archive-composition-v1"),
            ),
        ),
        phases=("observe", "replay", "quiescent"),
        measurement_scope=MeasurementScope.CGROUP,
        concurrency=1,
        admission="memory-headroom",
        quiescence_ms=500,
        budgets=(
            WorkloadBudget(
                measure=BudgetMeasure.PEAK_RSS_BYTES,
                maximum=512 * 1024 * 1024,
                semantics=BudgetSemantics.REGRESSION_GATE,
            ),
            WorkloadBudget(
                measure=BudgetMeasure.CANCELLATION_LATENCY_MS,
                maximum=1_000,
                semantics=BudgetSemantics.MEASURE_ONLY,
            ),
        ),
    )


def test_workload_receipt_distinguishes_zero_from_unavailable_and_is_stable() -> None:
    spec = _spec()
    phases = (
        WorkloadPhaseObservation(
            name="observe",
            wall_ms=250,
            peak_rss_bytes=256 * 1024 * 1024,
            anon_bytes=0,
            file_cache_bytes=128 * 1024 * 1024,
            read_io_bytes=1_000,
            unavailable=("peak_pss_bytes", "cancellation_latency_ms"),
        ),
        WorkloadPhaseObservation(
            name="replay",
            unavailable=("peak_rss_bytes", "cancellation_latency_ms"),
        ),
        WorkloadPhaseObservation(
            name="quiescent",
            current_rss_bytes=64 * 1024 * 1024,
            cleanup_complete=True,
            quiescent=True,
        ),
    )
    verdicts = evaluate_budgets(spec, phases)
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:abc",
        runtime_id="python:3.13",
        archive_id="archive:one",
        generation_id="generation:seven",
        frame_id="frame:latest",
        phases=phases,
        evidence_refs=("artifact:resource-samples.jsonl",),
        cleanup_complete=True,
    )

    assert verdicts[0].verdict is BudgetVerdict.PASS
    assert verdicts[1].verdict is BudgetVerdict.MEASUREMENT_UNAVAILABLE
    assert phases[0].to_payload()["anon_bytes"] == 0
    assert "peak_pss_bytes" not in phases[0].to_payload()
    assert receipt.receipt_id == receipt.receipt_id
    assert receipt.to_payload()["receipt_id"] == receipt.receipt_id


def test_physical_budget_cannot_be_expressed_as_a_semantic_result_cap() -> None:
    with pytest.raises(ValueError, match="cannot narrow logical result"):
        WorkloadEnvelopeSpec(
            workload_id="query:exact-session-actions",
            family_id="query",
            version=1,
            inputs=(WorkloadInputRef(input_id="archive:test", corpus_id="archive:test"),),
            phases=("query",),
            semantic_result="first-100",
        )


def test_phase_rejects_measurement_reported_as_both_present_and_unavailable() -> None:
    with pytest.raises(ValueError, match="both observed and unavailable"):
        WorkloadPhaseObservation(
            name="query",
            peak_rss_bytes=1,
            unavailable=("peak_rss_bytes",),
        )


def test_budget_aggregation_is_dimension_aware_and_may_target_one_phase() -> None:
    spec = WorkloadEnvelopeSpec(
        workload_id="query:actions",
        family_id="query",
        version=1,
        inputs=(WorkloadInputRef(input_id="archive:test", corpus_id="archive:test"),),
        phases=("rank", "render"),
        budgets=(
            WorkloadBudget(BudgetMeasure.WALL_MS, maximum=12),
            WorkloadBudget(BudgetMeasure.PEAK_RSS_BYTES, maximum=8),
            WorkloadBudget(BudgetMeasure.WALL_MS, maximum=5, phase="render"),
        ),
    )

    results = evaluate_budgets(
        spec,
        (
            WorkloadPhaseObservation(name="rank", wall_ms=7, peak_rss_bytes=8),
            WorkloadPhaseObservation(name="render", wall_ms=5, peak_rss_bytes=4),
        ),
    )

    assert [result.observed for result in results] == [12, 8, 5]
    assert [result.aggregation for result in results] == [
        BudgetAggregation.SUM,
        BudgetAggregation.MAXIMUM,
        BudgetAggregation.SUM,
    ]


def test_exact_session_actions_canary_uses_shared_tier_and_receipt_contract() -> None:
    spec = exact_session_actions_canary_spec(
        profile_id="workload-profile:sha256:profile",
        archive_id="archive:sha256:corpus",
    )
    workload_input = spec.inputs[0]

    assert workload_input.scale_tier == WorkloadScaleTier.CI_ACTIVATION.value
    assert workload_input.selectivity_tier == WorkloadSelectivityTier.EXACT_ONE.value
    passing = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:head",
        runtime_id="sqlite:test",
        archive_id="archive:sha256:corpus",
        generation_id="generation:test",
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="seed"),
            WorkloadPhaseObservation(name="query", sqlite_vm_steps=49_900),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=True, quiescent=True),
        ),
        cleanup_complete=True,
    )
    exceeded = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:mutant",
        runtime_id="sqlite:test",
        archive_id="archive:sha256:corpus",
        generation_id="generation:test",
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="seed"),
            WorkloadPhaseObservation(name="query", sqlite_vm_steps=50_001),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=True, quiescent=True),
        ),
        cleanup_complete=True,
    )

    assert passing.budget_results[0].verdict is BudgetVerdict.PASS
    assert exceeded.budget_results[0].verdict is BudgetVerdict.EXCEEDED
    assert exceeded.spec.semantic_result == "complete"
