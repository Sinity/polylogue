"""Tests for the experiment analysis projection (rxdo.9.10, mechanism J).

``_StcExperimentDefinition`` fixtures below are local dataclasses matching
the structural ``ExperimentDefinitionLike`` protocol -- standing in for the
stc-produced ``ExperimentDefinition`` assertion that has not landed yet. Two
differently-shaped fixtures (``_prompt_eval_definition`` and
``_routing_harness_definition``) exercise the SAME ``analyze_experiment``
path, proving no per-consumer special-case executor exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.insights.judgment.experiments import OutcomeRecord, analyze_experiment


@dataclass(frozen=True, slots=True)
class _StcExperimentDefinition:
    definition_ref: str
    definition_version: int
    arms: tuple[str, ...]
    paired: bool
    confirmatory: bool
    frame_ref: str
    exclusions: tuple[str, ...]
    has_assignment_receipts: bool
    has_exposure_receipts: bool
    metric_ref: str


def _prompt_eval_definition(*, causal: bool, confirmatory: bool = True) -> _StcExperimentDefinition:
    return _StcExperimentDefinition(
        definition_ref="prompt-eval-v3",
        definition_version=1,
        arms=("control", "treatment"),
        paired=True,
        confirmatory=confirmatory,
        frame_ref="cohort:prompt-eval-frame",
        exclusions=(),
        has_assignment_receipts=causal,
        has_exposure_receipts=causal,
        metric_ref="metric:cost-per-session",
    )


def _routing_harness_definition(*, causal: bool) -> _StcExperimentDefinition:
    return _StcExperimentDefinition(
        definition_ref="routing-harness-v1",
        definition_version=2,
        arms=("sol", "terra"),
        paired=False,
        confirmatory=True,
        frame_ref="cohort:routing-harness-frame",
        exclusions=(),
        has_assignment_receipts=causal,
        has_exposure_receipts=causal,
        metric_ref="metric:latency-p50",
    )


def test_two_different_consumers_share_the_same_analysis_path() -> None:
    outcomes = [
        OutcomeRecord(subject_ref="s1", arm="control", metric_value=1.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s1", arm="treatment", metric_value=0.5, observed_at_ms=2),
    ]
    prompt_eval_result = analyze_experiment(_prompt_eval_definition(causal=True), outcomes)
    routing_result = analyze_experiment(
        _routing_harness_definition(causal=True),
        [
            OutcomeRecord(subject_ref="t1", arm="sol", metric_value=200.0, observed_at_ms=1),
            OutcomeRecord(subject_ref="t2", arm="terra", metric_value=180.0, observed_at_ms=2),
        ],
    )
    assert prompt_eval_result.design == "causal"
    assert routing_result.design == "causal"
    assert prompt_eval_result.analysis_ref != routing_result.analysis_ref


def test_full_assignment_and_exposure_receipts_analyze_end_to_end_with_paired_metrics() -> None:
    """AC: an stc two-arm fixture with assignments/exposures/outcomes reproduces declared paired metrics."""

    outcomes = [
        OutcomeRecord(subject_ref="s1", arm="control", metric_value=10.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s1", arm="treatment", metric_value=6.0, observed_at_ms=2),
        OutcomeRecord(subject_ref="s2", arm="control", metric_value=8.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s2", arm="treatment", metric_value=4.0, observed_at_ms=2),
    ]
    result = analyze_experiment(_prompt_eval_definition(causal=True), outcomes)
    assert result.design == "causal"
    assert result.claim_class == "confirmatory"
    assert result.paired
    # paired diff = mean((10-6), (8-4)) = mean(4, 4) = 4.0
    assert result.paired_mean_difference == 4.0
    assert result.arm_means["control"] == 9.0
    assert result.arm_means["treatment"] == 5.0
    assert result.n_by_arm == {"control": 2, "treatment": 2}


def test_missing_assignment_or_exposure_renders_observational_never_causal() -> None:
    """AC: an otherwise-identical cohort-pair fixture without assignment/exposure
    renders observational and cannot emit a causal claim."""

    outcomes = [
        OutcomeRecord(subject_ref="s1", arm="control", metric_value=10.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s1", arm="treatment", metric_value=6.0, observed_at_ms=2),
    ]
    result = analyze_experiment(_prompt_eval_definition(causal=False), outcomes)
    assert result.design == "observational"
    assert result.claim_class == "exploratory"


def test_post_exposure_definition_change_forces_exploratory() -> None:
    """AC: post-exposure metric changes render exploratory/new-version."""

    outcomes = [
        OutcomeRecord(subject_ref="s1", arm="control", metric_value=10.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s1", arm="treatment", metric_value=6.0, observed_at_ms=2),
    ]
    result = analyze_experiment(
        _prompt_eval_definition(causal=True, confirmatory=True),
        outcomes,
        post_exposure_definition_change=True,
    )
    assert result.design == "causal"  # design is unaffected...
    assert result.claim_class == "exploratory"  # ...but claim class downgrades


def test_excluded_subjects_are_dropped_from_arm_computation() -> None:
    outcomes = [
        OutcomeRecord(subject_ref="s1", arm="control", metric_value=10.0, observed_at_ms=1),
        OutcomeRecord(subject_ref="s2", arm="control", metric_value=1000.0, observed_at_ms=1),
    ]
    result = analyze_experiment(_prompt_eval_definition(causal=True), outcomes, excluded_subject_refs=["s2"])
    assert result.arm_means["control"] == 10.0
    assert result.excluded_count == 1
