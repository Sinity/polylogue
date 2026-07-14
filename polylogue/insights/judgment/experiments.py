"""Experiment analysis projection over stc definitions and cohort relations (rxdo.9.10, mechanism J).

This module owns NO experiment identity, table, or lifecycle. It projects a
versioned, typed ``ExperimentDefinition`` assertion (produced by the stc
lifecycle lane -- not yet landed as of this writing) plus outcome receipts
into one analysis artifact. A pair of cohorts alone is an observational
comparison unless the experiment lifecycle proves assignment AND exposure.

``ExperimentDefinitionLike`` is a *structural* ``Protocol``, not a new stored
type: it lets this projection be written and exercised against the exact
shape stc is expected to produce before that assertion kind exists
(tracked upstream; this module has no dependency on stc's concrete class).
Any object satisfying the protocol -- from a PROMPT_EVAL definition, a
curriculum A/B definition, or a routing/harness comparison -- shares this one
path; there is no per-consumer special-case executor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    subject_ref: str
    arm: str
    metric_value: float
    observed_at_ms: int


class ExperimentDefinitionLike(Protocol):
    """Structural shape this projection consumes from an stc ``ExperimentDefinition``.

    Declared as read-only ``@property`` members (not plain attribute
    annotations) so an immutable ``@dataclass(frozen=True)`` fixture -- the
    shape a real stc-produced definition is expected to take -- satisfies the
    protocol without mypy demanding a settable attribute.
    """

    @property
    def definition_ref(self) -> str: ...
    @property
    def definition_version(self) -> int: ...
    @property
    def arms(self) -> tuple[str, ...]: ...
    @property
    def paired(self) -> bool: ...
    @property
    def confirmatory(self) -> bool: ...
    @property
    def frame_ref(self) -> str: ...
    @property
    def exclusions(self) -> tuple[str, ...]: ...
    @property
    def has_assignment_receipts(self) -> bool: ...
    @property
    def has_exposure_receipts(self) -> bool: ...
    @property
    def metric_ref(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ExperimentAnalysisResult:
    analysis_ref: str
    definition_ref: str
    definition_version: int
    design: Literal["causal", "observational"]
    claim_class: Literal["confirmatory", "exploratory"]
    paired: bool
    arm_means: Mapping[str, float]
    paired_mean_difference: float | None
    n_by_arm: Mapping[str, int]
    excluded_count: int


def analyze_experiment(
    definition: ExperimentDefinitionLike,
    outcomes: Sequence[OutcomeRecord],
    *,
    excluded_subject_refs: Sequence[str] = (),
    post_exposure_definition_change: bool = False,
) -> ExperimentAnalysisResult:
    """Lower a definition + outcome receipts into one analysis artifact.

    ``design`` is ``"causal"`` only when the definition carries BOTH
    assignment and exposure receipts; an otherwise-identical cohort pair with
    only outcomes renders ``"observational"`` no matter how suggestive the
    difference. ``claim_class`` downgrades to ``"exploratory"`` whenever
    design is observational OR the definition changed after exposures already
    ran (post-hoc reinterpretation guard) -- confirmatory status is earned by
    the lifecycle, not asserted.
    """

    excluded = set(excluded_subject_refs)
    kept = [outcome for outcome in outcomes if outcome.subject_ref not in excluded]

    design: Literal["causal", "observational"] = (
        "causal" if (definition.has_assignment_receipts and definition.has_exposure_receipts) else "observational"
    )
    claim_class: Literal["confirmatory", "exploratory"] = (
        "confirmatory"
        if (definition.confirmatory and design == "causal" and not post_exposure_definition_change)
        else "exploratory"
    )

    by_arm: dict[str, list[float]] = {arm: [] for arm in definition.arms}
    for record in kept:
        by_arm.setdefault(record.arm, []).append(record.metric_value)

    arm_means = {arm: (sum(values) / len(values) if values else math.nan) for arm, values in by_arm.items()}
    n_by_arm = {arm: len(values) for arm, values in by_arm.items()}

    paired_diff: float | None = None
    if definition.paired and len(definition.arms) == 2:
        arm_a, arm_b = definition.arms
        subjects_a = {o.subject_ref: o.metric_value for o in kept if o.arm == arm_a}
        subjects_b = {o.subject_ref: o.metric_value for o in kept if o.arm == arm_b}
        shared = subjects_a.keys() & subjects_b.keys()
        if shared:
            paired_diff = sum(subjects_a[subject] - subjects_b[subject] for subject in shared) / len(shared)

    analysis_ref = f"experiment-analysis:{definition.definition_ref}@v{definition.definition_version}"
    return ExperimentAnalysisResult(
        analysis_ref=analysis_ref,
        definition_ref=definition.definition_ref,
        definition_version=definition.definition_version,
        design=design,
        claim_class=claim_class,
        paired=definition.paired,
        arm_means=arm_means,
        paired_mean_difference=paired_diff,
        n_by_arm=n_by_arm,
        excluded_count=len(excluded),
    )


__all__ = ["ExperimentAnalysisResult", "ExperimentDefinitionLike", "OutcomeRecord", "analyze_experiment"]
