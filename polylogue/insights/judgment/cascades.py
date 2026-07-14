"""Judge cascades: cheap agent screens route to sparse operator gold (rxdo.9.15, mechanism O).

Routing policy over calibration (mechanism L) and comparative verdicts
(mechanism K): agent judges screen judgment demand cheaply and in parallel;
disagreement, uncertainty, non-decisive verdicts, or declared stakes route to
the operator, whose verdicts double as calibration gold. The existing
accept/reject lifecycle (37t.12) is unchanged as the durable-claim acceptance
gate -- this module only decides who looks at a comparative verdict next.
Agent judgments always remain candidates; promotion stays gated
(:func:`polylogue.storage.sqlite.archive_tiers.user_write.upsert_assertion`'s
promotion chokepoint already enforces this at the storage layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.calibration import CalibrationReport
from polylogue.insights.judgment.types import ComparativeJudgment

RouteDecision: TypeAlias = Literal["agent_screen_pass", "operator_review"]

DEFAULT_MIN_AGREEMENT_RATE = 0.8
DEFAULT_MIN_GOLD_OVERLAP = 5

_NON_DECISIVE_VERDICTS: frozenset[ComparativeVerdict] = frozenset(
    {ComparativeVerdict.ABSTAIN, ComparativeVerdict.INSUFFICIENT_EVIDENCE, ComparativeVerdict.INCOMPARABLE}
)


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    """The routing receipt: where this verdict goes next, and why."""

    route: RouteDecision
    reason: str
    calibration_known: bool


def route_judgment(
    judgment: ComparativeJudgment,
    calibration: CalibrationReport | None,
    *,
    agents_disagreed: bool = False,
    quota_selected: bool = False,
    high_stakes: bool = False,
    min_agreement_rate: float = DEFAULT_MIN_AGREEMENT_RATE,
    min_gold_overlap: int = DEFAULT_MIN_GOLD_OVERLAP,
) -> CascadeDecision:
    """Decide whether one agent-screen verdict stops here or routes to the operator.

    ``calibration`` MUST be looked up for the judgment's exact
    ``(actor_ref, execution_context_id, dimension)`` stratum -- passing a
    calibration report from any other stratum (e.g. a sibling execution
    context) is a caller bug this function cannot detect; it can only refuse
    to grant a pass when calibration is literally ``None`` or has no gold
    overlap, which is exactly what "unseen context" looks like from here.
    """

    if isinstance(judgment.verdict, ComparativeVerdict) and judgment.verdict in _NON_DECISIVE_VERDICTS:
        return CascadeDecision(
            route="operator_review",
            reason=f"non-decisive verdict: {judgment.verdict.value}",
            calibration_known=calibration is not None,
        )
    if high_stakes:
        return CascadeDecision(
            route="operator_review", reason="declared high-stakes item", calibration_known=calibration is not None
        )
    if agents_disagreed:
        return CascadeDecision(
            route="operator_review",
            reason="agent judges disagreed on this item",
            calibration_known=calibration is not None,
        )
    if quota_selected:
        return CascadeDecision(
            route="operator_review",
            reason="elicitation exploration quota selected this item for gold",
            calibration_known=calibration is not None,
        )
    if calibration is None or calibration.agreement_rate is None:
        return CascadeDecision(
            route="operator_review",
            reason="no calibration known for this actor/execution-context/dimension stratum",
            calibration_known=False,
        )
    if calibration.n_gold_overlap < min_gold_overlap:
        return CascadeDecision(
            route="operator_review",
            reason=f"insufficient gold coverage ({calibration.n_gold_overlap} < {min_gold_overlap})",
            calibration_known=True,
        )
    if calibration.agreement_rate < min_agreement_rate:
        return CascadeDecision(
            route="operator_review",
            reason=f"agreement rate {calibration.agreement_rate:.2f} below threshold {min_agreement_rate:.2f}",
            calibration_known=True,
        )
    return CascadeDecision(
        route="agent_screen_pass", reason="calibrated actor, covered context, decisive verdict", calibration_known=True
    )


__all__ = [
    "CascadeDecision",
    "DEFAULT_MIN_AGREEMENT_RATE",
    "DEFAULT_MIN_GOLD_OVERLAP",
    "RouteDecision",
    "route_judgment",
]
