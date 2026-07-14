"""Tests for judge cascades (rxdo.9.15, mechanism O)."""

from __future__ import annotations

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.calibration import CalibrationKey, CalibrationReport
from polylogue.insights.judgment.cascades import route_judgment
from polylogue.insights.judgment.types import ComparativeJudgment, JudgeIdentity

_JUDGE = JudgeIdentity(actor_ref="agent:sonnet", execution_context_id="ctx-a")


def _judgment(verdict: ComparativeVerdict) -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id="j1",
        items=("finding:a", "finding:b"),
        dimension="quality",
        verdict=verdict,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )


def _good_calibration() -> CalibrationReport:
    return CalibrationReport(
        key=CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="quality"),
        n_gold_overlap=10,
        agreement_rate=0.95,
        tie_rate=0.0,
        incomparable_rate=0.0,
        abstain_rate=0.0,
        insufficient_evidence_rate=0.0,
        n_total_verdicts=20,
    )


def test_non_decisive_verdicts_always_route_to_operator() -> None:
    for verdict in (
        ComparativeVerdict.ABSTAIN,
        ComparativeVerdict.INSUFFICIENT_EVIDENCE,
        ComparativeVerdict.INCOMPARABLE,
    ):
        decision = route_judgment(_judgment(verdict), _good_calibration())
        assert decision.route == "operator_review"


def test_unseen_execution_context_never_inherits_a_confident_pass() -> None:
    """AC: an unseen execution context never inherits a confident pass."""

    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), calibration=None)
    assert decision.route == "operator_review"
    assert not decision.calibration_known


def test_low_calibration_routes_to_operator() -> None:
    poor = CalibrationReport(
        key=CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="quality"),
        n_gold_overlap=10,
        agreement_rate=0.4,
        tie_rate=0.0,
        incomparable_rate=0.0,
        abstain_rate=0.0,
        insufficient_evidence_rate=0.0,
        n_total_verdicts=20,
    )
    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), poor)
    assert decision.route == "operator_review"
    assert decision.calibration_known


def test_insufficient_gold_coverage_routes_to_operator_even_with_high_agreement() -> None:
    sparse = CalibrationReport(
        key=CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="quality"),
        n_gold_overlap=1,
        agreement_rate=1.0,
        tie_rate=0.0,
        incomparable_rate=0.0,
        abstain_rate=0.0,
        insufficient_evidence_rate=0.0,
        n_total_verdicts=20,
    )
    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), sparse)
    assert decision.route == "operator_review"


def test_disagreement_routes_to_operator_regardless_of_calibration() -> None:
    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), _good_calibration(), agents_disagreed=True)
    assert decision.route == "operator_review"


def test_quota_selected_routes_to_operator_regardless_of_calibration() -> None:
    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), _good_calibration(), quota_selected=True)
    assert decision.route == "operator_review"


def test_well_calibrated_covered_decisive_verdict_stops_at_agent_screen() -> None:
    """AC: well-calibrated covered cases may stop at the agent screen with a receipt."""

    decision = route_judgment(_judgment(ComparativeVerdict.PREFER_LEFT), _good_calibration())
    assert decision.route == "agent_screen_pass"
    assert decision.calibration_known
    assert decision.reason
