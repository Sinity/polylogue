"""Tests for judge calibration (rxdo.9.12, mechanism L)."""

from __future__ import annotations

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.calibration import CalibrationKey, compute_calibration
from polylogue.insights.judgment.types import ComparativeJudgment, JudgeIdentity

_GOLD_JUDGE = JudgeIdentity(actor_ref="user:local", execution_context_id="operator")
_AGENT_CTX_A = JudgeIdentity(actor_ref="agent:sonnet", execution_context_id="ctx-a")
_AGENT_CTX_B = JudgeIdentity(actor_ref="agent:sonnet", execution_context_id="ctx-b")


def _judgment(
    judge: JudgeIdentity, items: tuple[str, str], verdict: ComparativeVerdict, judgment_id: str
) -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id=judgment_id,
        items=items,
        dimension="correctness",
        verdict=verdict,
        judge=judge,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )


def test_same_actor_two_contexts_have_separable_calibration() -> None:
    gold = [
        _judgment(_GOLD_JUDGE, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "gold-1"),
        _judgment(_GOLD_JUDGE, ("finding:c", "finding:d"), ComparativeVerdict.PREFER_RIGHT, "gold-2"),
    ]
    candidates = [
        # ctx-a agrees with gold on both overlap items
        _judgment(_AGENT_CTX_A, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "a-1"),
        _judgment(_AGENT_CTX_A, ("finding:c", "finding:d"), ComparativeVerdict.PREFER_RIGHT, "a-2"),
        # ctx-b disagrees with gold on both overlap items
        _judgment(_AGENT_CTX_B, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_RIGHT, "b-1"),
        _judgment(_AGENT_CTX_B, ("finding:c", "finding:d"), ComparativeVerdict.PREFER_LEFT, "b-2"),
    ]
    reports = compute_calibration(candidates, gold)

    key_a = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    key_b = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-b", dimension="correctness")
    assert reports[key_a].agreement_rate == 1.0
    assert reports[key_b].agreement_rate == 0.0


def test_missing_gold_overlap_is_unknown_not_inherited() -> None:
    gold = [_judgment(_GOLD_JUDGE, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "gold-1")]
    candidates = [
        # no overlap with gold's item set
        _judgment(_AGENT_CTX_A, ("finding:x", "finding:y"), ComparativeVerdict.PREFER_LEFT, "a-1"),
    ]
    reports = compute_calibration(candidates, gold)
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate is None
    assert reports[key].n_gold_overlap == 0


def test_no_gold_at_all_is_unknown() -> None:
    candidates = [_judgment(_AGENT_CTX_A, ("finding:a", "finding:b"), ComparativeVerdict.TIE, "a-1")]
    reports = compute_calibration(candidates, [])
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate is None


def test_verdict_mix_rates_are_preserved() -> None:
    candidates = [
        _judgment(_AGENT_CTX_A, ("finding:a", "finding:b"), ComparativeVerdict.TIE, "a-1"),
        _judgment(_AGENT_CTX_A, ("finding:c", "finding:d"), ComparativeVerdict.INCOMPARABLE, "a-2"),
        _judgment(_AGENT_CTX_A, ("finding:e", "finding:f"), ComparativeVerdict.ABSTAIN, "a-3"),
        _judgment(_AGENT_CTX_A, ("finding:g", "finding:h"), ComparativeVerdict.INSUFFICIENT_EVIDENCE, "a-4"),
    ]
    reports = compute_calibration(candidates, [])
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    report = reports[key]
    assert report.tie_rate == 0.25
    assert report.incomparable_rate == 0.25
    assert report.abstain_rate == 0.25
    assert report.insufficient_evidence_rate == 0.25
    assert report.n_total_verdicts == 4


def test_no_cross_context_pooling_function_is_exposed() -> None:
    """AC: reports refuse unsupported cross-context pooling -- there is no pooling API."""

    import polylogue.insights.judgment.calibration as calibration_module

    assert not hasattr(calibration_module, "pool_calibration")
    assert not hasattr(calibration_module, "pooled_calibration")
