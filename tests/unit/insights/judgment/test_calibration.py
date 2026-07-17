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


def test_order_swapped_items_with_matching_winner_are_treated_as_agreement() -> None:
    """Blinding randomizes left/right placement per judgment; the same real
    winner recorded as PREFER_LEFT under one item order and PREFER_RIGHT
    under the reversed order must agree, not be misread as a raw-label
    mismatch."""

    gold = [_judgment(_GOLD_JUDGE, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "gold-1")]
    candidates = [
        # opposite left/right placement, but items[1] ("finding:a") still wins
        _judgment(_AGENT_CTX_A, ("finding:b", "finding:a"), ComparativeVerdict.PREFER_RIGHT, "a-1"),
    ]
    reports = compute_calibration(candidates, gold)
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate == 1.0


def test_order_swapped_items_with_same_raw_label_but_different_winner_disagree() -> None:
    """Same raw PREFER_LEFT label under reversed item order names the
    opposite winner and must be scored as disagreement."""

    gold = [_judgment(_GOLD_JUDGE, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "gold-1")]
    candidates = [
        # same raw label as gold, but items order is reversed so the winner
        # (items[0] = "finding:b") is actually the opposite of gold's winner
        _judgment(_AGENT_CTX_A, ("finding:b", "finding:a"), ComparativeVerdict.PREFER_LEFT, "a-1"),
    ]
    reports = compute_calibration(candidates, gold)
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate == 0.0


def test_nwise_ordering_gold_agrees_with_pairwise_candidate_same_winner() -> None:
    """A 2-item n-wise ORDERING gold verdict and a pairwise PREFER_LEFT/RIGHT
    candidate verdict on the same item pair are both legal per
    ComparativeJudgment's own validation. When they name the same
    real-world winner they must agree -- not be silently scored as
    disagreement because one is stored as an ordering tuple and the other
    as a bare verdict label."""

    gold = [
        ComparativeJudgment(
            judgment_id="gold-1",
            items=("finding:a", "finding:b"),
            dimension="correctness",
            verdict=("finding:a", "finding:b"),  # n-wise ordering: a beats b
            judge=_GOLD_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )
    ]
    candidates = [
        # pairwise verdict on the same pair, same winner ("finding:a")
        _judgment(_AGENT_CTX_A, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_LEFT, "a-1"),
    ]
    reports = compute_calibration(candidates, gold)
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate == 1.0


def test_nwise_ordering_gold_disagrees_with_pairwise_candidate_different_winner() -> None:
    """Same cross-representation overlap as above, but the pairwise
    candidate names the opposite winner -- must be scored as disagreement,
    not accidentally coerced to agreement by the shared representation."""

    gold = [
        ComparativeJudgment(
            judgment_id="gold-1",
            items=("finding:a", "finding:b"),
            dimension="correctness",
            verdict=("finding:a", "finding:b"),  # n-wise ordering: a beats b
            judge=_GOLD_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )
    ]
    candidates = [
        # pairwise verdict on the same pair, opposite winner ("finding:b")
        _judgment(_AGENT_CTX_A, ("finding:a", "finding:b"), ComparativeVerdict.PREFER_RIGHT, "a-1"),
    ]
    reports = compute_calibration(candidates, gold)
    key = CalibrationKey(actor_ref="agent:sonnet", execution_context_id="ctx-a", dimension="correctness")
    assert reports[key].agreement_rate == 0.0


def test_no_cross_context_pooling_function_is_exposed() -> None:
    """AC: reports refuse unsupported cross-context pooling -- there is no pooling API."""

    import polylogue.insights.judgment.calibration as calibration_module

    assert not hasattr(calibration_module, "pool_calibration")
    assert not hasattr(calibration_module, "pooled_calibration")
