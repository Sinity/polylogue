"""Tests for the comparative-judgment value shape (rxdo.9.11, mechanism K)."""

from __future__ import annotations

import pytest

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.types import (
    NON_DIRECTED_VERDICTS,
    ComparativeJudgment,
    JudgeIdentity,
    all_items,
    decompose_to_pairwise,
    undirected_pair_kind,
)

_JUDGE = JudgeIdentity(actor_ref="user:local", execution_context_id="ctx-1")


def _pairwise(verdict: ComparativeVerdict, *, judgment_id: str = "j1") -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id=judgment_id,
        items=("finding:a", "finding:b"),
        dimension="correctness",
        verdict=verdict,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )


def test_pairwise_prefer_left_requires_two_items() -> None:
    with pytest.raises(ValueError, match="exactly 2 items"):
        ComparativeJudgment(
            judgment_id="j",
            items=("finding:a", "finding:b", "finding:c"),
            dimension="correctness",
            verdict=ComparativeVerdict.PREFER_LEFT,
            judge=_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )


def test_ordering_verdict_must_be_a_permutation_of_items() -> None:
    with pytest.raises(ValueError, match="permutation"):
        ComparativeJudgment(
            judgment_id="j",
            items=("finding:a", "finding:b", "finding:c"),
            dimension="correctness",
            verdict=("finding:a", "finding:b"),  # missing finding:c
            judge=_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )


def test_items_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ComparativeJudgment(
            judgment_id="j",
            items=("finding:a", "finding:a"),
            dimension="correctness",
            verdict=ComparativeVerdict.TIE,
            judge=_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )


def test_needs_at_least_two_items() -> None:
    with pytest.raises(ValueError, match="at least 2 items"):
        ComparativeJudgment(
            judgment_id="j",
            items=("finding:a",),
            dimension="correctness",
            verdict=ComparativeVerdict.ABSTAIN,
            judge=_JUDGE,
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
        )


_SORTED_NON_DIRECTED_VERDICTS: list[ComparativeVerdict] = sorted(NON_DIRECTED_VERDICTS, key=lambda v: v.value)


@pytest.mark.parametrize("verdict", _SORTED_NON_DIRECTED_VERDICTS)
def test_non_directed_verdicts_decompose_to_zero_edges(verdict: ComparativeVerdict) -> None:
    """AC: abstain/insufficient-evidence/tie/incomparable never become weak preferences."""

    judgment = _pairwise(verdict)
    assert decompose_to_pairwise(judgment) == ()


def test_prefer_left_decomposes_to_one_directed_edge() -> None:
    judgment = _pairwise(ComparativeVerdict.PREFER_LEFT)
    components = decompose_to_pairwise(judgment)
    assert len(components) == 1
    assert components[0].winner_ref == "finding:a"
    assert components[0].loser_ref == "finding:b"


def test_prefer_right_decomposes_to_one_directed_edge() -> None:
    judgment = _pairwise(ComparativeVerdict.PREFER_RIGHT)
    components = decompose_to_pairwise(judgment)
    assert len(components) == 1
    assert components[0].winner_ref == "finding:b"
    assert components[0].loser_ref == "finding:a"


def test_nwise_ordering_decomposes_plackett_luce_style_to_all_implied_pairs() -> None:
    judgment = ComparativeJudgment(
        judgment_id="j-nwise",
        items=("finding:a", "finding:b", "finding:c", "finding:d"),
        dimension="usefulness",
        verdict=("finding:a", "finding:b", "finding:c", "finding:d"),
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )
    components = decompose_to_pairwise(judgment)
    # C(4, 2) = 6 implied pairs
    pairs = {(c.winner_ref, c.loser_ref) for c in components}
    assert pairs == {
        ("finding:a", "finding:b"),
        ("finding:a", "finding:c"),
        ("finding:a", "finding:d"),
        ("finding:b", "finding:c"),
        ("finding:b", "finding:d"),
        ("finding:c", "finding:d"),
    }


def test_tie_and_incomparable_remain_distinct_pair_kinds() -> None:
    tie = _pairwise(ComparativeVerdict.TIE, judgment_id="j-tie")
    incomparable = _pairwise(ComparativeVerdict.INCOMPARABLE, judgment_id="j-inc")
    assert undirected_pair_kind(tie) == ("finding:a", "finding:b")
    assert undirected_pair_kind(incomparable) == ("finding:a", "finding:b")
    # Both resolve to the same sorted pair shape, but callers distinguish by
    # judgment.verdict identity -- this helper only reports "these two were
    # compared and yielded a non-directed pairwise verdict".
    assert tie.verdict is not incomparable.verdict


def test_abstain_and_insufficient_evidence_have_no_undirected_pair_kind() -> None:
    """Distinguishes non-directed-but-still-a-relation (tie/incomparable) from no-relation-recorded."""

    abstain = _pairwise(ComparativeVerdict.ABSTAIN)
    assert undirected_pair_kind(abstain) is None


def test_all_items_dedupes_and_sorts_across_a_judgment_set() -> None:
    j1 = _pairwise(ComparativeVerdict.PREFER_LEFT, judgment_id="j1")
    j2 = ComparativeJudgment(
        judgment_id="j2",
        items=("finding:b", "finding:c"),
        dimension="correctness",
        verdict=ComparativeVerdict.TIE,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )
    assert all_items([j1, j2]) == ("finding:a", "finding:b", "finding:c")
