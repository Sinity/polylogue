"""Tests for ranker:<hash> aggregation over judgment sets (rxdo.9.13, mechanism M)."""

from __future__ import annotations

import pytest

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.rankers import (
    RankerDefinition,
    bradley_terry_mle,
    fit_ranker,
)
from polylogue.insights.judgment.types import ComparativeJudgment, JudgeIdentity

_JUDGE = JudgeIdentity(actor_ref="user:local", execution_context_id="operator")


def _j(
    judgment_id: str, items: tuple[str, str], verdict: ComparativeVerdict, dimension: str = "quality"
) -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id=judgment_id,
        items=items,
        dimension=dimension,
        verdict=verdict,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
    )


def test_bradley_terry_mle_recovers_a_clear_ordering() -> None:
    # a beats b beats c, repeatedly, unambiguously
    wins = {
        ("a", "b"): 8.0,
        ("b", "a"): 2.0,
        ("b", "c"): 8.0,
        ("c", "b"): 2.0,
        ("a", "c"): 9.0,
        ("c", "a"): 1.0,
    }
    scores = bradley_terry_mle(["a", "b", "c"], wins)
    assert scores["a"] is not None and scores["b"] is not None and scores["c"] is not None
    assert scores["a"] > scores["b"] > scores["c"]


def test_bradley_terry_mle_returns_none_for_unobserved_items() -> None:
    scores = bradley_terry_mle(["a", "b", "isolated"], {("a", "b"): 5.0, ("b", "a"): 5.0})
    assert scores["isolated"] is None
    assert scores["a"] is not None


def test_fit_ranker_default_output_is_a_partial_order_exposing_every_condition() -> None:
    judgments = [
        # component 1: a beats b, tie between b and c
        _j("j1", ("a", "b"), ComparativeVerdict.PREFER_LEFT),
        _j("j2", ("b", "c"), ComparativeVerdict.TIE),
        # a preference cycle within component 1: c > d > a is impossible since
        # a/c not directly linked yet -- build an explicit 3-cycle instead
        _j("j3", ("x", "y"), ComparativeVerdict.PREFER_LEFT),
        _j("j4", ("y", "z"), ComparativeVerdict.PREFER_LEFT),
        _j("j5", ("z", "x"), ComparativeVerdict.PREFER_LEFT),
        # an incomparable pair, its own disconnected component
        _j("j6", ("p", "q"), ComparativeVerdict.INCOMPARABLE),
        # a fully disconnected singleton-pair component the ranker never touched
        _j("j7", ("m", "n"), ComparativeVerdict.ABSTAIN),
    ]
    definition = RankerDefinition(
        engine="bradley_terry_mle", dimension="quality", judgment_ids=tuple(j.judgment_id for j in judgments)
    )
    result = fit_ranker(definition, judgments)

    # disconnected components: {a,b,c}, {x,y,z}, {p,q}, {m,n} => 4 components
    assert len(result.components) == 4
    component_sets = {frozenset(c) for c in result.components}
    assert frozenset({"a", "b", "c"}) in component_sets
    assert frozenset({"x", "y", "z"}) in component_sets
    assert frozenset({"p", "q"}) in component_sets
    assert frozenset({"m", "n"}) in component_sets

    # the 3-cycle is detected
    assert any(set(cycle) == {"x", "y", "z"} for cycle in result.cycles)

    # tie and incomparable are visible and distinct
    assert ("b", "c") in result.ties
    assert ("p", "q") in result.incomparable_pairs
    assert ("b", "c") not in result.incomparable_pairs
    assert ("p", "q") not in result.ties

    # default: no total rank without a declared completion policy
    assert result.total_rank is None


def test_seeded_preference_cycle_is_visible_not_silently_broken() -> None:
    judgments = [
        _j("j1", ("a", "b"), ComparativeVerdict.PREFER_LEFT),
        _j("j2", ("b", "c"), ComparativeVerdict.PREFER_LEFT),
        _j("j3", ("c", "a"), ComparativeVerdict.PREFER_LEFT),
    ]
    definition = RankerDefinition(engine="bradley_terry_mle", dimension="quality", judgment_ids=("j1", "j2", "j3"))
    result = fit_ranker(definition, judgments)
    assert len(result.cycles) == 1
    assert set(result.cycles[0]) == {"a", "b", "c"}


def test_require_total_rank_fails_without_a_declared_completion_policy() -> None:
    definition = RankerDefinition(engine="bradley_terry_mle", dimension="quality", judgment_ids=("j1",))
    judgments = [_j("j1", ("a", "b"), ComparativeVerdict.PREFER_LEFT)]
    with pytest.raises(ValueError, match="completion_policy"):
        fit_ranker(definition, judgments, require_total_rank=True)


def test_a_declared_completion_policy_yields_a_total_rank_and_a_distinct_ranker_ref() -> None:
    judgments = [
        _j("j1", ("a", "b"), ComparativeVerdict.PREFER_LEFT),
        _j("j2", ("b", "c"), ComparativeVerdict.PREFER_LEFT),
    ]
    without_policy = RankerDefinition(engine="bradley_terry_mle", dimension="quality", judgment_ids=("j1", "j2"))
    with_policy = RankerDefinition(
        engine="bradley_terry_mle",
        dimension="quality",
        judgment_ids=("j1", "j2"),
        completion_policy="score_desc_stable",
    )
    assert without_policy.ranker_ref != with_policy.ranker_ref

    result = fit_ranker(with_policy, judgments, require_total_rank=True)
    assert result.total_rank == ("a", "b", "c")
    assert result.completion_policy == "score_desc_stable"


def test_unsupported_completion_policy_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unsupported completion policy"):
        RankerDefinition(
            engine="bradley_terry_mle", dimension="quality", judgment_ids=("j1",), completion_policy="vibes"
        )


def test_ranker_ref_is_stable_and_content_addressed() -> None:
    a = RankerDefinition(engine="win_rate", dimension="quality", judgment_ids=("j2", "j1"))
    b = RankerDefinition(engine="win_rate", dimension="quality", judgment_ids=("j1", "j2"))
    assert a.ranker_ref == b.ranker_ref  # judgment id order is not semantic
    assert a.ranker_ref.startswith("ranker:")

    different_engine = RankerDefinition(engine="bradley_terry_mle", dimension="quality", judgment_ids=("j1", "j2"))
    assert different_engine.ranker_ref != a.ranker_ref


def test_stale_judgment_id_is_excluded_not_silently_included() -> None:
    judgments = [
        _j("j1", ("a", "b"), ComparativeVerdict.PREFER_LEFT),
        _j("j2", ("c", "d"), ComparativeVerdict.PREFER_LEFT),
    ]
    definition = RankerDefinition(engine="win_rate", dimension="quality", judgment_ids=("j1",))
    result = fit_ranker(definition, judgments)
    all_items_seen = {item for component in result.components for item in component}
    assert all_items_seen == {"a", "b"}
