"""Tests for active elicitation sessions (rxdo.9.14, mechanism N)."""

from __future__ import annotations

import pytest

from polylogue.insights.judgment.elicitation import ElicitationSession, ExplorationQuota

_ITEMS = ("maj1", "maj2", "maj3", "maj4", "min1", "min2")


def _seed_session(*, quota_fraction: float, rng_seed: int = 7) -> ElicitationSession:
    session = ElicitationSession(
        session_id="s1",
        item_refs=_ITEMS,
        dimension="quality",
        budget_total=0,
        quota=ExplorationQuota(fraction=quota_fraction, min_coverage_threshold=1),
        rng_seed=rng_seed,
    )
    # Majority region: well covered, close-together estimates (the "closest
    # latent estimate" tie-break will always favor a majority-majority pair).
    session.observations.update({"maj1": 5, "maj2": 5, "maj3": 5, "maj4": 5, "min1": 0, "min2": 0})
    # maj1/maj2 is the UNIQUE closest-estimate pair (diff 0); every other pair
    # has a strictly larger gap, so exploitation's tie-break never matters.
    session.estimates.update({"maj1": 1.0, "maj2": 1.0, "maj3": 5.0, "maj4": 8.0, "min1": 20.0, "min2": -20.0})
    return session


def test_disabling_the_quota_changes_the_production_selection() -> None:
    """AC: disabling the quota changes the production selection and fails the (buggy) test."""

    disabled = _seed_session(quota_fraction=0.0)
    enabled = _seed_session(quota_fraction=0.6)

    disabled_pick = disabled.select_next()
    enabled_pick = enabled.select_next()
    assert disabled_pick is not None
    assert enabled_pick is not None

    # Without a quota, exploitation always wins: the closest-estimate pair is
    # maj1/maj2 (both estimate 1.0), never touching the minority region.
    assert {disabled_pick.left_ref, disabled_pick.right_ref} == {"maj1", "maj2"}
    assert disabled_pick.mode == "exploitation"

    # With a nonzero quota and an under-covered minority region, the very
    # first pick is forced into exploration and must touch the minority.
    assert enabled_pick.mode == "exploration"
    assert {"min1", "min2"} & {enabled_pick.left_ref, enabled_pick.right_ref}

    # The two sessions genuinely produce different selections.
    assert {disabled_pick.left_ref, disabled_pick.right_ref} != {enabled_pick.left_ref, enabled_pick.right_ref}


def test_a_seeded_minority_region_receives_its_declared_exploration_budget() -> None:
    session = _seed_session(quota_fraction=1.0)  # always explore while under-covered
    seen_minority = False
    for _ in range(4):
        pick = session.select_next()
        assert pick is not None
        if {"min1", "min2"} & {pick.left_ref, pick.right_ref}:
            seen_minority = True
        session.record_verdict(f"j-{pick.left_ref}-{pick.right_ref}", pick.left_ref, pick.right_ref, left_won=True)
    assert seen_minority


def test_select_next_returns_none_once_all_pairs_are_exhausted() -> None:
    session = ElicitationSession(
        session_id="s2",
        item_refs=("a", "b"),
        dimension="quality",
        budget_total=0,
        quota=ExplorationQuota(fraction=0.0),
    )
    pick = session.select_next()
    assert pick is not None
    session.record_verdict("j1", pick.left_ref, pick.right_ref, left_won=True)
    assert session.select_next() is None


def test_budget_caps_selection_even_with_pairs_remaining() -> None:
    session = ElicitationSession(
        session_id="s3",
        item_refs=("a", "b", "c"),
        dimension="quality",
        budget_total=1,
        quota=ExplorationQuota(fraction=0.0),
    )
    pick = session.select_next()
    assert pick is not None
    session.record_verdict("j1", pick.left_ref, pick.right_ref, left_won=True)
    assert session.select_next() is None  # budget exhausted, pairs still remain


def test_selection_receipt_is_reconstructible_without_hidden_labels() -> None:
    session = _seed_session(quota_fraction=0.0)
    pick = session.select_next()
    assert pick is not None
    assert pick.receipt_hash
    assert pick.candidate_pool_size > 0
    # the receipt names only item refs -- no provenance fields present
    assert pick.left_ref in _ITEMS
    assert pick.right_ref in _ITEMS


def test_session_rejects_duplicate_item_refs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ElicitationSession(
            session_id="s4",
            item_refs=("a", "a"),
            dimension="quality",
            budget_total=0,
            quota=ExplorationQuota(fraction=0.0),
        )


def test_quota_fraction_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ExplorationQuota(fraction=1.5)
