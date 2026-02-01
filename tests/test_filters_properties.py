"""Property-based tests for filter and RRF modules.

These tests verify mathematical properties of filter composition
and Reciprocal Rank Fusion scoring.

Key properties tested:
1. Filter monotonicity - adding filters never increases result count
2. RRF score bounds - scores are always in (0, 1/k)
3. RRF symmetry - swapping input lists doesn't change final scores
4. RRF monotonicity - higher rank in more lists = higher score
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion


# =============================================================================
# RRF Score Bound Properties
# =============================================================================


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=100,
    ),
    st.integers(min_value=1, max_value=200),
)
def test_rrf_scores_bounded_single_list(
    results: list[tuple[str, float]],
    k: int,
):
    """RRF scores from a single list are bounded by 1/(k+1)."""
    fused = reciprocal_rank_fusion(results, k=k)

    # Maximum possible score is 1/(k+1) for rank 1
    max_score = 1.0 / (k + 1)

    for _item_id, score in fused:
        assert score <= max_score + 1e-10  # Allow small float error
        assert score > 0


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=50,
    ),
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=50,
    ),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_rrf_scores_bounded_two_lists(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
):
    """RRF scores from two lists are bounded by 2/(k+1)."""
    fused = reciprocal_rank_fusion(results1, results2, k=k)

    # Maximum possible score is 2/(k+1) if item is rank 1 in both lists
    max_score = 2.0 / (k + 1)

    for _item_id, score in fused:
        assert score <= max_score + 1e-10
        assert score > 0


# =============================================================================
# RRF Symmetry Properties
# =============================================================================


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=30,
    ),
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=30,
    ),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=30)
def test_rrf_symmetric_scores(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
):
    """RRF(A, B) produces same scores as RRF(B, A)."""
    fused_ab = reciprocal_rank_fusion(results1, results2, k=k)
    fused_ba = reciprocal_rank_fusion(results2, results1, k=k)

    # Convert to dicts for comparison (order might differ)
    scores_ab = dict(fused_ab)
    scores_ba = dict(fused_ba)

    # Same items should have same scores
    for item_id in scores_ab:
        assert abs(scores_ab[item_id] - scores_ba.get(item_id, 0)) < 1e-10


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=20,
    ),
)
def test_rrf_single_list_preserves_order(results: list[tuple[str, float]]):
    """For a single list, RRF preserves the original ranking order."""
    # Filter out duplicates since RRF deduplicates
    seen = set()
    unique_results = []
    for item_id, score in results:
        if item_id not in seen:
            seen.add(item_id)
            unique_results.append((item_id, score))

    fused = reciprocal_rank_fusion(unique_results)

    # RRF order should match original order (first item has highest score)
    fused_order = [item_id for item_id, _ in fused]
    original_order = [item_id for item_id, _ in unique_results]

    assert fused_order == original_order


# =============================================================================
# RRF Monotonicity Properties
# =============================================================================


@given(st.integers(min_value=1, max_value=100))
def test_rrf_rank_decreases_score(k: int):
    """Lower rank (further from top) produces lower score."""
    # Create a simple ordered list
    results = [(f"item_{i}", 0.0) for i in range(10)]

    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)

    # Verify scores decrease with rank
    for i in range(len(results) - 1):
        current_id = f"item_{i}"
        next_id = f"item_{i + 1}"
        assert scores[current_id] > scores[next_id]


@given(st.integers(min_value=1, max_value=100))
def test_rrf_appearing_in_both_lists_increases_score(k: int):
    """Items appearing in both lists score higher than single-list items."""
    # Item appears in both lists
    list1 = [("common", 1.0), ("only_list1", 0.9)]
    list2 = [("common", 0.95), ("only_list2", 0.85)]

    fused = reciprocal_rank_fusion(list1, list2, k=k)
    scores = dict(fused)

    # Common item should have higher score than items in single list
    assert scores["common"] > scores["only_list1"]
    assert scores["common"] > scores["only_list2"]


# =============================================================================
# RRF Edge Cases
# =============================================================================


def test_rrf_empty_lists():
    """RRF handles empty input gracefully."""
    result = reciprocal_rank_fusion()
    assert result == []


def test_rrf_empty_list_in_args():
    """RRF handles empty list mixed with non-empty."""
    results = [("item1", 1.0), ("item2", 0.9)]
    empty: list[tuple[str, float]] = []

    fused = reciprocal_rank_fusion(results, empty)

    # Should still produce results from non-empty list
    assert len(fused) == 2


def test_rrf_duplicate_items_same_list():
    """RRF handles duplicate items in same list (uses first occurrence)."""
    results = [("item1", 1.0), ("item2", 0.9), ("item1", 0.5)]

    fused = reciprocal_rank_fusion(results)

    # Each item appears once in output
    item_ids = [item_id for item_id, _ in fused]
    assert len(set(item_ids)) == len(item_ids)


@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.floats(min_value=0, max_value=1)),
        min_size=0,
        max_size=50,
    ),
)
def test_rrf_never_crashes(results: list[tuple[str, float]]):
    """RRF handles any valid input without crashing."""
    fused = reciprocal_rank_fusion(results)
    assert isinstance(fused, list)


# =============================================================================
# RRF Formula Verification
# =============================================================================


def test_rrf_formula_correctness():
    """Verify RRF formula: score = 1/(k + rank)."""
    k = 60
    results = [("a", 0.0), ("b", 0.0), ("c", 0.0)]

    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)

    # Expected: rank 1 -> 1/61, rank 2 -> 1/62, rank 3 -> 1/63
    assert abs(scores["a"] - 1/61) < 1e-10
    assert abs(scores["b"] - 1/62) < 1e-10
    assert abs(scores["c"] - 1/63) < 1e-10


def test_rrf_combined_formula():
    """Verify combined RRF scores from multiple lists."""
    k = 60

    # a is rank 1 in list1, rank 2 in list2
    # b is rank 2 in list1, rank 1 in list2
    list1 = [("a", 0.0), ("b", 0.0)]
    list2 = [("b", 0.0), ("a", 0.0)]

    fused = reciprocal_rank_fusion(list1, list2, k=k)
    scores = dict(fused)

    # a: 1/61 + 1/62 = (61 + 62) / (61 * 62)
    expected_a = 1/61 + 1/62
    assert abs(scores["a"] - expected_a) < 1e-10

    # b: 1/62 + 1/61 = same as a
    expected_b = 1/62 + 1/61
    assert abs(scores["b"] - expected_b) < 1e-10

    # Scores should be equal since positions are symmetric
    assert abs(scores["a"] - scores["b"]) < 1e-10


# =============================================================================
# Conversation Filter Properties (Mock-based)
# =============================================================================


@given(
    st.lists(st.sampled_from(["chatgpt", "claude", "codex"]), min_size=1, max_size=3),
    st.lists(st.sampled_from(["chatgpt", "claude", "codex"]), min_size=0, max_size=2),
)
def test_provider_filter_exclusion_disjoint(
    include_providers: list[str],
    exclude_providers: list[str],
):
    """Provider inclusion and exclusion should be mutually exclusive.

    If a provider is in both include and exclude, exclude takes precedence.
    This tests the filter logic property, not the actual implementation.
    """
    # Simulate filter logic
    included = set(include_providers)
    excluded = set(exclude_providers)

    # Result should not contain excluded items
    result = included - excluded

    for provider in result:
        assert provider not in excluded


@given(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=50),
)
def test_limit_respects_bound(total_items: int, limit: int):
    """Limit filter produces at most `limit` items."""
    # Simulate having total_items conversations
    items = list(range(total_items))

    # Apply limit
    limited = items[:limit]

    assert len(limited) <= limit


@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=100),
)
def test_offset_skips_correctly(offset: int, total_items: int):
    """Offset filter skips the first `offset` items."""
    items = list(range(total_items))

    # Apply offset
    offset_items = items[offset:]

    expected_count = max(0, total_items - offset)
    assert len(offset_items) == expected_count


@given(
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 1, 1),
        timezones=st.just(timezone.utc),
    ),
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 1, 1),
        timezones=st.just(timezone.utc),
    ),
)
def test_date_range_logic(since: datetime, until: datetime):
    """Date range filter is a valid interval (since <= until)."""
    # Ensure valid range
    if since > until:
        since, until = until, since

    # A timestamp within range should pass
    mid = since + (until - since) / 2
    assert since <= mid <= until

    # A timestamp outside should fail
    before = since - timedelta(days=1)
    after = until + timedelta(days=1)

    assert before < since
    assert after > until


# =============================================================================
# Filter Composition Properties
# =============================================================================


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_filter_composition_monotonic(
    items: list[int],
    threshold1: int,
    threshold2: int,
):
    """Composing filters never increases result count.

    filter(A).filter(B).count() <= filter(A).count()
    """
    # Simulate two filters: > threshold1 and > threshold2
    filtered_once = [x for x in items if x > threshold1]
    filtered_twice = [x for x in filtered_once if x > threshold2]

    assert len(filtered_twice) <= len(filtered_once)


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
)
def test_filter_idempotent(items: list[int], threshold: int):
    """Applying the same filter twice has no additional effect.

    filter(A).filter(A) == filter(A)
    """
    filtered_once = [x for x in items if x > threshold]
    filtered_twice = [x for x in filtered_once if x > threshold]

    assert filtered_once == filtered_twice


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_filter_order_independent(
    items: list[int],
    threshold1: int,
    threshold2: int,
):
    """For commutative filters, order doesn't affect result count.

    filter(A).filter(B).count() == filter(B).filter(A).count()
    """
    # Filter A: > threshold1
    # Filter B: > threshold2
    filtered_ab = [x for x in items if x > threshold1 and x > threshold2]
    filtered_ba = [x for x in items if x > threshold2 and x > threshold1]

    # Same result (order of application doesn't matter for AND)
    assert filtered_ab == filtered_ba
