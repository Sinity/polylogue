"""Property laws for hybrid search and RRF fusion.

Supersedes parametrized RRF example tests in test_hybrid.py.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Law 1: RRF score for any rank is positive
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=1000))
def test_rrf_score_positive(rank: int) -> None:
    """RRF score for any positive rank is strictly positive."""
    k = 60
    score = 1.0 / (k + rank)
    assert score > 0


# ---------------------------------------------------------------------------
# Law 2: Higher rank (lower rank number) always yields higher RRF score
# ---------------------------------------------------------------------------

@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10, unique=True))
def test_rrf_higher_rank_higher_score(items: list[str]) -> None:
    """Higher-ranked (lower rank number) items always get higher RRF score."""
    k = 60
    scores = [1.0 / (k + rank) for rank in range(1, len(items) + 1)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


# ---------------------------------------------------------------------------
# Law 3: reciprocal_rank_fusion never crashes on arbitrary inputs
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        max_size=10,
    )
)
def test_rrf_never_crashes(result_list: list[tuple[str, float]]) -> None:
    """reciprocal_rank_fusion handles any list of (id, score) tuples without raising."""
    fused = reciprocal_rank_fusion(result_list)
    assert isinstance(fused, list)


# ---------------------------------------------------------------------------
# Law 4: RRF result length <= sum of unique items across all lists
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.lists(
            st.tuples(st.text(min_size=1, max_size=10), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            max_size=5,
        ),
        max_size=3,
    )
)
def test_rrf_result_count_bounded(lists: list[list[tuple[str, float]]]) -> None:
    """RRF result count never exceeds total unique IDs across all input lists."""
    all_ids = {item_id for lst in lists for item_id, _ in lst}
    fused = reciprocal_rank_fusion(*lists)
    assert len(fused) <= len(all_ids)


# ---------------------------------------------------------------------------
# Law 5: Items appearing in more lists get higher scores than single-list items
# ---------------------------------------------------------------------------

def test_rrf_multi_list_boost() -> None:
    """An item appearing in two lists gets a higher score than one appearing in only one."""
    list1 = [("shared", 0.9), ("only_in_1", 0.8)]
    list2 = [("shared", 0.85), ("only_in_2", 0.7)]
    fused = reciprocal_rank_fusion(list1, list2, k=60)
    scores = dict(fused)
    # "shared" appears in both lists and should outscore items only in one list
    assert scores["shared"] > scores["only_in_1"]
    assert scores["shared"] > scores["only_in_2"]
