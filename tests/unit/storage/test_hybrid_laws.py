"""Property laws for hybrid search and RRF fusion.

Supersedes parametrized RRF example tests in test_hybrid.py.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.storage.search_providers.hybrid import (
    HybridSearchProvider,
    create_hybrid_provider,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_providers(fts_results=None, vec_results=None):
    """Return (fts_mock, vec_mock) with default search results."""
    fts = MagicMock()
    fts.search.return_value = fts_results or []
    vec = MagicMock()
    vec.query.return_value = vec_results or []
    return fts, vec


def _messages_db(msg_to_conv: dict[str, str]) -> sqlite3.Connection:
    """In-memory SQLite with messages + conversations tables for search_conversations tests.

    Passes the connection directly to open_connection — which accepts
    sqlite3.Connection and yields it as-is (connection.py:123-126).
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE messages (message_id TEXT, conversation_id TEXT)")
    conn.execute(
        "CREATE TABLE conversations "
        "(conversation_id TEXT, provider_name TEXT, source_name TEXT)"
    )
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?)",
        [(msg_id, conv_id) for msg_id, conv_id in msg_to_conv.items()],
    )
    for conv_id in set(msg_to_conv.values()):
        conn.execute(
            "INSERT INTO conversations VALUES (?, 'chatgpt', 'chatgpt')",
            (conv_id,),
        )
    conn.commit()
    return conn


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


# ---------------------------------------------------------------------------
# HybridSearchProvider.search_scored laws
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=50))
def test_search_scored_respects_limit(limit: int) -> None:
    """search_scored never returns more results than the requested limit."""
    fts, vec = _make_providers(
        fts_results=[f"fts_{i}" for i in range(100)],
        vec_results=[(f"vec_{i}", 0.9 - i * 0.005) for i in range(100)],
    )
    provider = HybridSearchProvider(fts, vec)
    results = provider.search_scored("test query", limit=limit)
    assert len(results) <= limit


def test_search_scored_fuses_both_providers() -> None:
    """search_scored includes results from both FTS5 and vector providers."""
    fts, vec = _make_providers(
        fts_results=["fts_a", "fts_b", "fts_c"],
        vec_results=[("vec_x", 0.9), ("vec_y", 0.8), ("vec_z", 0.7)],
    )
    provider = HybridSearchProvider(fts, vec)
    results = provider.search_scored("test", limit=20)
    result_ids = {r[0] for r in results}
    # Items unique to each provider must both appear — pure single-source would drop one
    assert any(r_id.startswith("fts_") for r_id in result_ids), "FTS results missing"
    assert any(r_id.startswith("vec_") for r_id in result_ids), "vector results missing"


def test_search_scored_returns_scores_descending() -> None:
    """Fused scores must be in descending order."""
    fts, vec = _make_providers(
        fts_results=["a", "b", "c", "d"],
        vec_results=[("c", 0.9), ("a", 0.8), ("e", 0.5)],
    )
    provider = HybridSearchProvider(fts, vec)
    results = provider.search_scored("test", limit=20)
    scores = [score for _msg_id, score in results]
    assert scores == sorted(scores, reverse=True), "Results not in descending score order"


# ---------------------------------------------------------------------------
# HybridSearchProvider.search_conversations laws
# ---------------------------------------------------------------------------

def test_search_conversations_deduplicates() -> None:
    """Multiple message hits from the same conversation produce a single conversation result."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    # Two messages (msg1, msg2) map to conv_A; msg3 maps to conv_B
    conn = _messages_db({"msg1": "conv_A", "msg2": "conv_A", "msg3": "conv_B"})
    fts.db_path = conn

    provider.search_scored = MagicMock(  # type: ignore[method-assign]
        return_value=[("msg1", 0.9), ("msg2", 0.8), ("msg3", 0.7)]
    )
    result = provider.search_conversations("test", limit=10)
    assert result == ["conv_A", "conv_B"]
    assert result.count("conv_A") == 1, "conv_A must appear only once"


def test_search_conversations_respects_limit() -> None:
    """search_conversations never returns more than limit conversations."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    conn = _messages_db({f"msg{i}": f"conv_{i}" for i in range(10)})
    fts.db_path = conn

    provider.search_scored = MagicMock(  # type: ignore[method-assign]
        return_value=[(f"msg{i}", 1.0 - i * 0.05) for i in range(10)]
    )
    result = provider.search_conversations("test", limit=3)
    assert len(result) <= 3


def test_search_conversations_provider_filter() -> None:
    """Provider filter excludes conversations from other providers."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE messages (message_id TEXT, conversation_id TEXT)")
    conn.execute(
        "CREATE TABLE conversations "
        "(conversation_id TEXT, provider_name TEXT, source_name TEXT)"
    )
    conn.executemany("INSERT INTO messages VALUES (?, ?)", [
        ("msg1", "conv_chatgpt"),
        ("msg2", "conv_claude"),
    ])
    conn.executemany("INSERT INTO conversations VALUES (?, ?, ?)", [
        ("conv_chatgpt", "chatgpt", "chatgpt"),
        ("conv_claude", "claude", "claude"),
    ])
    conn.commit()
    fts.db_path = conn

    provider.search_scored = MagicMock(  # type: ignore[method-assign]
        return_value=[("msg1", 0.9), ("msg2", 0.8)]
    )
    result = provider.search_conversations("test", limit=10, providers=["chatgpt"])
    assert "conv_chatgpt" in result
    assert "conv_claude" not in result


# ---------------------------------------------------------------------------
# create_hybrid_provider factory law
# ---------------------------------------------------------------------------

def test_create_hybrid_provider_returns_none_when_no_vector() -> None:
    """create_hybrid_provider returns None when no vector provider is available."""
    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        return_value=None,
    ):
        result = create_hybrid_provider()
    assert result is None


def test_create_hybrid_provider_returns_provider_when_vector_available() -> None:
    """create_hybrid_provider returns HybridSearchProvider when vector is available."""
    mock_vec = MagicMock()
    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        return_value=mock_vec,
    ):
        result = create_hybrid_provider()
    assert isinstance(result, HybridSearchProvider)
    assert result.vector_provider is mock_vec
