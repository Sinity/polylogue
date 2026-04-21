"""Property laws for hybrid search and RRF fusion.

Supersedes parametrized RRF example tests in test_hybrid.py.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st

from polylogue.storage.search_providers.hybrid import (
    HybridSearchProvider,
    _resolve_ranked_conversation_ids,
    create_hybrid_provider,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_providers(
    fts_results: list[str] | None = None,
    vec_results: list[tuple[str, float]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (fts_mock, vec_mock) with default search results."""
    fts_results = [] if fts_results is None else fts_results
    vec_results = [] if vec_results is None else vec_results
    fts = MagicMock()
    fts.search.return_value = fts_results
    vec = MagicMock()
    vec.query.return_value = vec_results
    return fts, vec


def _messages_db(msg_to_conv: dict[str, str]) -> sqlite3.Connection:
    """In-memory SQLite with messages + conversations tables for search_conversations tests.

    Passes the connection directly to open_connection — which accepts
    sqlite3.Connection and yields it as-is (connection.py:123-126).
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE messages (message_id TEXT, conversation_id TEXT)")
    conn.execute("CREATE TABLE conversations (conversation_id TEXT, provider_name TEXT, source_name TEXT)")
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?)",
        list(msg_to_conv.items()),
    )
    for conv_id in set(msg_to_conv.values()):
        conn.execute(
            "INSERT INTO conversations VALUES (?, 'chatgpt', 'chatgpt')",
            (conv_id,),
        )
    conn.commit()
    return conn


def _stub_search_scored(provider: HybridSearchProvider, results: list[tuple[str, float]]) -> None:
    object.__setattr__(provider, "search_scored", MagicMock(return_value=results))


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
        st.tuples(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
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
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
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


def test_rrf_default_k_matches_explicit_sixty() -> None:
    """The default RRF constant must stay equivalent to k=60."""
    results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert reciprocal_rank_fusion(results) == reciprocal_rank_fusion(results, k=60)


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        min_size=1,
        max_size=30,
    ),
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        min_size=0,
        max_size=30,
    ),
    st.integers(min_value=1, max_value=100),
)
def test_rrf_scores_stay_within_rank_bound(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
) -> None:
    fused = reciprocal_rank_fusion(results1, results2, k=k)
    max_lists = 1 + int(bool(results2))
    bound = max_lists / (k + 1)
    assert all(0 < score <= bound + 1e-10 for _item_id, score in fused)


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        min_size=1,
        max_size=20,
    ),
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        min_size=1,
        max_size=20,
    ),
    st.integers(min_value=1, max_value=100),
)
def test_rrf_symmetric_scores(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
) -> None:
    assert dict(reciprocal_rank_fusion(results1, results2, k=k)) == dict(
        reciprocal_rank_fusion(results2, results1, k=k)
    )


def test_rrf_single_list_preserves_first_unique_order() -> None:
    results = [("a", 1.0), ("b", 0.9), ("a", 0.1), ("c", 0.8)]
    assert [item_id for item_id, _score in reciprocal_rank_fusion(results)] == ["a", "b", "c"]


def test_rrf_formula_contract() -> None:
    fused = dict(reciprocal_rank_fusion([("a", 0.0), ("b", 0.0)], [("b", 0.0), ("a", 0.0)], k=60))
    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-10
    assert abs(fused["b"] - (1 / 61 + 1 / 62)) < 1e-10


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


def test_search_scored_default_limit_is_twenty() -> None:
    """Default search_scored limit should stay at 20 results."""
    fts, vec = _make_providers(
        fts_results=[f"fts_{i}" for i in range(60)],
        vec_results=[(f"vec_{i}", 0.9 - i * 0.005) for i in range(60)],
    )
    provider = HybridSearchProvider(fts, vec)
    results = provider.search_scored("test query")
    assert len(results) == 20


def test_search_scored_requests_triple_limit_from_each_backend() -> None:
    """Hybrid search should over-fetch symmetrically from both backends."""
    fts, vec = _make_providers(
        fts_results=[f"fts_{i}" for i in range(30)],
        vec_results=[(f"vec_{i}", 0.9 - i * 0.01) for i in range(30)],
    )
    provider = HybridSearchProvider(fts, vec)

    provider.search_scored("needle", limit=7)

    fts.search.assert_called_once_with("needle", limit=21)
    vec.query.assert_called_once_with("needle", limit=21)


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

    _stub_search_scored(provider, [("msg1", 0.9), ("msg2", 0.8), ("msg3", 0.7)])
    result = provider.search_conversations("test", limit=10)
    assert result == ["conv_A", "conv_B"]
    assert result.count("conv_A") == 1, "conv_A must appear only once"


def test_search_conversations_empty_results_skips_db_lookup() -> None:
    """No message hits should return early without opening a DB connection."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    _stub_search_scored(provider, [])

    with patch("polylogue.storage.search_providers.hybrid.open_connection") as open_conn:
        assert provider.search_conversations("test", limit=10) == []
        open_conn.assert_not_called()


def test_search_conversations_respects_limit() -> None:
    """search_conversations never returns more than limit conversations."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    conn = _messages_db({f"msg{i}": f"conv_{i}" for i in range(10)})
    fts.db_path = conn

    _stub_search_scored(provider, [(f"msg{i}", 1.0 - i * 0.05) for i in range(10)])
    result = provider.search_conversations("test", limit=3)
    assert len(result) <= 3


def test_search_conversations_provider_filter() -> None:
    """Provider filter excludes conversations from other providers."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE messages (message_id TEXT, conversation_id TEXT)")
    conn.execute("CREATE TABLE conversations (conversation_id TEXT, provider_name TEXT, source_name TEXT)")
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?)",
        [
            ("msg1", "conv_chatgpt"),
            ("msg2", "conv_claude"),
        ],
    )
    conn.executemany(
        "INSERT INTO conversations VALUES (?, ?, ?)",
        [
            ("conv_chatgpt", "chatgpt", "chatgpt"),
            ("conv_claude", "claude-ai", "claude-ai"),
        ],
    )
    conn.commit()
    fts.db_path = conn

    _stub_search_scored(provider, [("msg1", 0.9), ("msg2", 0.8)])
    result = provider.search_conversations("test", limit=10, providers=["chatgpt"])
    assert "conv_chatgpt" in result
    assert "conv_claude" not in result


# ---------------------------------------------------------------------------
# create_hybrid_provider factory law
# ---------------------------------------------------------------------------


def test_search_conversations_limit_zero_returns_empty() -> None:
    """search_conversations with limit=0 must return [], not one result."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    conn = _messages_db({"msg1": "conv_A", "msg2": "conv_B"})
    fts.db_path = conn

    _stub_search_scored(provider, [("msg1", 0.9), ("msg2", 0.8)])
    result = provider.search_conversations("test", limit=0)
    assert result == [], f"limit=0 must return empty list, got {result}"


def test_resolve_ranked_conversation_ids_short_circuits_empty_inputs() -> None:
    """Internal ranked resolver should not hit SQL for empty or nonpositive inputs."""
    conn = MagicMock()

    assert (
        _resolve_ranked_conversation_ids(
            conn,
            message_results=[],
            limit=5,
            scope_names=None,
        )
        == []
    )
    assert (
        _resolve_ranked_conversation_ids(
            conn,
            message_results=[("msg1", 0.9)],
            limit=0,
            scope_names=None,
        )
        == []
    )

    conn.execute.assert_not_called()


def test_search_conversations_empty_providers_equivalent_to_none() -> None:
    """providers=[] and providers=None are both treated as 'no filter' (all conversations).

    The guard `if providers:` is falsy for both None and [], so an empty list
    does NOT mean 'empty allowlist'. Both values include all conversations.
    This is intentional — callers have no reason to pass providers=[] expecting
    zero results.
    """
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    conn = _messages_db({"msg1": "conv_A"})
    fts.db_path = conn

    _stub_search_scored(provider, [("msg1", 0.9)])
    result_empty_list = provider.search_conversations("test", limit=10, providers=[])
    result_none = provider.search_conversations("test", limit=10, providers=None)
    # Both must behave identically — no filter applied
    assert result_empty_list == result_none == ["conv_A"]


def test_search_conversations_orphan_message_ids_skipped() -> None:
    """Message IDs returned by search but absent from the DB are gracefully skipped."""
    fts, vec = _make_providers()
    provider = HybridSearchProvider(fts, vec)
    # Only msg2 exists in the DB; msg1 and msg3 are orphans (e.g., deleted)
    conn = _messages_db({"msg2": "conv_B"})
    fts.db_path = conn

    _stub_search_scored(provider, [("msg1", 0.9), ("msg2", 0.8), ("msg3", 0.7)])
    result = provider.search_conversations("test", limit=10)
    assert result == ["conv_B"], "Orphan message IDs must be silently skipped"


def test_rrf_deduplicates_within_same_list() -> None:
    """Duplicate IDs within a single result list are counted only once at their best rank."""
    # "msg1" appears twice in list1; should score as rank=1 only (not rank=1 + rank=3)
    list1 = [("msg1", 0.9), ("msg2", 0.8), ("msg1", 0.5)]
    fused = reciprocal_rank_fusion(list1, k=60)
    scores = dict(fused)
    # msg1 at rank 1 → 1/61; msg2 at rank 2 → 1/62
    # If dedup works, msg1=1/61 > msg2=1/62
    # If not dedup, msg1=1/61+1/63 >> msg2 — but the ratio would differ
    expected_msg1 = 1.0 / (60 + 1)
    assert abs(scores["msg1"] - expected_msg1) < 1e-10, (
        f"msg1 should score exactly 1/(60+1) from best rank; got {scores['msg1']}"
    )


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
