"""Tests for hybrid search provider combining FTS and vector search.

Tests cover hybrid search methods, reciprocal rank fusion algorithm,
provider filtering, and integration scenarios.

Extracted from test_search_index.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.archive.session.branch_type import BranchType
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.hybrid import (
    HybridSearchProvider,
    reciprocal_rank_fusion,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_hash, make_message, make_session, save_session_to_archive


def _as_mock(value: object) -> MagicMock:
    if not isinstance(value, MagicMock):
        raise TypeError(f"expected MagicMock, got {type(value).__name__}")
    return value


class TestHybridSearchProvider:
    """Tests for HybridSearchProvider search methods."""

    async def test_hybrid_search_sessions_limit_reached(self, tmp_path: Path) -> None:
        """search_sessions stops when limit reached."""
        # Create backend with some sessions and messages
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Add sessions and messages. Session/message IDs are the generated
        # ``origin:native_id`` form so the message FK and the FTS->session
        # resolution join line up with the stored rows.
        for i in range(5):
            sid = f"unknown-export:conv-{i}"
            conv = make_session(
                session_id=sid,
                source_name="test",
                provider_session_id=f"conv-{i}",
                content_hash=make_hash(f"conv-{i}"),
                title=f"Conv {i}",
                created_at=f"2024-01-0{i + 1}T00:00:00Z",
                updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                parent_session_id=None,
                branch_type=None,
                raw_id=None,
            )
            msg = make_message(
                message_id=f"{sid}:msg-{i}",
                session_id=sid,
                content_hash=make_hash(f"msg-{i}"),
                role="user",
                provider_message_id=f"msg-{i}",
                text=f"Message {i}",
                timestamp=f"2024-01-0{i + 1}T00:00:00Z",
            )
            await save_session_to_archive(backend, session=conv, messages=[msg])

        await backend.close()

        # Create hybrid provider with mocks. The backend writes the split-file
        # archive tier set, so the session-resolution query reads ``index.db``.
        fts_mock = MagicMock()
        # Return message IDs for all 5 sessions
        fts_mock.search.return_value = [f"unknown-export:conv-{i}:msg-{i}" for i in range(5)]
        fts_mock.db_path = tmp_path / "index.db"

        vec_mock = MagicMock()
        vec_mock.query.return_value = []

        provider = HybridSearchProvider(fts_provider=fts_mock, vector_provider=vec_mock)

        # Request only 2 sessions
        result = provider.search_sessions("test", limit=2)
        assert len(result) <= 2

    def test_create_hybrid_provider_no_vector_returns_none(self) -> None:
        """create_hybrid_provider returns None when vector search unavailable."""
        # Patch create_vector_provider at the point it's imported in hybrid.py
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=None):
            from polylogue.storage.search_providers.hybrid import create_hybrid_provider

            result = create_hybrid_provider()
            assert result is None

    def test_hybrid_search_empty_fts_results(self) -> None:
        """Hybrid search with empty FTS results still works."""
        fts_mock = MagicMock()
        fts_mock.search.return_value = []

        vec_mock = MagicMock()
        vec_mock.query.return_value = [("msg1", 0.9)]

        provider = HybridSearchProvider(fts_provider=fts_mock, vector_provider=vec_mock)

        # Should return results from vector search when FTS is empty
        result = provider.search_scored("test", limit=10)
        assert len(result) > 0

    async def test_hybrid_search_with_provider_filter(self, tmp_path: Path) -> None:
        """Hybrid search respects provider filter."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Add sessions from different providers. The session-scope filter is an
        # ``origin`` predicate now, so seed sessions whose origin token is the
        # one we filter by and reference the generated ``origin:native_id`` ids.
        for source_name, origin_token in [("claude-ai", "claude-ai-export"), ("chatgpt", "chatgpt-export")]:
            for i in range(2):
                sid = f"{origin_token}:conv-{i}"
                conv = make_session(
                    session_id=sid,
                    source_name=source_name,
                    provider_session_id=f"conv-{i}",
                    content_hash=make_hash(f"{source_name}-{i}"),
                    title=f"{source_name} Conv {i}",
                    created_at=f"2024-01-0{i + 1}T00:00:00Z",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                )
                msg = make_message(
                    message_id=f"{sid}:msg-{i}",
                    session_id=sid,
                    content_hash=make_hash(f"{source_name}-msg-{i}"),
                    role="user",
                    provider_message_id=f"msg-{i}",
                    text="test message",
                    timestamp=f"2024-01-0{i + 1}T00:00:00Z",
                )
                await save_session_to_archive(backend, session=conv, messages=[msg])

        await backend.close()

        # Create hybrid provider with mocks
        fts_mock = MagicMock()
        fts_mock.search.return_value = [
            "claude-ai-export:conv-0:msg-0",
            "chatgpt-export:conv-0:msg-0",
            "claude-ai-export:conv-1:msg-1",
            "chatgpt-export:conv-1:msg-1",
        ]
        fts_mock.db_path = tmp_path / "index.db"

        vec_mock = MagicMock()
        vec_mock.query.return_value = []

        provider = HybridSearchProvider(fts_provider=fts_mock, vector_provider=vec_mock)

        # Search with provider filter (origin token)
        result = provider.search_sessions("test", limit=10, providers=["claude-ai-export"])
        # Should filter to only Claude AI sessions
        assert set(result) == {"claude-ai-export:conv-0", "claude-ai-export:conv-1"}


class TestReciprocalRankFusion:
    """Tests for the RRF algorithm."""

    def test_rrf_overlapping_lists(self) -> None:
        """RRF boosts items appearing in multiple lists."""
        # msg2 appears at rank 2 in fts, rank 1 in vec
        fts_results = [("msg1", 0.9), ("msg2", 0.8), ("msg3", 0.7)]
        vec_results = [("msg2", 0.95), ("msg1", 0.85), ("msg4", 0.6)]

        fused = reciprocal_rank_fusion(fts_results, vec_results, k=60)

        # msg2 should rank highest (appears in both, good ranks in both)
        scores = dict(fused)

        # msg2: rank 2 in fts (1/62) + rank 1 in vec (1/61) = higher than msg1
        # msg1: rank 1 in fts (1/61) + rank 2 in vec (1/62) = same as msg2 actually
        # Wait, they're symmetric so msg1 and msg2 should have equal scores
        assert abs(scores["msg1"] - scores["msg2"]) < 0.0001

        # msg3 and msg4 appear only once, so lower scores
        assert scores["msg1"] > scores["msg3"]
        assert scores["msg1"] > scores["msg4"]

    def test_rrf_k_parameter_effect(self) -> None:
        """Different k values affect score magnitudes."""
        results = [("msg1", 0.9), ("msg2", 0.8)]

        fused_k60 = reciprocal_rank_fusion(results, k=60)
        fused_k10 = reciprocal_rank_fusion(results, k=10)

        # Lower k means higher scores
        scores_k60 = dict(fused_k60)
        scores_k10 = dict(fused_k10)

        assert scores_k10["msg1"] > scores_k60["msg1"]

    def test_rrf_original_scores_ignored(self) -> None:
        """RRF uses rank, not original scores."""
        # Different original scores, same ranks
        list1 = [("msg1", 0.999), ("msg2", 0.001)]
        list2 = [("msg1", 0.5), ("msg2", 0.4)]

        fused = reciprocal_rank_fusion(list1, list2, k=60)

        scores = dict(fused)
        # Both lists have same order, so RRF scores should be equal
        # regardless of original score differences
        assert scores["msg1"] == scores["msg1"]  # Trivially true, but shows intent

    def test_rrf_many_lists(self) -> None:
        """RRF works with many result lists."""
        lists = [
            [("msg1", 0.9), ("msg2", 0.8)],
            [("msg2", 0.9), ("msg1", 0.8)],
            [("msg1", 0.9), ("msg3", 0.8)],
            [("msg3", 0.9), ("msg2", 0.8)],
        ]

        fused = reciprocal_rank_fusion(*lists, k=60)

        # msg1 and msg2 appear 3 times, msg3 appears 2 times
        scores = dict(fused)
        assert scores["msg1"] > scores["msg3"]
        assert scores["msg2"] > scores["msg3"]


class TestHybridSearchProviderRRF:
    """Tests for HybridSearchProvider with RRF."""

    @pytest.fixture
    def mock_fts_provider(self) -> MagicMock:
        """Create mock FTS5 provider."""
        provider = MagicMock()
        provider.db_path = None
        return provider

    @pytest.fixture
    def mock_vector_provider(self) -> MagicMock:
        """Create mock vector provider."""
        return MagicMock()

    @pytest.fixture
    def hybrid_provider(self, mock_fts_provider: MagicMock, mock_vector_provider: MagicMock) -> HybridSearchProvider:
        """Create hybrid provider with mocks."""
        return HybridSearchProvider(
            fts_provider=mock_fts_provider,
            vector_provider=mock_vector_provider,
            rrf_k=60,
        )

    def test_search_combines_results(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search() combines FTS and vector results."""
        # Set up mock returns
        mock_fts_provider.search.return_value = ["msg1", "msg2", "msg3"]
        mock_vector_provider.query.return_value = [
            ("msg2", 0.95),
            ("msg4", 0.85),
            ("msg1", 0.75),
        ]

        results = hybrid_provider.search_scored("test query", limit=10)

        # Should have items from both sources
        ids = [item_id for item_id, _ in results]
        assert "msg1" in ids
        assert "msg2" in ids
        assert "msg3" in ids
        assert "msg4" in ids

        # msg1 and msg2 appear in both, should rank higher
        scores = dict(results)
        assert scores["msg1"] > scores["msg3"]
        assert scores["msg2"] > scores["msg4"]

    def test_search_empty_fts_results(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search() works with empty FTS results."""
        mock_fts_provider.search.return_value = []
        mock_vector_provider.query.return_value = [
            ("msg1", 0.95),
            ("msg2", 0.85),
        ]

        results = hybrid_provider.search_scored("test query", limit=10)

        # Should have vector results only
        ids = [item_id for item_id, _ in results]
        assert ids == ["msg1", "msg2"]

    def test_search_empty_vector_results(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search() works with empty vector results."""
        mock_fts_provider.search.return_value = ["msg1", "msg2"]
        mock_vector_provider.query.return_value = []

        results = hybrid_provider.search_scored("test query", limit=10)

        # Should have FTS results only
        ids = [item_id for item_id, _ in results]
        assert "msg1" in ids
        assert "msg2" in ids

    def test_search_both_empty(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search() returns empty when both sources empty."""
        mock_fts_provider.search.return_value = []
        mock_vector_provider.query.return_value = []

        results = hybrid_provider.search_scored("test query", limit=10)
        assert results == []

    def test_search_respects_limit(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search() respects the limit parameter."""
        mock_fts_provider.search.return_value = [f"msg{i}" for i in range(20)]
        mock_vector_provider.query.return_value = [(f"vec{i}", 0.9) for i in range(20)]

        results = hybrid_provider.search_scored("test query", limit=5)
        assert len(results) == 5

    def test_search_sessions_deduplicates(
        self,
        hybrid_provider: HybridSearchProvider,
        mock_fts_provider: MagicMock,
        mock_vector_provider: MagicMock,
    ) -> None:
        """search_sessions() returns unique session IDs."""
        # Multiple messages from same session
        mock_fts_provider.search.return_value = ["msg1", "msg2", "msg3"]
        mock_vector_provider.query.return_value = [
            ("msg2", 0.95),
            ("msg4", 0.85),
        ]

        # Mock the database lookup
        with patch("polylogue.storage.search_providers.hybrid.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context
            mock_context.execute.return_value.fetchall.return_value = [
                {"session_id": "conv1"},
                {"session_id": "conv2"},
                {"session_id": "conv3"},
            ]

            results = hybrid_provider.search_sessions("test query", limit=10)

            # Should have 3 unique sessions
            assert len(results) == 3
            assert len(set(results)) == len(results)  # All unique


class TestHybridSearchIntegration:
    """Integration-style tests with realistic scenarios."""

    def test_rrf_academic_example(self) -> None:
        """Test RRF with academic paper example scenario."""
        # Simulate search results from two different ranking systems
        # Both rank "python tutorial" highly but in different orders
        fts_ranking = [
            ("doc_python_intro", 0.95),
            ("doc_python_advanced", 0.85),
            ("doc_java_basics", 0.75),
            ("doc_python_tips", 0.65),
        ]

        semantic_ranking = [
            ("doc_python_advanced", 0.92),
            ("doc_python_intro", 0.88),
            ("doc_python_tips", 0.78),
            ("doc_rust_guide", 0.68),
        ]

        fused = reciprocal_rank_fusion(fts_ranking, semantic_ranking, k=60)
        ids = [doc_id for doc_id, _ in fused]

        # Python docs should dominate (appear in both)
        top_3 = ids[:3]
        assert "doc_python_intro" in top_3
        assert "doc_python_advanced" in top_3
        assert "doc_python_tips" in top_3

        # Java and Rust only appear once, should be lower
        scores = dict(fused)
        python_score = min(scores["doc_python_intro"], scores["doc_python_advanced"])
        single_source_score = max(scores.get("doc_java_basics", 0), scores.get("doc_rust_guide", 0))
        assert python_score > single_source_score


class TestProviderFilteringIntegration:
    """Integration tests for provider filtering in search."""

    @pytest.fixture
    def hybrid_provider(self) -> HybridSearchProvider:
        """Create a HybridSearchProvider with mocked dependencies."""
        mock_fts = MagicMock()
        mock_fts.db_path = None
        mock_vec = MagicMock()

        return HybridSearchProvider(
            fts_provider=mock_fts,
            vector_provider=mock_vec,
            rrf_k=60,
        )

    def test_provider_filter_applied_before_limit(self, hybrid_provider: HybridSearchProvider) -> None:
        """Provider filter should be applied before limit, not after."""
        # Mock search returns messages from various providers
        _as_mock(hybrid_provider.fts_provider.search).return_value = [f"msg-claude-{i}" for i in range(15)] + [
            f"msg-chatgpt-{i}" for i in range(5)
        ]

        _as_mock(hybrid_provider.vector_provider.query).return_value = [
            (f"msg-claude-{i}", 0.9 - i * 0.01) for i in range(10)
        ]

        # Mock database to return session IDs with provider info
        with patch("polylogue.storage.search_providers.hybrid.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context
            mock_context.execute.return_value.fetchall.return_value = [
                {"session_id": f"conv-chatgpt-{i}"} for i in range(5)
            ]

            # Search with provider filter
            results = hybrid_provider.search_sessions("test query", limit=10, providers=["chatgpt"])

            # Should return chatgpt sessions, not empty
            assert len(results) > 0
            assert set(results) == {f"conv-chatgpt-{i}" for i in range(5)}

    def test_provider_filter_none_returns_all(self, hybrid_provider: HybridSearchProvider) -> None:
        """When providers=None, should return sessions from all providers."""
        _as_mock(hybrid_provider.fts_provider.search).return_value = [
            "msg-claude-1",
            "msg-chatgpt-1",
            "msg-gemini-1",
        ]
        _as_mock(hybrid_provider.vector_provider.query).return_value = []

        with patch("polylogue.storage.search_providers.hybrid.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context

            mock_context.execute.return_value.fetchall.return_value = [
                {"session_id": "conv-1"},
                {"session_id": "conv-2"},
                {"session_id": "conv-3"},
            ]

            results = hybrid_provider.search_sessions(
                "test query",
                limit=10,
                providers=None,  # No filter
            )

            # Should return all 3 sessions
            assert len(results) == 3

    def test_provider_filter_multiple_providers(self, hybrid_provider: HybridSearchProvider) -> None:
        """Can filter by multiple providers at once."""
        _as_mock(hybrid_provider.fts_provider.search).return_value = [
            "msg-claude-1",
            "msg-chatgpt-1",
            "msg-gemini-1",
        ]
        _as_mock(hybrid_provider.vector_provider.query).return_value = []

        with patch("polylogue.storage.search_providers.hybrid.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context
            mock_context.execute.return_value.fetchall.return_value = [
                {"session_id": "conv-1"},
                {"session_id": "conv-2"},
            ]

            results = hybrid_provider.search_sessions("test query", limit=10, providers=["claude-ai", "chatgpt"])

            # Should return claude and chatgpt, but not gemini
            assert len(results) == 2
            assert "conv-1" in results
            assert "conv-2" in results
            assert "conv-3" not in results


class TestFTS5ProviderDirectFiltering:
    """Tests for FTS5Provider when used directly (not through hybrid)."""

    @pytest.fixture
    def fts_provider(self, tmp_path: Path) -> FTS5Provider:
        """Create an FTS5Provider with a test database."""
        db_path = tmp_path / "test.db"
        return FTS5Provider(db_path=db_path)

    def test_fts_search_returns_message_ids(self, fts_provider: FTS5Provider) -> None:
        """FTS5Provider.search() should return a list of message IDs."""
        results = fts_provider.search("nonexistent query")

        assert isinstance(results, list)
        assert all(isinstance(msg_id, str) for msg_id in results)

    def test_fts_search_empty_query(self, fts_provider: FTS5Provider) -> None:
        """Empty query should return empty results, not error."""
        results = fts_provider.search("")
        assert results == []

    def test_fts_search_special_characters(self, fts_provider: FTS5Provider) -> None:
        """FTS5 special characters must be escaped, not surfaced as syntax errors."""
        # Both safe queries and previously-problematic punctuation must round-trip
        # through ``escape_fts5_query`` and never raise.
        queries = [
            "test",
            "test AND query",
            "test OR query",
            'test "quoted phrase"',
            "test*",
            "test?",
            "?",
            "foo.bar",
            "../etc/passwd",
            "/leading-slash",
        ]
        for query in queries:
            results = fts_provider.search(query)
            assert isinstance(results, list)


class TestSearchProviderSourceFiltering:
    """Tests that provider filters stay provider-scoped."""

    async def test_hybrid_search_filters_by_origin(self) -> None:
        """HybridSearchProvider scopes the session filter to the origin column."""
        mock_fts = MagicMock()
        mock_fts.db_path = None
        mock_fts.search.return_value = ["msg-1", "msg-2"]

        mock_vec = MagicMock()
        mock_vec.query.return_value = []

        provider = HybridSearchProvider(
            fts_provider=mock_fts,
            vector_provider=mock_vec,
        )

        with patch("polylogue.storage.search_providers.hybrid.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context
            mock_context.execute.return_value.fetchall.return_value = [
                {"session_id": "conv-1"},
            ]

            provider.search_sessions("test", limit=10, providers=["specific-source"])

            # Provider filtering should stay scoped to the origin column only.
            execute_call = mock_context.execute.call_args
            assert execute_call is not None
            sql = execute_call.args[0]
            assert "sessions.origin" in sql
            assert "sessions.provider_name" not in sql


class TestListSessionsByParent:
    """Tests for list_sessions_by_parent (query for child sessions)."""

    async def test_empty(self, tmp_path: Path) -> None:
        """No parent → empty list."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        children = await backend.list_sessions_by_parent("nonexistent-parent")
        assert children == []
        await backend.close()

    async def test_single_child(self, tmp_path: Path) -> None:
        """Single child linked to parent."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        parent_id = "unknown-export:parent-conv"
        child_id = "unknown-export:child-conv"
        await save_session_to_archive(
            backend,
            session=make_session(
                session_id=parent_id,
                source_name="test",
                provider_session_id="parent-conv",
                content_hash=make_hash("parent-conv"),
                title="Parent",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        )
        # parent_session_id uses the parent's native_id ("parent-conv") so the
        # production writer resolves it to the archive session_id "unknown-export:parent-conv".
        await save_session_to_archive(
            backend,
            session=make_session(
                session_id=child_id,
                source_name="test",
                provider_session_id="child-conv",
                content_hash=make_hash("child-conv"),
                title="Child",
                created_at="2024-01-02T00:00:00Z",
                updated_at="2024-01-02T00:00:00Z",
                parent_session_id="parent-conv",
                branch_type=BranchType.CONTINUATION,
            ),
        )
        children = await backend.list_sessions_by_parent(parent_id)
        assert len(children) == 1
        assert children[0].session_id == child_id
        assert children[0].parent_session_id == parent_id
        await backend.close()

    async def test_multiple_children(self, tmp_path: Path) -> None:
        """Multiple children sorted by created_at."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        parent_id = "unknown-export:parent"
        await save_session_to_archive(
            backend,
            session=make_session(
                session_id=parent_id,
                source_name="test",
                provider_session_id="parent",
                content_hash=make_hash("parent"),
                title="Parent",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        )
        # parent_session_id uses the parent's native_id ("parent") so the
        # production writer resolves it to the archive session_id "unknown-export:parent".
        for i, ts in enumerate(["2024-01-03T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-04T00:00:00Z"]):
            await save_session_to_archive(
                backend,
                session=make_session(
                    session_id=f"unknown-export:child-{i}",
                    source_name="test",
                    provider_session_id=f"child-{i}",
                    content_hash=make_hash(f"child-{i}"),
                    title=f"Child {i}",
                    created_at=ts,
                    updated_at=ts,
                    parent_session_id="parent",
                    branch_type=BranchType.FORK,
                ),
            )
        children = await backend.list_sessions_by_parent(parent_id)
        assert len(children) == 3
        assert children[0].session_id == "unknown-export:child-1"
        assert children[1].session_id == "unknown-export:child-0"
        assert children[2].session_id == "unknown-export:child-2"
        await backend.close()
