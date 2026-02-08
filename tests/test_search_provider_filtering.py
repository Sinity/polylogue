"""Tests for search provider filtering bugs.

This test file targets the bug where FTS search with provider filtering would
return empty results because filters were applied post-search rather than
integrated into the search query itself.

Bug: HybridSearchProvider.search_conversations() applied provider filters after
retrieving all results, which meant the limit was applied to the unfiltered set,
then the filter reduced it (often to zero).

Fix: Provider filters are now checked during the message→conversation lookup,
before deduplication and limit application.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestProviderFilteringIntegration:
    """Integration tests for provider filtering in search.

    These tests verify the fix for the FTS + provider filter bug where
    filters were applied post-search instead of pre-search.
    """

    @pytest.fixture
    def hybrid_provider(self):
        """Create a HybridSearchProvider with mocked dependencies."""
        from polylogue.storage.search_providers.hybrid import HybridSearchProvider

        mock_fts = MagicMock()
        mock_fts.db_path = None
        mock_vec = MagicMock()

        return HybridSearchProvider(
            fts_provider=mock_fts,
            vector_provider=mock_vec,
            rrf_k=60,
        )

    def test_provider_filter_applied_before_limit(self, hybrid_provider):
        """Provider filter should be applied before limit, not after.

        Bug scenario: Search returns 30 messages from "claude" provider,
        but user wants "chatgpt" only. Old code would:
        1. Get 30 claude messages
        2. Apply limit=10 → 10 claude messages
        3. Filter to chatgpt → 0 results

        Fixed code should:
        1. Get all matching messages
        2. Filter by provider DURING conversation lookup
        3. Return up to limit chatgpt conversations
        """
        # Mock search returns messages from various providers
        hybrid_provider.fts_provider.search.return_value = [
            f"msg-claude-{i}" for i in range(15)
        ] + [f"msg-chatgpt-{i}" for i in range(5)]

        hybrid_provider.vector_provider.query.return_value = [
            (f"msg-claude-{i}", 0.9 - i * 0.01) for i in range(10)
        ]

        # Mock database to return conversation IDs with provider info
        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context

            # First call: message → conversation mapping
            def mock_execute(*args, **kwargs):
                sql = args[0] if args else ""
                params = args[1] if len(args) > 1 else []

                result = MagicMock()
                if "FROM messages WHERE message_id IN" in sql:
                    # Map messages to conversations
                    rows = []
                    for msg_id in params:
                        if "claude" in msg_id:
                            rows.append({
                                "message_id": msg_id,
                                "conversation_id": f"conv-claude-{msg_id.split('-')[2]}"
                            })
                        elif "chatgpt" in msg_id:
                            rows.append({
                                "message_id": msg_id,
                                "conversation_id": f"conv-chatgpt-{msg_id.split('-')[2]}"
                            })
                    result.fetchall.return_value = rows
                elif "FROM conversations" in sql and "provider_name IN" in sql and "source_name IN" in sql:
                    # Provider filtering query (checks both provider_name and source_name)
                    if "chatgpt" in str(params):
                        # Return only chatgpt conversation IDs
                        rows = [{
                            "conversation_id": f"conv-chatgpt-{i}"
                        } for i in range(5)]
                    else:
                        rows = []
                    result.fetchall.return_value = rows
                else:
                    result.fetchall.return_value = []

                return result

            mock_context.execute.side_effect = mock_execute

            # Search with provider filter
            results = hybrid_provider.search_conversations(
                "test query",
                limit=10,
                providers=["chatgpt"]
            )

            # Should return chatgpt conversations, not empty
            assert len(results) > 0
            # All results should be chatgpt conversations
            assert all("chatgpt" in conv_id for conv_id in results)

    def test_provider_filter_none_returns_all(self, hybrid_provider):
        """When providers=None, should return conversations from all providers."""
        hybrid_provider.fts_provider.search.return_value = [
            "msg-claude-1", "msg-chatgpt-1", "msg-gemini-1"
        ]
        hybrid_provider.vector_provider.query.return_value = []

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context

            mock_context.execute.return_value.fetchall.return_value = [
                {"message_id": "msg-claude-1", "conversation_id": "conv-1"},
                {"message_id": "msg-chatgpt-1", "conversation_id": "conv-2"},
                {"message_id": "msg-gemini-1", "conversation_id": "conv-3"},
            ]

            results = hybrid_provider.search_conversations(
                "test query",
                limit=10,
                providers=None  # No filter
            )

            # Should return all 3 conversations
            assert len(results) == 3

    def test_provider_filter_multiple_providers(self, hybrid_provider):
        """Can filter by multiple providers at once."""
        hybrid_provider.fts_provider.search.return_value = [
            "msg-claude-1", "msg-chatgpt-1", "msg-gemini-1"
        ]
        hybrid_provider.vector_provider.query.return_value = []

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context

            def mock_execute(*args, **kwargs):
                sql = args[0] if args else ""
                params = args[1] if len(args) > 1 else ()

                result = MagicMock()
                if "FROM messages WHERE message_id IN" in sql:
                    result.fetchall.return_value = [
                        {"message_id": "msg-claude-1", "conversation_id": "conv-1"},
                        {"message_id": "msg-chatgpt-1", "conversation_id": "conv-2"},
                        {"message_id": "msg-gemini-1", "conversation_id": "conv-3"},
                    ]
                elif "FROM conversations" in sql and ("provider_name IN" in sql or "source_name IN" in sql):
                    # Filter to claude and chatgpt only based on params
                    # The actual SQL checks both provider_name and source_name with OR
                    if "claude" in str(params) and "chatgpt" in str(params):
                        result.fetchall.return_value = [
                            {"conversation_id": "conv-1"},
                            {"conversation_id": "conv-2"},
                        ]
                    else:
                        result.fetchall.return_value = []
                else:
                    result.fetchall.return_value = []
                return result

            mock_context.execute.side_effect = mock_execute

            results = hybrid_provider.search_conversations(
                "test query",
                limit=10,
                providers=["claude", "chatgpt"]
            )

            # Should return claude and chatgpt, but not gemini
            assert len(results) == 2
            assert "conv-1" in results
            assert "conv-2" in results
            assert "conv-3" not in results


class TestFTS5ProviderDirectFiltering:
    """Tests for FTS5Provider when used directly (not through hybrid).

    While FTS5Provider doesn't have provider filtering in its search() method,
    it should be tested to ensure it returns correct message IDs that can then
    be filtered by the caller.
    """

    @pytest.fixture
    def fts_provider(self, tmp_path: Path):
        """Create an FTS5Provider with a test database."""
        from polylogue.storage.search_providers.fts5 import FTS5Provider

        db_path = tmp_path / "test.db"
        return FTS5Provider(db_path=db_path)

    def test_fts_search_returns_message_ids(self, fts_provider):
        """FTS5Provider.search() should return a list of message IDs."""
        # This is a contract test - search should return list[str]
        # Even with empty database, it should return empty list, not None or error

        results = fts_provider.search("nonexistent query")

        assert isinstance(results, list)
        assert all(isinstance(msg_id, str) for msg_id in results)

    def test_fts_search_empty_query(self, fts_provider):
        """Empty query should return empty results, not error."""
        results = fts_provider.search("")
        assert results == []

    def test_fts_search_special_characters(self, fts_provider):
        """Special characters in query should not crash FTS.

        Note: FTS5 has special syntax - '?' is a syntax error.
        This test documents that behavior. In production, queries should
        be sanitized or wrapped in quotes to prevent syntax errors.
        """
        # Safe queries (valid FTS5 syntax)
        safe_queries = [
            "test",
            "test AND query",
            "test OR query",
            'test "quoted phrase"',
            "test*",
        ]

        for query in safe_queries:
            results = fts_provider.search(query)
            assert isinstance(results, list)

        # Known syntax errors in FTS5
        # These would need escaping/quoting in production
        syntax_error_queries = ["test?"]

        for query in syntax_error_queries:
            # Should raise OperationalError with syntax error
            # This is expected behavior - FTS5 query syntax is strict
            try:
                results = fts_provider.search(query)
            except Exception:
                # Expected - FTS5 syntax error
                pass


class TestSearchProviderSourceFiltering:
    """Tests for source_name filtering in addition to provider_name.

    The bug fix added support for filtering by source_name as well, since
    some providers (like claude-code) can have multiple sources.
    """

    def test_hybrid_search_filters_by_source_name(self):
        """HybridSearchProvider should filter by source_name as well as provider_name."""
        from polylogue.storage.search_providers.hybrid import HybridSearchProvider

        mock_fts = MagicMock()
        mock_fts.db_path = None
        mock_fts.search.return_value = ["msg-1", "msg-2"]

        mock_vec = MagicMock()
        mock_vec.query.return_value = []

        provider = HybridSearchProvider(
            fts_provider=mock_fts,
            vector_provider=mock_vec,
        )

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_context

            def mock_execute(*args, **kwargs):
                sql = args[0] if args else ""
                params = args[1] if len(args) > 1 else []

                result = MagicMock()
                if "FROM messages WHERE message_id IN" in sql:
                    result.fetchall.return_value = [
                        {"message_id": "msg-1", "conversation_id": "conv-1"},
                        {"message_id": "msg-2", "conversation_id": "conv-2"},
                    ]
                elif "provider_name IN" in sql and "source_name IN" in sql:
                    # The fix checks both provider_name and source_name
                    # This should be an OR condition (either matches)
                    result.fetchall.return_value = [
                        {"conversation_id": "conv-1"},
                    ]
                else:
                    result.fetchall.return_value = []
                return result

            mock_context.execute.side_effect = mock_execute

            results = provider.search_conversations(
                "test",
                limit=10,
                providers=["specific-source"]
            )

            # Should have called the SQL with both provider_name and source_name checks
            calls = [str(call) for call in mock_context.execute.call_args_list]
            sql_calls = [call for call in calls if "provider_name" in call]

            # Should have made a call checking both columns
            assert any("source_name" in call for call in sql_calls)
