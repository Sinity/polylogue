"""MCP server core tests — repository integration, search/list/get/stats tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from polylogue.lib.models import Conversation, Message
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, MessageRecord
from tests.integration.conftest import make_mock_filter

# =============================================================================
# Test data tables (SCREAMING_CASE constants)
# =============================================================================

# Stats tool test configurations
STATS_CONFIGS = [
    (100, 5000, {"claude": 50, "chatgpt": 30, "claude-code": 20}, 10, 200, 1048576, 1.0, "normal stats"),
    (0, 0, {}, 0, 0, 0, 0, "zero stats"),
    (5, 20, {"test": 5}, 0, 0, None, 0, "none db_size"),
]

# List tool filter test data
LIST_FILTER_CASES = [
    ("since", "2024-06-01", "since filter", lambda f: f.since),
    ("invalid_limit_type", [1, 2], "limit", "error expected"),
]

# Prompt edge case test data
PROMPT_EDGE_CASES = [
    (
        "analyze_errors_provider_filter",
        "analyze_errors",
        {"provider": "claude"},
        lambda f: f.provider,
        "provider filter on analyze_errors",
    ),
    (
        "analyze_errors_since_filter",
        "analyze_errors",
        {"since": "2024-01-01"},
        lambda f: f.since,
        "since filter on analyze_errors",
    ),
    (
        "extract_code_language_filter",
        "extract_code",
        {"language": "python"},
        None,
        "language filter on extract_code",
    ),
]


# =============================================================================
# MERGED FROM test_mcp_server_integration.py
# =============================================================================


class TestRepositoryViewMethod:
    """Tests for ConversationRepository.view() with ID resolution."""

    async def test_view_resolves_partial_id(self):
        """view() should call resolve_id for ID resolution."""
        # Create mock backend
        backend = Mock(spec=SQLiteBackend)

        # Mock ID resolution - returns full ID
        backend.resolve_id.return_value = "full-conv-id-12345"

        # view() will call resolve_id and then get()
        # Return None to test just the ID resolution call
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Test view with partial ID
        await repo.view("12345")

        # Should call resolve_id first
        backend.resolve_id.assert_called_once_with("12345")
        # Then try get_conversation with resolved ID
        backend.get_conversation.assert_called_once_with("full-conv-id-12345")

    async def test_view_uses_resolved_id_fallback(self):
        """view() should fall back to original ID if resolve fails."""
        backend = Mock(spec=SQLiteBackend)

        # Resolve returns None (no match found)
        backend.resolve_id.return_value = None
        # get_conversation also returns None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Test with ID that won't resolve
        await repo.view("nonexistent")

        # Should try resolve
        backend.resolve_id.assert_called_once_with("nonexistent")
        # Then try original ID
        backend.get_conversation.assert_called_once_with("nonexistent")

    async def test_view_returns_none_if_not_found(self):
        """view() should return None if conversation not found."""
        backend = Mock(spec=SQLiteBackend)
        backend.resolve_id.return_value = None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        result = await repo.view("missing-id")

        assert result is None


class TestRepositoryDataInsertion:
    """Tests for proper data insertion using backend methods."""

    async def test_save_conversation_uses_backend_methods(self):
        """save_conversation() should use await backend.save_conversation() and backend.save_messages()."""
        backend = Mock(spec=SQLiteBackend)
        backend.get_conversation = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])
        backend.get_attachments = AsyncMock(return_value=[])

        # Setup async context manager for transaction()
        backend.transaction = MagicMock()
        backend.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
        backend.transaction.return_value.__aexit__ = AsyncMock(return_value=False)

        # Mock the async backend methods that save_via_backend calls
        backend.save_conversation_record = AsyncMock()
        backend.save_messages = AsyncMock()
        backend.prune_attachments = AsyncMock()
        backend.save_attachments = AsyncMock()

        repo = ConversationRepository(backend)

        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "conv-1"
        conv_record.content_hash = "hash123"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "msg-1"
        msg_record.content_hash = "hash456"

        # Perform save operation
        result = await repo.save_conversation(
            conversation=conv_record,
            messages=[msg_record],
            attachments=[],
        )

        # Should use backend.save_conversation_record
        backend.save_conversation_record.assert_called_once_with(conv_record)

        # Should use backend.save_messages
        backend.save_messages.assert_called_once_with([msg_record])

        # Should return counts dict
        assert isinstance(result, dict)
        assert "conversations" in result
        assert "messages" in result

    async def test_backend_direct_insertion(self):
        """Test using backend directly for fixture data insertion."""
        # For test setup, use backend methods directly instead of repository.save()
        backend = Mock(spec=SQLiteBackend)

        # Mock the async backend methods
        backend.save_conversation_record = AsyncMock()
        backend.save_messages = AsyncMock()

        # Simulate what test setup would do:
        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "test-conv-1"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "test-msg-1"
        msg_record.conversation_id = "test-conv-1"

        # Use backend methods, not repository.save()
        await backend.save_conversation_record(conv_record)
        await backend.save_messages([msg_record])

        # Verify backend methods were called with correct names
        backend.save_conversation_record.assert_called_once_with(conv_record)
        backend.save_messages.assert_called_once_with([msg_record])


class TestRepositoryIntegration:
    """Integration tests between repository and backend."""

    def test_repository_wraps_backend_operations(self):
        """Repository should wrap backend for thread-safe operations.

        Write safety is provided by SQLite's BEGIN IMMEDIATE transactions
        in the backend layer, not by a Python-level lock.
        """
        backend = Mock(spec=SQLiteBackend)

        repo = ConversationRepository(backend)

        # Repository should have reference to backend
        assert repo.backend == backend
        assert hasattr(repo, "_backend")

    def test_repository_methods_exist(self):
        """Repository should have the documented methods."""
        backend = Mock(spec=SQLiteBackend)
        repo = ConversationRepository(backend)

        # Check methods exist
        assert hasattr(repo, "view")
        assert hasattr(repo, "get")
        assert hasattr(repo, "save_conversation")
        assert hasattr(repo, "search")
        assert callable(repo.view)
        assert callable(repo.get)
        assert callable(repo.save_conversation)


# =============================================================================
# Tool Tests
# =============================================================================


class TestSearchTool:
    """Tests for search tool execution."""

    @pytest.mark.asyncio
    async def test_search_with_valid_query(self, sample_conversation):
        """Search returns matching conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.search.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                # Get the tool function from the server
                search_fn = server._tool_manager._tools["search"].fn
                result = await search_fn(query="hello", limit=10)

                # Parse the JSON result
                results = json.loads(result)
                assert len(results) == 1
                assert results[0]["id"] == "test:conv-123"
                assert results[0]["provider"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_search_with_limit(self):
        """Search respects limit parameter."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.search.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                filter_instance = make_mock_filter(results=[])
                MockFilter.return_value = filter_instance

                server = _build_server()
                result = await server._tool_manager._tools["search"].fn(query="test", limit=5)

                # Verify filter was called with clamped limit
                filter_instance.limit.assert_called()
                # Parse result to verify valid JSON
                parsed = json.loads(result)
                assert parsed == []

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Search handles empty results gracefully."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.search.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await server._tool_manager._tools["search"].fn(query="nonexistent", limit=10)

                results = json.loads(result)
                assert results == []


class TestListTool:
    """Tests for list_conversations tool execution."""

    @pytest.mark.asyncio
    async def test_list_returns_conversations(self, sample_conversation):
        """List returns recent conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                result = await server._tool_manager._tools["list_conversations"].fn(limit=10)

                results = json.loads(result)
                assert len(results) == 1
                assert results[0]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_list_with_limit(self):
        """List respects limit parameter."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                filter_instance = make_mock_filter(results=[])
                MockFilter.return_value = filter_instance

                server = _build_server()
                result = await server._tool_manager._tools["list_conversations"].fn(limit=25)

                filter_instance.limit.assert_called()
                parsed = json.loads(result)
                assert parsed == []

    @pytest.mark.asyncio
    async def test_list_with_provider_filter(self):
        """List filters by provider."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                filter_instance = make_mock_filter(results=[])
                MockFilter.return_value = filter_instance

                server = _build_server()
                result = await server._tool_manager._tools["list_conversations"].fn(provider="claude", limit=10)

                filter_instance.provider.assert_called_once_with("claude")
                parsed = json.loads(result)
                assert parsed == []


class TestGetTool:
    """Tests for get_conversation tool execution."""

    def test_get_returns_conversation(self, sample_conversation):
        """Get returns full conversation with messages."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.view.return_value = sample_conversation
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_conversation"].fn(id="test:conv-123")

            conv = json.loads(result)

            assert conv["id"] == "test:conv-123"
            assert "messages" in conv
            assert len(conv["messages"]) == 2

    def test_get_not_found(self):
        """Get returns error dict for non-existent conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_conversation"].fn(id="nonexistent")

            result_dict = json.loads(result)
            assert "error" in result_dict
            assert "not found" in result_dict["error"].lower()

    def test_get_returns_full_messages(self):
        """Get returns full message text without truncation."""
        from polylogue.mcp.server import _build_server

        long_text = "A" * 2000
        conv = Conversation(
            id="test:long",
            provider="test",
            title="Long Message",
            messages=[
                Message(id="m1", role="assistant", text=long_text),
            ],
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.view.return_value = conv
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_conversation"].fn(id="test:long")

            result_dict = json.loads(result)
            msg_text = result_dict["messages"][0]["text"]

            assert msg_text == long_text


class TestStatsTool:
    """Tests for stats tool."""

    @pytest.mark.parametrize(
        "total_conversations,total_messages,providers,embedded_convs,embedded_msgs,db_size,expected_mb,desc",
        STATS_CONFIGS,
    )
    def test_stats_configurations(
        self,
        total_conversations,
        total_messages,
        providers,
        embedded_convs,
        embedded_msgs,
        db_size,
        expected_mb,
        desc,
    ):
        """Stats tool handles various configuration states."""
        from polylogue.lib.stats import ArchiveStats
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_archive_stats.return_value = ArchiveStats(
                total_conversations=total_conversations,
                total_messages=total_messages,
                providers=providers,
                embedded_conversations=embedded_convs if embedded_convs else 0,
                embedded_messages=embedded_msgs if embedded_msgs else 0,
                db_size_bytes=db_size,
            )
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["stats"].fn()

            data = json.loads(result)
            assert data["total_conversations"] == total_conversations
            assert data["total_messages"] == total_messages
            assert data["db_size_mb"] == expected_mb
