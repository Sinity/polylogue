"""MCP server core tests — repository integration, search/list/get/stats tools."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from polylogue.lib.models import Conversation, Message
from polylogue.storage.backends.async_sqlite import SQLiteBackend
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

QUERY_TOOL_CASES = [
    (
        "search-basic",
        "search",
        {"query": "hello", "limit": 10},
        1,
        {
            "contains": ("hello",),
            "limit": (10,),
        },
        "test:conv-123",
    ),
    (
        "search-provider-since",
        "search",
        {"query": "hello", "provider": "claude", "since": "2024-01-01", "limit": 5},
        1,
        {
            "contains": ("hello",),
            "provider": ("claude",),
            "since": ("2024-01-01",),
            "limit": (5,),
        },
        "test:conv-123",
    ),
    (
        "list-basic",
        "list_conversations",
        {"limit": 10},
        1,
        {
            "limit": (10,),
        },
        "test:conv-123",
    ),
    (
        "list-with-filters",
        "list_conversations",
        {"provider": "claude", "since": "2024-01-01", "tag": "bug", "title": "incident", "limit": 2},
        1,
        {
            "provider": ("claude",),
            "since": ("2024-01-01",),
            "tag": ("bug",),
            "title": ("incident",),
            "limit": (2,),
        },
        "test:conv-123",
    ),
]


def _invoke_surface(fn, /, *args, **kwargs):
    """Call an MCP surface whether it is sync or async."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


async def _invoke_surface_async(fn, /, *args, **kwargs):
    """Await an MCP surface from async tests."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def _make_repo_mock() -> MagicMock:
    """Create a repository mock with async methods for MCP handlers."""
    repo = MagicMock()
    repo.view = AsyncMock(return_value=None)
    repo.get = AsyncMock(return_value=None)
    repo.get_archive_stats = AsyncMock(return_value=MagicMock())
    repo.add_tag = AsyncMock(return_value=None)
    repo.remove_tag = AsyncMock(return_value=None)
    repo.list_tags = AsyncMock(return_value={})
    repo.get_metadata = AsyncMock(return_value={})
    repo.update_metadata = AsyncMock(return_value=None)
    repo.delete_metadata = AsyncMock(return_value=None)
    repo.delete_conversation = AsyncMock(return_value=False)
    repo.resolve_id = AsyncMock(return_value=None)
    repo.get_summary = AsyncMock(return_value=None)
    repo.get_conversation_stats = AsyncMock(return_value={})
    repo.get_session_tree = AsyncMock(return_value=[])
    repo.get_stats_by = AsyncMock(return_value={})
    return repo

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


async def _insert_conversation(
    repo: ConversationRepository,
    *,
    conversation_id: str,
    provider: str,
    provider_conversation_id: str,
    text: str,
) -> None:
    conversation = ConversationRecord(
        conversation_id=conversation_id,
        provider_name=provider,
        provider_conversation_id=provider_conversation_id,
        title=f"{provider} conversation",
        content_hash=f"hash-{conversation_id}",
    )
    message = MessageRecord(
        message_id=f"{conversation_id}:m1",
        conversation_id=conversation_id,
        role="user",
        text=text,
        content_hash=f"hash-{conversation_id}:m1",
    )
    await repo.save_conversation(conversation, [message], [])


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

        # Setup async context manager for connection() — used by lightweight hash check
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)  # No existing conversation
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        backend.connection = MagicMock()
        backend.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        backend.connection.return_value.__aexit__ = AsyncMock(return_value=False)

        # Mock the async backend methods that save_via_backend calls
        backend.save_conversation_record = AsyncMock()
        backend.save_messages = AsyncMock()
        backend.upsert_conversation_stats = AsyncMock()
        backend.prune_attachments = AsyncMock()
        backend.save_attachments = AsyncMock()

        repo = ConversationRepository(backend)

        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="chatgpt",
            provider_conversation_id="provider-conv-1",
            title="Conversation",
            content_hash="hash123",
        )

        msg_record = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            provider_message_id="provider-msg-1",
            role="user",
            text="hello",
            content_hash="hash456",
            provider_name="chatgpt",
        )

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


class TestQueryTools:
    """Shared contract tests for search and list_conversations tools."""

    @pytest.mark.parametrize(
        "case_id,tool_name,args,expected_len,expected_calls,expected_first_id",
        QUERY_TOOL_CASES,
    )
    @pytest.mark.asyncio
    async def test_query_tool_filter_contract(
        self,
        simple_conversation,
        case_id,
        tool_name,
        args,
        expected_len,
        expected_calls,
        expected_first_id,
    ):
        """Query tools return JSON lists and invoke expected filter-chain methods."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = _make_repo_mock()
            mock_repo.search.return_value = [simple_conversation]
            mock_repo.list.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                filter_instance = make_mock_filter(results=[simple_conversation])
                MockFilter.return_value = filter_instance

                server = _build_server()
                raw = await server._tool_manager._tools[tool_name].fn(**args)

        payload = json.loads(raw)
        assert isinstance(payload, list), f"{case_id}: expected list payload"
        assert len(payload) == expected_len, f"{case_id}: unexpected payload length"
        assert payload[0]["id"] == expected_first_id, f"{case_id}: wrong conversation returned"

        for method_name, method_args in expected_calls.items():
            getattr(filter_instance, method_name).assert_called_once_with(*method_args)


class TestGetTool:
    """Tests for get_conversation tool execution."""

    def test_get_returns_conversation(self, simple_conversation):
        """Get returns full conversation with messages."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = _make_repo_mock()
            mock_repo.view.return_value = simple_conversation
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = _invoke_surface(server._tool_manager._tools["get_conversation"].fn, id="test:conv-123")

            conv = json.loads(result)

            assert conv["id"] == "test:conv-123"
            assert "messages" in conv
            assert len(conv["messages"]) == 2

    def test_get_not_found(self):
        """Get returns error dict for non-existent conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = _make_repo_mock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = _invoke_surface(server._tool_manager._tools["get_conversation"].fn, id="nonexistent")

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
            mock_repo = _make_repo_mock()
            mock_repo.view.return_value = conv
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = _invoke_surface(server._tool_manager._tools["get_conversation"].fn, id="test:long")

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
            mock_repo = _make_repo_mock()
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
            result = _invoke_surface(server._tool_manager._tools["stats"].fn, )

            data = json.loads(result)
            assert data["total_conversations"] == total_conversations
            assert data["total_messages"] == total_messages
            assert data["db_size_mb"] == expected_mb


class TestMCPToolValidation:
    """Test MCP tool parameter validation."""

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self):
        """Search tool handles empty query gracefully."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = _make_repo_mock()
            mock_repo.search.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await _invoke_surface_async(server._tool_manager._tools["search"].fn, query="", limit=10)

                # Should return valid JSON result (empty list or error)
                assert result is not None
                parsed = json.loads(result)
                assert isinstance(parsed, (list, dict))

    @pytest.mark.asyncio
    async def test_list_with_invalid_limit(self, tmp_path):
        """Negative limits are clamped and still return a valid list payload."""
        from polylogue.mcp.server import _build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-invalid-limit.db")
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:one",
                provider="chatgpt",
                provider_conversation_id="one",
                text="first result",
            )
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:two",
                provider="chatgpt",
                provider_conversation_id="two",
                text="second result",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = _build_server()
                result = await _invoke_surface_async(server._tool_manager._tools["list_conversations"].fn, limit=-1)
                parsed = json.loads(result)
                assert isinstance(parsed, list)
                assert len(parsed) == 1
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_get_with_nonexistent_id(self):
        """Get tool handles nonexistent conversation ID gracefully."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = _make_repo_mock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = await _invoke_surface_async(server._tool_manager._tools["get_conversation"].fn, id="nonexistent-id-xyz")

            assert result is not None
            parsed = json.loads(result)
            # Should return error dict or empty
            assert isinstance(parsed, dict)


class TestMCPRealRepositoryPaths:
    """Integration tests that exercise real repository + filter behavior."""

    @pytest.mark.asyncio
    async def test_search_uses_real_repository_and_filter_stack(self, tmp_path):
        """Search tool should return persisted conversations from a real temp DB."""
        from polylogue.mcp.server import _build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-search.db")
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:needle",
                provider="chatgpt",
                provider_conversation_id="needle",
                text="finding a needle in a haystack",
            )
            await _insert_conversation(
                repo,
                conversation_id="claude:other",
                provider="claude",
                provider_conversation_id="other",
                text="something unrelated",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = _build_server()
                result = await _invoke_surface_async(server._tool_manager._tools["search"].fn, query="needle", limit=10)

            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["id"] == "chatgpt:needle"
            assert parsed[0]["provider"] == "chatgpt"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_applies_provider_filter_on_real_repository(self, tmp_path):
        """List tool provider filter should scope results in a real repository."""
        from polylogue.mcp.server import _build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-list.db")
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:one",
                provider="chatgpt",
                provider_conversation_id="one",
                text="chatgpt content",
            )
            await _insert_conversation(
                repo,
                conversation_id="claude:one",
                provider="claude",
                provider_conversation_id="one",
                text="claude content",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = _build_server()
                result = await _invoke_surface_async(server._tool_manager._tools["list_conversations"].fn, provider="claude", limit=10)

            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["id"] == "claude:one"
            assert parsed[0]["provider"] == "claude"
        finally:
            await backend.close()
