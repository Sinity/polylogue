"""Tests for MCP server tools, resources, and prompts.

Tests the FastMCP-based server in polylogue/mcp/server.py.

Covers:
- Tool execution (search, list_conversations, get_conversation, stats)
- Resource reading (stats, conversations, conversation/{id})
- Prompt generation (analyze_errors, summarize_week, extract_code)
- Serialization helpers and filtering logic
- Repository integration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, MessageRecord

# =============================================================================
# Fixtures & Helpers
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock ConversationRepository."""
    repo = MagicMock()

    # Default behaviors
    repo.list.return_value = []
    repo.search.return_value = []
    repo.view.return_value = None
    repo.get.return_value = None
    repo.resolve_id.return_value = None
    repo.get_archive_stats.return_value = MagicMock()

    return repo


def make_mock_filter(results=None, **method_overrides):
    """Create a pre-configured mock ConversationFilter.

    Args:
        results: List of conversations to return from .list()
        **method_overrides: Set side_effect for any method (e.g., since=ValueError("Bad date"))

    Returns:
        Configured MagicMock filter instance with chaining support.
    """
    f = MagicMock()
    # All these methods should return self for chaining
    for method in ("provider", "contains", "after", "before", "tags", "title", "since", "limit", "tag"):
        getattr(f, method).return_value = f

    # Set list() to return provided results
    f.list.return_value = results or []

    # Apply any method-specific overrides (for side_effect, e.g.)
    for method_name, override_value in method_overrides.items():
        method = getattr(f, method_name)
        if isinstance(override_value, Exception):
            method.side_effect = override_value
        else:
            method.return_value = override_value

    return f


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return Conversation(
        id="test:conv-123",
        provider="chatgpt",
        title="Test Conversation",
        messages=[
            Message(
                id="msg-1",
                role="user",
                text="Hello, how are you?",
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            ),
            Message(
                id="msg-2",
                role="assistant",
                text="I'm doing well, thank you!",
                timestamp=datetime(2024, 1, 15, 10, 30, 30, tzinfo=timezone.utc),
            ),
        ],
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )


# =============================================================================
# Test data tables (SCREAMING_CASE constants)
# =============================================================================

# Stats tool test configurations
STATS_CONFIGS = [
    (100, 5000, {"claude": 50, "chatgpt": 30, "claude-code": 20}, 10, 200, 1048576, 1.0, "normal stats"),
    (0, 0, {}, 0, 0, 0, 0, "zero stats"),
    (5, 20, {"test": 5}, 0, 0, None, 0, "none db_size"),
]

# Search tool filter test data
SEARCH_FILTER_CASES = [
    ("provider", "claude", "provider filter", lambda f: f.provider),
    ("since", "2024-01-01", "since filter", lambda f: f.since),
    ("limit_normal", 5, "normal limit", lambda f: f.limit),
    ("limit_max", 99999, "limit above max", lambda f: f.limit),
    ("limit_negative", -5, "negative limit", lambda f: f.limit),
]

# Search error cases
SEARCH_ERROR_CASES = [
    ("invalid_since", {"query": "test", "since": "not-a-date"}, "date", "Invalid date in error"),
    ("invalid_limit_type", {"query": "test", "limit": "not-a-number"}, "limit", "limit in error message"),
]

# List tool filter test data
LIST_FILTER_CASES = [
    ("since", "2024-06-01", "since filter", lambda f: f.since),
    ("invalid_limit_type", [1, 2], "limit", "error expected"),
]

# Serialization edge case test data
SERIALIZATION_CASES = [
    (
        "no_timestamps",
        Conversation(id="t1", provider="test", title="No Times", messages=[]),
        {"created_at": None, "updated_at": None, "message_count": 0},
        "_conversation_to_dict",
    ),
    (
        "empty_messages",
        Conversation(id="t2", provider="test", title="Empty", messages=[]),
        {"messages": []},
        "_conversation_to_full_dict",
    ),
    (
        "empty_role",
        Conversation(
            id="t3",
            provider="test",
            title="Empty Role",
            messages=[Message(id="m1", role="", text="test")],
        ),
        {"messages": [{"role": "unknown"}]},
        "_conversation_to_full_dict",
    ),
    (
        "null_text",
        Conversation(
            id="t4",
            provider="test",
            title="Null Text",
            messages=[Message(id="m1", role="user", text=None)],
        ),
        {"messages": [{"text": ""}]},
        "_conversation_to_full_dict",
    ),
    (
        "null_timestamp",
        Conversation(
            id="t5",
            provider="test",
            title="No TS",
            messages=[Message(id="m1", role="user", text="hi")],
        ),
        {"messages": [{"timestamp": None}]},
        "_conversation_to_full_dict",
    ),
    (
        "unusual_role",
        Conversation(
            id="t6",
            provider="test",
            title="Unusual Role",
            messages=[Message(id="m1", role="tool", text="test")],
        ),
        {"messages": [{"role": "tool"}]},
        "_conversation_to_full_dict",
    ),
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

    def test_view_resolves_partial_id(self):
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
        result = repo.view("12345")

        # Should call resolve_id first
        backend.resolve_id.assert_called_once_with("12345")
        # Then try get_conversation with resolved ID
        backend.get_conversation.assert_called_once_with("full-conv-id-12345")

    def test_view_uses_resolved_id_fallback(self):
        """view() should fall back to original ID if resolve fails."""
        backend = Mock(spec=SQLiteBackend)

        # Resolve returns None (no match found)
        backend.resolve_id.return_value = None
        # get_conversation also returns None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Test with ID that won't resolve
        result = repo.view("nonexistent")

        # Should try resolve
        backend.resolve_id.assert_called_once_with("nonexistent")
        # Then try original ID
        backend.get_conversation.assert_called_once_with("nonexistent")

    def test_view_returns_none_if_not_found(self):
        """view() should return None if conversation not found."""
        backend = Mock(spec=SQLiteBackend)
        backend.resolve_id.return_value = None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        result = repo.view("missing-id")

        assert result is None


class TestRepositoryDataInsertion:
    """Tests for proper data insertion using backend methods."""

    def test_save_conversation_uses_backend_methods(self):
        """save_conversation() should use backend.save_conversation() and backend.save_messages()."""
        backend = Mock(spec=SQLiteBackend)
        backend.get_conversation.return_value = None
        backend.get_messages.return_value = []
        backend.get_attachments.return_value = []

        repo = ConversationRepository(backend)

        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "conv-1"
        conv_record.content_hash = "hash123"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "msg-1"
        msg_record.content_hash = "hash456"

        # Perform save operation
        result = repo.save_conversation(
            conversation=conv_record,
            messages=[msg_record],
            attachments=[],
        )

        # Should use backend.save_conversation
        backend.save_conversation.assert_called()

        # Should use backend.save_messages
        backend.save_messages.assert_called()

        # Should return counts dict
        assert isinstance(result, dict)
        assert "conversations" in result
        assert "messages" in result

    def test_backend_direct_insertion(self):
        """Test using backend directly for fixture data insertion."""
        # For test setup, use backend methods directly instead of repository.save()
        backend = Mock(spec=SQLiteBackend)

        # Simulate what test setup would do:
        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "test-conv-1"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "test-msg-1"
        msg_record.conversation_id = "test-conv-1"

        # Use backend methods, not repository.save()
        backend.save_conversation(conv_record)
        backend.save_messages([msg_record])

        # Verify backend methods were called
        backend.save_conversation.assert_called_once_with(conv_record)
        backend.save_messages.assert_called_once_with([msg_record])


class TestRepositoryIntegration:
    """Integration tests between repository and backend."""

    def test_repository_wraps_backend_operations(self):
        """Repository should wrap backend for thread-safe operations."""
        backend = Mock(spec=SQLiteBackend)

        repo = ConversationRepository(backend)

        # Repository should have reference to backend
        assert repo.backend == backend
        assert hasattr(repo, "_write_lock")

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

    def test_search_with_valid_query(self, sample_conversation):
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
                result = search_fn(query="hello", limit=10)

                # Parse the JSON result
                results = json.loads(result)
                assert len(results) == 1
                assert results[0]["id"] == "test:conv-123"
                assert results[0]["provider"] == "chatgpt"

    def test_search_with_limit(self):
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
                result = server._tool_manager._tools["search"].fn(query="test", limit=5)

                # Verify filter was called with clamped limit
                filter_instance.limit.assert_called()
                # Parse result to verify valid JSON
                parsed = json.loads(result)
                assert parsed == []

    def test_search_empty_results(self):
        """Search handles empty results gracefully."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.search.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = server._tool_manager._tools["search"].fn(query="nonexistent", limit=10)

                results = json.loads(result)
                assert results == []


class TestListTool:
    """Tests for list_conversations tool execution."""

    def test_list_returns_conversations(self, sample_conversation):
        """List returns recent conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                result = server._tool_manager._tools["list_conversations"].fn(limit=10)

                results = json.loads(result)
                assert len(results) == 1
                assert results[0]["message_count"] == 2

    def test_list_with_limit(self):
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
                result = server._tool_manager._tools["list_conversations"].fn(limit=25)

                filter_instance.limit.assert_called()
                parsed = json.loads(result)
                assert parsed == []

    def test_list_with_provider_filter(self):
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
                result = server._tool_manager._tools["list_conversations"].fn(provider="claude", limit=10)

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
        from polylogue.mcp.server import _build_server
        from polylogue.lib.stats import ArchiveStats

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


# =============================================================================
# Resource Tests
# =============================================================================


class TestStatsResource:
    """Tests for polylogue://stats resource."""

    def test_stats_returns_archive_statistics(self):
        """Stats resource returns conversation and message counts."""
        from polylogue.mcp.server import _build_server
        from polylogue.lib.stats import ArchiveStats

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_archive_stats.return_value = ArchiveStats(
                total_conversations=2,
                total_messages=4,
                providers={"chatgpt": 2},
            )
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._resource_manager._resources["polylogue://stats"].fn()

            stats = json.loads(result)
            assert stats["total_conversations"] == 2
            assert stats["total_messages"] == 4


class TestConversationsResource:
    """Tests for polylogue://conversations resource."""

    def test_conversations_resource_returns_list(self, sample_conversation):
        """Conversations resource returns all conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                result = server._resource_manager._resources["polylogue://conversations"].fn()

                convs = json.loads(result)
                assert len(convs) == 1


class TestConversationResource:
    """Tests for polylogue://conversation/{id} resource."""

    def test_single_conversation_resource(self, sample_conversation):
        """Single conversation resource returns full conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get.return_value = sample_conversation
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn(conv_id="test:conv-123")

            conv = json.loads(result)
            assert conv["id"] == "test:conv-123"
            assert "messages" in conv

    def test_conversation_resource_not_found(self):
        """Conversation resource returns error for missing conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn(conv_id="nonexistent")

            result_dict = json.loads(result)
            assert "error" in result_dict


# =============================================================================
# Prompt Tests
# =============================================================================


class TestAnalyzeErrorsPrompt:
    """Tests for analyze_errors prompt."""

    def test_analyze_errors_with_conversations(self, sample_conversation):
        """analyze_errors generates prompt from error conversations."""
        from polylogue.mcp.server import _build_server

        # Modify sample to have error message
        sample_conversation.messages[0].text = "Got an error while running"

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                result = server._prompt_manager._prompts["analyze_errors"].fn()

                assert isinstance(result, str)
                assert "error" in result.lower()

    def test_analyze_errors_limits_error_contexts_to_20(self):
        """analyze_errors stops collecting after 20 error snippets."""
        from polylogue.mcp.server import _build_server

        # Create conversation with 30 error messages
        msgs = [
            Message(
                id=f"m{i}",
                role="user",
                text=f"error #{i} occurred",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            )
            for i in range(30)
        ]
        big_conv = Conversation(
            id="big", provider="test", title="Errors", messages=msgs
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [big_conv]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[big_conv])

                server = _build_server()
                result = server._prompt_manager._prompts["analyze_errors"].fn()

                # Should have "20 error instances found"
                assert "20 error instances" in result

    def test_analyze_errors_no_matches(self):
        """analyze_errors handles zero matching conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = server._prompt_manager._prompts["analyze_errors"].fn()

                assert "0 conversations" in result


class TestSummarizeWeekPrompt:
    """Tests for summarize_week prompt."""

    def test_summarize_week_empty(self):
        """summarize_week handles no conversations in past week."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = server._prompt_manager._prompts["summarize_week"].fn()

                assert "0 conversations" in result
                assert "0 messages" in result


class TestExtractCodePrompt:
    """Tests for extract_code prompt."""

    def test_extract_code_no_code_blocks(self):
        """extract_code handles conversations without code."""
        from polylogue.mcp.server import _build_server

        conv = Conversation(
            id="nocode",
            provider="test",
            title="No Code",
            messages=[Message(id="m1", role="user", text="Just text, no code")],
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [conv]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = server._prompt_manager._prompts["extract_code"].fn()

                assert "0 code blocks" in result

    def test_extract_code_with_language_filter(self):
        """extract_code filters by language."""
        from polylogue.mcp.server import _build_server

        conv = Conversation(
            id="code",
            provider="test",
            title="Code",
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    text="```python\nprint('hi')\n```\n```javascript\nconsole.log('hi')\n```",
                )
            ],
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [conv]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = server._prompt_manager._prompts["extract_code"].fn(language="python")

                assert "python" in result.lower()

    def test_extract_code_null_message_text(self):
        """extract_code skips messages with None text."""
        from polylogue.mcp.server import _build_server

        conv = Conversation(
            id="nulltext",
            provider="test",
            title="Null",
            messages=[Message(id="m1", role="assistant", text=None)],
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [conv]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = server._prompt_manager._prompts["extract_code"].fn()

                assert isinstance(result, str)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestConversationSerialization:
    """Tests for conversation serialization helpers."""

    def test_conversation_to_dict(self, sample_conversation):
        """_conversation_to_dict serializes conversation metadata."""
        from polylogue.mcp.server import _conversation_to_dict

        result = _conversation_to_dict(sample_conversation)

        assert result["id"] == "test:conv-123"
        assert result["provider"] == "chatgpt"
        assert result["message_count"] == 2
        assert "created_at" in result
        assert "updated_at" in result

    def test_conversation_to_full_dict(self, sample_conversation):
        """_conversation_to_full_dict includes messages."""
        from polylogue.mcp.server import _conversation_to_full_dict

        result = _conversation_to_full_dict(sample_conversation)

        assert "messages" in result
        assert len(result["messages"]) == 2

        msg = result["messages"][0]
        assert "id" in msg
        assert "role" in msg
        assert "text" in msg
        assert "timestamp" in msg

    @pytest.mark.parametrize(
        "case_id,conv,expected_fields,func_name",
        SERIALIZATION_CASES,
    )
    def test_serialization_edge_cases(
        self,
        case_id,
        conv,
        expected_fields,
        func_name,
    ):
        """Serialization handles edge cases."""
        if func_name == "_conversation_to_dict":
            from polylogue.mcp.server import _conversation_to_dict

            result = _conversation_to_dict(conv)
        else:
            from polylogue.mcp.server import _conversation_to_full_dict

            result = _conversation_to_full_dict(conv)

        for key, expected_value in expected_fields.items():
            if key == "messages":
                assert key in result
                if expected_value:
                    for i, expected_msg in enumerate(expected_value):
                        for msg_key, msg_val in expected_msg.items():
                            assert result[key][i][msg_key] == msg_val
                else:
                    assert result[key] == expected_value
            elif key == "created_at" or key == "updated_at":
                assert result[key] == expected_value
            else:
                assert result[key] == expected_value, f"Failed for case {case_id}: {key}"


class TestClampLimit:
    """Tests for _clamp_limit helper."""

    def test_clamp_limit_normal(self):
        """_clamp_limit passes through normal values."""
        from polylogue.mcp.server import _clamp_limit

        assert _clamp_limit(10) == 10
        assert _clamp_limit(1) == 1
        assert _clamp_limit(5000) == 5000

    def test_clamp_limit_max(self):
        """_clamp_limit caps at _MAX_LIMIT."""
        from polylogue.mcp.server import _clamp_limit

        assert _clamp_limit(99999) == 10000
        assert _clamp_limit(10001) == 10000

    def test_clamp_limit_min(self):
        """_clamp_limit floors at 1."""
        from polylogue.mcp.server import _clamp_limit

        assert _clamp_limit(0) == 1
        assert _clamp_limit(-5) == 1

    def test_clamp_limit_type_coercion(self):
        """_clamp_limit coerces types and handles errors."""
        from polylogue.mcp.server import _clamp_limit

        # Valid string
        assert _clamp_limit("10") == 10

        # Invalid types return default
        assert _clamp_limit("not-a-number") == 10
        assert _clamp_limit([1, 2]) == 10


# =============================================================================
# Search Tool Filter Tests (Parametrized)
# =============================================================================


class TestSearchToolFilters:
    """Tests for search tool edge cases and filters."""

    @pytest.mark.parametrize(
        "filter_type,filter_value,desc,filter_checker",
        SEARCH_FILTER_CASES,
    )
    def test_search_filter_application(
        self,
        filter_type,
        filter_value,
        desc,
        filter_checker,
    ):
        """Search applies filters correctly."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.search.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                filter_instance = make_mock_filter(results=[])
                MockFilter.return_value = filter_instance

                server = _build_server()

                args = {"query": "test"}
                if filter_type.startswith("limit"):
                    args["limit"] = filter_value
                else:
                    args[filter_type] = filter_value

                result = server._tool_manager._tools["search"].fn(**args)

                assert isinstance(result, str)
                filter_instance = MockFilter.return_value
                if filter_type == "limit_normal":
                    filter_checker(filter_instance).assert_called_once_with(5)
                elif filter_type == "limit_max":
                    filter_checker(filter_instance).assert_called_once_with(10000)
                elif filter_type == "limit_negative":
                    filter_checker(filter_instance).assert_called_once_with(1)
                else:
                    filter_checker(filter_instance).assert_called_once_with(filter_value)

    @pytest.mark.parametrize(
        "case_type,args,error_keyword,assertion_desc",
        SEARCH_ERROR_CASES,
    )
    def test_search_error_cases(
        self,
        case_type,
        args,
        error_keyword,
        assertion_desc,
    ):
        """Search handles invalid parameters."""
        from polylogue.mcp.server import _build_server

        if case_type == "invalid_since":
            with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
                mock_repo = MagicMock()
                mock_get_repo.return_value = mock_repo

                with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                    MockFilter.return_value = make_mock_filter(since=ValueError("Invalid date"))

                    server = _build_server()
                    # Tool should raise or return error dict
                    try:
                        result = server._tool_manager._tools["search"].fn(**args)
                        # If no exception, check result contains error info
                        parsed = json.loads(result)
                        # Tools that catch errors may return dict with error key
                    except ValueError:
                        pass  # Expected for invalid date
        else:
            with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
                mock_repo = MagicMock()
                mock_get_repo.return_value = mock_repo

                with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                    MockFilter.return_value = make_mock_filter(results=[])

                    server = _build_server()
                    # For invalid_limit_type, tool should handle gracefully
                    try:
                        result = server._tool_manager._tools["search"].fn(**args)
                        parsed = json.loads(result)
                    except (TypeError, ValueError):
                        pass  # Expected for invalid limit type


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestServerBuilding:
    """Tests for server construction."""

    def test_build_server_returns_fastmcp_instance(self):
        """_build_server returns a FastMCP server instance."""
        from polylogue.mcp.server import _build_server

        server = _build_server()
        assert server is not None
        assert hasattr(server, "_tool_manager")
        assert hasattr(server, "_resource_manager")
        assert hasattr(server, "_prompt_manager")

    def test_server_has_all_tools(self):
        """Built server has all expected tools."""
        from polylogue.mcp.server import _build_server

        server = _build_server()
        tool_names = set(server._tool_manager._tools.keys())

        assert "search" in tool_names
        assert "list_conversations" in tool_names
        assert "get_conversation" in tool_names
        assert "stats" in tool_names

    def test_server_has_all_resources(self):
        """Built server has all expected resources and templates."""
        from polylogue.mcp.server import _build_server

        server = _build_server()
        resource_uris = set(server._resource_manager._resources.keys())
        template_uris = set(server._resource_manager._templates.keys())

        assert "polylogue://stats" in resource_uris
        assert "polylogue://conversations" in resource_uris
        assert "polylogue://conversation/{conv_id}" in template_uris

    def test_server_has_all_prompts(self):
        """Built server has all expected prompts."""
        from polylogue.mcp.server import _build_server

        server = _build_server()
        prompt_names = set(server._prompt_manager._prompts.keys())

        assert "analyze_errors" in prompt_names
        assert "summarize_week" in prompt_names
        assert "extract_code" in prompt_names
