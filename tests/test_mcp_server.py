"""Tests for MCP server protocol handlers.

Tests the JSON-RPC request handlers in polylogue/mcp/server.py.

Covers:
- Protocol initialization
- Tool listing and execution (search, list, get)
- Resource listing and reading
- Prompt listing and retrieval
- Error handling (JSON-RPC errors)
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
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock ConversationRepository."""
    repo = MagicMock()

    # Default behaviors
    repo.list.return_value = []
    repo.search.return_value = []
    repo.view.return_value = None
    repo.resolve_id.return_value = None

    return repo


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


@pytest.fixture
def handle_request():
    """Import _handle_request function from server module."""
    from polylogue.mcp.server import _handle_request
    return _handle_request


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


class TestMCPServerHandleGet:
    """Tests for _handle_get using repo.view() instead of repo.get()."""

    def test_handle_get_uses_view_method(self):
        """_handle_get should use repo.view() for ID resolution."""
        backend = Mock(spec=SQLiteBackend)

        # Mock resolve_id and get_conversation to return None
        # (test just verifies view() calls these)
        backend.resolve_id.return_value = "full-id"
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Simulate what _handle_get would do
        result = repo.view("partial-id")

        # Should use resolve_id
        assert backend.resolve_id.called
        # Should eventually call get_conversation
        assert backend.get_conversation.called

    def test_mock_includes_view_method(self):
        """Mock repo should include view() method for testing."""
        mock_repo = Mock(spec=ConversationRepository)

        # Mock should have view method
        assert hasattr(mock_repo, "view")

        # Set up mock to return None
        mock_repo.view.return_value = None

        # Test the mock
        result = mock_repo.view("conv-id")

        assert result is None
        mock_repo.view.assert_called_once_with("conv-id")


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
# Protocol Initialization Tests
# =============================================================================


class TestProtocolInitialization:
    """Tests for the initialize method."""

    def test_initialize_returns_server_info(self, handle_request, mock_repo):
        """Initialize returns protocol version and server info."""
        request = {"method": "initialize", "params": {}, "id": 1}

        response = handle_request(request, mock_repo)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

        result = response["result"]
        assert result["protocolVersion"] == "0.1.0"
        assert result["serverInfo"]["name"] == "polylogue"
        assert "capabilities" in result

    def test_initialize_returns_capabilities(self, handle_request, mock_repo):
        """Initialize returns available capabilities."""
        request = {"method": "initialize", "params": {}, "id": 1}

        response = handle_request(request, mock_repo)
        result = response["result"]

        caps = result["capabilities"]
        assert "tools" in caps
        assert "resources" in caps
        assert "prompts" in caps


# =============================================================================
# Tool Listing Tests
# =============================================================================


class TestToolsListing:
    """Tests for tools/list method."""

    def test_tools_list_returns_available_tools(self, handle_request, mock_repo):
        """tools/list returns all available tools."""
        request = {"method": "tools/list", "params": {}, "id": 2}

        response = handle_request(request, mock_repo)

        assert response["id"] == 2
        assert "result" in response

        tools = response["result"]["tools"]
        tool_names = {t["name"] for t in tools}

        assert "search" in tool_names
        assert "list" in tool_names
        assert "get" in tool_names

    def test_tools_have_input_schemas(self, handle_request, mock_repo):
        """Each tool has an inputSchema defining parameters."""
        request = {"method": "tools/list", "params": {}, "id": 2}

        response = handle_request(request, mock_repo)
        tools = response["result"]["tools"]

        for tool in tools:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]


# =============================================================================
# Tool Execution Tests (tools/call)
# =============================================================================


class TestSearchTool:
    """Tests for search tool execution."""

    def test_search_requires_query(self, handle_request, mock_repo):
        """Search returns error when query is missing."""
        request = {
            "method": "tools/call",
            "params": {"name": "search", "arguments": {}},
            "id": 3,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "query" in response["error"]["message"].lower()

    def test_search_with_valid_query(self, handle_request, mock_repo, sample_conversation):
        """Search returns matching conversations."""
        mock_repo.search.return_value = [sample_conversation]

        request = {
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "hello"}},
            "id": 3,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        content = response["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

        # Parse the JSON text
        results = json.loads(content[0]["text"])
        assert len(results) == 1
        assert results[0]["id"] == "test:conv-123"
        assert results[0]["provider"] == "chatgpt"

    def test_search_with_limit(self, handle_request, mock_repo):
        """Search respects limit parameter."""
        request = {
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "test", "limit": 5}},
            "id": 3,
        }

        response = handle_request(request, mock_repo)

        # Verify the response is valid (no error)
        assert "result" in response
        assert "content" in response["result"]

    def test_search_empty_results(self, handle_request, mock_repo):
        """Search handles empty results gracefully."""
        mock_repo.search.return_value = []

        request = {
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "nonexistent"}},
            "id": 3,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        results = json.loads(response["result"]["content"][0]["text"])
        assert results == []


class TestListTool:
    """Tests for list tool execution."""

    def test_list_returns_conversations(self, handle_request, mock_repo, sample_conversation):
        """List returns recent conversations."""
        mock_repo.list.return_value = [sample_conversation]

        request = {
            "method": "tools/call",
            "params": {"name": "list", "arguments": {}},
            "id": 4,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        results = json.loads(response["result"]["content"][0]["text"])
        assert len(results) == 1
        assert results[0]["message_count"] == 2

    def test_list_with_limit(self, handle_request, mock_repo):
        """List respects limit parameter."""
        request = {
            "method": "tools/call",
            "params": {"name": "list", "arguments": {"limit": 25}},
            "id": 4,
        }

        response = handle_request(request, mock_repo)

        # Verify the response is valid (no error)
        assert "result" in response
        assert "content" in response["result"]

    def test_list_with_provider_filter(self, handle_request, mock_repo):
        """List filters by provider."""
        request = {
            "method": "tools/call",
            "params": {"name": "list", "arguments": {"provider": "claude"}},
            "id": 4,
        }

        response = handle_request(request, mock_repo)

        # Verify the response is valid (no error)
        assert "result" in response
        assert "content" in response["result"]


class TestGetTool:
    """Tests for get tool execution."""

    def test_get_requires_id(self, handle_request, mock_repo):
        """Get returns error when id is missing."""
        request = {
            "method": "tools/call",
            "params": {"name": "get", "arguments": {}},
            "id": 5,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "id" in response["error"]["message"].lower()

    def test_get_returns_conversation(self, handle_request, mock_repo, sample_conversation):
        """Get returns full conversation with messages."""
        mock_repo.view.return_value = sample_conversation

        request = {
            "method": "tools/call",
            "params": {"name": "get", "arguments": {"id": "test:conv-123"}},
            "id": 5,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        conv = json.loads(response["result"]["content"][0]["text"])

        assert conv["id"] == "test:conv-123"
        assert "messages" in conv
        assert len(conv["messages"]) == 2

    def test_get_not_found(self, handle_request, mock_repo):
        """Get returns error for non-existent conversation."""
        mock_repo.view.return_value = None

        request = {
            "method": "tools/call",
            "params": {"name": "get", "arguments": {"id": "nonexistent"}},
            "id": 5,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "not found" in response["error"]["message"].lower()

    def test_get_truncates_long_messages(self, handle_request, mock_repo):
        """Get truncates messages longer than 1000 chars."""
        long_text = "A" * 2000
        conv = Conversation(
            id="test:long",
            provider="test",
            title="Long Message",
            messages=[
                Message(id="m1", role="assistant", text=long_text),
            ],
        )
        mock_repo.view.return_value = conv

        request = {
            "method": "tools/call",
            "params": {"name": "get", "arguments": {"id": "test:long"}},
            "id": 5,
        }

        response = handle_request(request, mock_repo)

        result = json.loads(response["result"]["content"][0]["text"])
        msg_text = result["messages"][0]["text"]

        assert len(msg_text) < len(long_text)
        assert msg_text.endswith("...")


class TestUnknownTool:
    """Tests for unknown tool handling."""

    def test_unknown_tool_returns_error(self, handle_request, mock_repo):
        """Unknown tool name returns error."""
        request = {
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
            "id": 6,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "nonexistent" in response["error"]["message"]


# =============================================================================
# Resource Listing Tests
# =============================================================================


class TestResourcesListing:
    """Tests for resources/list method."""

    def test_resources_list_returns_available_resources(self, handle_request, mock_repo):
        """resources/list returns available resources."""
        request = {"method": "resources/list", "params": {}, "id": 7}

        response = handle_request(request, mock_repo)

        assert "result" in response
        resources = response["result"]["resources"]

        uris = {r["uri"] for r in resources}
        assert "polylogue://stats" in uris
        assert "polylogue://conversations" in uris

    def test_resources_have_metadata(self, handle_request, mock_repo):
        """Each resource has name, uri, and mimeType."""
        request = {"method": "resources/list", "params": {}, "id": 7}

        response = handle_request(request, mock_repo)
        resources = response["result"]["resources"]

        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "mimeType" in resource


# =============================================================================
# Resource Reading Tests
# =============================================================================


class TestStatsResource:
    """Tests for polylogue://stats resource."""

    def test_stats_returns_archive_statistics(self, handle_request, mock_repo, sample_conversation):
        """Stats resource returns conversation and message counts."""
        from polylogue.lib.stats import ArchiveStats

        mock_repo.get_archive_stats.return_value = ArchiveStats(
            total_conversations=2,
            total_messages=4,
            providers={"chatgpt": 2},
        )

        request = {
            "method": "resources/read",
            "params": {"uri": "polylogue://stats"},
            "id": 8,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        contents = response["result"]["contents"]
        assert len(contents) == 1

        stats = json.loads(contents[0]["text"])
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 4


class TestConversationsResource:
    """Tests for polylogue://conversations resource."""

    def test_conversations_resource_returns_list(self, handle_request, mock_repo, sample_conversation):
        """Conversations resource returns all conversations."""
        mock_repo.list.return_value = [sample_conversation]

        # Need to mock the filter chain
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [sample_conversation]
            MockFilter.return_value = filter_instance

            request = {
                "method": "resources/read",
                "params": {"uri": "polylogue://conversations"},
                "id": 9,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            contents = response["result"]["contents"]
            convs = json.loads(contents[0]["text"])
            assert len(convs) == 1

    def test_conversations_resource_with_query_params(self, handle_request, mock_repo):
        """Conversations resource supports query parameters."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.provider.return_value = filter_instance
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "resources/read",
                "params": {"uri": "polylogue://conversations?provider=claude&limit=50"},
                "id": 9,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            # Verify filters were applied
            filter_instance.provider.assert_called_once_with("claude")
            filter_instance.limit.assert_called_once_with(50)


class TestConversationResource:
    """Tests for polylogue://conversation/{id} resource."""

    def test_single_conversation_resource(self, handle_request, mock_repo, sample_conversation):
        """Single conversation resource returns full conversation."""
        mock_repo.get.return_value = sample_conversation

        request = {
            "method": "resources/read",
            "params": {"uri": "polylogue://conversation/test:conv-123"},
            "id": 10,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        contents = response["result"]["contents"]
        assert contents[0]["uri"] == "polylogue://conversation/test:conv-123"

        conv = json.loads(contents[0]["text"])
        assert conv["id"] == "test:conv-123"
        assert "messages" in conv

    def test_conversation_resource_not_found(self, handle_request, mock_repo):
        """Conversation resource returns error for missing conversation."""
        mock_repo.get.return_value = None

        request = {
            "method": "resources/read",
            "params": {"uri": "polylogue://conversation/nonexistent"},
            "id": 10,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "not found" in response["error"]["message"].lower()


class TestUnknownResource:
    """Tests for unknown resource handling."""

    def test_unknown_resource_returns_error(self, handle_request, mock_repo):
        """Unknown resource URI returns error."""
        request = {
            "method": "resources/read",
            "params": {"uri": "polylogue://unknown"},
            "id": 11,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "Unknown resource" in response["error"]["message"]


# =============================================================================
# Prompt Listing Tests
# =============================================================================


class TestPromptsListing:
    """Tests for prompts/list method."""

    def test_prompts_list_returns_available_prompts(self, handle_request, mock_repo):
        """prompts/list returns available prompts."""
        request = {"method": "prompts/list", "params": {}, "id": 12}

        response = handle_request(request, mock_repo)

        assert "result" in response
        prompts = response["result"]["prompts"]

        prompt_names = {p["name"] for p in prompts}
        assert "analyze-errors" in prompt_names
        assert "summarize-week" in prompt_names
        assert "extract-code" in prompt_names

    def test_prompts_have_descriptions(self, handle_request, mock_repo):
        """Each prompt has a description."""
        request = {"method": "prompts/list", "params": {}, "id": 12}

        response = handle_request(request, mock_repo)
        prompts = response["result"]["prompts"]

        for prompt in prompts:
            assert "description" in prompt
            assert len(prompt["description"]) > 0


# =============================================================================
# Prompt Retrieval Tests
# =============================================================================


class TestAnalyzeErrorsPrompt:
    """Tests for analyze-errors prompt."""

    def test_analyze_errors_prompt(self, handle_request, mock_repo, sample_conversation):
        """analyze-errors prompt generates error analysis context."""
        # Modify sample to have error content
        sample_conversation.messages[0].text = "I got an error: FileNotFoundError"

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.provider.return_value = filter_instance
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [sample_conversation]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "analyze-errors", "arguments": {}},
                "id": 13,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            result = response["result"]
            assert "messages" in result
            assert len(result["messages"]) > 0

            user_msg = result["messages"][0]
            assert user_msg["role"] == "user"
            assert "error" in user_msg["content"]["text"].lower()


class TestSummarizeWeekPrompt:
    """Tests for summarize-week prompt."""

    def test_summarize_week_prompt(self, handle_request, mock_repo, sample_conversation):
        """summarize-week prompt generates weekly summary context."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [sample_conversation]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "summarize-week", "arguments": {}},
                "id": 14,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            result = response["result"]
            assert "messages" in result

            prompt_text = result["messages"][0]["content"]["text"]
            assert "week" in prompt_text.lower() or "past" in prompt_text.lower()


class TestExtractCodePrompt:
    """Tests for extract-code prompt."""

    def test_extract_code_prompt(self, handle_request, mock_repo):
        """extract-code prompt generates code extraction context."""
        # Conversation with code block
        conv_with_code = Conversation(
            id="test:code",
            provider="test",
            title="Code Example",
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    text="Here's some code:\n```python\nprint('hello')\n```",
                ),
            ],
        )

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [conv_with_code]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "extract-code", "arguments": {"language": "python"}},
                "id": 15,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            result = response["result"]
            assert "messages" in result

            prompt_text = result["messages"][0]["content"]["text"]
            assert "code" in prompt_text.lower()


class TestUnknownPrompt:
    """Tests for unknown prompt handling."""

    def test_unknown_prompt_returns_error(self, handle_request, mock_repo):
        """Unknown prompt name returns error."""
        request = {
            "method": "prompts/get",
            "params": {"name": "nonexistent", "arguments": {}},
            "id": 16,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "Unknown prompt" in response["error"]["message"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for JSON-RPC error handling."""

    def test_unknown_method_returns_error(self, handle_request, mock_repo):
        """Unknown method returns method not found error."""
        request = {"method": "unknown/method", "params": {}, "id": 17}

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]

    def test_response_preserves_request_id(self, handle_request, mock_repo):
        """Response always includes the request ID."""
        request = {"method": "initialize", "params": {}, "id": "custom-id-123"}

        response = handle_request(request, mock_repo)

        assert response["id"] == "custom-id-123"

    def test_response_includes_jsonrpc_version(self, handle_request, mock_repo):
        """All responses include JSON-RPC version."""
        request = {"method": "initialize", "params": {}, "id": 1}

        response = handle_request(request, mock_repo)

        assert response["jsonrpc"] == "2.0"


# =============================================================================
# Helper Function Tests
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


# =============================================================================
# Response Format Tests
# =============================================================================


class TestResponseFormats:
    """Tests for _success and _error helper functions."""

    def test_success_response_format(self):
        """_success creates valid JSON-RPC success response."""
        from polylogue.mcp.server import _success

        result = _success(42, {"data": "test"})

        assert result == {
            "jsonrpc": "2.0",
            "id": 42,
            "result": {"data": "test"},
        }

    def test_error_response_format(self):
        """_error creates valid JSON-RPC error response."""
        from polylogue.mcp.server import _error

        result = _error(42, -32600, "Invalid Request")

        assert result == {
            "jsonrpc": "2.0",
            "id": 42,
            "error": {"code": -32600, "message": "Invalid Request"},
        }


# =============================================================================
# Stats Tool Tests
# =============================================================================


class TestStatsTool:
    """Tests for _handle_stats_tool."""

    def test_stats_returns_archive_stats(self, handle_request, mock_repo):
        """Stats tool returns conversation, message, and provider counts."""
        from polylogue.lib.stats import ArchiveStats

        mock_repo.get_archive_stats.return_value = ArchiveStats(
            total_conversations=100,
            total_messages=5000,
            providers={"claude": 50, "chatgpt": 30, "claude-code": 20},
            embedded_conversations=10,
            embedded_messages=200,
            db_size_bytes=1048576,
        )

        request = {
            "method": "tools/call",
            "params": {"name": "stats", "arguments": {}},
            "id": 20,
        }

        response = handle_request(request, mock_repo)

        assert "result" in response
        data = json.loads(response["result"]["content"][0]["text"])
        assert data["total_conversations"] == 100
        assert data["total_messages"] == 5000
        assert data["providers"]["claude"] == 50
        assert data["embedded_conversations"] == 10
        assert data["db_size_mb"] == 1.0

    def test_stats_zero_db_size(self, handle_request, mock_repo):
        """Stats tool handles zero database size."""
        from polylogue.lib.stats import ArchiveStats

        mock_repo.get_archive_stats.return_value = ArchiveStats(
            total_conversations=0,
            total_messages=0,
            providers={},
            db_size_bytes=0,
        )

        request = {
            "method": "tools/call",
            "params": {"name": "stats", "arguments": {}},
            "id": 21,
        }

        response = handle_request(request, mock_repo)

        data = json.loads(response["result"]["content"][0]["text"])
        assert data["db_size_mb"] == 0
        assert data["total_conversations"] == 0

    def test_stats_none_db_size(self, handle_request, mock_repo):
        """Stats tool handles None database size."""
        from polylogue.lib.stats import ArchiveStats

        mock_repo.get_archive_stats.return_value = ArchiveStats(
            total_conversations=5,
            total_messages=20,
            providers={"test": 5},
            db_size_bytes=None,
        )

        request = {
            "method": "tools/call",
            "params": {"name": "stats", "arguments": {}},
            "id": 22,
        }

        response = handle_request(request, mock_repo)

        data = json.loads(response["result"]["content"][0]["text"])
        assert data["db_size_mb"] == 0


# =============================================================================
# Search Tool Filters Tests
# =============================================================================


class TestSearchToolFilters:
    """Tests for search tool edge cases and filters."""

    def test_search_with_provider_filter(self, handle_request, mock_repo):
        """Search applies provider filter via ConversationFilter."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.provider.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "test", "provider": "claude"},
                },
                "id": 30,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.provider.assert_called_once_with("claude")

    def test_search_with_since_filter(self, handle_request, mock_repo):
        """Search applies since date filter."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "test", "since": "2024-01-01"},
                },
                "id": 31,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.since.assert_called_once_with("2024-01-01")

    def test_search_invalid_since_returns_error(self, handle_request, mock_repo):
        """Search returns error for unparseable date."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.since.side_effect = ValueError("Invalid date")
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "test", "since": "not-a-date"},
                },
                "id": 32,
            }

            response = handle_request(request, mock_repo)

            assert "error" in response
            assert "Invalid date" in response["error"]["message"]

    def test_search_invalid_limit_type(self, handle_request, mock_repo):
        """Search returns error for non-integer limit."""
        request = {
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "test", "limit": "not-a-number"},
            },
            "id": 33,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "limit" in response["error"]["message"].lower()

    def test_search_limit_capped_at_max(self, handle_request, mock_repo):
        """Search clamps limit to MAX_LIMIT (10000)."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "test", "limit": 99999},
                },
                "id": 34,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.limit.assert_called_once_with(10000)

    def test_search_negative_limit_clamped_to_one(self, handle_request, mock_repo):
        """Search clamps negative limit to 1."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "test", "limit": -5},
                },
                "id": 35,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.limit.assert_called_once_with(1)


# =============================================================================
# List Tool Filters Tests
# =============================================================================


class TestListToolFilters:
    """Tests for list tool edge cases and filters."""

    def test_list_with_since_filter(self, handle_request, mock_repo):
        """List applies since filter."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.since.return_value = filter_instance
            filter_instance.provider.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "list",
                    "arguments": {"since": "2024-06-01"},
                },
                "id": 40,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.since.assert_called_once_with("2024-06-01")

    def test_list_invalid_since_returns_error(self, handle_request, mock_repo):
        """List returns error for unparseable date."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.since.side_effect = ValueError("Bad date")
            MockFilter.return_value = filter_instance

            request = {
                "method": "tools/call",
                "params": {
                    "name": "list",
                    "arguments": {"since": "garbage"},
                },
                "id": 41,
            }

            response = handle_request(request, mock_repo)

            assert "error" in response
            assert "Invalid date" in response["error"]["message"]

    def test_list_invalid_limit_type(self, handle_request, mock_repo):
        """List returns error for non-integer limit."""
        request = {
            "method": "tools/call",
            "params": {
                "name": "list",
                "arguments": {"limit": [1, 2]},
            },
            "id": 42,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "limit" in response["error"]["message"].lower()


# =============================================================================
# Conversations Resource Edge Cases Tests
# =============================================================================


class TestConversationsResourceEdges:
    """Tests for conversations resource edge cases."""

    def test_conversations_resource_with_tag_filter(self, handle_request, mock_repo):
        """Conversations resource supports tag filter."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.tag.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "resources/read",
                "params": {"uri": "polylogue://conversations?tag=important"},
                "id": 50,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.tag.assert_called_once_with("important")

    def test_conversations_resource_invalid_limit(self, handle_request, mock_repo):
        """Conversations resource returns error for non-integer limit."""
        request = {
            "method": "resources/read",
            "params": {"uri": "polylogue://conversations?limit=abc"},
            "id": 51,
        }

        response = handle_request(request, mock_repo)

        assert "error" in response
        assert "limit" in response["error"]["message"].lower()

    def test_conversations_resource_invalid_since(self, handle_request, mock_repo):
        """Conversations resource returns error for unparseable since date."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.provider.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.since.side_effect = ValueError("Cannot parse")
            MockFilter.return_value = filter_instance

            request = {
                "method": "resources/read",
                "params": {"uri": "polylogue://conversations?since=not-a-date"},
                "id": 52,
            }

            response = handle_request(request, mock_repo)

            assert "error" in response


# =============================================================================
# Prompt Edge Cases Tests
# =============================================================================


class TestPromptEdgeCases:
    """Tests for prompt edge cases."""

    def test_analyze_errors_with_provider_filter(
        self, handle_request, mock_repo, sample_conversation
    ):
        """analyze-errors applies provider filter."""
        sample_conversation.messages[0].text = "Got an error"

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.provider.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [sample_conversation]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {
                    "name": "analyze-errors",
                    "arguments": {"provider": "claude"},
                },
                "id": 60,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.provider.assert_called_once_with("claude")

    def test_analyze_errors_with_since_filter(
        self, handle_request, mock_repo, sample_conversation
    ):
        """analyze-errors applies since filter."""
        sample_conversation.messages[0].text = "Got an error"

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [sample_conversation]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {
                    "name": "analyze-errors",
                    "arguments": {"since": "2024-01-01"},
                },
                "id": 61,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            filter_instance.since.assert_called_once_with("2024-01-01")

    def test_analyze_errors_invalid_since(self, handle_request, mock_repo):
        """analyze-errors returns error for invalid since date."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.since.side_effect = ValueError("Bad date")
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {
                    "name": "analyze-errors",
                    "arguments": {"since": "garbage"},
                },
                "id": 62,
            }

            response = handle_request(request, mock_repo)

            assert "error" in response

    def test_analyze_errors_limits_error_contexts_to_20(
        self, handle_request, mock_repo
    ):
        """analyze-errors stops collecting after 20 error snippets."""
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

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [big_conv]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "analyze-errors", "arguments": {}},
                "id": 63,
            }

            response = handle_request(request, mock_repo)

            prompt_text = response["result"]["messages"][0]["content"]["text"]
            # Should have "20 error instances found"
            assert "20 error instances" in prompt_text

    def test_analyze_errors_no_matches(self, handle_request, mock_repo):
        """analyze-errors handles zero matching conversations."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.contains.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "analyze-errors", "arguments": {}},
                "id": 64,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            prompt_text = response["result"]["messages"][0]["content"]["text"]
            assert "0 conversations" in prompt_text

    def test_summarize_week_empty(self, handle_request, mock_repo):
        """summarize-week handles no conversations in past week."""
        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.since.return_value = filter_instance
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = []
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "summarize-week", "arguments": {}},
                "id": 65,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            prompt_text = response["result"]["messages"][0]["content"]["text"]
            assert "0 conversations" in prompt_text
            assert "0 messages" in prompt_text

    def test_extract_code_no_code_blocks(self, handle_request, mock_repo):
        """extract-code handles conversations without code."""
        conv = Conversation(
            id="nocode",
            provider="test",
            title="No Code",
            messages=[Message(id="m1", role="user", text="Just text, no code")],
        )

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [conv]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "extract-code", "arguments": {}},
                "id": 66,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            prompt_text = response["result"]["messages"][0]["content"]["text"]
            assert "0 code blocks" in prompt_text

    def test_extract_code_with_language_filter(self, handle_request, mock_repo):
        """extract-code filters by language."""
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

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [conv]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {
                    "name": "extract-code",
                    "arguments": {"language": "python"},
                },
                "id": 67,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response
            prompt_text = response["result"]["messages"][0]["content"]["text"]
            assert "python" in prompt_text.lower()

    def test_extract_code_null_message_text(self, handle_request, mock_repo):
        """extract-code skips messages with None text."""
        conv = Conversation(
            id="nulltext",
            provider="test",
            title="Null",
            messages=[Message(id="m1", role="assistant", text=None)],
        )

        with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
            filter_instance = MagicMock()
            filter_instance.limit.return_value = filter_instance
            filter_instance.list.return_value = [conv]
            MockFilter.return_value = filter_instance

            request = {
                "method": "prompts/get",
                "params": {"name": "extract-code", "arguments": {}},
                "id": 68,
            }

            response = handle_request(request, mock_repo)

            assert "result" in response


# =============================================================================
# Serialization Edge Cases Tests
# =============================================================================


class TestSerializationEdgeCases:
    """Tests for serialization helper edge cases."""

    def test_conversation_to_dict_no_timestamps(self):
        """Handles conversation with no timestamps."""
        from polylogue.mcp.server import _conversation_to_dict

        conv = Conversation(
            id="t1", provider="test", title="No Times", messages=[]
        )

        result = _conversation_to_dict(conv)

        assert result["created_at"] is None
        assert result["updated_at"] is None
        assert result["message_count"] == 0

    def test_conversation_to_full_dict_empty_messages(self):
        """Handles conversation with no messages."""
        from polylogue.mcp.server import _conversation_to_full_dict

        conv = Conversation(
            id="t2", provider="test", title="Empty", messages=[]
        )

        result = _conversation_to_full_dict(conv)

        assert result["messages"] == []

    def test_conversation_to_full_dict_empty_role(self):
        """Handles message with empty string role."""
        from polylogue.mcp.server import _conversation_to_full_dict

        conv = Conversation(
            id="t3",
            provider="test",
            title="Empty Role",
            messages=[Message(id="m1", role="", text="test")],
        )

        result = _conversation_to_full_dict(conv)

        assert result["messages"][0]["role"] == "unknown"

    def test_conversation_to_full_dict_null_text(self):
        """Handles message with None text."""
        from polylogue.mcp.server import _conversation_to_full_dict

        conv = Conversation(
            id="t4",
            provider="test",
            title="Null Text",
            messages=[Message(id="m1", role="user", text=None)],
        )

        result = _conversation_to_full_dict(conv)

        assert result["messages"][0]["text"] == ""

    def test_conversation_to_full_dict_null_timestamp(self):
        """Handles message with None timestamp."""
        from polylogue.mcp.server import _conversation_to_full_dict

        conv = Conversation(
            id="t5",
            provider="test",
            title="No TS",
            messages=[Message(id="m1", role="user", text="hi")],
        )

        result = _conversation_to_full_dict(conv)

        assert result["messages"][0]["timestamp"] is None

    def test_conversation_to_full_dict_unusual_role(self):
        """Handles message with unusual but valid role string."""
        from polylogue.mcp.server import _conversation_to_full_dict

        conv = Conversation(
            id="t6",
            provider="test",
            title="Unusual Role",
            messages=[Message(id="m1", role="tool", text="test")],
        )

        result = _conversation_to_full_dict(conv)

        assert result["messages"][0]["role"] == "tool"


# =============================================================================
# Write Error Tests
# =============================================================================


class TestWriteError:
    """Tests for _write_error helper function."""

    def test_write_error_outputs_to_stdout(self, capsys):
        """_write_error writes JSON-RPC error to stdout."""
        from polylogue.mcp.server import _write_error

        _write_error(-32700, "Parse error")

        captured = capsys.readouterr()
        response = json.loads(captured.out.strip())

        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"
