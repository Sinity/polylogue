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
from unittest.mock import MagicMock, patch

import pytest

from polylogue.lib.models import Conversation, Message

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
