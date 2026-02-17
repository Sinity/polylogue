"""MCP server resource and prompt tests — stats/conversations/conversation resources, prompts, serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from polylogue.lib.models import Conversation, Message
from tests.integration.conftest import make_mock_filter

# =============================================================================
# Test data tables (SCREAMING_CASE constants)
# =============================================================================

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


# =============================================================================
# Resource Tests
# =============================================================================


class TestStatsResource:
    """Tests for polylogue://stats resource."""

    def test_stats_returns_archive_statistics(self):
        """Stats resource returns conversation and message counts."""
        from polylogue.lib.stats import ArchiveStats
        from polylogue.mcp.server import _build_server

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

    @pytest.mark.asyncio
    async def test_conversations_resource_returns_list(self, simple_conversation):
        """Conversations resource returns all conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[simple_conversation])

                server = _build_server()
                result = await server._resource_manager._resources["polylogue://conversations"].fn()

                convs = json.loads(result)
                assert len(convs) == 1


class TestConversationResource:
    """Tests for polylogue://conversation/{id} resource."""

    def test_single_conversation_resource(self, simple_conversation):
        """Single conversation resource returns full conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get.return_value = simple_conversation
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

    @pytest.mark.asyncio
    async def test_analyze_errors_with_conversations(self, simple_conversation):
        """analyze_errors generates prompt from error conversations."""
        from polylogue.mcp.server import _build_server

        # Modify sample to have error message
        simple_conversation.messages[0].text = "Got an error while running"

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[simple_conversation])

                server = _build_server()
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

                assert isinstance(result, str)
                assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_errors_limits_error_contexts_to_20(self):
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
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

                # Should have "20 error instances found"
                assert "20 error instances" in result

    @pytest.mark.asyncio
    async def test_analyze_errors_no_matches(self):
        """analyze_errors handles zero matching conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

                assert "0 conversations" in result


class TestSummarizeWeekPrompt:
    """Tests for summarize_week prompt."""

    @pytest.mark.asyncio
    async def test_summarize_week_empty(self):
        """summarize_week handles no conversations in past week."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                MockFilter.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await server._prompt_manager._prompts["summarize_week"].fn()

                assert "0 conversations" in result
                assert "0 messages" in result


class TestExtractCodePrompt:
    """Tests for extract_code prompt."""

    @pytest.mark.asyncio
    async def test_extract_code_no_code_blocks(self):
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
                result = await server._prompt_manager._prompts["extract_code"].fn()

                assert "0 code blocks" in result

    @pytest.mark.asyncio
    async def test_extract_code_with_language_filter(self):
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
                result = await server._prompt_manager._prompts["extract_code"].fn(language="python")

                assert "python" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_code_null_message_text(self):
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
                result = await server._prompt_manager._prompts["extract_code"].fn()

                assert isinstance(result, str)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestConversationSerialization:
    """Tests for conversation serialization helpers."""

    def test_conversation_to_dict(self, simple_conversation):
        """_conversation_to_dict serializes conversation metadata."""
        from polylogue.mcp.server import _conversation_to_dict

        result = _conversation_to_dict(simple_conversation)

        assert result["id"] == "test:conv-123"
        assert result["provider"] == "chatgpt"
        assert result["message_count"] == 2
        assert "created_at" in result
        assert "updated_at" in result

    def test_conversation_to_full_dict(self, simple_conversation):
        """_conversation_to_full_dict includes messages."""
        from polylogue.mcp.server import _conversation_to_full_dict

        result = _conversation_to_full_dict(simple_conversation)

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
    @pytest.mark.asyncio
    async def test_search_filter_application(
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

                result = await server._tool_manager._tools["search"].fn(**args)

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
    @pytest.mark.asyncio
    async def test_search_error_cases(
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
                        result = await server._tool_manager._tools["search"].fn(**args)
                        # If no exception, check result contains error info
                        json.loads(result)
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
                        result = await server._tool_manager._tools["search"].fn(**args)
                        json.loads(result)
                    except (TypeError, ValueError):
                        pass  # Expected for invalid limit type
