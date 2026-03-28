"""Unit contracts for MCP server resources, prompts, exports, and payload surfaces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.lib.models import Conversation, Message
from tests.infra.mcp import (
    EXPECTED_PROMPT_NAMES,
    EXPECTED_RESOURCE_TEMPLATE_URIS,
    EXPECTED_RESOURCE_URIS,
    EXPECTED_TOOL_NAMES,
    invoke_surface,
    invoke_surface_async,
    make_mock_filter,
    make_repo_mock,
    make_simple_conversation,
)

SERIALIZATION_CASES = [
    (
        "no_timestamps",
        Conversation(id="t1", provider="test", title="No Times", messages=[]),
        {"created_at": None, "updated_at": None, "message_count": 0},
        "summary",
    ),
    (
        "empty_messages",
        Conversation(id="t2", provider="test", title="Empty", messages=[]),
        {"messages": []},
        "detail",
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
        "detail",
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
        "detail",
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
        "detail",
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
        "detail",
    ),
]


@pytest.fixture
def simple_conversation() -> Conversation:
    return make_simple_conversation()


class TestServerSurfaceRegistration:
    """Server registration should expose the documented MCP surfaces."""

    def test_build_server_exposes_managers(self):
        from polylogue.mcp.server import _build_server

        server = _build_server()
        assert server is not None
        assert hasattr(server, "_tool_manager")
        assert hasattr(server, "_resource_manager")
        assert hasattr(server, "_prompt_manager")

    @pytest.mark.parametrize(
        ("surface_attr", "actual_getter", "expected"),
        [
            ("tools", lambda server: set(server._tool_manager._tools.keys()), EXPECTED_TOOL_NAMES),
            ("resources", lambda server: set(server._resource_manager._resources.keys()), EXPECTED_RESOURCE_URIS),
            (
                "resource_templates",
                lambda server: set(server._resource_manager._templates.keys()),
                EXPECTED_RESOURCE_TEMPLATE_URIS,
            ),
            ("prompts", lambda server: set(server._prompt_manager._prompts.keys()), EXPECTED_PROMPT_NAMES),
        ],
    )
    def test_server_surface_contract(self, surface_attr, actual_getter, expected):
        from polylogue.mcp.server import _build_server

        server = _build_server()
        actual = actual_getter(server)
        missing = expected - actual
        assert not missing, f"Missing {surface_attr}: {sorted(missing)}"


class TestResourceSurfaces:
    def test_stats_returns_archive_statistics(self):
        from polylogue.lib.stats import ArchiveStats
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_archive_stats = AsyncMock(
                return_value=ArchiveStats(
                    total_conversations=2,
                    total_messages=4,
                    providers={"chatgpt": 2},
                )
            )
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(server._resource_manager._resources["polylogue://stats"].fn)

        stats = json.loads(result)
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 4

    @pytest.mark.asyncio
    async def test_conversations_resource_returns_list(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

                server = _build_server()
                result = await invoke_surface_async(server._resource_manager._resources["polylogue://conversations"].fn)

        convs = json.loads(result)
        assert len(convs) == 1
        assert convs[0]["id"] == simple_conversation.id

    def test_single_conversation_resource(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get = AsyncMock(return_value=simple_conversation)
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(
                server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn,
                conv_id="test:conv-123",
            )

        conv = json.loads(result)
        assert conv["id"] == "test:conv-123"
        assert "messages" in conv

    def test_conversation_resource_not_found(self):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get = AsyncMock(return_value=None)
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(
                server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn,
                conv_id="nonexistent",
            )

        result_dict = json.loads(result)
        assert "error" in result_dict

    def test_tags_resource(self):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.list_tags.return_value = {"feature": 10, "bug": 5}
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(server._resource_manager._resources["polylogue://tags"].fn)

        parsed = json.loads(result)
        assert parsed == {"feature": 10, "bug": 5}

    def test_health_resource(self):
        from polylogue.mcp.server import _build_server

        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "All systems operational"

        with patch("polylogue.mcp.server._get_config") as mock_get_config, patch(
            "polylogue.health.get_health"
        ) as mock_get_health:
            mock_get_config.return_value = MagicMock()
            mock_get_health.return_value = mock_report

            server = _build_server()
            result = invoke_surface(server._resource_manager._resources["polylogue://health"].fn)

        parsed = json.loads(result)
        assert len(parsed["checks"]) == 1
        assert parsed["summary"] == "All systems operational"


class TestPromptSurfaces:
    @pytest.mark.asyncio
    async def test_analyze_errors_with_conversations(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        simple_conversation.messages[0].text = "Got an error while running"

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

                server = _build_server()
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

        assert isinstance(result, str)
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_errors_limits_error_contexts_to_20(self):
        from polylogue.mcp.server import _build_server

        msgs = [
            Message(
                id=f"m{i}",
                role="user",
                text=f"error #{i} occurred",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            )
            for i in range(30)
        ]
        big_conv = Conversation(id="big", provider="test", title="Errors", messages=msgs)

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = [big_conv]
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[big_conv])

                server = _build_server()
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

        assert "20 error instances" in result

    @pytest.mark.asyncio
    async def test_analyze_errors_no_matches(self):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await server._prompt_manager._prompts["analyze_errors"].fn()

        assert "0 conversations" in result

    @pytest.mark.asyncio
    async def test_summarize_week_empty(self):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            mock_get_repo.return_value = mock_repo

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                server = _build_server()
                result = await server._prompt_manager._prompts["summarize_week"].fn()

        assert "0 conversations" in result
        assert "0 messages" in result

    @pytest.mark.asyncio
    async def test_extract_code_no_code_blocks(self):
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

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = await server._prompt_manager._prompts["extract_code"].fn()

        assert "0 code blocks" in result

    @pytest.mark.asyncio
    async def test_extract_code_with_language_filter(self):
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

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = await server._prompt_manager._prompts["extract_code"].fn(language="python")

        assert "python" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_code_null_message_text(self):
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

            with patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                server = _build_server()
                result = await server._prompt_manager._prompts["extract_code"].fn()

        assert isinstance(result, str)

    def test_compare_conversations_prompt(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.side_effect = [simple_conversation, simple_conversation]
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(
                server._prompt_manager._prompts["compare_conversations"].fn,
                id1="test:conv-1",
                id2="test:conv-2",
            )

        assert "Compare" in result
        assert "Conversation 1" in result
        assert "Conversation 2" in result

    @pytest.mark.asyncio
    async def test_extract_patterns_prompt(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo, patch(
            "polylogue.lib.filters.ConversationFilter"
        ) as mock_filter_cls:
            mock_repo = make_repo_mock()
            mock_get_repo.return_value = mock_repo
            mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

            server = _build_server()
            result = await invoke_surface_async(server._prompt_manager._prompts["extract_patterns"].fn)

        assert isinstance(result, str)
        assert "patterns" in result.lower()


class TestExportConversationTool:
    def test_export_markdown(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo, patch(
            "polylogue.lib.formatting.format_conversation"
        ) as mock_format:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = simple_conversation
            mock_get_repo.return_value = mock_repo
            mock_format.return_value = "# Test Conversation\n\nFormatted content"

            server = _build_server()
            result = invoke_surface(
                server._tool_manager._tools["export_conversation"].fn,
                id="test:conv-123",
                format="markdown",
            )

        assert "Test Conversation" in result
        mock_format.assert_called_once()
        call_args = mock_format.call_args
        assert call_args[0][0] == simple_conversation
        assert call_args[0][1] == "markdown"

    def test_export_not_found(self):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = invoke_surface(server._tool_manager._tools["export_conversation"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_export_invalid_format_falls_back_to_markdown(self, simple_conversation):
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo, patch(
            "polylogue.lib.formatting.format_conversation"
        ) as mock_format:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = simple_conversation
            mock_get_repo.return_value = mock_repo
            mock_format.return_value = "# Content"

            server = _build_server()
            invoke_surface(
                server._tool_manager._tools["export_conversation"].fn,
                id="test:conv-123",
                format="invalid_format",
            )

        assert mock_format.call_args.args[1] == "markdown"


class TestTypedPayloads:
    def test_conversation_to_summary_dict(self, simple_conversation):
        from polylogue.mcp.payloads import MCPConversationSummaryPayload

        result = MCPConversationSummaryPayload.from_conversation(simple_conversation).model_dump(mode="json")

        assert result["id"] == "test:conv-123"
        assert result["provider"] == "chatgpt"
        assert result["message_count"] == 2
        assert "created_at" in result
        assert "updated_at" in result

    def test_conversation_to_full_dict(self, simple_conversation):
        from polylogue.mcp.payloads import MCPConversationDetailPayload

        result = MCPConversationDetailPayload.from_conversation(simple_conversation).model_dump(mode="json")

        assert "messages" in result
        assert len(result["messages"]) == 2
        msg = result["messages"][0]
        assert {"id", "role", "text", "timestamp"} <= msg.keys()

    @pytest.mark.parametrize(("case_id", "conv", "expected_fields", "payload_kind"), SERIALIZATION_CASES)
    def test_serialization_edge_cases(self, case_id, conv, expected_fields, payload_kind):
        if payload_kind == "summary":
            from polylogue.mcp.payloads import MCPConversationSummaryPayload

            result = MCPConversationSummaryPayload.from_conversation(conv).model_dump(mode="json")
        else:
            from polylogue.mcp.payloads import MCPConversationDetailPayload

            result = MCPConversationDetailPayload.from_conversation(conv).model_dump(mode="json")

        for key, expected_value in expected_fields.items():
            if key == "messages":
                assert key in result
                if expected_value:
                    for index, expected_msg in enumerate(expected_value):
                        for msg_key, msg_val in expected_msg.items():
                            assert result[key][index][msg_key] == msg_val, f"Failed for case {case_id}: {msg_key}"
                else:
                    assert result[key] == expected_value
            else:
                assert result[key] == expected_value, f"Failed for case {case_id}: {key}"


class TestClampLimit:
    @pytest.mark.parametrize(
        ("raw_limit", "expected"),
        [
            (10, 10),
            (1, 1),
            (5000, 5000),
            (99999, 99999),
            (10001, 10001),
            (0, 1),
            (-5, 1),
            ("10", 10),
            ("not-a-number", 10),
            ([1, 2], 10),
        ],
    )
    def test_clamp_limit_contract(self, raw_limit, expected):
        from polylogue.mcp.server import _clamp_limit

        assert _clamp_limit(raw_limit) == expected
