"""Unit contracts for MCP server resources, prompts, exports, and payload surfaces."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Conversation, ConversationSummary
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    EXPECTED_PROMPT_NAMES,
    EXPECTED_RESOURCE_TEMPLATE_URIS,
    EXPECTED_RESOURCE_URIS,
    EXPECTED_TOOL_NAMES,
    MCPServerUnderTest,
    invoke_surface,
    invoke_surface_async,
    make_mock_filter,
    make_query_store_mock,
    make_simple_conversation,
    make_tag_store_mock,
)

SerializationCase = tuple[str, Conversation, dict[str, object], str]


def _summary_for(conversation: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(str(conversation.id)),
        provider=Provider.from_string(str(conversation.provider)),
        title=conversation.display_title,
        message_count=len(conversation.messages),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


SERIALIZATION_CASES: list[SerializationCase] = [
    (
        "no_timestamps",
        make_conv(id="t1", provider=Provider.UNKNOWN, title="No Times", messages=[]),
        {"created_at": None, "updated_at": None, "message_count": 0},
        "summary",
    ),
    (
        "empty_messages",
        make_conv(id="t2", provider=Provider.UNKNOWN, title="Empty", messages=[]),
        {"messages": []},
        "detail",
    ),
    (
        "empty_role",
        make_conv(
            id="t3",
            provider=Provider.UNKNOWN,
            title="Empty Role",
            messages=[make_msg(id="m1", role=Role.UNKNOWN, text="test")],
        ),
        {"messages": [{"role": "unknown"}]},
        "detail",
    ),
    (
        "null_text",
        make_conv(
            id="t4",
            provider=Provider.UNKNOWN,
            title="Null Text",
            messages=[make_msg(id="m1", role=Role.USER, text=None)],
        ),
        {"messages": [{"text": ""}]},
        "detail",
    ),
    (
        "null_timestamp",
        make_conv(
            id="t5",
            provider=Provider.UNKNOWN,
            title="No TS",
            messages=[make_msg(id="m1", role=Role.USER, text="hi")],
        ),
        {"messages": [{"timestamp": None}]},
        "detail",
    ),
    (
        "unusual_role",
        make_conv(
            id="t6",
            provider=Provider.UNKNOWN,
            title="Unusual Role",
            messages=[make_msg(id="m1", role=Role.TOOL, text="test")],
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

    def testbuild_server_exposes_managers(self: object, mcp_server: MCPServerUnderTest) -> None:
        assert mcp_server is not None
        assert hasattr(mcp_server, "_tool_manager")
        assert hasattr(mcp_server, "_resource_manager")
        assert hasattr(mcp_server, "_prompt_manager")

    def test_dynamic_product_tools_publish_explicit_signatures(self: object, mcp_server: MCPServerUnderTest) -> None:
        signature = inspect.signature(mcp_server._tool_manager._tools["session_profiles"].fn)

        assert "provider" in signature.parameters
        assert "limit" in signature.parameters
        assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

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
    def test_server_surface_contract(
        self: object,
        surface_attr: str,
        actual_getter: Callable[[MCPServerUnderTest], set[str]],
        expected: set[str],
        mcp_server: MCPServerUnderTest,
    ) -> None:
        actual = actual_getter(mcp_server)
        missing = expected - actual
        assert not missing, f"Missing {surface_attr}: {sorted(missing)}"

    def test_read_role_omits_mutation_and_maintenance_tools(self: object) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="read")
        tools = set(server._tool_manager._tools.keys())

        assert "search" in tools
        assert "get_messages" in tools
        assert "session_profiles" in tools
        assert "add_tag" not in tools
        assert "set_metadata" not in tools
        assert "rebuild_index" not in tools
        assert "rebuild_session_products" not in tools

    def test_write_role_omits_admin_maintenance_tools(self: object) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="write")
        tools = set(server._tool_manager._tools.keys())

        assert "add_tag" in tools
        assert "set_metadata" in tools
        assert "rebuild_index" not in tools
        assert "rebuild_session_products" not in tools


class TestResourceSurfaces:
    def test_stats_returns_archive_statistics(self: object, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.archive.stats import ArchiveStats

        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.storage_stats = AsyncMock(
                return_value=ArchiveStats(
                    total_conversations=2,
                    total_messages=4,
                    providers={"chatgpt": 2},
                )
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://stats"].fn)

        stats = json.loads(result)
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 4

    @pytest.mark.asyncio
    async def test_conversations_resource_returns_list(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

                result = await invoke_surface_async(
                    mcp_server._resource_manager._resources["polylogue://conversations"].fn
                )

        convs = json.loads(result)
        assert len(convs) == 1
        assert convs[0]["id"] == simple_conversation.id

    def test_single_conversation_resource(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=_summary_for(simple_conversation))
            mock_ops.get_conversation_stats = AsyncMock(
                return_value={"total_messages": len(simple_conversation.messages)}
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn,
                conv_id="test:conv-123",
            )

        conv = json.loads(result)
        assert conv["id"] == "test:conv-123"
        assert conv["message_count"] == 2
        assert "messages" not in conv

    def test_conversation_resource_not_found(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=None)
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://conversation/{conv_id}"].fn,
                conv_id="nonexistent",
            )

        result_dict = json.loads(result)
        assert "error" in result_dict

    def test_tags_resource(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store:
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.list_tags.return_value = {"feature": 10, "bug": 5}
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://tags"].fn)

        parsed = json.loads(result)
        assert parsed == {"feature": 10, "bug": 5}

    def test_readiness_resource(self: object, mcp_server: MCPServerUnderTest) -> None:
        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "All systems operational"

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.readiness.get_readiness") as mock_get_readiness,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_readiness.return_value = mock_report

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://readiness"].fn)

        parsed = json.loads(result)
        assert len(parsed["checks"]) == 1
        assert parsed["summary"] == "All systems operational"


class TestPromptSurfaces:
    @pytest.mark.asyncio
    async def test_analyze_errors_with_conversations(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        simple_conversation.messages.to_list()[0].text = "Got an error while running"

        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert isinstance(result, str)
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_errors_limits_error_contexts_to_20(self: object, mcp_server: MCPServerUnderTest) -> None:
        msgs = [
            make_msg(
                id=f"m{i}",
                role=Role.USER,
                text=f"error #{i} occurred",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            )
            for i in range(30)
        ]
        big_conv = make_conv(id="big", provider=Provider.UNKNOWN, title="Errors", messages=msgs)

        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[big_conv])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "20 error instances" in result

    @pytest.mark.asyncio
    async def test_analyze_errors_no_matches(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "0 conversations" in result

    @pytest.mark.asyncio
    async def test_summarize_week_empty(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["summarize_week"].fn)

        assert "0 conversations" in result
        assert "0 messages" in result

    @pytest.mark.asyncio
    async def test_extract_code_no_code_blocks(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="nocode",
            provider=Provider.UNKNOWN,
            title="No Code",
            messages=[make_msg(id="m1", role=Role.USER, text="Just text, no code")],
        )

        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_code"].fn)

        assert "0 code blocks" in result

    @pytest.mark.asyncio
    async def test_extract_code_with_language_filter(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="code",
            provider=Provider.UNKNOWN,
            title="Code",
            messages=[
                make_msg(
                    id="m1",
                    role=Role.ASSISTANT,
                    text="```python\nprint('hi')\n```\n```javascript\nconsole.log('hi')\n```",
                )
            ],
        )

        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(
                    mcp_server._prompt_manager._prompts["extract_code"].fn,
                    language="python",
                )

        assert "python" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_code_null_message_text(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="nulltext",
            provider=Provider.UNKNOWN,
            title="Null",
            messages=[make_msg(id="m1", role=Role.ASSISTANT, text=None)],
        )

        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_code"].fn)

        assert isinstance(result, str)

    def test_compare_conversations_prompt(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_query_store = make_query_store_mock()
            mock_query_store.get_eager = AsyncMock(side_effect=[simple_conversation, simple_conversation])
            mock_get_query_store.return_value = mock_query_store

            result = invoke_surface(
                mcp_server._prompt_manager._prompts["compare_conversations"].fn,
                id1="test:conv-1",
                id2="test:conv-2",
            )

        assert "Compare" in result
        assert "Conversation 1" in result
        assert "Conversation 2" in result

    @pytest.mark.asyncio
    async def test_extract_patterns_prompt(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.archive.filter.filters.ConversationFilter") as mock_filter_cls,
        ):
            mock_get_query_store.return_value = MagicMock()
            mock_filter_cls.return_value = make_mock_filter(results=[simple_conversation])

            result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_patterns"].fn)

        assert isinstance(result, str)
        assert "patterns" in result.lower()


class TestExportConversationTool:
    def test_get_messages_tool_applies_content_projection(
        self: object,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        projected_input = make_conv(
            id="test:conv-123",
            provider=Provider.CHATGPT,
            title="Projected Conversation",
            messages=[
                make_msg(
                    id="msg-1",
                    role="assistant",
                    text="Alpha\n\n```python\nprint('x')\n```\n\nOmega",
                )
            ],
        )

        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=_summary_for(projected_input))
            projected = projected_input.with_content_projection(ContentProjectionSpec(include_code=False))
            mock_ops.get_messages_paginated = AsyncMock(
                return_value=(list(projected.messages), len(projected.messages))
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                conversation_id="test:conv-123",
                no_code_blocks=True,
            )

        payload = json.loads(result)
        assert payload["messages"][0]["text"] == "Alpha\n\nOmega"
        projection = mock_ops.get_messages_paginated.await_args.kwargs["content_projection"]
        assert projection.include_code is False

    def test_export_markdown(self: object, simple_conversation: Conversation, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops,
            patch("polylogue.rendering.formatting.format_conversation") as mock_format,
        ):
            mock_ops = MagicMock()
            mock_ops.get_conversation = AsyncMock(return_value=simple_conversation)
            mock_get_archive_ops.return_value = mock_ops
            mock_format.return_value = "# Test Conversation\n\nFormatted content"

            result = invoke_surface(
                mcp_server._tool_manager._tools["export_conversation"].fn,
                id="test:conv-123",
                format="markdown",
            )

        assert "Test Conversation" in result
        mock_format.assert_called_once()
        call_args = mock_format.call_args
        assert call_args[0][0] == simple_conversation
        assert call_args[0][1] == "markdown"

    def test_export_not_found(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation = AsyncMock(return_value=None)
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["export_conversation"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_export_invalid_format_falls_back_to_markdown(
        self: object,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops,
            patch("polylogue.rendering.formatting.format_conversation") as mock_format,
        ):
            mock_ops = MagicMock()
            mock_ops.get_conversation = AsyncMock(return_value=simple_conversation)
            mock_get_archive_ops.return_value = mock_ops
            mock_format.return_value = "# Content"

            invoke_surface(
                mcp_server._tool_manager._tools["export_conversation"].fn,
                id="test:conv-123",
                format="invalid_format",
            )

        assert mock_format.call_args.args[1] == "markdown"


class TestTypedPayloads:
    def test_conversation_to_summary_dict(self: object, simple_conversation: Conversation) -> None:
        from polylogue.mcp.payloads import MCPConversationSummaryPayload

        result = MCPConversationSummaryPayload.from_conversation(simple_conversation).model_dump(mode="json")

        assert result["id"] == "test:conv-123"
        assert result["provider"] == "chatgpt"
        assert result["message_count"] == 2
        assert "created_at" in result
        assert "updated_at" in result

    def test_conversation_to_full_dict(self: object, simple_conversation: Conversation) -> None:
        from polylogue.mcp.payloads import MCPConversationDetailPayload

        result = MCPConversationDetailPayload.from_conversation(simple_conversation).model_dump(mode="json")

        assert "messages" in result
        assert len(result["messages"]) == 2
        msg = result["messages"][0]
        assert {"id", "role", "text", "timestamp"} <= msg.keys()

    def test_conversation_to_full_dict_applies_projection(self: object) -> None:
        from polylogue.mcp.payloads import MCPConversationDetailPayload

        conversation = make_conv(
            id="projected",
            provider=Provider.CHATGPT,
            title="Projected",
            messages=[
                make_msg(
                    id="m1",
                    role="assistant",
                    text="Alpha\n\n```python\nprint('x')\n```\n\nOmega",
                )
            ],
        )

        result = MCPConversationDetailPayload.from_conversation(
            conversation,
            content_projection=ContentProjectionSpec.prose_only(),
        ).model_dump(mode="json")

        assert result["messages"][0]["text"] == "Alpha\n\nOmega"

    @pytest.mark.parametrize(("case_id", "conv", "expected_fields", "payload_kind"), SERIALIZATION_CASES)
    def test_serialization_edge_cases(
        self: object,
        case_id: str,
        conv: Conversation,
        expected_fields: dict[str, object],
        payload_kind: str,
    ) -> None:
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
                    assert isinstance(expected_value, list)
                    for index, expected_msg in enumerate(expected_value):
                        assert isinstance(expected_msg, dict)
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
            # Values above the 1000 cap clamp to 1000.
            (5000, 1000),
            (99999, 1000),
            (10001, 1000),
            (1000, 1000),
            (0, 1),
            (-5, 1),
            ("10", 10),
            ("not-a-number", 10),
            ([1, 2], 10),
        ],
    )
    def test_clamp_limit_contract(self: object, raw_limit: object, expected: int) -> None:
        from polylogue.mcp.server_support import _clamp_limit

        assert _clamp_limit(raw_limit) == expected
