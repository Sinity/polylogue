"""Unit tests for the build_context_pack MCP tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Conversation, ConversationSummary
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.mcp.context_pack import select_context_pack_conversations
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    EXPECTED_TOOL_NAMES,
    MCPServerUnderTest,
    invoke_surface,
)


def _summary_for(conversation: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(str(conversation.id)),
        provider=Provider.from_string(str(conversation.provider)),
        title=conversation.display_title,
        message_count=len(conversation.messages),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


class TestBuildContextPackRegistration:
    """Verify the build_context_pack tool is registered and callable."""

    def test_tool_name_is_in_expected_set(self) -> None:
        """The tool is listed in EXPECTED_TOOL_NAMES."""
        assert "build_context_pack" in EXPECTED_TOOL_NAMES

    def test_tool_is_registered_on_server(self, mcp_server: MCPServerUnderTest) -> None:
        """The tool is present in the server tool manager."""
        tools = mcp_server._tool_manager._tools
        assert "build_context_pack" in tools
        assert callable(tools["build_context_pack"].fn)

    def test_tool_accepts_valid_parameters(self, mcp_server: MCPServerUnderTest) -> None:
        """The tool can be called with valid parameters."""
        from polylogue.mcp.server_support import _set_runtime_services

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn

        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Test Context Pack",
            messages=[make_msg(id="m1", role=Role.USER, text="hello")],
        )

        mock_services = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_action_events_batch = AsyncMock(return_value={})
        mock_repo.list_summaries = AsyncMock(return_value=[_summary_for(conv)])
        mock_repo.list = AsyncMock(return_value=[conv])
        mock_repo.list_by_query = AsyncMock(return_value=[conv])
        mock_repo.view = AsyncMock(return_value=conv)
        mock_repo.get = AsyncMock(return_value=conv)
        mock_repo.resolve_id = AsyncMock(return_value="test:conv-1")
        mock_repo.get_messages_paginated = AsyncMock(
            return_value=(
                conv.messages if hasattr(conv, "messages") else [],
                len(conv.messages) if hasattr(conv, "messages") else 0,
            )
        )

        mock_services.get_repository.return_value = mock_repo
        mock_services.get_config.return_value = MagicMock()
        mock_services.get_backend.return_value = MagicMock()

        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_conversations=3, detail_level="summary")
            parsed = json.loads(result)
            assert "project" in parsed
            assert "date_range" in parsed
            assert "query_context" in parsed
            assert "conversations" in parsed
            assert "provenance" in parsed
            assert parsed["provenance"]["source"] == "polylogue"
        finally:
            _set_runtime_services(None)

    @pytest.mark.asyncio
    async def test_context_pack_broadens_zero_result_archaeology_query(self) -> None:
        """A pasted multi-token archaeology query falls back to recall terms."""
        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Replay Replacement History",
            messages=[make_msg(id="m1", role=Role.USER, text="event_replacements")],
        )
        seen_queries: list[tuple[str, ...]] = []

        async def query_conversations(spec: ConversationQuerySpec) -> list[Conversation]:
            seen_queries.append(spec.query_terms)
            if spec.query_terms == ("event_replacements",):
                return [conv]
            return []

        def coerce_limit(value: object) -> int:
            return int(str(value))

        selection = await select_context_pack_conversations(
            query_conversations,
            coerce_limit,
            project_path=None,
            project_repo=None,
            since="2026-04-01",
            until=None,
            provider=None,
            query="supersedes_event_id event_replacements equivalence_key",
            limit=5,
        )

        assert selection.conversations == [conv]
        assert selection.match_strategy == "term_recall"
        assert ("supersedes_event_id event_replacements equivalence_key",) in seen_queries
        assert ("event_replacements",) in seen_queries

    @pytest.mark.asyncio
    async def test_context_pack_relaxes_project_filter_after_recall_miss(self) -> None:
        """Project filters are relaxed only after strict and in-project recall miss."""
        conv = make_conv(
            id="test:conv-2",
            provider=Provider.CLAUDE_AI,
            title="Target Vision Archaeology",
            messages=[make_msg(id="m1", role=Role.USER, text="source_material_id")],
        )

        async def query_conversations(spec: ConversationQuerySpec) -> list[Conversation]:
            if spec.cwd_prefix is None and spec.query_terms == ("source_material_id",):
                return [conv]
            return []

        def coerce_limit(value: object) -> int:
            return int(str(value))

        selection = await select_context_pack_conversations(
            query_conversations,
            coerce_limit,
            project_path="/realm/project/sinex",
            project_repo=None,
            since="2026-04-01",
            until=None,
            provider=None,
            query="source_material_id anchor_byte",
            limit=5,
        )

        assert selection.conversations == [conv]
        assert selection.match_strategy == "relaxed_project_term_recall"
        assert selection.relaxed_filters == ("project_path",)

    def test_summary_detail_omits_messages(self, mcp_server: MCPServerUnderTest) -> None:
        """Summary detail level omits message bodies."""
        from polylogue.mcp.server_support import _set_runtime_services

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn

        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Summary Test",
            messages=[make_msg(id="m1", role=Role.USER, text="hello")],
        )

        mock_services = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_action_events_batch = AsyncMock(return_value={})
        mock_repo.list = AsyncMock(return_value=[conv])
        mock_repo.list_by_query = AsyncMock(return_value=[conv])
        mock_repo.view = AsyncMock(return_value=conv)
        mock_repo.get = AsyncMock(return_value=conv)
        mock_repo.resolve_id = AsyncMock(return_value="test:conv-1")
        mock_repo.list_summaries = AsyncMock(return_value=[_summary_for(conv)])

        mock_services.get_repository.return_value = mock_repo
        mock_services.get_config.return_value = MagicMock()
        mock_services.get_backend.return_value = MagicMock()

        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_conversations=1, detail_level="summary")
            parsed = json.loads(result)
            convs = parsed.get("conversations", [])
            if convs:
                assert len(convs[0].get("messages", [])) == 0
        finally:
            _set_runtime_services(None)

    def test_redact_paths_defaults_to_true(self, mcp_server: MCPServerUnderTest) -> None:
        """Provenance.redacted is True by default."""
        from polylogue.mcp.server_support import _set_runtime_services

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn

        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Redaction Test",
            messages=[],
        )

        mock_services = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_action_events_batch = AsyncMock(return_value={})
        mock_repo.list = AsyncMock(return_value=[conv])
        mock_repo.list_by_query = AsyncMock(return_value=[conv])
        mock_repo.view = AsyncMock(return_value=conv)
        mock_repo.get = AsyncMock(return_value=conv)
        mock_repo.resolve_id = AsyncMock(return_value="test:conv-1")
        mock_repo.list_summaries = AsyncMock(return_value=[_summary_for(conv)])

        mock_services.get_repository.return_value = mock_repo
        mock_services.get_config.return_value = MagicMock()
        mock_services.get_backend.return_value = MagicMock()

        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_conversations=1, detail_level="summary")
            parsed = json.loads(result)
            assert parsed["provenance"]["redacted"] is True
        finally:
            _set_runtime_services(None)

    def test_unresolved_work_with_action_events(self, mcp_server: MCPServerUnderTest) -> None:
        """Conversations with action events appear in unresolved_work."""
        from datetime import datetime, timezone

        from polylogue.mcp.server_support import _set_runtime_services

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn

        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Has Actions",
            messages=[],
        )

        mock_event = MagicMock()
        mock_event.event_id = "evt-1"
        mock_event.message_id = "msg-1"
        mock_event.timestamp = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        mock_event.cwd_path = "/realm/project/foo"
        mock_event.branch_names = ("feature/test",)
        mock_event.affected_paths = ("src/main.py",)
        mock_event.normalized_tool_name = "bash"
        mock_event.tool_name = "bash"

        mock_services = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_action_events_batch = AsyncMock(return_value={"test:conv-1": (mock_event,)})
        mock_repo.list = AsyncMock(return_value=[conv])
        mock_repo.list_by_query = AsyncMock(return_value=[conv])
        mock_repo.view = AsyncMock(return_value=conv)
        mock_repo.get = AsyncMock(return_value=conv)
        mock_repo.resolve_id = AsyncMock(return_value="test:conv-1")
        mock_repo.list_summaries = AsyncMock(return_value=[_summary_for(conv)])

        mock_services.get_repository.return_value = mock_repo
        mock_services.get_config.return_value = MagicMock()
        mock_services.get_backend.return_value = MagicMock()

        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_conversations=1, detail_level="summary")
            parsed = json.loads(result)

            unresolved = parsed.get("unresolved_work", [])
            assert len(unresolved) > 0
            assert unresolved[0]["conversation_id"] == "test:conv-1"
            assert unresolved[0]["tool_use_count"] == 1
            assert unresolved[0]["reason"] is not None

            project = parsed.get("project", {})
            assert project.get("branch") == "feature/test"
            assert len(project.get("cwd_paths", [])) > 0

            action_summaries = parsed.get("action_summaries", [])
            assert len(action_summaries) > 0
            assert action_summaries[0]["tool_name"] == "bash"
            assert action_summaries[0]["count"] == 1
        finally:
            _set_runtime_services(None)


class TestContextPackModels:
    """Smoke test the Pydantic models directly."""

    def test_payload_default_construction(self) -> None:
        from polylogue.mcp.context_pack import ContextPackPayload

        payload = ContextPackPayload()
        assert payload.provenance.source == "polylogue"
        assert payload.provenance.redacted is True
        assert isinstance(payload.conversations, list)
        assert len(payload.conversations) == 0

    def test_redact_path_home_directory(self) -> None:
        import os

        from polylogue.mcp.context_pack import redact_path

        home = os.path.expanduser("~")
        result = redact_path(home + "/projects/foo")
        assert result == "~/projects/foo"
        assert not result.startswith(home)

    def test_redact_path_non_home_unchanged(self) -> None:
        from polylogue.mcp.context_pack import redact_path

        result = redact_path("/tmp/scratch")
        assert result == "/tmp/scratch"

    def test_redact_path_exact_home(self) -> None:
        import os

        from polylogue.mcp.context_pack import redact_path

        home = os.path.expanduser("~")
        result = redact_path(home)
        assert result == "~"
