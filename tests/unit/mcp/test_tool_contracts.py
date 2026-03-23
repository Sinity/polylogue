"""Unit contracts for MCP tool surfaces backed by repository/config mocks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.lib.query_spec import ConversationQuerySpec
from polylogue.lib.stats import ArchiveStats
from tests.infra.mcp import (
    invoke_surface,
    invoke_surface_async,
    make_repo_mock,
    make_simple_conversation,
)

STATS_CONFIGS = [
    (100, 5000, {"claude-ai": 50, "chatgpt": 30, "claude-code": 20}, 10, 200, 1048576, 1.0),
    (0, 0, {}, 0, 0, 0, 0),
    (5, 20, {"test": 5}, 0, 0, None, 0),
]

QUERY_TOOL_CASES = [
    (
        "search",
        {"query": "hello", "limit": 10},
        {
            "contains": ("hello",),
            "limit": (10,),
        },
    ),
    (
        "search",
        {"query": "hello", "provider": "claude-ai", "since": "2024-01-01", "limit": 5},
        {
            "contains": ("hello",),
            "provider": ("claude-ai",),
            "since": ("2024-01-01",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "path": "/realm/project/polylogue/README.md", "limit": 5},
        {
            "contains": ("hello",),
            "path": ("/realm/project/polylogue/README.md",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "action": "search", "exclude_action": "git", "limit": 5},
        {
            "contains": ("hello",),
            "action": ("search",),
            "exclude_action": ("git",),
            "limit": (5,),
        },
    ),
    (
        "list_conversations",
        {"limit": 10},
        {
            "limit": (10,),
        },
    ),
    (
        "list_conversations",
        {"provider": "claude-ai", "since": "2024-01-01", "tag": "bug", "title": "incident", "limit": 2},
        {
            "provider": ("claude-ai",),
            "since": ("2024-01-01",),
            "tag": ("bug",),
            "title": ("incident",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"path": "/realm/project/polylogue/README.md", "limit": 2},
        {
            "path": ("/realm/project/polylogue/README.md",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action": "file_edit", "exclude_action": "web", "limit": 2},
        {
            "action": ("file_edit",),
            "exclude_action": ("web",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action": "none", "limit": 2},
        {
            "action": ("none",),
            "limit": (2,),
        },
    ),
]


@pytest.fixture
def simple_conversation() -> Conversation:
    return make_simple_conversation()


class TestQueryTools:
    @pytest.mark.parametrize(("tool_name", "args", "expected_calls"), QUERY_TOOL_CASES)
    @pytest.mark.asyncio
    async def test_query_tool_filter_contract(self, simple_conversation, tool_name, args, expected_calls, mcp_server):
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.query_conversations = AsyncMock(return_value=[simple_conversation])
            mock_get_archive_ops.return_value = mock_ops

            raw = await mcp_server._tool_manager._tools[tool_name].fn(**args)

        payload = json.loads(raw)
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["id"] == simple_conversation.id
        mock_ops.query_conversations.assert_awaited_once()
        spec = mock_ops.query_conversations.await_args.args[0]
        assert isinstance(spec, ConversationQuerySpec)
        for method_name, method_args in expected_calls.items():
            expected_value = method_args[0] if len(method_args) == 1 else method_args
            if method_name == "contains":
                assert spec.query_terms == (expected_value,)
            elif method_name == "provider":
                assert tuple(str(provider) for provider in spec.providers) == (expected_value,)
            elif method_name == "since":
                assert spec.since == expected_value
            elif method_name == "path":
                assert spec.path_terms == (expected_value,)
            elif method_name == "action":
                assert spec.action_terms == (expected_value,)
            elif method_name == "exclude_action":
                assert spec.excluded_action_terms == (expected_value,)
            elif method_name == "tag":
                assert spec.tags == (expected_value,)
            elif method_name == "title":
                assert spec.title == expected_value
            elif method_name == "limit":
                assert spec.limit == expected_value

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, mcp_server):
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.query_conversations = AsyncMock(return_value=[])
            mock_get_archive_ops.return_value = mock_ops

            result = await invoke_surface_async(mcp_server._tool_manager._tools["search"].fn, query="", limit=10)

        parsed = json.loads(result)
        assert isinstance(parsed, (list, dict))


class TestGetConversationTool:
    def test_get_returns_conversation(self, simple_conversation, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = simple_conversation
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation"].fn, id="test:conv-123")

        conv = json.loads(result)
        assert conv["id"] == "test:conv-123"
        assert len(conv["messages"]) == 2

    def test_get_not_found(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_get_returns_full_messages(self, mcp_server):
        long_text = "A" * 2000
        conv = Conversation(
            id="test:long",
            provider="test",
            title="Long Message",
            messages=[Message(id="m1", role="assistant", text=long_text)],
        )

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = conv
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation"].fn, id="test:long")

        assert json.loads(result)["messages"][0]["text"] == long_text

    @pytest.mark.asyncio
    async def test_get_with_nonexistent_id(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            result = await invoke_surface_async(mcp_server._tool_manager._tools["get_conversation"].fn, id="nonexistent-id-xyz")

        assert isinstance(json.loads(result), dict)


class TestStatsTool:
    @pytest.mark.parametrize(
        (
            "total_conversations",
            "total_messages",
            "providers",
            "embedded_convs",
            "embedded_msgs",
            "db_size",
            "expected_mb",
        ),
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
        mcp_server,
    ):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.get_archive_stats.return_value = ArchiveStats(
                total_conversations=total_conversations,
                total_messages=total_messages,
                providers=providers,
                embedded_conversations=embedded_convs,
                embedded_messages=embedded_msgs,
                db_size_bytes=db_size,
            )
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["stats"].fn)

        data = json.loads(result)
        assert data["total_conversations"] == total_conversations
        assert data["total_messages"] == total_messages
        assert data["db_size_mb"] == expected_mb


class TestMutationTools:
    def test_add_tag_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.add_tag.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["add_tag"].fn, conversation_id="test:conv-123", tag="important")

        parsed = json.loads(result)
        assert parsed == {"status": "ok", "conversation_id": "test:conv-123", "tag": "important"}
        mock_repo.add_tag.assert_called_once_with("test:conv-123", "important")

    def test_add_tag_error(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.add_tag.side_effect = ValueError("Invalid tag")
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["add_tag"].fn, conversation_id="test:conv-123", tag="invalid")

        parsed = json.loads(result)
        assert "Invalid tag" in parsed["error"]

    def test_remove_tag_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.remove_tag.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["remove_tag"].fn, conversation_id="test:conv-123", tag="important")

        parsed = json.loads(result)
        assert parsed == {"status": "ok", "conversation_id": "test:conv-123", "tag": "important"}
        mock_repo.remove_tag.assert_called_once_with("test:conv-123", "important")

    def test_remove_tag_error(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.remove_tag.side_effect = RuntimeError("Backend error")
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["remove_tag"].fn, conversation_id="test:conv-123", tag="important")

        assert "error" in json.loads(result)

    def test_list_tags_returns_counts(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.list_tags.return_value = {"bug": 3, "feature": 5, "urgent": 1}
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn)

        assert json.loads(result) == {"bug": 3, "feature": 5, "urgent": 1}

    def test_list_tags_with_provider(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.list_tags.return_value = {"claude-ai": 2}
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn, provider="claude-ai")

        assert json.loads(result) == {"claude-ai": 2}
        mock_repo.list_tags.assert_called_once_with(provider="claude-ai")

    def test_get_metadata_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.get_metadata.return_value = {"key": "value", "count": 42}
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_metadata"].fn, conversation_id="test:conv-123")

        assert json.loads(result) == {"key": "value", "count": 42}

    def test_set_metadata_string_value(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.update_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                conversation_id="test:conv-123",
                key="author",
                value="john",
            )

        assert json.loads(result)["status"] == "ok"
        mock_repo.update_metadata.assert_called_once_with("test:conv-123", "author", "john")

    def test_set_metadata_json_value(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.update_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                conversation_id="test:conv-123",
                key="config",
                value='{"nested": true}',
            )

        assert json.loads(result)["status"] == "ok"
        mock_repo.update_metadata.assert_called_once_with("test:conv-123", "config", {"nested": True})

    def test_delete_metadata_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.delete_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_metadata"].fn,
                conversation_id="test:conv-123",
                key="author",
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["key"] == "author"
        mock_repo.delete_metadata.assert_called_once_with("test:conv-123", "author")

    def test_delete_requires_confirm(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="test:conv-123",
                confirm=False,
            )

        parsed = json.loads(result)
        assert "confirm=true" in parsed["error"]
        mock_repo.delete_conversation.assert_not_called()

    def test_delete_with_confirm(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.delete_conversation.return_value = True
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="test:conv-123",
                confirm=True,
            )

        assert json.loads(result)["status"] == "deleted"
        mock_repo.delete_conversation.assert_called_once_with("test:conv-123")

    def test_delete_not_found(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.delete_conversation.return_value = False
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="nonexistent",
                confirm=True,
            )

        assert json.loads(result)["status"] == "not_found"

    def test_summary_returns_metadata(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_summary = ConversationSummary(
                id="test:conv-123",
                provider="chatgpt",
                title="Test Conv",
                message_count=None,
                created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            )
            mock_repo.resolve_id.return_value = "test:conv-123"
            mock_repo.get_summary.return_value = mock_summary
            mock_repo.queries.get_conversation_stats.return_value = {"total_messages": 5}
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation_summary"].fn, id="test:conv-123")

        parsed = json.loads(result)
        assert parsed["id"] == "test:conv-123"
        assert parsed["provider"] == "chatgpt"
        assert parsed["title"] == "Test Conv"
        assert parsed["message_count"] == 5

    def test_summary_not_found(self, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.resolve_id.return_value = None
            mock_repo.get_summary.return_value = None
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation_summary"].fn, id="nonexistent")

        assert "not found" in json.loads(result)["error"].lower()

    def test_session_tree_returns_list(self, simple_conversation, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.get_session_tree.return_value = [simple_conversation]
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_session_tree"].fn, conversation_id="test:conv-123")

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == "test:conv-123"

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            ("provider", {"chatgpt": 10, "claude-ai": 5}),
            ("month", {"2024-01": 15, "2024-02": 20}),
        ],
    )
    def test_stats_by_group(self, group_by, expected, mcp_server):
        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.queries.get_stats_by.return_value = expected
            mock_get_repo.return_value = mock_repo

            result = invoke_surface(mcp_server._tool_manager._tools["get_stats_by"].fn, group_by=group_by)

        assert json.loads(result) == expected

    def test_health_check_success(self, mcp_server):
        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"
        mock_check.count = 100
        mock_check.detail = "All good"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "Healthy"
        mock_report.cached = False

        with patch("polylogue.mcp.server._get_config") as mock_get_config, patch(
            "polylogue.health_archive.get_health"
        ) as mock_get_health:
            mock_get_config.return_value = MagicMock()
            mock_get_health.return_value = mock_report

            result = invoke_surface(mcp_server._tool_manager._tools["health_check"].fn)

        parsed = json.loads(result)
        assert parsed["summary"] == "Healthy"
        assert parsed["checks"][0]["name"] == "database"

    def test_rebuild_index_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_config") as mock_get_config, patch(
            "polylogue.mcp.server._get_repo"
        ) as mock_get_repo, patch("polylogue.pipeline.services.indexing.IndexService") as mock_service_cls:
            mock_get_config.return_value = MagicMock()
            mock_get_repo.return_value = make_repo_mock()
            mock_service = MagicMock()
            mock_service.rebuild_index = AsyncMock(return_value=True)
            mock_service.get_index_status = AsyncMock(return_value={"exists": True, "count": 500})
            mock_service_cls.return_value = mock_service

            result = invoke_surface(mcp_server._tool_manager._tools["rebuild_index"].fn)

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["index_exists"] is True
        assert parsed["indexed_messages"] == 500

    def test_update_index_success(self, mcp_server):
        with patch("polylogue.mcp.server._get_config") as mock_get_config, patch(
            "polylogue.mcp.server._get_repo"
        ) as mock_get_repo, patch("polylogue.pipeline.services.indexing.IndexService") as mock_service_cls:
            mock_get_config.return_value = MagicMock()
            mock_get_repo.return_value = make_repo_mock()
            mock_service = MagicMock()
            mock_service.update_index = AsyncMock(return_value=True)
            mock_service_cls.return_value = mock_service

            result = invoke_surface(
                mcp_server._tool_manager._tools["update_index"].fn,
                conversation_ids=["test:conv-1", "test:conv-2"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["conversation_count"] == 2
