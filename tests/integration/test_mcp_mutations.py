"""MCP server mutation tool tests — tags, metadata, delete, summary, session tree, stats-by, health, index."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from polylogue.lib.models import ConversationSummary

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


# =============================================================================
# Tier 2: Mutation Tool Tests
# =============================================================================


class TestAddTagTool:
    """Tests for add_tag tool."""

    def test_add_tag_success(self):
        """add_tag succeeds and returns status=ok."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.add_tag.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["add_tag"].fn(
                conversation_id="test:conv-123", tag="important"
            )

            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            assert parsed["conversation_id"] == "test:conv-123"
            assert parsed["tag"] == "important"
            mock_repo.add_tag.assert_called_once_with("test:conv-123", "important")

    def test_add_tag_error(self):
        """add_tag returns error JSON on exception."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.add_tag.side_effect = ValueError("Invalid tag")
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["add_tag"].fn(
                conversation_id="test:conv-123", tag="invalid"
            )

            parsed = json.loads(result)
            assert "error" in parsed
            assert "Invalid tag" in parsed["error"]


class TestRemoveTagTool:
    """Tests for remove_tag tool."""

    def test_remove_tag_success(self):
        """remove_tag succeeds and returns status=ok."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.remove_tag.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["remove_tag"].fn(
                conversation_id="test:conv-123", tag="important"
            )

            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            assert parsed["conversation_id"] == "test:conv-123"
            assert parsed["tag"] == "important"
            mock_repo.remove_tag.assert_called_once_with("test:conv-123", "important")

    def test_remove_tag_error(self):
        """remove_tag returns error JSON on exception."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.remove_tag.side_effect = RuntimeError("Backend error")
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["remove_tag"].fn(
                conversation_id="test:conv-123", tag="important"
            )

            parsed = json.loads(result)
            assert "error" in parsed


class TestListTagsTool:
    """Tests for list_tags tool."""

    def test_list_tags_returns_dict(self):
        """list_tags returns tag counts as JSON."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_tags.return_value = {"bug": 3, "feature": 5, "urgent": 1}
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["list_tags"].fn()

            parsed = json.loads(result)
            assert parsed["bug"] == 3
            assert parsed["feature"] == 5
            assert parsed["urgent"] == 1

    def test_list_tags_with_provider(self):
        """list_tags passes provider kwarg through."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_tags.return_value = {"claude": 2}
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["list_tags"].fn(provider="claude")

            parsed = json.loads(result)
            assert parsed == {"claude": 2}
            mock_repo.list_tags.assert_called_once_with(provider="claude")


class TestGetMetadataTool:
    """Tests for get_metadata tool."""

    def test_get_metadata_success(self):
        """get_metadata returns metadata dict as JSON."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_metadata.return_value = {"key": "value", "count": 42}
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_metadata"].fn(conversation_id="test:conv-123")

            parsed = json.loads(result)
            assert parsed["key"] == "value"
            assert parsed["count"] == 42


class TestSetMetadataTool:
    """Tests for set_metadata tool."""

    def test_set_metadata_string_value(self):
        """set_metadata with string value calls repo.update_metadata."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.update_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["set_metadata"].fn(
                conversation_id="test:conv-123", key="author", value="john"
            )

            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            mock_repo.update_metadata.assert_called_once_with("test:conv-123", "author", "john")

    def test_set_metadata_json_value(self):
        """set_metadata parses JSON value."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.update_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["set_metadata"].fn(
                conversation_id="test:conv-123", key="config", value='{"nested": true}'
            )

            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            # Verify that the JSON string was parsed before calling update_metadata
            mock_repo.update_metadata.assert_called_once_with("test:conv-123", "config", {"nested": True})


class TestDeleteMetadataTool:
    """Tests for delete_metadata tool."""

    def test_delete_metadata_success(self):
        """delete_metadata succeeds and returns status=ok."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.delete_metadata.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["delete_metadata"].fn(
                conversation_id="test:conv-123", key="author"
            )

            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            assert parsed["key"] == "author"
            mock_repo.delete_metadata.assert_called_once_with("test:conv-123", "author")


class TestDeleteConversationTool:
    """Tests for delete_conversation tool."""

    def test_delete_requires_confirm(self):
        """delete_conversation without confirm=True returns error."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["delete_conversation"].fn(
                conversation_id="test:conv-123", confirm=False
            )

            parsed = json.loads(result)
            assert "error" in parsed
            assert "confirm=true" in parsed["error"]
            mock_repo.delete_conversation.assert_not_called()

    def test_delete_with_confirm(self):
        """delete_conversation with confirm=True deletes and returns status."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.delete_conversation.return_value = True
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["delete_conversation"].fn(
                conversation_id="test:conv-123", confirm=True
            )

            parsed = json.loads(result)
            assert parsed["status"] == "deleted"
            mock_repo.delete_conversation.assert_called_once_with("test:conv-123")

    def test_delete_not_found(self):
        """delete_conversation returns not_found if conversation missing."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.delete_conversation.return_value = False
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["delete_conversation"].fn(
                conversation_id="nonexistent", confirm=True
            )

            parsed = json.loads(result)
            assert parsed["status"] == "not_found"


# =============================================================================
# Tier 3: Enhanced Read Tool Tests
# =============================================================================


class TestGetConversationSummaryTool:
    """Tests for get_conversation_summary tool."""

    def test_summary_returns_metadata(self):
        """get_conversation_summary returns summary dict with id/provider/title/message_count."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_summary = ConversationSummary(
                id="test:conv-123",
                provider="chatgpt",
                title="Test Conv",
                message_count=5,
                created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            )
            mock_repo.resolve_id.return_value = "test:conv-123"
            mock_repo.get_summary.return_value = mock_summary
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_conversation_summary"].fn(id="test:conv-123")

            parsed = json.loads(result)
            assert parsed["id"] == "test:conv-123"
            assert parsed["provider"] == "chatgpt"
            assert parsed["title"] == "Test Conv"
            assert parsed["message_count"] == 5

    def test_summary_not_found(self):
        """get_conversation_summary returns error if not found."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.resolve_id.return_value = None
            mock_repo.get_summary.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_conversation_summary"].fn(id="nonexistent")

            parsed = json.loads(result)
            assert "error" in parsed
            assert "not found" in parsed["error"].lower()


class TestGetSessionTreeTool:
    """Tests for get_session_tree tool."""

    def test_session_tree_returns_list(self, sample_conversation):
        """get_session_tree returns list of conversations."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_session_tree.return_value = [sample_conversation]
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_session_tree"].fn(conversation_id="test:conv-123")

            parsed = json.loads(result)
            assert isinstance(parsed, list)
            assert len(parsed) == 1
            assert parsed[0]["id"] == "test:conv-123"


class TestGetStatsByTool:
    """Tests for get_stats_by tool."""

    def test_stats_by_provider(self):
        """get_stats_by with provider grouping returns dict."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [("chatgpt", 10), ("claude", 5)]
            mock_conn.execute.return_value = mock_cursor
            mock_repo.backend._get_connection.return_value = mock_conn
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_stats_by"].fn(group_by="provider")

            parsed = json.loads(result)
            assert parsed["chatgpt"] == 10
            assert parsed["claude"] == 5

    def test_stats_by_month(self):
        """get_stats_by with month grouping returns dict."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [("2024-01", 15), ("2024-02", 20)]
            mock_conn.execute.return_value = mock_cursor
            mock_repo.backend._get_connection.return_value = mock_conn
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["get_stats_by"].fn(group_by="month")

            parsed = json.loads(result)
            assert parsed["2024-01"] == 15
            assert parsed["2024-02"] == 20


class TestHealthCheckTool:
    """Tests for health_check tool."""

    def test_health_check_success(self):
        """health_check returns checks list and summary."""
        from polylogue.mcp.server import _build_server

        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"
        mock_check.count = 100
        mock_check.detail = "All good"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "Healthy"
        mock_report.cached = False

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            with patch("polylogue.health.get_health") as mock_get_health:
                mock_get_config.return_value = MagicMock()
                mock_get_health.return_value = mock_report

                server = _build_server()
                result = server._tool_manager._tools["health_check"].fn()

                parsed = json.loads(result)
                assert "checks" in parsed
                assert len(parsed["checks"]) == 1
                assert parsed["checks"][0]["name"] == "database"
                assert parsed["summary"] == "Healthy"


# =============================================================================
# Tier 4: Pipeline Tool Tests
# =============================================================================


class TestRebuildIndexTool:
    """Tests for rebuild_index tool."""

    def test_rebuild_index_success(self):
        """rebuild_index returns status=ok and index info."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
                with patch("polylogue.pipeline.services.indexing.IndexService") as MockIndexService:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    mock_repo = MagicMock()
                    mock_conn = MagicMock()
                    mock_repo.backend._get_connection.return_value = mock_conn
                    mock_get_repo.return_value = mock_repo

                    mock_service = MagicMock()
                    mock_service.rebuild_index.return_value = True
                    mock_service.get_index_status.return_value = {"exists": True, "count": 500}
                    MockIndexService.return_value = mock_service

                    server = _build_server()
                    result = server._tool_manager._tools["rebuild_index"].fn()

                    parsed = json.loads(result)
                    assert parsed["status"] == "ok"
                    assert parsed["index_exists"] is True
                    assert parsed["indexed_messages"] == 500


class TestUpdateIndexTool:
    """Tests for update_index tool."""

    def test_update_index_success(self):
        """update_index returns status=ok with conversation_count."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
                with patch("polylogue.pipeline.services.indexing.IndexService") as MockIndexService:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    mock_repo = MagicMock()
                    mock_conn = MagicMock()
                    mock_repo.backend._get_connection.return_value = mock_conn
                    mock_get_repo.return_value = mock_repo

                    mock_service = MagicMock()
                    mock_service.update_index.return_value = True
                    MockIndexService.return_value = mock_service

                    server = _build_server()
                    result = server._tool_manager._tools["update_index"].fn(
                        conversation_ids=["test:conv-1", "test:conv-2"]
                    )

                    parsed = json.loads(result)
                    assert parsed["status"] == "ok"
                    assert parsed["conversation_count"] == 2
