"""MCP server export and completeness tests — export tool, new resources, new prompts, safe_call, tool inventory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from polylogue.lib.models import Conversation, Message
from tests.integration.conftest import make_mock_filter


# =============================================================================
# Tier 5: Export Tool Tests
# =============================================================================


class TestExportConversationTool:
    """Tests for export_conversation tool."""

    def test_export_markdown(self, sample_conversation):
        """export_conversation with markdown format calls _format_conversation."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            with patch("polylogue.cli.query._format_conversation") as mock_format:
                mock_repo = MagicMock()
                mock_repo.view.return_value = sample_conversation
                mock_get_repo.return_value = mock_repo
                mock_format.return_value = "# Test Conversation\n\nFormatted content"

                server = _build_server()
                result = server._tool_manager._tools["export_conversation"].fn(
                    id="test:conv-123", format="markdown"
                )

                assert "Test Conversation" in result
                mock_format.assert_called_once()
                call_args = mock_format.call_args
                assert call_args[0][0] == sample_conversation
                assert call_args[0][1] == "markdown"

    def test_export_not_found(self):
        """export_conversation returns error if conversation not found."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.view.return_value = None
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._tool_manager._tools["export_conversation"].fn(id="nonexistent")

            parsed = json.loads(result)
            assert "error" in parsed
            assert "not found" in parsed["error"].lower()

    def test_export_invalid_format(self, sample_conversation):
        """export_conversation falls back to markdown for invalid format."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            with patch("polylogue.cli.query._format_conversation") as mock_format:
                mock_repo = MagicMock()
                mock_repo.view.return_value = sample_conversation
                mock_get_repo.return_value = mock_repo
                mock_format.return_value = "# Content"

                server = _build_server()
                result = server._tool_manager._tools["export_conversation"].fn(
                    id="test:conv-123", format="invalid_format"
                )

                # Verify markdown was used as fallback
                call_args = mock_format.call_args
                assert call_args[0][1] == "markdown"


# =============================================================================
# New Resource Tests
# =============================================================================


class TestNewResources:
    """Tests for new resources."""

    def test_tags_resource(self):
        """polylogue://tags resource returns tag counts."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_tags.return_value = {"feature": 10, "bug": 5}
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._resource_manager._resources["polylogue://tags"].fn()

            parsed = json.loads(result)
            assert parsed["feature"] == 10
            assert parsed["bug"] == 5

    def test_health_resource(self):
        """polylogue://health resource returns health status."""
        from polylogue.mcp.server import _build_server

        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "All systems operational"

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            with patch("polylogue.health.get_health") as mock_get_health:
                mock_get_config.return_value = MagicMock()
                mock_get_health.return_value = mock_report

                server = _build_server()
                result = server._resource_manager._resources["polylogue://health"].fn()

                parsed = json.loads(result)
                assert "checks" in parsed
                assert parsed["summary"] == "All systems operational"


# =============================================================================
# New Prompt Tests
# =============================================================================


class TestNewPrompts:
    """Tests for new prompts."""

    def test_compare_conversations_prompt(self, sample_conversation):
        """compare_conversations prompt returns comparison text."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.view.side_effect = [sample_conversation, sample_conversation]
            mock_get_repo.return_value = mock_repo

            server = _build_server()
            result = server._prompt_manager._prompts["compare_conversations"].fn(
                id1="test:conv-1", id2="test:conv-2"
            )

            assert isinstance(result, str)
            assert "Compare" in result
            assert "Conversation 1" in result
            assert "Conversation 2" in result

    def test_extract_patterns_prompt(self, sample_conversation):
        """extract_patterns prompt returns pattern analysis text."""
        from polylogue.mcp.server import _build_server

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            with patch("polylogue.lib.filters.ConversationFilter") as MockFilter:
                mock_repo = MagicMock()
                mock_get_repo.return_value = mock_repo
                MockFilter.return_value = make_mock_filter(results=[sample_conversation])

                server = _build_server()
                result = server._prompt_manager._prompts["extract_patterns"].fn()

                assert isinstance(result, str)
                assert "patterns" in result.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSafeCall:
    """Tests for _safe_call error wrapper."""

    def test_safe_call_returns_result(self):
        """_safe_call returns result when fn succeeds."""
        from polylogue.mcp.server import _safe_call

        def success_fn():
            return "ok"

        result = _safe_call("test_fn", success_fn)
        assert result == "ok"

    def test_safe_call_catches_exception(self):
        """_safe_call catches exceptions and returns error JSON."""
        from polylogue.mcp.server import _safe_call

        def failing_fn():
            raise ValueError("Something went wrong")

        result = _safe_call("test_fn", failing_fn)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Something went wrong" in parsed["error"]
        assert parsed["tool"] == "test_fn"


# =============================================================================
# Updated Server Building Test
# =============================================================================


class TestServerHasAllNewTools:
    """Updated test to verify all tools exist including new ones."""

    def test_server_has_all_tools_v2(self):
        """Built server has all expected tools (original + new)."""
        from polylogue.mcp.server import _build_server

        server = _build_server()
        tool_names = set(server._tool_manager._tools.keys())

        # Original tools
        assert "search" in tool_names
        assert "list_conversations" in tool_names
        assert "get_conversation" in tool_names
        assert "stats" in tool_names

        # New Tier 2 mutation tools
        assert "add_tag" in tool_names
        assert "remove_tag" in tool_names
        assert "list_tags" in tool_names
        assert "get_metadata" in tool_names
        assert "set_metadata" in tool_names
        assert "delete_metadata" in tool_names
        assert "delete_conversation" in tool_names

        # New Tier 3 enhanced read tools
        assert "get_conversation_summary" in tool_names
        assert "get_session_tree" in tool_names
        assert "get_stats_by" in tool_names

        # New Tier 3 health check
        assert "health_check" in tool_names

        # New Tier 4 pipeline tools
        assert "rebuild_index" in tool_names
        assert "update_index" in tool_names

        # New Tier 5 export tool
        assert "export_conversation" in tool_names
