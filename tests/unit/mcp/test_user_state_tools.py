"""MCP user-state tool contracts (#867)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock, make_query_store_mock


def test_list_marks_returns_typed_items(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_marks = AsyncMock(
            return_value=[
                {
                    "conversation_id": "test:conv-123",
                    "mark_type": "star",
                    "created_at": "2026-05-15T00:00:00+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["list_marks"].fn,
            mark_type="star",
            conversation_id="test:conv-123",
        )

    parsed = json.loads(result)
    assert parsed["total"] == 1
    assert parsed["items"][0]["mark_type"] == "star"
    mock_poly.list_marks.assert_awaited_once_with(mark_type="star", conversation_id="test:conv-123")


def test_add_mark_resolves_conversation_and_reports_idempotency(mcp_server: MCPServerUnderTest) -> None:
    with (
        patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
    ):
        mock_poly = make_polylogue_mock()
        mock_poly.add_mark = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly
        mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")

        result = invoke_surface(
            mcp_server._tool_manager._tools["add_mark"].fn,
            conversation_id="conv-123",
            mark_type="pin",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "unchanged"
    assert parsed["conversation_id"] == "test:conv-123"
    assert parsed["key"] == "pin"
    assert parsed["detail"] == "already_present"
    mock_poly.add_mark.assert_awaited_once_with("test:conv-123", "pin")


def test_add_mark_rejects_unknown_type(mcp_server: MCPServerUnderTest) -> None:
    result = invoke_surface(
        mcp_server._tool_manager._tools["add_mark"].fn,
        conversation_id="conv-123",
        mark_type="later",
    )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "star, pin, archive" in parsed["error"]


def test_remove_mark_reports_missing_mark(mcp_server: MCPServerUnderTest) -> None:
    with (
        patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
    ):
        mock_poly = make_polylogue_mock()
        mock_poly.remove_mark = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly
        mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")

        result = invoke_surface(
            mcp_server._tool_manager._tools["remove_mark"].fn,
            conversation_id="conv-123",
            mark_type="star",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "not_found"
    assert parsed["detail"] == "mark_not_present"
    mock_poly.remove_mark.assert_awaited_once_with("test:conv-123", "star")


def test_remove_mark_rejects_unknown_type(mcp_server: MCPServerUnderTest) -> None:
    result = invoke_surface(
        mcp_server._tool_manager._tools["remove_mark"].fn,
        conversation_id="conv-123",
        mark_type="later",
    )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "star, pin, archive" in parsed["error"]


def test_list_saved_views_decodes_query_json(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_views = AsyncMock(
            return_value=[
                {
                    "view_id": "view-1",
                    "name": "Claude Code",
                    "query_json": '{"provider":"claude-code","query":"Hello"}',
                    "created_at": "2026-05-15T00:00:00+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["list_saved_views"].fn)

    parsed = json.loads(result)
    assert parsed["total"] == 1
    assert parsed["items"][0]["query"] == {"provider": "claude-code", "query": "Hello"}
    mock_poly.list_views.assert_awaited_once_with()


def test_save_saved_view_validates_query_spec(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.save_view = AsyncMock(return_value=True)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["save_saved_view"].fn,
            name="Claude Code",
            query_json='{"provider":"claude-code"}',
            view_id="view-1",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["key"] == "view-1"
    mock_poly.save_view.assert_awaited_once_with("view-1", "Claude Code", '{"provider":"claude-code"}')


def test_save_saved_view_default_id_is_content_addressed(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.save_view = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["save_saved_view"].fn,
            name="Claude Code",
            query_json='{"provider":"claude-code"}',
        )

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["outcome"] == "updated"
    assert str(parsed["key"]).startswith("saved-view-")
    saved_id = mock_poly.save_view.await_args.args[0]
    assert saved_id == parsed["key"]
    assert saved_id != "claude-code"


def test_save_saved_view_rejects_unknown_query_param(mcp_server: MCPServerUnderTest) -> None:
    result = invoke_surface(
        mcp_server._tool_manager._tools["save_saved_view"].fn,
        name="bad",
        query_json='{"not_a_filter":true}',
        view_id="bad",
    )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "ConversationQuerySpec" in parsed["error"]
    assert "not_a_filter" in parsed["detail"]


def test_delete_saved_view_reports_status(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.delete_view = AsyncMock(return_value=True)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["delete_saved_view"].fn, view_id="view-1")

    parsed = json.loads(result)
    assert parsed["status"] == "deleted"
    assert parsed["key"] == "view-1"
    mock_poly.delete_view.assert_awaited_once_with("view-1")
