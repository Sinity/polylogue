"""MCP user-state tool contracts (#867)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def test_list_marks_returns_typed_items(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_marks = AsyncMock(
            return_value=[
                {
                    "target_type": "session",
                    "target_id": "test:conv-123",
                    "session_id": "test:conv-123",
                    "message_id": "",
                    "mark_type": "star",
                    "created_at": "2026-05-15T00:00:00+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["list_marks"].fn,
            mark_type="star",
            session_id="test:conv-123",
        )

    parsed = json.loads(result)
    assert parsed["total"] == 1
    assert parsed["items"][0]["target_type"] == "session"
    assert parsed["items"][0]["target_id"] == "test:conv-123"
    assert parsed["items"][0]["mark_type"] == "star"
    mock_poly.list_marks.assert_awaited_once_with(
        mark_type="star",
        session_id="test:conv-123",
        target_type=None,
        target_id=None,
        message_id=None,
    )


def test_add_mark_resolves_session_and_reports_idempotency(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
        mock_poly.add_mark = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["add_mark"].fn,
            session_id="conv-123",
            mark_type="pin",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "unchanged"
    assert parsed["session_id"] == "test:conv-123"
    assert parsed["key"] == "pin"
    assert parsed["detail"] == "already_present"
    mock_poly.add_mark.assert_awaited_once_with(
        "test:conv-123",
        "pin",
        target_type="session",
        target_id=None,
        message_id=None,
    )


def test_add_mark_rejects_unknown_type(mcp_server: MCPServerUnderTest) -> None:
    result = invoke_surface(
        mcp_server._tool_manager._tools["add_mark"].fn,
        session_id="conv-123",
        mark_type="later",
    )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "star, pin, archive" in parsed["error"]


def test_remove_mark_reports_missing_mark(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
        mock_poly.remove_mark = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["remove_mark"].fn,
            session_id="conv-123",
            mark_type="star",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "not_found"
    assert parsed["detail"] == "mark_not_present"
    mock_poly.remove_mark.assert_awaited_once_with(
        "test:conv-123",
        "star",
        target_type="session",
        target_id=None,
        message_id=None,
    )


def test_remove_mark_rejects_unknown_type(mcp_server: MCPServerUnderTest) -> None:
    result = invoke_surface(
        mcp_server._tool_manager._tools["remove_mark"].fn,
        session_id="conv-123",
        mark_type="later",
    )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "star, pin, archive" in parsed["error"]


def test_add_mark_accepts_message_target(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
        mock_poly.add_mark = AsyncMock(return_value=True)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["add_mark"].fn,
            session_id="conv-123",
            mark_type="pin",
            target_type="message",
            message_id="msg-1",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    mock_poly.add_mark.assert_awaited_once_with(
        "test:conv-123",
        "pin",
        target_type="message",
        target_id=None,
        message_id="msg-1",
    )


def test_annotations_roundtrip_through_typed_mcp_payloads(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
        mock_poly.save_annotation = AsyncMock(return_value=True)
        mock_poly.list_annotations = AsyncMock(
            return_value=[
                {
                    "annotation_id": "ann-1",
                    "target_type": "message",
                    "target_id": "msg-1",
                    "session_id": "test:conv-123",
                    "message_id": "msg-1",
                    "note_text": "Important",
                    "created_at": "2026-05-15T00:00:00+00:00",
                    "updated_at": "2026-05-15T00:00:01+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        saved = invoke_surface(
            mcp_server._tool_manager._tools["save_annotation"].fn,
            annotation_id="ann-1",
            session_id="conv-123",
            note_text="Important",
            target_type="message",
            message_id="msg-1",
        )
        listed = invoke_surface(
            mcp_server._tool_manager._tools["list_annotations"].fn,
            session_id="test:conv-123",
        )

    saved_payload = json.loads(saved)
    listed_payload = json.loads(listed)
    assert saved_payload["status"] == "ok"
    assert saved_payload["outcome"] == "added"
    assert listed_payload["total"] == 1
    assert listed_payload["items"][0]["target_type"] == "message"
    assert listed_payload["items"][0]["message_id"] == "msg-1"
    mock_poly.save_annotation.assert_awaited_once_with(
        "ann-1",
        "test:conv-123",
        "Important",
        target_type="message",
        target_id=None,
        message_id="msg-1",
    )


def test_delete_annotation_reports_status(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.delete_annotation = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["delete_annotation"].fn, annotation_id="ann-1")

    parsed = json.loads(result)
    assert parsed["status"] == "not_found"
    assert parsed["detail"] == "annotation_not_found"
    mock_poly.delete_annotation.assert_awaited_once_with("ann-1")


def test_list_saved_views_decodes_query_json(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_views = AsyncMock(
            return_value=[
                {
                    "view_id": "view-1",
                    "name": "Claude Code",
                    "query_json": '{"origin":"claude-code-session","query":"Hello"}',
                    "created_at": "2026-05-15T00:00:00+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["list_saved_views"].fn)

    parsed = json.loads(result)
    assert parsed["total"] == 1
    assert parsed["items"][0]["query"] == {"origin": "claude-code-session", "query": "Hello"}
    mock_poly.list_views.assert_awaited_once_with()


def test_save_saved_view_validates_query_spec(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.save_view = AsyncMock(return_value=True)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["save_saved_view"].fn,
            name="Claude Code",
            query_json='{"origin":"claude-code-session"}',
            view_id="view-1",
        )

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["key"] == "view-1"
    mock_poly.save_view.assert_awaited_once_with("view-1", "Claude Code", '{"origin":"claude-code-session"}')


def test_save_saved_view_default_id_is_content_addressed(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.save_view = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["save_saved_view"].fn,
            name="Claude Code",
            query_json='{"origin":"claude-code-session"}',
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
    assert "SessionQuerySpec" in parsed["error"]
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


def test_recall_pack_tools_roundtrip_typed_payloads(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.create_recall_pack = AsyncMock(return_value=True)
        mock_poly.list_recall_packs = AsyncMock(
            return_value=[
                {
                    "pack_id": "pack-1",
                    "label": "Handoff",
                    "session_ids_json": '["conv-1"]',
                    "payload_json": (
                        '{"items":[{"target_type":"session","target_id":"conv-1","status":"resolved"}],'
                        '"resolved_count":1,"degraded_count":0}'
                    ),
                    "created_at": "2026-05-15T00:00:00+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        saved = invoke_surface(
            mcp_server._tool_manager._tools["save_recall_pack"].fn,
            pack_id="pack-1",
            label="Handoff",
            payload_json=(
                '{"items":['
                '{"target_type":"session","session_id":"conv-1"},'
                '{"target_type":"annotation","annotation_id":"ann-1"}'
                "]}"
            ),
        )
        listed = invoke_surface(mcp_server._tool_manager._tools["list_recall_packs"].fn)

    saved_payload = json.loads(saved)
    listed_payload = json.loads(listed)
    assert saved_payload["status"] == "ok"
    assert saved_payload["outcome"] == "added"
    assert listed_payload["total"] == 1
    assert listed_payload["items"][0]["payload"]["resolved_count"] == 1
    mock_poly.create_recall_pack.assert_awaited_once_with(
        "pack-1",
        "Handoff",
        '{"items":[{"session_id":"conv-1","target_type":"session"},{"annotation_id":"ann-1","target_type":"annotation"}]}',
    )


def test_delete_recall_pack_reports_status(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.delete_recall_pack = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["delete_recall_pack"].fn, pack_id="pack-1")

    parsed = json.loads(result)
    assert parsed["status"] == "not_found"
    assert parsed["detail"] == "recall_pack_not_found"
    mock_poly.delete_recall_pack.assert_awaited_once_with("pack-1")


def test_workspace_tools_roundtrip_typed_payloads(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.save_workspace = AsyncMock(return_value=True)
        mock_poly.list_workspaces = AsyncMock(
            return_value=[
                {
                    "workspace_id": "workspace-1",
                    "name": "Investigation",
                    "mode": "compare",
                    "open_targets_json": ('[{"target_type":"session","target_id":"conv-1","status":"resolved"}]'),
                    "layout_json": '{"panes":[{"width":0.5},{"width":0.5}]}',
                    "active_target_json": '{"target_type":"session","target_id":"conv-1","status":"resolved"}',
                    "created_at": "2026-05-15T00:00:00+00:00",
                    "updated_at": "2026-05-15T00:00:01+00:00",
                }
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        saved = invoke_surface(
            mcp_server._tool_manager._tools["save_workspace"].fn,
            workspace_id="workspace-1",
            name="Investigation",
            mode="compare",
            open_targets_json='[{"target_type":"session","session_id":"conv-1"}]',
            layout_json='{"panes":[{"width":0.5},{"width":0.5}]}',
            active_target_json='{"target_type":"session","session_id":"conv-1"}',
        )
        listed = invoke_surface(mcp_server._tool_manager._tools["list_workspaces"].fn)

    saved_payload = json.loads(saved)
    listed_payload = json.loads(listed)
    assert saved_payload["status"] == "ok"
    assert saved_payload["outcome"] == "added"
    assert listed_payload["total"] == 1
    assert listed_payload["items"][0]["mode"] == "compare"
    assert listed_payload["items"][0]["layout"] == {"panes": [{"width": 0.5}, {"width": 0.5}]}
    mock_poly.save_workspace.assert_awaited_once_with(
        "workspace-1",
        "Investigation",
        "compare",
        '[{"session_id":"conv-1","target_type":"session"}]',
        '{"panes":[{"width":0.5},{"width":0.5}]}',
        '{"session_id":"conv-1","target_type":"session"}',
    )


def test_delete_workspace_reports_status(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.delete_workspace = AsyncMock(return_value=False)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(mcp_server._tool_manager._tools["delete_workspace"].fn, workspace_id="workspace-1")

    parsed = json.loads(result)
    assert parsed["status"] == "not_found"
    assert parsed["detail"] == "workspace_not_found"
    mock_poly.delete_workspace.assert_awaited_once_with("workspace-1")
