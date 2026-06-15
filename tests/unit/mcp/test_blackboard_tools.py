"""MCP blackboard tool contracts (#1697)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.archive.blackboard import BlackboardNote
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def _note(**overrides: object) -> BlackboardNote:
    base: dict[str, object] = {
        "note_id": "note-1",
        "kind": "finding",
        "title": "t",
        "content": "c",
        "scope_repo": "polylogue",
        "target_type": None,
        "target_id": None,
        "created_at_ms": 111,
        "updated_at_ms": 222,
    }
    base.update(overrides)
    return BlackboardNote(**base)  # type: ignore[arg-type]


def test_blackboard_list_returns_typed_items(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_blackboard_notes = AsyncMock(return_value=[_note(), _note(note_id="note-2", kind="blocker")])
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["blackboard_list"].fn,
            kind=None,
            scope_repo="polylogue",
            unresolved=False,
            limit=20,
        )

    parsed = json.loads(result)
    assert parsed["total"] == 2
    assert [item["note_id"] for item in parsed["items"]] == ["note-1", "note-2"]
    assert parsed["items"][0]["kind"] == "finding"
    assert parsed["items"][1]["kind"] == "blocker"
    mock_poly.list_blackboard_notes.assert_awaited_once_with(
        kind=None,
        scope_repo="polylogue",
        unresolved=False,
        limit=20,
    )


def test_blackboard_post_passes_fields_and_returns_note(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.post_blackboard_note = AsyncMock(return_value=_note(kind="handoff", title="pick up"))
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["blackboard_post"].fn,
            kind="handoff",
            title="pick up",
            content="MCP tools remain",
            scope_repo="polylogue",
            related_sessions=["claude-code:abc"],
        )

    parsed = json.loads(result)
    assert parsed["kind"] == "handoff"
    assert parsed["title"] == "pick up"
    mock_poly.post_blackboard_note.assert_awaited_once_with(
        kind="handoff",
        title="pick up",
        content="MCP tools remain",
        scope_repo="polylogue",
        scope_session=None,
        scope_issue=None,
        scope_path=None,
        related_sessions=("claude-code:abc",),
    )


def test_blackboard_post_reports_invalid_kind_as_error(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.post_blackboard_note = AsyncMock(side_effect=ValueError("kind must be one of [...]"))
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["blackboard_post"].fn,
            kind="bogus",
            title="t",
            content="c",
        )

    parsed = json.loads(result)
    assert parsed["is_error"] is True
    assert "kind must be one of" in parsed["message"]
