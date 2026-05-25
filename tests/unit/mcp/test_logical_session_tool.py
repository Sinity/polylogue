from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.insights.topology import ConversationRef, LogicalSession
from polylogue.types import ConversationId
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def test_logical_session_tool_returns_compact_envelope(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_logical_session = AsyncMock(
            return_value=LogicalSession(
                conversation_id=ConversationId("fork"),
                root_id=ConversationId("root"),
                thread=(
                    ConversationRef(conversation_id=ConversationId("root"), source_name="claude-code", depth=0),
                    ConversationRef(conversation_id=ConversationId("fork"), source_name="claude-code", depth=1),
                ),
                siblings=(),
                descendants=(),
                cycle_detected=False,
            )
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["get_logical_session"].fn,
            conversation_id="fork",
        )

    parsed = json.loads(result)
    assert parsed["conversation_id"] == "fork"
    assert parsed["root_id"] == "root"
    assert [item["conversation_id"] for item in parsed["thread"]] == ["root", "fork"]
    assert parsed["cycle_detected"] is False


def test_logical_session_tool_returns_not_found(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_logical_session = AsyncMock(return_value=None)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["get_logical_session"].fn,
            conversation_id="missing",
        )

    parsed = json.loads(result)
    assert parsed["code"] == "not_found"
