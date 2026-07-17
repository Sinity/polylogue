from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.core.types import SessionId
from polylogue.insights.topology import LogicalSession, SessionRef
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def test_logical_session_tool_returns_compact_envelope(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_logical_session = AsyncMock(
            return_value=LogicalSession(
                session_id=SessionId("fork"),
                root_id=SessionId("root"),
                thread=(
                    SessionRef(session_id=SessionId("root"), origin="claude-code", depth=0),
                    SessionRef(session_id=SessionId("fork"), origin="claude-code", depth=1),
                ),
                siblings=(),
                descendants=(),
                cycle_detected=False,
            )
        )
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["get_logical_session"].fn,
            session_id="fork",
        )

    parsed = json.loads(result)
    assert parsed["session_id"] == "fork"
    assert parsed["root_id"] == "root"
    assert [item["session_id"] for item in parsed["thread"]] == ["root", "fork"]
    assert parsed["cycle_detected"] is False


def test_logical_session_tool_returns_not_found(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_logical_session = AsyncMock(return_value=None)
        mock_get_polylogue.return_value = mock_poly

        result = invoke_surface(
            mcp_server._tool_manager._tools["get_logical_session"].fn,
            session_id="missing",
        )

    parsed = json.loads(result)
    assert parsed["code"] == "not_found"
