"""polylogue-9layg: MCP `read` must serve the messages view, not refuse it.

`read` accepted only the summary and topology views, so a caller asking for
messages — the view the CLI serves — got `unsupported read view` rather than
the session's messages.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.storage.runtime import LineageCompleteness
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock

_SESSION_ID = "codex-session:native-1"


async def _invoke_read(mcp_server: MCPServerUnderTest, *, view: str) -> dict[str, object]:
    with patch("polylogue.mcp.server._get_polylogue") as get_polylogue:
        poly = make_polylogue_mock(resolved_id=_SESSION_ID)
        poly.get_messages_paginated = AsyncMock(
            return_value=((), 0, LineageCompleteness(complete=True, truncation_reason=None))
        )
        get_polylogue.return_value = poly
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["read"].fn,
            ref=_SESSION_ID,
            view=view,
        )
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.asyncio
async def test_read_serves_the_messages_view(mcp_server: MCPServerUnderTest) -> None:
    """Anti-vacuity: drop the messages branch and this returns the
    `unsupported read view` error instead of a messages payload."""
    payload = await _invoke_read(mcp_server, view="messages")

    assert "unsupported read view" not in json.dumps(payload)
    assert payload.get("session_id") == _SESSION_ID
    assert "messages" in payload


@pytest.mark.asyncio
async def test_read_still_refuses_a_view_it_does_not_serve(mcp_server: MCPServerUnderTest) -> None:
    """Widening must not turn the view name into a free-for-all."""
    payload = await _invoke_read(mcp_server, view="not-a-view")

    assert "unsupported read view" in json.dumps(payload)
