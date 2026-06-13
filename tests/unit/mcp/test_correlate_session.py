"""Tests for the correlate_session MCP tool (#1690)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.models import Session
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def _make_session_with_messages() -> Session:
    """Build a mock session with tool-call messages for testing."""
    conv = make_conv(
        id="test-session-1",
        provider="claude-code",
        title="Test Session",
        messages=[
            make_msg(
                id="msg-1",
                role="user",
                text="Fix issue #1690 in Sinity/polylogue",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
            make_msg(
                id="msg-2",
                role="assistant",
                text="I'll read src/main.py first.",
                timestamp=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
            ),
        ],
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
    )
    # Add content_blocks to messages
    for msg in conv.messages:
        if msg.id == "msg-2":
            msg.blocks = [
                {
                    "type": "tool_use",
                    "name": "Read",
                    "affected_paths": ["src/main.py"],
                }
            ]
        elif msg.id == "msg-1":
            msg.blocks = []
    return conv


@pytest.mark.asyncio
async def test_correlate_session_not_found(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=None)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_session"].fn,
            session_id="nonexistent",
        )

    payload = json.loads(raw)
    assert "error" in payload or "code" in payload


@pytest.mark.asyncio
async def test_correlate_session_returns_result_shape(
    mcp_server: MCPServerUnderTest,
) -> None:
    conv = _make_session_with_messages()
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_session"].fn,
            session_id="test-session-1",
            repo_path="/nonexistent/path",
        )

    payload = json.loads(raw)
    # Should have a root key (MCPRootPayload)
    root = payload.get("root", payload)
    assert root["session_id"] == "test-session-1"
    assert "window_start" in root
    assert "window_end" in root
    assert "commits" in root
    assert "issue_refs" in root
    assert "pr_refs" in root
    assert "file_paths" in root


@pytest.mark.asyncio
async def test_correlate_session_extracts_issue_refs(
    mcp_server: MCPServerUnderTest,
) -> None:
    conv = _make_session_with_messages()
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_session"].fn,
            session_id="test-session-1",
            repo_path="/nonexistent/path",
        )

    payload = json.loads(raw)
    root = payload.get("root", payload)
    # The message text contains "#1690" and "Sinity/polylogue"
    issue_refs = root.get("issue_refs", [])
    pr_refs = root.get("pr_refs", [])
    all_refs = issue_refs + pr_refs
    numbers = {r["number"] for r in all_refs}
    assert 1690 in numbers


@pytest.mark.asyncio
async def test_correlate_session_no_git_graceful(
    mcp_server: MCPServerUnderTest,
) -> None:
    """When git is not available, the tool still returns a valid result."""
    conv = _make_session_with_messages()
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_session"].fn,
            session_id="test-session-1",
            repo_path="/nonexistent/path/should/not/exist",
        )

    payload = json.loads(raw)
    root = payload.get("root", payload)
    assert root["commits"] == []


@pytest.mark.asyncio
async def test_correlate_session_respects_confidence_threshold(
    mcp_server: MCPServerUnderTest,
) -> None:
    conv = _make_session_with_messages()
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_session"].fn,
            session_id="test-session-1",
            repo_path="/nonexistent/path",
            confidence_threshold=0.9,
        )

    payload = json.loads(raw)
    root = payload.get("root", payload)
    # With a high threshold and no git, commits should be empty
    assert root["commits"] == []
