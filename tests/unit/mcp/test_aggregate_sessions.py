"""Thin-wrapper tests for the aggregate_sessions MCP tool (#1691 / polylogue-9e5.24).

The GROUP BY math itself moved to
polylogue.insights.archive_rollups.aggregate_session_profiles_by_dimension
and is pinned directly (no MCP/mock scaffolding) in
tests/unit/insights/test_archive_rollups.py. This module only proves the
MCP tool delegates to the facade method and formats its result/error
envelope correctly; tests/unit/mcp/test_analysis_primitives_facade_parity.py
is the concrete facade-vs-MCP equivalence proof.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


@pytest.mark.asyncio
async def test_aggregate_sessions_delegates_to_facade_method(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.aggregate_sessions = AsyncMock(
            return_value={"group_by": "workflow_shape", "total_sessions": 3, "buckets": {"chat": 2, "agentic_loop": 1}}
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="workflow_shape",
            origin="claude-code-session",
        )

    payload = json.loads(raw)
    assert payload == {"group_by": "workflow_shape", "total_sessions": 3, "buckets": {"chat": 2, "agentic_loop": 1}}
    mock_poly.aggregate_sessions.assert_awaited_once_with(
        group_by="workflow_shape",
        since=None,
        until=None,
        provider="claude-code",
    )


@pytest.mark.asyncio
async def test_aggregate_sessions_maps_invalid_group_by_value_error_to_error_envelope(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.aggregate_sessions = AsyncMock(
            side_effect=ValueError(
                "Unknown group_by: 'invalid_key'. Supported: workflow_shape, terminal_state, origin."
            )
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="invalid_key",
        )

    payload = json.loads(raw)
    assert "message" in payload
    assert "invalid_key" in payload["message"]


@pytest.mark.asyncio
async def test_aggregate_sessions_default_group_by_is_workflow_shape(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.aggregate_sessions = AsyncMock(
            return_value={"group_by": "workflow_shape", "total_sessions": 0, "buckets": {}}
        )
        mock_get_polylogue.return_value = mock_poly

        await invoke_surface_async(mcp_server._tool_manager._tools["aggregate_sessions"].fn)

    mock_poly.aggregate_sessions.assert_awaited_once_with(
        group_by="workflow_shape",
        since=None,
        until=None,
        provider=None,
    )
