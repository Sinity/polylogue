"""Tests for distilled-bundle MCP tools: get_postmortem_bundle (#2436)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.insights.postmortem import PostmortemScope, compile_postmortem_bundle
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


@pytest.mark.asyncio
async def test_get_postmortem_bundle_delegates_and_serializes(mcp_server: MCPServerUnderTest) -> None:
    bundle = compile_postmortem_bundle([], {}, scope=PostmortemScope(matched_session_count=0))
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.postmortem_bundle = AsyncMock(return_value=bundle)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["get_postmortem_bundle"].fn,
            since="2026-01-01",
        )

    payload = json.loads(raw)
    assert "scope" in payload
    assert "schema_version" in payload
    assert mock_poly.postmortem_bundle.await_count == 1
    # The candidate scope must not inherit the MCP default page limit (10),
    # which would silently cap the postmortem below its own analysis cap.
    resolved_spec = mock_poly.postmortem_bundle.await_args.args[0]
    assert resolved_spec.limit is None


@pytest.mark.asyncio
async def test_get_pathologies_delegates_and_serializes(mcp_server: MCPServerUnderTest) -> None:
    from polylogue.insights.pathology import PathologyFinding, PathologyReport

    report = PathologyReport(
        findings=(PathologyFinding(kind="wasted_loop", session_id="s1", severity="medium", detail="3 failed turns"),),
        counts_by_kind={"wasted_loop": 1},
        session_count=1,
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.pathology_report = AsyncMock(return_value=report)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["get_pathologies"].fn,
            since="2026-01-01",
        )

    payload = json.loads(raw)
    assert payload["counts_by_kind"] == {"wasted_loop": 1}
    assert payload["findings"][0]["kind"] == "wasted_loop"
    # candidate scope must not inherit the MCP default page limit
    assert mock_poly.pathology_report.await_args.args[0].limit is None
