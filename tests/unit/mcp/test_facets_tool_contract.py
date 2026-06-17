"""Runtime contracts for the MCP ``facets`` tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.surfaces.payloads import FacetsResponse
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


class TestMCPFacetsTool:
    @pytest.mark.asyncio
    async def test_forwards_query_filters_to_shared_contract(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.facets = AsyncMock(return_value=FacetsResponse(scoped_to_query=True))
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["facets"].fn,
                query="fts readiness",
                origin="claude-code-session",
            )

        parsed = json.loads(raw)
        assert parsed["scoped_to_query"] is True
        mock_poly.facets.assert_awaited_once()
        spec = mock_poly.facets.await_args.args[0]
        assert isinstance(spec, SessionQuerySpec)
        # The shared query-expression parser/lowerer splits a bare-word free-text
        # query into one FTS term per word, so "fts readiness" forwards as two.
        assert spec.query_terms == ("fts", "readiness")
        assert spec.origins == ("claude-code-session",)

    @pytest.mark.asyncio
    async def test_without_filters_requests_global_view(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.facets = AsyncMock(return_value=FacetsResponse(scoped_to_query=False))
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(mcp_server._tool_manager._tools["facets"].fn)

        parsed = json.loads(raw)
        assert parsed["scoped_to_query"] is False
        mock_poly.facets.assert_awaited_once_with(None)
