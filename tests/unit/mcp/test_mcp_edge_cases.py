"""MCP server edge case tests.

Tests Unicode handling, boundary parameters, error recovery,
and concurrent access patterns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from polylogue.mcp.server_support import _clamp_limit, _safe_call
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_mock_filter, make_polylogue_mock

# =============================================================================
# _clamp_limit boundary tests
# =============================================================================


class TestClampLimit:
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (10, 10),
            (1, 1),
            (0, 1),  # clamped to 1
            (-1, 1),  # clamped to 1
            (-100, 1),
            (1000, 1000),
        ],
    )
    def test_integer_inputs(self, input_val: int, expected: int) -> None:
        assert _clamp_limit(input_val) == expected

    def test_string_number(self) -> None:
        assert _clamp_limit("5") == 5

    def test_invalid_string(self) -> None:
        assert _clamp_limit("abc") == 10  # default

    def test_none(self) -> None:
        assert _clamp_limit(None) == 10  # default

    def test_float(self) -> None:
        assert _clamp_limit(3.7) == 3  # int(3.7) = 3


# =============================================================================
# _safe_call error wrapping
# =============================================================================


class TestSafeCall:
    def test_success_passes_through(self) -> None:
        result = _safe_call("test", lambda: '{"ok": true}')
        assert '"ok"' in result

    def test_exception_returns_structured_error(self) -> None:
        """A tool body that raises returns a typed error JSON instead of crashing.

        Returning rather than raising isolates per-tool failures from the
        stdio loop — the historical raise propagated through FastMCP and
        killed the server, taking every other tool offline (#1621).
        """
        import json

        def failing() -> str:
            raise RuntimeError("DB connection lost")

        result = _safe_call("test_tool", failing)
        body = json.loads(result)
        assert body["is_error"] is True
        assert body["code"] == "internal_error"
        assert body["tool"] == "test_tool"
        assert body["detail"] == "RuntimeError"
        assert "internal error" in body["message"]
        assert "RuntimeError" in body["message"]

    def test_traceback_not_in_output(self) -> None:
        def failing() -> str:
            raise ValueError("secret internal path /home/user/.db")

        result = _safe_call("test_tool", failing)
        assert "Traceback" not in result

    def test_raw_exception_text_not_leaked(self) -> None:
        """Exception message content is not exposed to MCP clients."""

        def failing() -> str:
            raise RuntimeError("secret connection string postgresql://admin:hunter2@db.internal")

        result = _safe_call("test_tool", failing)
        assert "secret" not in result
        assert "hunter2" not in result
        assert "admin" not in result
        assert "internal error" in result
        assert "RuntimeError" in result


# =============================================================================
# Unicode handling via MCP tool internals
# =============================================================================


async def _invoke_tool(mcp_server: MCPServerUnderTest, tool_name: str, **kwargs: object) -> str:
    """Invoke a registered MCP tool by name, matching the existing test pattern."""
    tool = mcp_server._tool_manager._tools[tool_name]
    return await invoke_surface_async(tool.fn, **kwargs)


class TestUnicodeHandling:
    @pytest.mark.asyncio
    async def test_unicode_tag(self, mcp_server: MCPServerUnderTest) -> None:
        """Unicode characters in tag names don't crash the server."""

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.add_tag = AsyncMock(return_value=True)
            mock_get_polylogue.return_value = mock_poly

            # This should not crash even with emoji/CJK/RTL
            for tag in ["bug-fix", "重要", "مهم", "critical"]:
                result = await _invoke_tool(
                    mcp_server,
                    "add_tag",
                    session_id="test-conv",
                    tag=tag,
                )
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_query(self, mcp_server: MCPServerUnderTest) -> None:
        """Empty query string doesn't crash search."""
        from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics

        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls,
        ):
            mock_poly = AsyncMock()
            mock_poly.search_session_hits = AsyncMock(return_value=[])
            mock_poly.diagnose_query_miss = AsyncMock(
                return_value=QueryMissDiagnostics(message="No sessions matched.", filters=(), reasons=())
            )
            mock_get_polylogue.return_value = mock_poly
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await _invoke_tool(
                mcp_server,
                "search",
                query="",
                limit=10,
            )
            assert isinstance(result, str)


# =============================================================================
# Boundary parameters
# =============================================================================


class TestBoundaryParameters:
    @pytest.mark.asyncio
    async def test_limit_zero(self, mcp_server: MCPServerUnderTest) -> None:
        """limit=0 is clamped to 1 (returns minimal results)."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls,
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.query_sessions = AsyncMock(return_value=[])
            mock_poly.diagnose_query_miss = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await _invoke_tool(
                mcp_server,
                "list_sessions",
                limit=0,
            )
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_limit_negative(self, mcp_server: MCPServerUnderTest) -> None:
        """Negative limit is clamped to 1."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls,
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.query_sessions = AsyncMock(return_value=[])
            mock_poly.diagnose_query_miss = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await _invoke_tool(
                mcp_server,
                "list_sessions",
                limit=-1,
            )
            assert isinstance(result, str)
