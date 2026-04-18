"""MCP server edge case tests.

Tests Unicode handling, boundary parameters, error recovery,
and concurrent access patterns.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.mcp.server_support import _clamp_limit, _safe_call

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

    def test_exception_returns_json_error(self) -> None:
        def failing() -> None:
            raise RuntimeError("DB connection lost")

        result = _safe_call("test_tool", failing)
        data = json.loads(result)
        assert "error" in data
        assert "DB connection lost" in data["error"]
        assert data["tool"] == "test_tool"

    def test_traceback_not_in_output(self) -> None:
        def failing() -> None:
            raise ValueError("secret internal path /home/user/.db")

        result = _safe_call("test_tool", failing)
        # _safe_call wraps in JSON, no raw traceback
        assert "Traceback" not in result


# =============================================================================
# Unicode handling via MCP tool internals
# =============================================================================


def _invoke_tool(mcp_server: Any, tool_name: str, **kwargs: object) -> Any:
    """Invoke a registered MCP tool by name, matching the existing test pattern."""
    tool = mcp_server._tool_manager._tools[tool_name]
    return tool.fn(**kwargs)


class TestUnicodeHandling:
    @pytest.mark.asyncio
    async def test_unicode_tag(self, mcp_server: Any) -> None:
        """Unicode characters in tag names don't crash the server."""
        from tests.infra.mcp import make_repo_mock

        with patch("polylogue.mcp.server._get_repo") as mock_get_repo:
            mock_repo = make_repo_mock()
            mock_repo.add_tag = AsyncMock(return_value=True)
            mock_get_repo.return_value = mock_repo

            # This should not crash even with emoji/CJK/RTL
            for tag in ["bug-fix", "重要", "مهم", "critical"]:
                result = await _invoke_tool(
                    mcp_server,
                    "add_tag",
                    conversation_id="test-conv",
                    tag=tag,
                )
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_query(self, mcp_server: Any) -> None:
        """Empty query string doesn't crash search."""
        from tests.infra.mcp import make_mock_filter, make_repo_mock

        with (
            patch("polylogue.mcp.server._get_repo") as mock_get_repo,
            patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls,
        ):
            mock_repo = make_repo_mock()
            mock_repo.search = AsyncMock(return_value=[])
            mock_get_repo.return_value = mock_repo
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
    async def test_limit_zero(self, mcp_server: Any) -> None:
        """limit=0 is clamped to 1 (returns minimal results)."""
        from tests.infra.mcp import make_mock_filter, make_repo_mock

        with (
            patch("polylogue.mcp.server._get_repo") as mock_get_repo,
            patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls,
        ):
            mock_repo = make_repo_mock()
            mock_repo.list = AsyncMock(return_value=[])
            mock_get_repo.return_value = mock_repo
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await _invoke_tool(
                mcp_server,
                "list_conversations",
                limit=0,
            )
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_limit_negative(self, mcp_server: Any) -> None:
        """Negative limit is clamped to 1."""
        from tests.infra.mcp import make_mock_filter, make_repo_mock

        with (
            patch("polylogue.mcp.server._get_repo") as mock_get_repo,
            patch("polylogue.lib.filters.ConversationFilter") as mock_filter_cls,
        ):
            mock_repo = make_repo_mock()
            mock_repo.list = AsyncMock(return_value=[])
            mock_get_repo.return_value = mock_repo
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await _invoke_tool(
                mcp_server,
                "list_conversations",
                limit=-1,
            )
            assert isinstance(result, str)
