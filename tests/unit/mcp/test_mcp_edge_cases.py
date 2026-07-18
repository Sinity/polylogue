"""MCP server edge case tests.

Tests Unicode handling, boundary parameters, error recovery,
and concurrent access patterns.
"""

from __future__ import annotations

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


# Unicode-safety and limit-boundary smoke coverage for the six cutover tools
# lives in test_tool_discovery.py's full parametrized sweep (every tool,
# minimal valid kwargs, asserts a valid JSON envelope with no unhandled
# InternalError). The per-retired-tool scenarios previously here (unicode tag
# via add_tag, empty query via search, limit=0/-1 via list_sessions) tested
# individual tool registrations that no longer exist.
