"""MCP security regression tests.

The MCP tool body wrappers (``_safe_call``/``_async_safe_call``) return a
typed error JSON instead of raising on failure, so one bad tool call cannot
take down the entire MCP server (#1621). The security invariant being
pinned here is that the *returned* payload never includes raw exception
messages, internal file paths, or tracebacks — only the exception class
name.
"""

from __future__ import annotations

from polylogue.mcp.server_support import _safe_call


class TestMcpSafeCall:
    """Tests for _safe_call — traceback exposure prevention."""

    def test_success_returns_result(self: object) -> None:
        result = _safe_call("test_tool", lambda: '{"ok": true}')
        assert result == '{"ok": true}'

    def test_error_returns_structured_payload(self: object) -> None:
        def failing() -> str:
            raise ValueError("test error message")

        result = _safe_call("test_tool", failing)
        assert "test_tool" in result
        assert "ValueError" in result
        # The raw exception message must not appear in the payload.
        assert "test error message" not in result

    def test_no_raw_traceback_in_error(self: object) -> None:
        def failing() -> str:
            raise RuntimeError("internal error")

        result = _safe_call("test_tool", failing)
        assert "Traceback" not in result

    def test_no_internal_paths_in_error(self: object) -> None:
        def failing() -> str:
            raise ImportError("No module named 'secret_module'")

        result = _safe_call("test_tool", failing)
        assert "/realm/" not in result
        assert "polylogue/" not in result
        assert "secret_module" not in result
