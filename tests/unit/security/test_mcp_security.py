"""MCP security regression tests."""

from __future__ import annotations

import json

from polylogue.mcp.server_support import _safe_call


class TestMcpSafeCall:
    """Tests for _safe_call — traceback exposure prevention."""

    def test_success_returns_result(self: object) -> None:
        result = _safe_call("test_tool", lambda: '{"ok": true}')
        assert result == '{"ok": true}'

    def test_error_returns_json(self: object) -> None:
        def failing() -> None:
            raise ValueError("test error message")

        result = _safe_call("test_tool", failing)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error"] == "internal MCP tool error"
        assert parsed["code"] == -32603  # JSON-RPC internal error
        assert parsed["detail"] == "ValueError"
        assert parsed["tool"] == "test_tool"

    def test_no_traceback_in_error_response(self: object) -> None:
        def failing() -> None:
            raise RuntimeError("internal error")

        result = _safe_call("test_tool", failing)
        assert result is not None
        parsed = json.loads(result)
        assert "traceback" not in parsed
        assert "Traceback" not in result
        assert "File " not in result  # No file paths leaked

    def test_no_internal_paths_in_error(self: object) -> None:
        def failing() -> None:
            raise ImportError("No module named 'secret_module'")

        result = _safe_call("test_tool", failing)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error"] == "internal MCP tool error"
        assert "/realm/" not in result
        assert "polylogue/" not in result
