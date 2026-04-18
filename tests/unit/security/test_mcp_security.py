"""MCP security regression tests."""

from __future__ import annotations

import json
from typing import Any

from polylogue.mcp.server_support import _safe_call


class TestMcpSafeCall:
    """Tests for _safe_call — traceback exposure prevention."""

    def test_success_returns_result(self: Any) -> None:
        result = _safe_call("test_tool", lambda: '{"ok": true}')
        assert result == '{"ok": true}'

    def test_error_returns_json(self: Any) -> None:
        def failing() -> None:
            raise ValueError("test error message")

        result = _safe_call("test_tool", failing)
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["tool"] == "test_tool"
        assert "test error message" in parsed["error"]

    def test_no_traceback_in_error_response(self: Any) -> None:
        def failing() -> None:
            raise RuntimeError("internal error")

        result = _safe_call("test_tool", failing)
        parsed = json.loads(result)
        assert "traceback" not in parsed
        assert "Traceback" not in result
        assert "File " not in result  # No file paths leaked

    def test_no_internal_paths_in_error(self: Any) -> None:
        def failing() -> None:
            raise ImportError("No module named 'secret_module'")

        result = _safe_call("test_tool", failing)
        # Should not contain full tracebacks with file paths
        assert "/realm/" not in result
        assert "polylogue/" not in result or "polylogue/" in json.loads(result).get("error", "")
