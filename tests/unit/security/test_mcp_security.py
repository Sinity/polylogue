"""MCP security regression tests."""

from __future__ import annotations

import pytest

from polylogue.mcp.server_support import _safe_call


class TestMcpSafeCall:
    """Tests for _safe_call — traceback exposure prevention."""

    def test_success_returns_result(self: object) -> None:
        result = _safe_call("test_tool", lambda: '{"ok": true}')
        assert result == '{"ok": true}'

    def test_error_raises_polylogue_error(self: object) -> None:
        def failing() -> None:
            raise ValueError("test error message")

        from polylogue.errors import PolylogueError

        with pytest.raises(PolylogueError, match="test_tool.*ValueError"):
            _safe_call("test_tool", failing)

    def test_no_raw_traceback_in_error(self: object) -> None:
        def failing() -> None:
            raise RuntimeError("internal error")

        from polylogue.errors import PolylogueError

        with pytest.raises(PolylogueError) as exc:
            _safe_call("test_tool", failing)
        assert "Traceback" not in str(exc.value)

    def test_no_internal_paths_in_error(self: object) -> None:
        def failing() -> None:
            raise ImportError("No module named 'secret_module'")

        from polylogue.errors import PolylogueError

        with pytest.raises(PolylogueError) as exc:
            _safe_call("test_tool", failing)
        assert "/realm/" not in str(exc.value)
        assert "polylogue/" not in str(exc.value)
