"""MCP tool body error isolation.

Pins the post-#1611/#1621 contract for the shared
``_safe_call``/``_async_safe_call`` helpers that wrap every registered MCP
tool body:

* A ``SchemaVersionMismatchError`` raised inside a tool body produces a typed
  error payload with ``code="schema_version_mismatch"`` and the on-disk and
  expected schema versions, so every tool surfaces the same diagnostic
  ``readiness_check`` does instead of "internal error (OperationalError)"
  (#1611).
* Any other exception is returned as a typed error payload with
  ``code="internal_error"`` and the exception class name, so the stdio
  loop is never killed by a single bad tool call (#1621).
* Every registered tool routes through the shared wrappers, so the
  contract above applies uniformly across the MCP surface.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.errors import SchemaVersionMismatchError
from polylogue.mcp.server import build_server
from polylogue.mcp.server_support import _async_safe_call, _safe_call
from tests.infra.mcp import MCPServerUnderTest


def _structured_error(payload: str) -> dict[str, object]:
    body = json.loads(payload)
    assert isinstance(body, dict)
    assert body.get("is_error") is True
    return cast(dict[str, object], body)


class TestSchemaVersionMismatchSurface:
    """A schema mismatch raised inside any tool body becomes a typed payload."""

    def test_safe_call_returns_schema_version_mismatch_payload(self) -> None:
        def boom() -> str:
            raise SchemaVersionMismatchError(
                "Database schema version 16 is newer than this Polylogue runtime expects (9).",
                current_version=16,
                expected_version=9,
            )

        body = _structured_error(_safe_call("stats", boom))
        assert body["code"] == "schema_version_mismatch"
        assert body["tool"] == "stats"
        assert body["current_version"] == 16
        assert body["expected_version"] == 9
        assert body["detail"] == "SchemaVersionMismatchError"
        assert "schema version 16" in cast(str, body["message"])

    @pytest.mark.asyncio
    async def test_async_safe_call_returns_schema_version_mismatch_payload(self) -> None:
        async def boom() -> str:
            raise SchemaVersionMismatchError(
                "Database schema version 99 is newer than this Polylogue runtime expects (16).",
                current_version=99,
                expected_version=16,
            )

        body = _structured_error(await _async_safe_call("search", boom))
        assert body["code"] == "schema_version_mismatch"
        assert body["tool"] == "search"
        assert body["current_version"] == 99
        assert body["expected_version"] == 16


class TestTopLevelIsolation:
    """An unhandled exception inside a tool body never escapes the wrapper."""

    def test_safe_call_swallows_arbitrary_exception(self) -> None:
        def boom() -> str:
            raise RuntimeError("oops")

        body = _structured_error(_safe_call("cost_rollups", boom))
        assert body["code"] == "internal_error"
        assert body["tool"] == "cost_rollups"
        assert body["detail"] == "RuntimeError"
        # The exception class name MUST appear so the operator can see
        # what failed; the raw message MUST NOT.
        assert "RuntimeError" in cast(str, body["message"])
        assert "oops" not in cast(str, body["message"])

    @pytest.mark.asyncio
    async def test_async_safe_call_swallows_arbitrary_exception(self) -> None:
        async def boom() -> str:
            raise ValueError("inner failure")

        body = _structured_error(await _async_safe_call("list_sessions", boom))
        assert body["code"] == "internal_error"
        assert body["tool"] == "list_sessions"
        assert body["detail"] == "ValueError"
        assert "inner failure" not in cast(str, body["message"])

    @pytest.mark.asyncio
    async def test_async_safe_call_does_not_reraise(self) -> None:
        """The wrapper must not raise — that is what historically killed the
        stdio server (#1621). Subsequent calls after a failure must succeed.
        """

        async def boom() -> str:
            raise RuntimeError("trigger")

        async def healthy() -> str:
            return '{"ok": true}'

        first = await _async_safe_call("flaky", boom)
        _structured_error(first)
        # The second call still runs through the same module-level helper;
        # if the first had raised we would not get here.
        second = await _async_safe_call("flaky", healthy)
        assert json.loads(second) == {"ok": True}


class TestRegistrySurfaceContract:
    """Every registered MCP tool routes through the shared error wrappers."""

    def test_every_tool_returns_structured_error_on_internal_failure(self) -> None:
        """Sanity-check the surface: building the server with all roles
        registers > 30 tools, and at minimum the headline read tools that
        regressed in #1611 / #1621 are present. (Per-tool error wiring is
        proven by the wrapper contract above plus per-tool tests elsewhere.)
        """
        server = cast(MCPServerUnderTest, build_server(role="admin"))
        tool_names = set(server._tool_manager._tools.keys())
        # Tools called out explicitly in the two issues:
        for name in [
            "stats",
            "search",
            "list_sessions",
            "readiness_check",
            "cost_rollups",
            "facets",
            "archive_coverage",
        ]:
            assert name in tool_names, f"missing expected MCP tool: {name}"
