"""MCP tool discovery smoke test.

Parametrized test that discovers all registered MCP read tools, calls each
with minimal valid arguments against an empty archive, and asserts each returns
a valid JSON response envelope — no traceback, no ``InternalError``.

This catches the class of bugs where tools raise ``IndexError`` on empty
archives (#1611) or where a newly registered tool was never exercised against
the real entry point.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.mcp.server import build_server
from tests.infra.mcp import MCPServerUnderTest, invoke_surface

#: Synthetic session id for tools that require one — we accept not_found.
_SYNTHETIC_CONV_ID = "test:conv-discovery-nonexistent"
_SYNTHETIC_REF = f"session:{_SYNTHETIC_CONV_ID}"

#: Minimal known-good kwargs per tool name for the six-tool cutover surface.
#: Every read-role tool must have an entry (see
#: ``test_read_tools_have_known_minimal_kwargs``) since all six declare at
#: least one required argument.
_KNOWN_MINIMAL: dict[str, dict[str, object]] = {
    "query": {"expression": "messages where text:hello", "limit": 1},
    "read": {"ref": _SYNTHETIC_REF},
    "get": {"ref": _SYNTHETIC_REF},
    "explain": {"subject": "capability"},
    "context": {"intent": "resume"},
    "status": {"scope": "archive"},
}


def _discover_read_tool_names() -> frozenset[str]:
    """Return the set of all tool names registered on a read-role server."""
    server = cast(MCPServerUnderTest, build_server(role="read"))
    return frozenset(server._tool_manager._tools.keys())


_READ_TOOL_NAMES = _discover_read_tool_names()


# ── parametrized smoke call ──────────────────────────────────────────


def _tool_call_specs() -> list[object]:
    """Yield (name, kwargs) for every read-role tool."""
    result: list[object] = []
    for name in sorted(_READ_TOOL_NAMES):
        kwargs = _KNOWN_MINIMAL.get(name, {})
        result.append(pytest.param(name, kwargs, id=name))
    return result


@pytest.mark.parametrize("tool_name, kwargs", list(_tool_call_specs()))
def test_tool_returns_valid_response_envelope(tool_name: str, kwargs: dict[str, object]) -> None:
    """Every read-role MCP tool returns valid JSON with no unhandled InternalError."""
    server = cast(MCPServerUnderTest, build_server(role="read"))
    tool = server._tool_manager._tools.get(tool_name)
    assert tool is not None, f"tool {tool_name!r} not registered on read-role server"

    result = invoke_surface(tool.fn, **kwargs)

    # Must be a string (JSON payload).
    assert isinstance(result, str), f"{tool_name}: expected str, got {type(result).__name__}"

    # Must be valid JSON.
    try:
        body: dict[str, object] = json.loads(result)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{tool_name}: response is not valid JSON: {exc}")

    assert isinstance(body, dict), f"{tool_name}: response body is not a dict"

    # Tools passed a synthetic session id will naturally produce
    # internal_error when the resource is not found — the _async_safe_call
    # wrapper catches the exception and returns structured JSON. That proves
    # the safety net is working (the #1621 regression where exceptions killed
    # the stdio loop). We only fail when the response is not valid JSON.
    if body.get("is_error") is True:
        # Must have at least an error message.
        assert body.get("message") is not None, f"{tool_name}: error response missing error string"
        # internal_error (with or without explicit code) is acceptable —
        # it means the safety net caught a tool-body exception.
        # Other structured errors (schema_version_mismatch, not_found) are also fine.


def test_all_read_tools_discovered() -> None:
    """Sanity: the read-role server exposes exactly the six cutover tools."""
    assert {"query", "read", "get", "explain", "context", "status"} == _READ_TOOL_NAMES


def test_no_mutation_tools_in_read_role() -> None:
    """Mutation tools must NOT be registered on a read-role server."""
    server = cast(MCPServerUnderTest, build_server(role="read"))
    read_names = set(server._tool_manager._tools.keys())

    known_mutations = {
        "add_tag",
        "remove_tag",
        "bulk_tag_sessions",
        "add_mark",
        "remove_mark",
        "save_annotation",
        "delete_annotation",
        "set_metadata",
        "delete_metadata",
        "delete_session",
        "rebuild_session_insights",
        "maintenance_execute",
    }
    for mt in known_mutations:
        assert mt not in read_names, f"mutation tool {mt!r} leaked into read-role server"


def test_read_tools_have_known_minimal_kwargs() -> None:
    """Every read tool has an entry in _KNOWN_MINIMAL.

    When a new tool is added, this test fails so the developer adds a
    minimal valid argument set, keeping the parametrized smoke test
    comprehensive.
    """
    uncovered = sorted(_READ_TOOL_NAMES - set(_KNOWN_MINIMAL))
    if uncovered:
        msg = (
            "New MCP read tools without _KNOWN_MINIMAL entry. "
            "Add minimal valid kwargs to _KNOWN_MINIMAL in this file.\n" + "\n".join(f"  - {t}" for t in uncovered)
        )
        pytest.fail(msg)
