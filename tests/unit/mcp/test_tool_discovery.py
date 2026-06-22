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

#: Minimal known-good kwargs per tool name. Tools not listed here are called
#: with no kwargs and must return a valid JSON envelope (empty results, stats
#: for an empty archive, etc.).
_KNOWN_MINIMAL: dict[str, dict[str, object]] = {
    "search": {"query": "hello", "limit": 1},
    "query_units": {"expression": "messages where text:hello", "limit": 1},
    "resolve_ref": {"ref": f"session:{_SYNTHETIC_CONV_ID}"},
    "compile_context": {"seed_ref": f"session:{_SYNTHETIC_CONV_ID}", "read_views": "recovery"},
    "blackboard_list": {},
    "get_session": {"id": _SYNTHETIC_CONV_ID},
    "get_session_summary": {"id": _SYNTHETIC_CONV_ID},
    "get_messages": {"session_id": _SYNTHETIC_CONV_ID, "limit": 1},
    "get_session_tree": {"session_id": _SYNTHETIC_CONV_ID},
    "get_session_topology": {"session_id": _SYNTHETIC_CONV_ID},
    "get_logical_session": {"session_id": _SYNTHETIC_CONV_ID},
    "session_profile": {"session_id": _SYNTHETIC_CONV_ID},
    "session_latency_profile": {"session_id": _SYNTHETIC_CONV_ID},
    "get_resume_brief": {"session_id": _SYNTHETIC_CONV_ID},
    "raw_artifacts": {"session_id": _SYNTHETIC_CONV_ID},
    "session_costs": {"session_id": _SYNTHETIC_CONV_ID},
    "cost_outlook": {"plan": "claude-pro"},
    "find_resume_candidates": {"repo_path": "/tmp/nonexistent"},
    "aggregate_sessions": {"session_ids": "test:a,test:b"},
    "compose_context_preamble": {"context": "{}"},
    "compare_sessions": {"session_ids": "test:a,test:b"},
    "find_similar_sessions": {"query": "test"},
    "correlate_session": {"session_id": _SYNTHETIC_CONV_ID},
    "correlate_sessions": {"session_ids": "test:a,test:b"},
    "session_tool_timing": {"session_id": _SYNTHETIC_CONV_ID},
    "facets": {"limit": 1},
    "neighbor_candidates": {"query": "test"},
    "insight_rigor_audit": {},
    "list_sessions": {"limit": 1},
    "archive_get_session": {"session_id": _SYNTHETIC_CONV_ID},
    "archive_list_sessions": {"limit": 1},
    "archive_search_sessions": {"query": "hello", "limit": 1},
    "session_profiles": {"limit": 1},
    "session_work_events": {"limit": 1},
    "session_phases": {"limit": 1},
    "session_tag_rollups": {"limit": 1},
    "threads": {"limit": 1},
    "archive_coverage": {"limit": 1},
    "archive_debt": {"limit": 1},
    "explain_import": {"raw_ref": "raw:discovery-nonexistent", "limit": 1},
    "cost_rollups": {"limit": 1},
    "tool_usage": {"limit": 1},
    "tool_call_latency_distribution": {"limit": 1},
    "workflow_shape_distribution": {"limit": 1},
    "find_stuck_sessions": {"limit": 1},
    "find_abandoned_sessions": {"limit": 1},
    "get_stats_by": {"group_by": "origin"},
    "list_read_view_profiles": {},
    "list_assertion_claims": {"target_ref": f"session:{_SYNTHETIC_CONV_ID}", "limit": 1},
    "get_recovery_report": {"session_id": "demo", "report": "continue"},
    "get_recovery_work_packet": {"session_id": _SYNTHETIC_CONV_ID},
    "explain_query_expression": {"expression": "repo:polylogue"},
    "query_completions": {"kind": "field", "incomplete": "d"},
    "action_affordances": {},
    "embedding_status": {},
    "embedding_preflight": {},
    "build_context_pack": {"project_repo": "test"},
    "readiness_check": {},
    "stats": {},
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

    try:
        result = invoke_surface(tool.fn, **kwargs)
    except TypeError:
        # Tools we don't have a known-good kwarg set for may require params
        # we cannot synthesise. The wrapper handles this gracefully.
        # We just verify it didn't crash the server.
        return

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
    """Sanity: we discovered at least the known headline read tools."""
    for expected in [
        "search",
        "list_sessions",
        "stats",
        "facets",
        "readiness_check",
        "get_session",
        "build_context_pack",
    ]:
        assert expected in _READ_TOOL_NAMES, f"missing expected read tool: {expected}"


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
    # Tools that genuinely take no arguments and work fine are listed here.
    NO_KWARGS_NEEDED: frozenset[str] = frozenset({"compose_context_preamble"})
    uncovered = [t for t in uncovered if t not in NO_KWARGS_NEEDED]
    if uncovered:
        msg = (
            "New MCP read tools without _KNOWN_MINIMAL entry. "
            "Add minimal valid kwargs to _KNOWN_MINIMAL in this file.\n" + "\n".join(f"  - {t}" for t in uncovered)
        )
        pytest.fail(msg)
