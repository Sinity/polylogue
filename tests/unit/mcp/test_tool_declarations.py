"""Independent contracts for the protocol-native MCP read algebra."""

from __future__ import annotations

import inspect
from pathlib import Path

from polylogue.agent_integration.spec import DEFAULT_READ_TOOLS
from polylogue.mcp.declarations.registry import (
    MCP_TOOL_DECLARATIONS,
    TARGET_DEFAULT_READ_ALGEBRA,
    declared_tool_names,
)
from polylogue.mcp.server import build_server
from tests.infra.mcp import MCP_TOOL_NAME_BASELINE, MCPServerUnderTest


def test_frozen_public_inventory_matches_exactly_six_read_transactions() -> None:
    """The test oracle is independent of the declaration-derived live set."""
    assert frozenset({"query", "read", "get", "explain", "context", "status"}) == MCP_TOOL_NAME_BASELINE
    assert frozenset(DEFAULT_READ_TOOLS) == MCP_TOOL_NAME_BASELINE
    assert declared_tool_names("read") == MCP_TOOL_NAME_BASELINE
    assert len(MCP_TOOL_NAME_BASELINE) == 6


def test_all_roles_expose_the_same_read_floor_until_privileged_cutover_lands() -> None:
    """No compatibility tool leaks into a broader role during read migration."""
    for role in ("read", "write", "review", "admin"):
        assert declared_tool_names(role) == MCP_TOOL_NAME_BASELINE


def test_target_algebra_has_no_separate_graph_transaction() -> None:
    assert tuple(item.name for item in TARGET_DEFAULT_READ_ALGEBRA) == DEFAULT_READ_TOOLS
    assert "graph" not in {item.name for item in TARGET_DEFAULT_READ_ALGEBRA}


def test_live_registration_matches_frozen_public_inventory() -> None:
    server = build_server(role="read")
    assert isinstance(server, MCPServerUnderTest)
    assert set(server._tool_manager._tools) == MCP_TOOL_NAME_BASELINE


def test_live_handlers_match_their_declaration_modules_and_public_names() -> None:
    server = build_server(role="read")
    declarations = {declaration.name: declaration for declaration in MCP_TOOL_DECLARATIONS}
    assert set(declarations) == MCP_TOOL_NAME_BASELINE
    for name, declaration in declarations.items():
        handler = server._tool_manager._tools[name].fn
        assert handler.__name__ == name
        assert handler.__module__ == declaration.registration.module
        assert declaration.registration.registrar == "register_cutover_read_tools"


def test_discovery_signatures_expose_real_resume_and_reference_inputs() -> None:
    server = build_server(role="read")
    signatures = {name: inspect.signature(tool.fn) for name, tool in server._tool_manager._tools.items()}
    assert {"expression", "continuation"} <= set(signatures["query"].parameters)
    assert {"ref", "view", "continuation"} <= set(signatures["read"].parameters)
    assert {"ref", "projection"} <= set(signatures["get"].parameters)
    assert {"subject", "expression", "ref"} <= set(signatures["explain"].parameters)
    assert {"intent", "query", "budget_tokens", "result_ref"} <= set(signatures["context"].parameters)
    assert {"scope", "include", "ref"} <= set(signatures["status"].parameters)


def test_generated_equivalence_map_tracks_the_cutover_declarations() -> None:
    import json

    payload = json.loads(Path("docs/generated/mcp-equivalence.json").read_text())
    surface = payload["compatibility_surface"]
    assert surface["tool_count"] == 6
    assert set(surface["tool_names"]) == MCP_TOOL_NAME_BASELINE
