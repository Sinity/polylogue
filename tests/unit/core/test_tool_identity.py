from __future__ import annotations

from polylogue.core.tool_identity import extract_mcp_server, parse_mcp_tool_name


def test_mcp_identity_preserves_server_and_server_local_tool_name() -> None:
    identity = parse_mcp_tool_name("mcp__github__search__issues")

    assert identity is not None
    assert identity.server == "github"
    assert identity.tool == "search__issues"
    assert extract_mcp_server(identity.raw_name) == "github"


def test_mcp_identity_rejects_names_without_both_structural_segments() -> None:
    assert parse_mcp_tool_name("mcp__github") is None
    assert parse_mcp_tool_name("mcp____tool") is None
    assert parse_mcp_tool_name("mcp__github__") is None
    assert parse_mcp_tool_name("search") is None
