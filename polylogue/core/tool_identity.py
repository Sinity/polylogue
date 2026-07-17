"""Structural identities encoded in normalized tool names.

This module contains protocol-level parsing only.  It deliberately does not
classify what a tool *does* from prose or from an open provider namespace.
"""

from __future__ import annotations

from dataclasses import dataclass

MCP_TOOL_PREFIX = "mcp__"


@dataclass(frozen=True, slots=True)
class MCPToolIdentity:
    """The server/tool coordinates carried by an MCP tool name."""

    raw_name: str
    server: str
    tool: str


def parse_mcp_tool_name(tool_name: str | None) -> MCPToolIdentity | None:
    """Parse ``mcp__<server>__<tool>`` without guessing missing segments.

    The tool segment may itself contain ``__``; only the first separator after
    the prefix divides the server identity from the server-local tool name.
    Malformed or non-MCP names return ``None`` so callers preserve an explicit
    unknown/fallback state instead of inventing a server.
    """

    if not tool_name or not tool_name.startswith(MCP_TOOL_PREFIX):
        return None
    remainder = tool_name[len(MCP_TOOL_PREFIX) :]
    server, separator, tool = remainder.partition("__")
    if not separator or not server or not tool:
        return None
    return MCPToolIdentity(raw_name=tool_name, server=server, tool=tool)


def extract_mcp_server(tool_name: str | None) -> str | None:
    """Return the structurally encoded MCP server, when present."""

    identity = parse_mcp_tool_name(tool_name)
    return identity.server if identity is not None else None


__all__ = ["extract_mcp_server", "MCP_TOOL_PREFIX", "MCPToolIdentity", "parse_mcp_tool_name"]
