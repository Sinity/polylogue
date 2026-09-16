"""Structural identities encoded in normalized tool names.

This module contains protocol-level parsing only.  It deliberately does not
classify what a tool *does* from prose or from an open provider namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

MCP_TOOL_PREFIX = "mcp__"

TOOL_PATH_INPUT_KEYS: tuple[str, ...] = ("file_path", "path", "notebook_path", "file", "filename")
"""Which ``tool_input`` keys can hold the operated-on path, most specific first.

One owner for a question that was answered five different ways with four
different key sets, the narrowest of which was the generated ``blocks.tool_path``
column and the FTS ``search_text`` that structured queries actually read --
so a NotebookEdit's ``notebook_path`` was visible to the renderer and invisible
to search (polylogue-7k3n0).
"""

TOOL_COMMAND_INPUT_KEYS: tuple[str, ...] = ("command", "cmd")
"""Which ``tool_input`` keys can hold the executed command, most specific first."""


def _first_non_empty_string(payload: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def tool_input_path(payload: Mapping[str, object]) -> str | None:
    """Return the operated-on path declared by a tool input, if any."""
    return _first_non_empty_string(payload, TOOL_PATH_INPUT_KEYS)


def tool_input_command(payload: Mapping[str, object]) -> str | None:
    """Return the command declared by a tool input, if any."""
    return _first_non_empty_string(payload, TOOL_COMMAND_INPUT_KEYS)


def sql_coalesced_json_extract(column: str, keys: tuple[str, ...]) -> str:
    """Render the SQL that reads ``keys`` out of a JSON ``column``, in order.

    Generated columns and the FTS projection are built from this so the stored
    authority cannot carry a narrower key set than the Python readers.
    """
    extracts = ", ".join(f"json_extract({column}, '$.{key}')" for key in keys)
    return f"COALESCE({extracts})" if len(keys) > 1 else extracts


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


__all__ = [
    "extract_mcp_server",
    "MCP_TOOL_PREFIX",
    "MCPToolIdentity",
    "parse_mcp_tool_name",
    "sql_coalesced_json_extract",
    "TOOL_COMMAND_INPUT_KEYS",
    "TOOL_PATH_INPUT_KEYS",
    "tool_input_command",
    "tool_input_path",
]
