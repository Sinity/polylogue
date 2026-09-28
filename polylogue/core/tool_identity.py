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

    The stored JSON keeps a lone surrogate as an exact ``\\uXXXX`` escape,
    but ``json_extract`` would decode it into text that is not valid UTF-8, and
    reading that column then fails. Such a string is projected in its escaped
    JSON spelling instead; the JSON column stays the exact authority.
    """
    extracts = ", ".join(_sql_text_projection(column, key) for key in keys)
    return f"COALESCE({extracts})" if len(keys) > 1 else extracts


def _sql_text_projection(column: str, key: str) -> str:
    path = f"'$.{key}'"
    quoted = f"({column} -> {path})"
    return (
        f"CASE WHEN json_type({column}, {path}) = 'text' AND {quoted} GLOB '*\\u[dD][89a-fA-F]*' "
        f"THEN json_extract({_sql_escape_surrogate_escapes(quoted)}, '$') "
        f"ELSE json_extract({column}, {path}) END"
    )


#: The spelling of every lone-surrogate escape prefix, ``\uD800``-``\uDFFF``
#: in either hex case.
_SURROGATE_ESCAPE_PREFIXES = tuple(f"\\u{d}{h}" for d in "dD" for h in "89abcdefABCDEF")


def _sql_escape_surrogate_escapes(json_string: str) -> str:
    """Rewrite a JSON string literal so decoding it keeps surrogates spelled out.

    SQLite's own JSON decoder then handles every other escape (short forms,
    ``\\u00XX`` controls, any other ``\\uXXXX``) and leaves every literal
    character, U+FFFF included, untouched. Escaped backslashes are first
    respelled as ``\\u005c`` so each remaining backslash starts a real escape;
    each surrogate escape then gains an escaped backslash, which decodes to
    the literal text ``\\uD8xx`` instead of text that is not valid UTF-8.
    """
    rewritten = f"replace({json_string}, '\\\\', '\\u005c')"
    for prefix in _SURROGATE_ESCAPE_PREFIXES:
        rewritten = f"replace({rewritten}, '{prefix}', '\\\\{prefix[1:]}')"
    return rewritten


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
