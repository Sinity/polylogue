"""MCP server implementation for Polylogue using the official MCP SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.server_prompts import register_prompts
from polylogue.mcp.server_resources import register_resources
from polylogue.mcp.server_support import (
    MCPRole,
    ServerCallbacks,
    _async_safe_call,
    _clamp_limit,
    _error_json,
    _extract_fenced_code,
    _get_config,
    _get_polylogue,
    _json_payload,
    _safe_call,
    _set_runtime_services,
)
from polylogue.mcp.server_tools import register_tools
from polylogue.services import RuntimeServices

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def build_server(*, role: MCPRole = "read") -> FastMCP:
    """Construct the FastMCP server with all tools, resources, and prompts."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "polylogue",
        instructions=(
            "Polylogue is an AI session archive. Use the tools to search, "
            "list, and retrieve sessions from ChatGPT, Claude, Codex, and "
            "other providers. Sessions include full message history. "
            f"This server is running with the {role!r} MCP role."
        ),
    )
    hooks = ServerCallbacks(
        json_payload=_json_payload,
        clamp_limit=_clamp_limit,
        safe_call=lambda fn_name, fn, *, session_id=None, session_ids=(): _safe_call(
            fn_name,
            fn,
            session_id=session_id,
            session_ids=session_ids,
        ),
        async_safe_call=lambda fn_name, fn, *, session_id=None, session_ids=(): _async_safe_call(
            fn_name,
            fn,
            session_id=session_id,
            session_ids=session_ids,
        ),
        error_json=lambda message, **extra: _error_json(message, **extra),
        get_config=lambda: _get_config(),
        get_polylogue=lambda: _get_polylogue(),
        extract_fenced_code=_extract_fenced_code,
        role=role,
    )
    register_tools(mcp, hooks)
    register_resources(mcp, hooks)
    register_prompts(mcp, hooks)
    return mcp


_server_instance: FastMCP | None = None
_server_instance_role: MCPRole | None = None


def _get_server(services: RuntimeServices | None = None, *, role: MCPRole = "read") -> FastMCP:
    global _server_instance, _server_instance_role
    if services is not None:
        _set_runtime_services(services)
    if _server_instance is None or _server_instance_role != role:
        _server_instance = build_server(role=role)
        _server_instance_role = role
    return _server_instance


def serve_stdio(services: RuntimeServices | None = None, *, role: MCPRole = "read") -> None:
    """Start MCP server with stdio transport."""
    from polylogue.mcp.call_log import start_mcp_call_log

    start_mcp_call_log()
    _get_server(services, role=role).run(transport="stdio")


__all__ = ["build_server", "serve_stdio"]
