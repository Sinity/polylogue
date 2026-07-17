"""MCP server implementation for Polylogue using the official MCP SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.declarations.adapter import DeclaredToolRegistrar
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
    _response_context,
    _safe_call,
    _set_runtime_services,
)
from polylogue.mcp.server_tools import register_tools
from polylogue.services import RuntimeServices

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _instructions_for_role(role: MCPRole) -> str:
    """Return compatibility guidance now and the standing manual after cutover."""
    from polylogue.agent_integration.manifest import target_surface_is_registered

    base = (
        "Polylogue is the local evidence system for prior AI work. "
        f"This server is running with the {role!r} MCP role; that role is a hard capability boundary."
    )
    if not target_surface_is_registered(role):
        return (
            f"{base} The six-tool manual package is staged but is not active because target tool-name "
            "registration and generated-schema verification have not both completed. Use live discovery "
            "and polylogue://capabilities/query."
        )
    from polylogue.agent_integration.assets import read_agent_asset

    return f"{base}\n\n{read_agent_asset('standing-manual.md')}"


def build_server(*, role: MCPRole = "read") -> FastMCP:
    """Construct the FastMCP server with all tools, resources, and prompts."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "polylogue",
        instructions=_instructions_for_role(role),
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
        response_context=_response_context,
        role=role,
    )
    declared_mcp = DeclaredToolRegistrar(mcp, role=role)
    register_tools(declared_mcp, hooks)
    declared_mcp.finalize()
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


__all__ = ["_instructions_for_role", "build_server", "serve_stdio"]
