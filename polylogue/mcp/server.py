"""MCP server implementation for Polylogue using the official MCP SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.server_prompts import register_prompts
from polylogue.mcp.server_resources import register_resources
from polylogue.mcp.server_support import (
    ServerCallbacks,
    _async_safe_call,
    _clamp_limit,
    _error_json,
    _extract_fenced_code,
    _get_config,
    _get_repo,
    _get_runtime_services,
    _json_payload,
    _safe_call,
    _set_runtime_services,
)
from polylogue.mcp.server_tools import register_tools
from polylogue.operations import ArchiveOperations
from polylogue.services import RuntimeServices

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _get_archive_ops() -> ArchiveOperations:
    """Return canonical archive operations for MCP read surfaces."""
    repo = _get_repo()
    services = _get_runtime_services()
    return ArchiveOperations(
        config=services.config if services is not None else None,
        repository=repo,
        backend=getattr(repo, "backend", None),
    )


def build_server() -> FastMCP:
    """Construct the FastMCP server with all tools, resources, and prompts."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "polylogue",
        instructions=(
            "Polylogue is an AI conversation archive. Use the tools to search, "
            "list, and retrieve conversations from ChatGPT, Claude, Codex, and "
            "other providers. Conversations include full message history. "
            "You can also manage tags, metadata, trigger indexing, and export conversations."
        ),
    )
    hooks = ServerCallbacks(
        json_payload=_json_payload,
        clamp_limit=_clamp_limit,
        safe_call=lambda fn_name, fn: _safe_call(fn_name, fn),
        async_safe_call=lambda fn_name, fn: _async_safe_call(fn_name, fn),
        error_json=lambda message, **extra: _error_json(message, **extra),
        get_repo=lambda: _get_repo(),
        get_config=lambda: _get_config(),
        get_archive_ops=lambda: _get_archive_ops(),
        extract_fenced_code=lambda text, language="": _extract_fenced_code(text, language),
    )
    register_tools(mcp, hooks)
    register_resources(mcp, hooks)
    register_prompts(mcp, hooks)
    return mcp


_server_instance: FastMCP | None = None


def _get_server(services: RuntimeServices | None = None) -> FastMCP:
    global _server_instance
    if services is not None:
        _set_runtime_services(services)
    if _server_instance is None:
        _server_instance = build_server()
    return _server_instance


def serve_stdio(services: RuntimeServices | None = None) -> None:
    """Start MCP server with stdio transport."""
    _get_server(services).run(transport="stdio")
