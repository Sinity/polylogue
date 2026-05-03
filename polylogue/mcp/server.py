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
    _get_runtime_services,
    _json_payload,
    _safe_call,
    _set_runtime_services,
)
from polylogue.mcp.server_tools import register_tools
from polylogue.operations import ArchiveOperations
from polylogue.protocols import ConversationQueryRuntimeStore, TagStore
from polylogue.services import RuntimeServices
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _get_repo() -> ConversationRepository:
    """Return the canonical server-level repository injection seam."""
    return _get_runtime_services().get_repository()


def _get_query_store() -> ConversationQueryRuntimeStore:
    """Return the query/runtime store used by read-heavy MCP surfaces."""
    return _get_repo()


def _get_tag_store() -> TagStore:
    """Return the tag/metadata store used by mutation/resource surfaces."""
    return _get_repo()


def _get_backend() -> SQLiteBackend:
    """Return the backend bound to the injected repository when available."""
    backend = getattr(_get_repo(), "backend", None)
    if isinstance(backend, SQLiteBackend):
        return backend
    return _get_runtime_services().get_backend()


def _get_archive_ops() -> ArchiveOperations:
    """Return canonical archive operations for MCP read surfaces."""
    services = _get_runtime_services()
    return ArchiveOperations(
        config=services.config if services is not None else None,
        repository=_get_repo(),
        backend=_get_backend(),
    )


def build_server(*, role: MCPRole = "read") -> FastMCP:
    """Construct the FastMCP server with all tools, resources, and prompts."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "polylogue",
        instructions=(
            "Polylogue is an AI conversation archive. Use the tools to search, "
            "list, and retrieve conversations from ChatGPT, Claude, Codex, and "
            "other providers. Conversations include full message history. "
            f"This server is running with the {role!r} MCP role."
        ),
    )
    hooks = ServerCallbacks(
        json_payload=_json_payload,
        clamp_limit=_clamp_limit,
        safe_call=lambda fn_name, fn: _safe_call(fn_name, fn),
        async_safe_call=lambda fn_name, fn: _async_safe_call(fn_name, fn),
        error_json=lambda message, **extra: _error_json(message, **extra),
        get_query_store=lambda: _get_query_store(),
        get_tag_store=lambda: _get_tag_store(),
        get_backend=lambda: _get_backend(),
        get_config=lambda: _get_config(),
        get_archive_ops=lambda: _get_archive_ops(),
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
    _get_server(services, role=role).run(transport="stdio")


__all__ = ["build_server", "serve_stdio"]
