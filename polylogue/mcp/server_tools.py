"""Stable MCP tool registration surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks


def register_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    from polylogue.mcp.server_cutover import register_cutover_privileged_tools, register_cutover_read_tools

    register_cutover_read_tools(mcp, hooks)
    register_cutover_privileged_tools(mcp, hooks)


__all__ = ["register_tools"]
