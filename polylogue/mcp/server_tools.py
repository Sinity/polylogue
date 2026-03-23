"""Stable MCP tool registration surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_product_tools import register_product_tools
from polylogue.mcp.server_query_tools import register_query_tools
from polylogue.mcp.server_read_tools import register_read_tools

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    register_query_tools(mcp, hooks)
    register_mutation_tools(mcp, hooks)
    register_read_tools(mcp, hooks)
    register_maintenance_tools(mcp, hooks)
    register_product_tools(mcp, hooks)


__all__ = ["register_tools"]
