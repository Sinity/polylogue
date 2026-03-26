"""Small public root for archive-product MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.server_product_tools_aggregate import register_aggregate_product_tools
from polylogue.mcp.server_product_tools_governance import register_governance_product_tools
from polylogue.mcp.server_product_tools_session import register_session_product_tools

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    register_session_product_tools(mcp, hooks)
    register_aggregate_product_tools(mcp, hooks)
    register_governance_product_tools(mcp, hooks)


__all__ = ["register_product_tools"]
