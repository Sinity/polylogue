"""MCP (Model Context Protocol) server for Polylogue."""

from __future__ import annotations

__all__ = ["serve_stdio"]


def serve_stdio() -> None:
    """Start MCP server with stdio transport."""
    from polylogue.mcp.server import serve_stdio as _serve

    _serve()
