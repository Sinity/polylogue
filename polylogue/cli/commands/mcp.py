"""MCP (Model Context Protocol) server command."""

from __future__ import annotations

import click

from polylogue.cli.types import AppEnv


@click.command("mcp")
@click.option("--transport", type=click.Choice(["stdio"]), default="stdio", help="Transport protocol")
@click.pass_obj
def mcp_command(env: AppEnv, transport: str) -> None:
    """Start MCP server for AI assistant integration.

    The MCP server provides conversation search and retrieval capabilities
    via the Model Context Protocol (stdio transport).
    """
    if transport != "stdio":
        env.ui.console.print(f"Unsupported transport: {transport}")
        raise SystemExit(1)

    # Lazy import to avoid loading mcp dependencies when not needed
    try:
        from polylogue.mcp.server import serve_stdio
    except ImportError as exc:
        env.ui.console.print(f"MCP dependencies not installed: {exc}")
        env.ui.console.print("Install with: pip install polylogue[mcp]")
        raise SystemExit(1) from None

    serve_stdio()


__all__ = ["mcp_command"]
