"""MCP server command."""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv


@click.command("mcp")
@click.option("--transport", type=click.Choice(["stdio"]), default="stdio", help="Transport protocol")
@click.pass_obj
def mcp_command(env: AppEnv, transport: str) -> None:
    """Start the MCP server for AI assistant integration."""
    if transport != "stdio":
        env.ui.console.print(f"Unsupported transport: {transport}")
        raise SystemExit(1)

    try:
        from polylogue.mcp.server import serve_stdio
    except ImportError as exc:
        env.ui.console.print(f"MCP dependencies not installed: {exc}")
        env.ui.console.print(
            "Install the base polylogue package in an environment that includes its runtime dependencies."
        )
        raise SystemExit(1) from None

    serve_stdio(env.services)


__all__ = ["mcp_command"]
