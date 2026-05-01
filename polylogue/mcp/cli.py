"""Standalone MCP server entrypoint."""

from __future__ import annotations

from typing import cast

import click

from polylogue.mcp.server_support import MCPRole


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--role",
    type=click.Choice(["read", "write", "admin"]),
    default="read",
    show_default=True,
    help="MCP capability role. Read omits mutation and maintenance tools.",
)
def main(role: str) -> None:
    """Start the Polylogue MCP stdio bridge."""
    try:
        from polylogue.mcp.server import serve_stdio
    except ImportError as exc:
        click.echo(f"MCP dependencies not installed: {exc}", err=True)
        click.echo(
            "Install the base polylogue package in an environment that includes its runtime dependencies.", err=True
        )
        raise SystemExit(1) from None

    serve_stdio(role=cast(MCPRole, role))


__all__ = ["main"]
