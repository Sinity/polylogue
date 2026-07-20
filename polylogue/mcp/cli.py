"""Standalone MCP server entrypoint."""

from __future__ import annotations

import click

from polylogue.mcp.server_support import MCPCapabilities


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Start the Polylogue MCP stdio bridge.

    Read-only by default. Write (``write``/``run``), judge, and maintenance
    dispatchers are explicit config opt-ins -- see ``[mcp]`` in
    ``polylogue.toml`` (``write_enabled``/``judge_enabled``/
    ``maintenance_enabled``) or the ``POLYLOGUE_MCP_WRITE_ENABLED`` /
    ``POLYLOGUE_MCP_JUDGE_ENABLED`` / ``POLYLOGUE_MCP_MAINTENANCE_ENABLED``
    environment variables. There is no CLI flag and no role ladder
    (polylogue-800m): each capability is an independent boolean.
    """
    try:
        from polylogue.mcp.server import serve_stdio
    except ImportError as exc:
        click.echo(f"MCP dependencies not installed: {exc}", err=True)
        click.echo(
            "Install the base polylogue package in an environment that includes its runtime dependencies.", err=True
        )
        raise SystemExit(1) from None

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    capabilities = MCPCapabilities(
        write=cfg.mcp_write_enabled,
        judge=cfg.mcp_judge_enabled,
        maintenance=cfg.mcp_maintenance_enabled,
    )
    serve_stdio(capabilities=capabilities)


__all__ = ["main"]
