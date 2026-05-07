"""Config command — show resolved polylogue configuration."""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv


@click.command("config")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["toml", "json"]),
    default="toml",
    help="Output format.",
)
@click.pass_obj
def config_command(env: AppEnv, output_format: str) -> None:
    """Show resolved Polylogue configuration with precedence sources."""
    from polylogue.config import format_config_toml, load_polylogue_config

    cfg = load_polylogue_config()

    if output_format == "json":
        import json

        env.ui.console.print(json.dumps(cfg, indent=2, default=str))
    else:
        env.ui.console.print(format_config_toml(cfg))


__all__ = ["config_command"]
