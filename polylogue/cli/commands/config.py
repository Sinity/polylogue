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

        # ``console.print`` interprets ``[brackets]`` as Rich markup, which
        # would mangle JSON output and TOML section headers. Print without
        # markup parsing to preserve exact bytes.
        env.ui.console.print(json.dumps(cfg.raw, indent=2, default=str), markup=False, highlight=False)
    else:
        env.ui.console.print(format_config_toml(cfg.raw), markup=False, highlight=False)


__all__ = ["config_command"]
