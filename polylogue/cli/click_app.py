"""CLI entrypoint (clean surface, adaptive UI)."""

from __future__ import annotations

import os
from pathlib import Path

import click

from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.config import config_command
from polylogue.cli.commands.export import export_command
from polylogue.cli.commands.health import health_command
from polylogue.cli.commands.index import index_command
from polylogue.cli.commands.run import run_command, sources_command
from polylogue.cli.commands.search import search_command
from polylogue.cli.commands.serve import serve_command
from polylogue.cli.commands.state import state_command
from polylogue.cli.formatting import announce_plain_mode, should_use_plain
from polylogue.cli.helpers import print_summary
from polylogue.cli.types import AppEnv
from polylogue.ui import create_ui


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--plain", is_flag=True, help="Force non-interactive plain output")
@click.option("--interactive", is_flag=True, help="Force interactive output")
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config.json")
@click.pass_context
def cli(ctx: click.Context, plain: bool, interactive: bool, config_path: Path | None) -> None:
    """Polylogue CLI."""
    use_plain = should_use_plain(plain=plain, interactive=interactive)
    env = AppEnv(ui=create_ui(use_plain), config_path=config_path)
    ctx.obj = env
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    forced_plain = bool(env_force and env_force.lower() not in {"0", "false", "no"})
    if use_plain and not plain and not interactive and not forced_plain:
        announce_plain_mode()
    if ctx.invoked_subcommand is None:
        print_summary(env)


cli.add_command(run_command)
cli.add_command(sources_command)
cli.add_command(index_command)
cli.add_command(search_command)
cli.add_command(completions_command)
cli.add_command(health_command)
cli.add_command(export_command)
cli.add_command(state_command)
cli.add_command(config_command)
cli.add_command(serve_command)


def main() -> None:
    cli()


__all__ = ["cli", "main"]
