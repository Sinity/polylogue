"""v666 CLI entrypoint (clean surface, adaptive UI)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

from ..ui import create_ui


@dataclass
class AppEnv:
    ui: object
    profile: Optional[str] = None
    config_path: Optional[Path] = None


def _should_use_plain(*, plain: bool, interactive: bool) -> bool:
    if plain:
        return True
    if interactive:
        return False
    return not (sys.stdout.isatty() and sys.stderr.isatty())


def _announce_plain_mode() -> None:
    sys.stderr.write("Plain output active (non-TTY). Use --interactive from a TTY to re-enable prompts.\n")


def _not_implemented(command: str) -> None:
    raise SystemExit(f"{command} is not implemented yet in the v666 rewrite.")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--plain", is_flag=True, help="Force non-interactive plain output")
@click.option("--interactive", is_flag=True, help="Force interactive output")
@click.option("--profile", help="Select a named profile")
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config.json")
@click.pass_context
def cli(ctx: click.Context, plain: bool, interactive: bool, profile: Optional[str], config_path: Optional[Path]) -> None:
    """Polylogue v666 CLI."""
    use_plain = _should_use_plain(plain=plain, interactive=interactive)
    env = AppEnv(ui=create_ui(use_plain), profile=profile, config_path=config_path)
    ctx.obj = env
    if use_plain and not plain and not interactive:
        _announce_plain_mode()


@cli.command()
@click.pass_obj
def plan(env: AppEnv) -> None:
    _not_implemented("plan")


@cli.command()
@click.option("--no-plan", is_flag=True, help="Skip plan preview")
@click.option("--strict-plan", is_flag=True, help="Fail if plan drift is detected")
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all")
@click.pass_obj
def run(env: AppEnv, no_plan: bool, strict_plan: bool, stage: str) -> None:
    _not_implemented("run")


@cli.command()
@click.pass_obj
def ingest(env: AppEnv) -> None:
    _not_implemented("ingest")


@cli.command()
@click.pass_obj
def render(env: AppEnv) -> None:
    _not_implemented("render")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines")
@click.option("--csv", type=click.Path(path_type=Path), help="Write CSV to file")
@click.option("--pick", is_flag=True, help="Interactive picker for results")
@click.option("--open", "open_result", is_flag=True, help="Open result path after selection")
@click.pass_obj
def search(env: AppEnv, json_output: bool, json_lines: bool, csv: Optional[Path], pick: bool, open_result: bool) -> None:
    _not_implemented("search")


@cli.command()
@click.option("--open", "open_result", is_flag=True, help="Open path in browser/editor")
@click.pass_obj
def open(env: AppEnv, open_result: bool) -> None:
    _not_implemented("open")


@cli.command()
@click.pass_obj
def health(env: AppEnv) -> None:
    _not_implemented("health")


@cli.command()
@click.pass_obj
def export(env: AppEnv) -> None:
    _not_implemented("export")


@cli.group()
def config() -> None:
    """Configuration commands."""


@config.command("init")
@click.option("--interactive", "interactive", is_flag=True, help="Run interactive config init")
@click.pass_obj
def config_init(env: AppEnv, interactive: bool) -> None:
    _not_implemented("config init")


@config.command("show")
@click.pass_obj
def config_show(env: AppEnv) -> None:
    _not_implemented("config show")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_obj
def config_set(env: AppEnv, key: str, value: str) -> None:
    _not_implemented("config set")


def main() -> None:
    cli()


__all__ = ["cli", "main"]
