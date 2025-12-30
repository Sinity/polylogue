"""v666 CLI entrypoint (clean surface, adaptive UI)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click

from ..ui import create_ui
from ..config import (
    Config,
    ConfigError,
    default_config,
    load_config,
    resolve_profile,
    update_config,
    update_profile,
    update_source,
    write_config,
)


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


def _fail(command: str, message: str) -> None:
    raise SystemExit(f"{command}: {message}")


def _load_effective_config(env: AppEnv) -> Tuple[Config, str]:
    config = load_config(env.config_path)
    profile_name, _ = resolve_profile(config, env.profile)
    return config, profile_name


def _print_summary(env: AppEnv) -> None:
    ui = env.ui
    try:
        config, profile_name = _load_effective_config(env)
    except ConfigError as exc:
        ui.console.print(f"[yellow]{exc}[/yellow]")
        ui.console.print("Run `polylogue config init` to create a config.")
        return
    ui.summary(
        "Polylogue",
        [
            f"Config: {config.path}",
            f"Archive root: {config.archive_root}",
            f"Profile: {profile_name}",
            "Last run: unavailable (run ledger not implemented yet)",
            "Health: unavailable (health checks not implemented yet)",
        ],
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
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
    if ctx.invoked_subcommand is None:
        _print_summary(env)


@cli.command()
@click.pass_obj
def plan(env: AppEnv) -> None:
    _fail("plan", "not implemented yet in the v666 rewrite")


@cli.command()
@click.option("--no-plan", is_flag=True, help="Skip plan preview")
@click.option("--strict-plan", is_flag=True, help="Fail if plan drift is detected")
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all")
@click.pass_obj
def run(env: AppEnv, no_plan: bool, strict_plan: bool, stage: str) -> None:
    _fail("run", "not implemented yet in the v666 rewrite")


@cli.command()
@click.pass_obj
def ingest(env: AppEnv) -> None:
    _fail("ingest", "not implemented yet in the v666 rewrite")


@cli.command()
@click.pass_obj
def render(env: AppEnv) -> None:
    _fail("render", "not implemented yet in the v666 rewrite")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines")
@click.option("--csv", type=click.Path(path_type=Path), help="Write CSV to file")
@click.option("--pick", is_flag=True, help="Interactive picker for results")
@click.option("--open", "open_result", is_flag=True, help="Open result path after selection")
@click.pass_obj
def search(env: AppEnv, json_output: bool, json_lines: bool, csv: Optional[Path], pick: bool, open_result: bool) -> None:
    _fail("search", "not implemented yet in the v666 rewrite")


@cli.command()
@click.option("--open", "open_result", is_flag=True, help="Open path in browser/editor")
@click.pass_obj
def open(env: AppEnv, open_result: bool) -> None:
    _fail("open", "not implemented yet in the v666 rewrite")


@cli.command()
@click.pass_obj
def health(env: AppEnv) -> None:
    _fail("health", "not implemented yet in the v666 rewrite")


@cli.command()
@click.pass_obj
def export(env: AppEnv) -> None:
    _fail("export", "not implemented yet in the v666 rewrite")


@cli.group()
def config() -> None:
    """Configuration commands."""


@config.command("init")
@click.option("--interactive", "interactive", is_flag=True, help="Run interactive config init")
@click.pass_obj
def config_init(env: AppEnv, interactive: bool) -> None:
    target = env.config_path
    config = default_config(target)
    if config.path.exists():
        _fail("config init", f"config already exists at {config.path}")
    if interactive and not env.ui.plain:
        if not env.ui.confirm(f"Write config to {config.path}?", default=True):
            env.ui.console.print("Init cancelled.")
            return
    write_config(config)
    env.ui.console.print(f"Config written to {config.path}")


@config.command("show")
@click.pass_obj
def config_show(env: AppEnv) -> None:
    try:
        config, profile_name = _load_effective_config(env)
    except ConfigError as exc:
        _fail("config show", str(exc))
    payload = config.as_dict()
    payload["active_profile"] = profile_name
    env.ui.console.print(json.dumps(payload, indent=2))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_obj
def config_set(env: AppEnv, key: str, value: str) -> None:
    try:
        config = load_config(env.config_path)
    except ConfigError as exc:
        _fail("config set", str(exc))
    try:
        if key == "archive_root":
            config = update_config(config, archive_root=Path(value))
        elif key.startswith("profile."):
            parts = key.split(".", 2)
            if len(parts) != 3:
                raise ConfigError("Profile updates require 'profile.<name>.<field>'")
            _, profile_name, field = parts
            config = update_profile(config, profile_name, field, value)
        elif key.startswith("source."):
            parts = key.split(".", 2)
            if len(parts) != 3:
                raise ConfigError("Source updates require 'source.<name>.<field>'")
            _, source_name, field = parts
            config = update_source(config, source_name, field, value)
        else:
            raise ConfigError(f"Unknown config key '{key}'")
    except ConfigError as exc:
        _fail("config set", str(exc))
    write_config(config)
    env.ui.console.print(f"Config updated: {config.path}")


def main() -> None:
    cli()


__all__ = ["cli", "main"]
