"""Config command — show resolved polylogue configuration."""

from __future__ import annotations

import click

from polylogue.cli.commands.completions import (
    action_affordances_command,
    completions_command,
    query_completions_command,
)
from polylogue.cli.commands.paths import paths_command
from polylogue.cli.shared.types import AppEnv


@click.group("config", invoke_without_command=True, no_args_is_help=False)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["toml", "json"]),
    default="toml",
    help="Output format.",
)
@click.option(
    "--show-layers",
    is_flag=True,
    default=False,
    help="Show the layer source for each config key (default/site/user/env/cli).",
)
@click.pass_context
def config_command(ctx: click.Context, output_format: str, show_layers: bool) -> None:
    """Show resolved Polylogue configuration with precedence sources."""
    if ctx.invoked_subcommand is not None:
        return
    env = ctx.obj
    assert isinstance(env, AppEnv)
    _show_config(env, output_format, show_layers)


config_command.add_command(completions_command)
config_command.add_command(query_completions_command)
config_command.add_command(action_affordances_command)
config_command.add_command(paths_command)


def _show_config(env: AppEnv, output_format: str, show_layers: bool) -> None:
    from polylogue.config import effective_config_payload, format_config_toml, load_polylogue_config

    cfg = load_polylogue_config()
    payload = effective_config_payload(cfg)

    if output_format == "json":
        import json

        # ``console.print`` interprets ``[brackets]`` as Rich markup, which
        # would mangle JSON output and TOML section headers. Print without
        # markup parsing to preserve exact bytes. Secret-bearing keys are
        # redacted by ``effective_config_payload``. ``soft_wrap=True`` is
        # load-bearing: without it Rich wraps long lines at the console
        # width by inserting a literal newline mid-line, which corrupts any
        # JSON string value (e.g. an inventory ``description``) long enough
        # to exceed that width into invalid JSON.
        env.ui.console.print(
            json.dumps(payload, indent=2, default=str),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return

    if show_layers:
        lines = ["# Polylogue config layer sources", ""]
        layers_section = payload["layers"]
        assert isinstance(layers_section, dict)
        lines.append("[layers]")
        for layer_name, descriptor in layers_section.items():
            lines.append(f"# {layer_name}: {descriptor}")
        lines.append("")
        lines.append("[values]")
        values_section = payload["values"]
        assert isinstance(values_section, dict)
        for key in sorted(values_section):
            info = values_section[key]
            assert isinstance(info, dict)
            value = info.get("value")
            layer = info.get("source_layer")
            owner = info.get("owner_class")
            reload_behavior = info.get("reload_behavior")
            lines.append(f"{key} = {value!r}  # layer = {layer}; owner = {owner}; reload = {reload_behavior}")
        env.ui.console.print("\n".join(lines), markup=False, highlight=False)
        return

    env.ui.console.print(format_config_toml(cfg.raw), markup=False, highlight=False)


__all__ = ["config_command"]
