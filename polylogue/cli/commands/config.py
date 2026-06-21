"""Config command — show resolved polylogue configuration."""

from __future__ import annotations

import click

from polylogue.cli.commands.completions import completions_command, query_completions_command
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
config_command.add_command(paths_command)


def _show_config(env: AppEnv, output_format: str, show_layers: bool) -> None:
    from polylogue.config import (
        describe_config_layers,
        format_config_toml,
        is_secret_config_key,
        load_polylogue_config,
        redact_config_mapping,
        redact_secret_value,
    )

    cfg = load_polylogue_config()

    def _display_value(key: str) -> object:
        raw_value = cfg.raw.get(key)
        if is_secret_config_key(key):
            return redact_secret_value(raw_value)
        return raw_value

    if show_layers:
        # ``console.print`` interprets ``[brackets]`` as Rich markup; disable
        # markup so JSON output and TOML section headers survive verbatim.
        # Secret-bearing keys are redacted before display so the layer dump
        # never reveals a cleartext secret.
        layer_paths = describe_config_layers()
        payload: dict[str, object] = {
            "layers": {
                "default": "built-in defaults",
                "site": layer_paths["site"],
                "user": layer_paths["user"],
                "env": "POLYLOGUE_* environment variables",
                "cli": "CLI overrides (per-invocation)",
            },
            "values": {
                key: {"value": _display_value(key), "layer": cfg.layer_of(key)} for key in sorted(cfg.raw.keys())
            },
        }
        if output_format == "json":
            import json

            env.ui.console.print(
                json.dumps(payload, indent=2, default=str),
                markup=False,
                highlight=False,
            )
        else:
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
            for key, info in values_section.items():
                assert isinstance(info, dict)
                value = info.get("value")
                layer = info.get("layer")
                lines.append(f"{key} = {value!r}  # layer = {layer}")
            env.ui.console.print("\n".join(lines), markup=False, highlight=False)
        return

    if output_format == "json":
        import json

        # ``console.print`` interprets ``[brackets]`` as Rich markup, which
        # would mangle JSON output and TOML section headers. Print without
        # markup parsing to preserve exact bytes. Secret-bearing keys are
        # redacted so the JSON dump never reveals a cleartext secret.
        env.ui.console.print(
            json.dumps(redact_config_mapping(cfg.raw), indent=2, default=str),
            markup=False,
            highlight=False,
        )
    else:
        env.ui.console.print(format_config_toml(cfg.raw), markup=False, highlight=False)


__all__ = ["config_command"]
