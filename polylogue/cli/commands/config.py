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
@click.option(
    "--show-layers",
    is_flag=True,
    default=False,
    help="Show the layer source for each config key (default/site/user/env/cli).",
)
@click.pass_obj
def config_command(env: AppEnv, output_format: str, show_layers: bool) -> None:
    """Show resolved Polylogue configuration with precedence sources."""
    from polylogue.config import describe_config_layers, format_config_toml, load_polylogue_config

    cfg = load_polylogue_config()

    if show_layers:
        # ``console.print`` interprets ``[brackets]`` as Rich markup; disable
        # markup so JSON output and TOML section headers survive verbatim.
        layer_paths = describe_config_layers()
        payload: dict[str, object] = {
            "layers": {
                "default": "built-in defaults",
                "site": layer_paths["site"],
                "user": layer_paths["user"],
                "env": "POLYLOGUE_* environment variables",
                "cli": "CLI overrides (per-invocation)",
            },
            "values": {key: {"value": cfg.raw.get(key), "layer": cfg.layer_of(key)} for key in sorted(cfg.raw.keys())},
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
        # markup parsing to preserve exact bytes.
        env.ui.console.print(json.dumps(cfg.raw, indent=2, default=str), markup=False, highlight=False)
    else:
        env.ui.console.print(format_config_toml(cfg.raw), markup=False, highlight=False)


__all__ = ["config_command"]
