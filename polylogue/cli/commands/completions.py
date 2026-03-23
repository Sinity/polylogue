"""Shell completions command."""

from __future__ import annotations

import click
from click.shell_completion import get_completion_class


@click.command("completions")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), required=True)
@click.pass_context
def completions_command(ctx: click.Context, shell: str) -> None:
    """Generate shell completion scripts."""
    root_cmd = ctx.find_root().command
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")

    comp = comp_cls(root_cmd, {}, "polylogue", "_POLYLOGUE_COMPLETE")
    click.echo(comp.source())


__all__ = ["completions_command"]
