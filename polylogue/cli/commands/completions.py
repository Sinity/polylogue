"""Completions command."""

from __future__ import annotations

import click
from click.shell_completion import get_completion_class


@click.command("completions")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), required=True)
@click.pass_context
def completions_command(ctx: click.Context, shell: str) -> None:
    """Generate shell completion scripts.

    Outputs the completion script for the specified shell to stdout.
    """
    # Use canonical program name, not sys.argv[0] which may be __main__.py
    prog_name = "polylogue"

    # Get the root command (the CLI app)
    root_cmd = ctx.find_root().command

    # Use Click's built-in completion generator
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")

    comp = comp_cls(root_cmd, {}, prog_name, "_POLYLOGUE_COMPLETE")
    click.echo(comp.source())
