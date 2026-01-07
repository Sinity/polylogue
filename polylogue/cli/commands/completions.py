"""Completions command."""

from __future__ import annotations

import os
import sys

import click


@click.command("completions")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), required=True)
def completions_command(shell: str) -> None:
    """Generate shell completion scripts."""
    script: str = ""
    prog_name = os.path.basename(sys.argv[0])
    env_name = "_POLYLOGUE_COMPLETE"

    if shell == "bash":
        script = f'eval "$({env_name}=bash_source {prog_name})"'
    elif shell == "zsh":
        script = f'eval "$({env_name}=zsh_source {prog_name})"'
    elif shell == "fish":
        script = f"eval (env {env_name}=fish_source {prog_name})"

    click.echo(f"# To enable completions, run:\n{script}")
