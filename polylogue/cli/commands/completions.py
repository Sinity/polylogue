"""Shell completions command.

Emits a shell-specific completion script on stdout that wires
``polylogue`` to the Click dynamic-completion protocol. The same source
of truth (the registered ``shell_complete`` callbacks on Click options
and arguments) drives bash, zsh, and fish.

Install (per shell):

* **bash** — append to ``~/.bashrc``::

      eval "$(polylogue completions --shell bash)"

* **zsh** — append to ``~/.zshrc`` (ensure ``compinit`` runs first)::

      eval "$(polylogue completions --shell zsh)"

* **fish** — load on every shell start::

      polylogue completions --shell fish | source

  Or persist to fish's per-user completions directory::

      polylogue completions --shell fish > ~/.config/fish/completions/polylogue.fish

Dynamic completers cover: session IDs, origin/source-family
tokens, tags, repository names, working-directory prefixes, action
categories and ordered action sequences, normalized tool names,
message types, and retrieval lanes. The matrix of completer × shell
is exercised by ``tests/unit/cli/test_completion_matrix.py``.
"""

from __future__ import annotations

import click
from click.shell_completion import get_completion_class

_INSTALL_EPILOG = """\
\b
Install:
  bash: add  eval "$(polylogue completions --shell bash)"  to ~/.bashrc
  zsh:  add  eval "$(polylogue completions --shell zsh)"   to ~/.zshrc
  fish: polylogue completions --shell fish > ~/.config/fish/completions/polylogue.fish

\b
Dynamic completers cover: session IDs, origins/source-families,
tags, repos, cwd prefixes, action categories and sequences, tool names,
message types, retrieval lanes.
"""


@click.command("completions", epilog=_INSTALL_EPILOG)
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
