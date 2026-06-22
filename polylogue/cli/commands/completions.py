"""Shell completions command.

Emits a shell-specific completion script on stdout that wires
``polylogue`` to the Click dynamic-completion protocol. The same source
of truth (the registered ``shell_complete`` callbacks on Click options
and arguments) drives bash, zsh, and fish.

Install (per shell):

* **bash** — append to ``~/.bashrc``::

      eval "$(polylogue config completions --shell bash)"

* **zsh** — append to ``~/.zshrc`` (ensure ``compinit`` runs first)::

      eval "$(polylogue config completions --shell zsh)"

* **fish** — load on every shell start::

      polylogue config completions --shell fish | source

  Or persist to fish's per-user completions directory::

      polylogue config completions --shell fish > ~/.config/fish/completions/polylogue.fish

Dynamic completers cover: session IDs, origin/source-family
tokens, tags, repository names, working-directory prefixes, action
categories and ordered action sequences, normalized tool names,
message types, and retrieval lanes. The matrix of completer × shell
is exercised by ``tests/unit/cli/test_completion_matrix.py``.
"""

from __future__ import annotations

import json

import click
from click.shell_completion import get_completion_class

from polylogue.archive.query.completions import QUERY_COMPLETION_KINDS, QueryCompletionError, query_completion_payload
from polylogue.operations.action_contracts import action_affordance_list_payload

_INSTALL_EPILOG = """\
\b
Install:
  bash: add  eval "$(polylogue config completions --shell bash)"  to ~/.bashrc
  zsh:  add  eval "$(polylogue config completions --shell zsh)"   to ~/.zshrc
  fish: polylogue config completions --shell fish > ~/.config/fish/completions/polylogue.fish

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


@click.command("query-completions")
@click.option("--kind", type=click.Choice(QUERY_COMPLETION_KINDS), required=True)
@click.option("--incomplete", default="", show_default=True, help="Current token or partial text to complete.")
@click.option("--unit", help="Structural or terminal query unit for unit-scoped completion.")
@click.option("--field", help="Query field for operator completion.")
def query_completions_command(kind: str, incomplete: str, unit: str | None, field: str | None) -> None:
    """Print shared query-builder completion metadata as JSON."""

    try:
        payload = query_completion_payload(kind, incomplete=incomplete, unit=unit, field=field)
    except QueryCompletionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@click.command("action-affordances")
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    help="Emit JSON. Accepted for consistency with other machine-readable commands.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json"]),
    help="Output format. Only JSON is currently supported.",
)
def action_affordances_command(json_flag: bool, output_format: str | None) -> None:
    """Print shared query-action affordance metadata as JSON."""

    _ = (json_flag, output_format)
    click.echo(action_affordance_list_payload().model_dump_json(indent=2))


__all__ = ["action_affordances_command", "completions_command", "query_completions_command"]
