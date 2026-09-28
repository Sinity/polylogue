"""Narrow root compatibility route for the private Fable packet command."""

from __future__ import annotations

import click


@click.group("insights")
def insights_alias_command() -> None:
    """Access the private Fable packet command."""


def _register_fable_packet() -> None:
    from polylogue.cli.commands.insights import ops_insights_command

    context = click.Context(ops_insights_command)
    command = ops_insights_command.get_command(context, "fable-packet")
    if command is not None:
        insights_alias_command.add_command(command)


_register_fable_packet()

__all__ = ["insights_alias_command"]
