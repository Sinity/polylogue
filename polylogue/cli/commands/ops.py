"""Operational command group."""

from __future__ import annotations

import click

from polylogue.cli.click_command_registration import register_ops_commands


@click.group("ops")
def ops_command() -> None:
    """Run operational archive, daemon, auth, embedding, and maintenance commands."""


register_ops_commands(ops_command)


__all__ = ["ops_command"]
