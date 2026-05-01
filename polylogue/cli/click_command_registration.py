"""Command registration for the root Click application."""

from __future__ import annotations

import click

from polylogue.cli.commands.auth import auth_command
from polylogue.cli.commands.browser_capture import browser_capture_command
from polylogue.cli.commands.check import check_command
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.diagnostics import diagnostics_group
from polylogue.cli.commands.export import export_command
from polylogue.cli.commands.insights import insights_command
from polylogue.cli.commands.mcp import mcp_command
from polylogue.cli.commands.neighbors import neighbors_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.resume import resume_command
from polylogue.cli.commands.run import run_command
from polylogue.cli.commands.schema import schema_command
from polylogue.cli.commands.tags import tags_command
from polylogue.cli.commands.watch import watch_command

ROOT_COMMANDS: tuple[click.Command, ...] = (
    run_command,
    browser_capture_command,
    check_command,
    reset_command,
    mcp_command,
    auth_command,
    completions_command,
    dashboard_command,
    neighbors_command,
    export_command,
    resume_command,
    insights_command,
    tags_command,
    schema_command,
    watch_command,
    diagnostics_group,
)


def register_root_commands(group: click.Group) -> None:
    """Attach the canonical root subcommands to the main CLI group."""
    for command in ROOT_COMMANDS:
        group.add_command(command)


__all__ = [
    "completions_command",
    "dashboard_command",
    "export_command",
    "mcp_command",
    "neighbors_command",
    "resume_command",
    "register_root_commands",
]
