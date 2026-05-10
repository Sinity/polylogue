"""Command registration for the root Click application."""

from __future__ import annotations

import click

from polylogue.cli.commands.auth import auth_command
from polylogue.cli.commands.backup import backup_command
from polylogue.cli.commands.check import check_command
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.config import config_command
from polylogue.cli.commands.context_pack import context_pack_command
from polylogue.cli.commands.cost import cost_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.diagnostics import diagnostics_group
from polylogue.cli.commands.export import export_command
from polylogue.cli.commands.ingest import ingest_command
from polylogue.cli.commands.insights import insights_command
from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.cli.commands.neighbors import neighbors_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.resume import resume_command
from polylogue.cli.commands.schema import schema_command
from polylogue.cli.commands.status import status_command
from polylogue.cli.commands.tags import tags_command

ROOT_COMMANDS: tuple[click.Command, ...] = (
    context_pack_command,
    backup_command,
    check_command,
    config_command,
    cost_command,
    reset_command,
    status_command,
    ingest_command,
    auth_command,
    completions_command,
    dashboard_command,
    neighbors_command,
    export_command,
    resume_command,
    insights_command,
    tags_command,
    schema_command,
    diagnostics_group,
    maintenance_group,
)


def register_root_commands(group: click.Group) -> None:
    """Attach the canonical root subcommands to the main CLI group."""
    for command in ROOT_COMMANDS:
        group.add_command(command)


__all__ = [
    "completions_command",
    "dashboard_command",
    "export_command",
    "maintenance_group",
    "neighbors_command",
    "register_root_commands",
    "resume_command",
]
