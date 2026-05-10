"""Verify the maintenance group is registered and reachable via CLI."""

from __future__ import annotations

import click
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.click_command_registration import maintenance_group
from polylogue.cli.commands.maintenance import plan_command, run_command


def test_maintenance_group_in_root_commands() -> None:
    """maintenance_group is registered in ROOT_COMMANDS."""
    from polylogue.cli.click_command_registration import ROOT_COMMANDS

    assert maintenance_group in ROOT_COMMANDS


def test_maintenance_group_is_click_group() -> None:
    """maintenance_group is a Click Group."""
    assert isinstance(maintenance_group, click.Group)


def test_maintenance_plan_is_click_command() -> None:
    """plan is a Click Command on the maintenance group."""
    assert isinstance(plan_command, click.Command)


def test_maintenance_run_is_click_command() -> None:
    """run is a Click Command on the maintenance group."""
    assert isinstance(run_command, click.Command)


def test_maintenance_appears_in_help() -> None:
    """polylogue --help includes the maintenance subcommand."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["--help"])
    assert result.exit_code == 0
    assert "maintenance" in result.output


def test_maintenance_group_has_plan_and_run() -> None:
    """maintenance group lists plan and run as subcommands."""
    ctx = click.Context(maintenance_group)
    cmds = maintenance_group.list_commands(ctx)
    assert "plan" in cmds
    assert "run" in cmds


def test_maintenance_plan_help_output() -> None:
    """polylogue maintenance plan --help shows plan help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["maintenance", "plan", "--help"])
    assert result.exit_code == 0
    assert "Dry-run" in result.output or "summary" in result.output.lower()


def test_maintenance_run_help_output() -> None:
    """polylogue maintenance run --help shows run help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["maintenance", "run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
