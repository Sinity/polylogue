"""Verify the maintenance group is registered and reachable via CLI."""

from __future__ import annotations

import click
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.commands.maintenance._blob_reference_closure import blob_reference_closure_command
from polylogue.cli.commands.maintenance._hook_payload_ref_reconciliation import hook_payload_ref_reconcile_command
from polylogue.cli.commands.maintenance._plan import plan_command
from polylogue.cli.commands.maintenance._raw_authority_recovery import raw_authority_recovery_command
from polylogue.cli.commands.maintenance._run import run_command
from polylogue.cli.commands.maintenance._run_preview import run_preview_command
from polylogue.cli.commands.maintenance._status import status_command


def _registered_maintenance_command() -> click.Command:
    from polylogue.cli.click_command_registration import OPS_COMMANDS

    for command in OPS_COMMANDS:
        if command.name == "maintenance":
            return command
    raise AssertionError("maintenance command is not registered under ops")


def test_maintenance_group_in_ops_commands() -> None:
    """maintenance_group is registered under polylogue ops."""
    assert _registered_maintenance_command() is not None


def test_maintenance_group_is_click_group() -> None:
    """maintenance_group is a Click Group."""
    assert isinstance(_registered_maintenance_command(), click.Group)


def test_maintenance_plan_is_click_command() -> None:
    """plan is a Click Command on the maintenance group."""
    assert isinstance(plan_command, click.Command)


def test_raw_authority_recovery_is_click_command() -> None:
    assert isinstance(raw_authority_recovery_command, click.Command)


def test_maintenance_run_is_click_command() -> None:
    """run is a Click Command on the maintenance group."""
    assert isinstance(run_command, click.Command)


def test_maintenance_run_preview_is_click_command() -> None:
    """run-preview is a Click Command on the maintenance group."""
    assert isinstance(run_preview_command, click.Command)


def test_maintenance_appears_in_ops_help() -> None:
    """polylogue ops --help includes the maintenance subcommand."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "--help"])
    assert result.exit_code == 0
    assert "maintenance" in result.output


def test_maintenance_group_has_plan_and_run() -> None:
    """maintenance group lists plan, run, and run-preview as subcommands."""
    maintenance_group = _registered_maintenance_command()
    ctx = click.Context(maintenance_group)
    cmds = maintenance_group.list_commands(ctx)  # type: ignore[attr-defined]
    assert "plan" in cmds
    assert "run" in cmds
    assert "run-preview" in cmds
    assert "raw-authority-recovery" in cmds


def test_maintenance_plan_help_output() -> None:
    """polylogue ops maintenance plan --help shows plan help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "maintenance", "plan", "--help"])
    assert result.exit_code == 0
    assert "Dry-run" in result.output or "summary" in result.output.lower()


def test_maintenance_run_help_output() -> None:
    """polylogue ops maintenance run --help shows run help.

    ``run`` is the lean apply-only command post-split (polylogue-oou3c): it
    no longer carries a ``--dry-run`` flag -- ``run-preview`` is the
    dedicated read-only twin instead.
    """
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "maintenance", "run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" not in result.output
    assert "--operation-id" in result.output


def test_maintenance_run_preview_help_output() -> None:
    """polylogue ops maintenance run-preview --help shows the preview help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "maintenance", "run-preview", "--help"])
    assert result.exit_code == 0
    assert "Read-only" in result.output or "read-only" in result.output.lower()
    assert "--operation-id" in result.output


def test_maintenance_status_is_click_command() -> None:
    """status is a Click Command on the maintenance group (#1197)."""
    assert isinstance(status_command, click.Command)


def test_hook_payload_reconcile_is_click_command() -> None:
    """hook-payload-ref-reconcile is a Click Command on the maintenance group."""
    assert isinstance(hook_payload_ref_reconcile_command, click.Command)


def test_blob_reference_closure_is_click_command() -> None:
    assert isinstance(blob_reference_closure_command, click.Command)


def test_maintenance_group_has_status() -> None:
    """maintenance group lists status as a subcommand (#1197)."""
    maintenance_group = _registered_maintenance_command()
    ctx = click.Context(maintenance_group)
    cmds = maintenance_group.list_commands(ctx)  # type: ignore[attr-defined]
    assert "status" in cmds
    assert "hook-payload-ref-reconcile" in cmds
    assert "blob-reference-closure" in cmds


def test_maintenance_status_help_output() -> None:
    """polylogue ops maintenance status --help shows the status help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "maintenance", "status", "--help"])
    assert result.exit_code == 0
    assert "--operation-id" in result.output
    assert "--all" in result.output
