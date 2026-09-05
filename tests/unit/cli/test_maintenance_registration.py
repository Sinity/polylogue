"""Verify the maintenance group is registered and reachable via CLI."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.commands.maintenance._blob_integrity import (
    blob_reference_prune_orphans_command,
    blob_reference_prune_orphans_preview_command,
    blob_reference_replace_from_source_command,
    blob_reference_replace_from_source_preview_command,
)
from polylogue.cli.commands.maintenance._blob_reference_closure import blob_reference_closure_command
from polylogue.cli.commands.maintenance._operation_recovery import operation_recovery_command
from polylogue.cli.commands.maintenance._plan import plan_command
from polylogue.cli.commands.maintenance._status import status_command
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.services import RuntimeServices
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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


def test_operation_recovery_is_click_command() -> None:
    assert isinstance(operation_recovery_command, click.Command)


def _recovery_env(tmp_path: Path) -> AppEnv:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    return AppEnv(services=RuntimeServices(config=Config(archive_root=archive_root, render_root=tmp_path, sources=[])))


def test_operation_recovery_adjudication_refuses_while_daemon_owns_writes(tmp_path: Path) -> None:
    """The offline CLI must not become a second writer beside a live daemon.

    Anti-vacuity: this patches the daemon-liveness probe, not the guard, so
    dropping ``offline_maintenance_block_reason`` from the command makes the
    adjudication proceed and fail with ``operation not found`` instead.
    """

    env = _recovery_env(tmp_path)
    with patch("polylogue.maintenance.offline_guard.running_daemon_pid", return_value=4321):
        result = CliRunner().invoke(
            operation_recovery_command,
            [
                "--operation-id",
                "operation:test",
                "--target-outcome",
                "session:test=applied",
                "--reason",
                "operator evidence",
                "--confirm",
            ],
            obj=env,
        )
    assert result.exit_code == 1
    assert "4321" in result.output
    assert "not found" not in result.output


def test_operation_recovery_adjudication_requires_confirm_and_reason(tmp_path: Path) -> None:
    """Adjudication fails closed without both operator authorizations."""

    env = _recovery_env(tmp_path)
    for extra in (["--confirm"], ["--reason", "operator evidence"]):
        result = CliRunner().invoke(
            operation_recovery_command,
            ["--operation-id", "operation:test", "--target-outcome", "session:test=applied", *extra],
            obj=env,
        )
        assert result.exit_code == 1
        assert "requires --confirm and --reason" in result.output


def test_operation_recovery_rejects_an_unknown_target_outcome_value(tmp_path: Path) -> None:
    """The CLI validates outcome values before touching audit authority."""

    env = _recovery_env(tmp_path)
    result = CliRunner().invoke(
        operation_recovery_command,
        [
            "--operation-id",
            "operation:test",
            "--target-outcome",
            "session:test=probably",
            "--reason",
            "operator evidence",
            "--confirm",
        ],
        obj=env,
    )
    assert result.exit_code == 1
    assert "applied|not-applied|unknown" in result.output


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
    assert "operation-recovery" in cmds
    assert "blob-conservation" in cmds


def test_ops_import_keeps_blob_conservation_unloaded() -> None:
    """Unrelated ops commands do not pay for the census implementation.

    Anti-vacuity: an eager import in ``ops`` puts the maintenance module in
    ``sys.modules`` during this reload.
    """
    sys.modules.pop("polylogue.maintenance.blob_conservation", None)
    importlib.reload(importlib.import_module("polylogue.cli.commands.ops"))

    assert "polylogue.maintenance.blob_conservation" not in sys.modules


def test_blob_conservation_uses_the_resolved_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy route reads the maintenance root instead of a local override.

    Anti-vacuity: restoring a required subcommand ``--archive-root`` option
    makes this invocation fail before the command can inspect the archive.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    result = CliRunner().invoke(root_cli, ["ops", "maintenance", "blob-conservation", "--output-format", "json"])

    assert result.exit_code == 0, result.output
    assert f'"archive_root": "{archive_root}"' in result.output


def test_blob_conservation_json_returns_failure_for_a_failed_census(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automation receives a nonzero status for JSON conservation failures.

    Anti-vacuity: returning directly after JSON output leaves Click with a
    success status despite the failing report.
    """
    from polylogue.cli.commands.maintenance._blob_conservation import blob_conservation_command
    from polylogue.maintenance import blob_conservation

    monkeypatch.setattr(
        blob_conservation,
        "check_blob_conservation",
        lambda *_args, **_kwargs: blob_conservation.BlobConservationReport(
            archive_root="/archive",
            referenced_blobs=1,
            present_blobs=0,
            orphan_blobs=0,
            dangling_references=1,
            recoverable_references=0,
            reserved_blobs=0,
            corrupt_blobs=0,
            invalid_namespace_entries=0,
            staged_in_flight=0,
        ),
    )

    result = CliRunner().invoke(blob_conservation_command, ["--output-format", "json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["ok"] is False


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


def test_blob_reference_closure_is_click_command() -> None:
    assert isinstance(blob_reference_closure_command, click.Command)


def test_blob_integrity_preview_and_apply_commands_are_distinct_click_routes() -> None:
    """The real Click registry exposes diagnostic and write commands separately."""

    for command in (
        blob_reference_replace_from_source_preview_command,
        blob_reference_prune_orphans_preview_command,
    ):
        assert isinstance(command, click.Command)
        assert "--apply" not in {option.name for option in command.params}

    for command in (
        blob_reference_replace_from_source_command,
        blob_reference_prune_orphans_command,
    ):
        assert isinstance(command, click.Command)

    maintenance_group = _registered_maintenance_command()
    ctx = click.Context(maintenance_group)
    commands = maintenance_group.list_commands(ctx)  # type: ignore[attr-defined]
    assert "blob-reference-replace-from-source-preview" in commands
    assert "blob-reference-replace-from-source" in commands
    assert "blob-reference-prune-orphans-preview" in commands
    assert "blob-reference-prune-orphans" in commands


def test_maintenance_group_has_status() -> None:
    """maintenance group lists status as a subcommand (#1197)."""
    maintenance_group = _registered_maintenance_command()
    ctx = click.Context(maintenance_group)
    cmds = maintenance_group.list_commands(ctx)  # type: ignore[attr-defined]
    assert "status" in cmds
    assert "blob-reference-closure" in cmds


def test_maintenance_status_help_output() -> None:
    """polylogue ops maintenance status --help shows the status help."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "maintenance", "status", "--help"])
    assert result.exit_code == 0
    assert "--operation-id" in result.output
    assert "--all" in result.output


def test_plan_reports_a_refused_scope_and_exits_nonzero(tmp_path: Path) -> None:
    """A permanently refused plan is not a clean archive.

    Anti-vacuity: removing the ``SystemExit(1)`` and the failure-sample loop
    from ``plan_command`` makes this red -- the command would print
    "Affected: 0 rows" and exit 0 for a request no target can apply.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with (
        patch("polylogue.cli.commands.maintenance._plan.archive_root", return_value=archive_root),
        patch("polylogue.cli.commands.maintenance._plan.render_root", return_value=tmp_path),
    ):
        result = CliRunner().invoke(
            root_cli,
            ["ops", "maintenance", "plan", "--target", "empty_sessions", "--origin", "claude-code-session"],
        )

    assert result.exit_code == 1
    assert "UnsupportedScopeDimension" in result.output


def test_plan_without_a_session_filter_is_not_a_scoped_request(tmp_path: Path) -> None:
    """An omitted ``--session-id`` is no narrowing, so the default plan runs.

    Anti-vacuity: dropping ``_coerce_session_ids``'s empty-tuple
    normalization makes this red -- Click's ``()`` would read as a session
    scope and ``superseded_raw_snapshots``, which honors no dimension, would
    refuse with exit 1.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with (
        patch("polylogue.cli.commands.maintenance._plan.archive_root", return_value=archive_root),
        patch("polylogue.cli.commands.maintenance._plan.render_root", return_value=tmp_path),
    ):
        result = CliRunner().invoke(
            root_cli,
            ["ops", "maintenance", "plan", "--target", "superseded_raw_snapshots"],
        )

    assert result.exit_code == 0
    assert "UnsupportedScopeDimension" not in result.output
