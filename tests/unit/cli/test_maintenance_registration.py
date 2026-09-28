"""Verify the maintenance group is registered and reachable via CLI."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
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


def test_maintenance_appears_in_ops_help() -> None:
    """polylogue ops --help includes the maintenance subcommand."""
    runner = CliRunner()
    result = runner.invoke(root_cli, ["ops", "--help"])
    assert result.exit_code == 0
    assert "maintenance" in result.output


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


def test_blob_conservation_json_returns_failure_for_a_failed_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automation receives a nonzero status for JSON conservation failures.

    Anti-vacuity: returning directly after JSON output leaves Click with a
    success status despite the failing report. ``catch_exceptions=False`` is
    what keeps that condition load-bearing -- a ``CliRunner`` that swallows an
    exception also reports ``exit_code == 1`` with empty output, so the status
    assertion alone passes for the wrong reason.

    The archive root is pinned because the command calls ``archive_root()``
    before the stubbed census: resolved ambiently, it depends on whatever a
    previously executed test left in the environment, which made this test pass
    alone and fail inside a wider selection.
    """
    from polylogue.cli.commands.maintenance._blob_conservation import blob_conservation_command
    from polylogue.maintenance import blob_conservation

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

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

    result = CliRunner().invoke(blob_conservation_command, ["--output-format", "json"], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["ok"] is False
