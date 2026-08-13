"""Regression coverage for the offline inode-preserving archive-root move."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.backup import backup_archive
from polylogue.operations.archive_root_relocation import ArchiveRootRelocationError, prepare_archive_root_relocation


def test_archive_root_relocation_is_a_real_maintenance_route(cli_workspace: dict[str, object]) -> None:
    """The production maintenance dispatcher exposes the explicit relocation route."""
    result = CliRunner().invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-root-relocation", "--help"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "inode-preserving" in result.output


def test_plan_refuses_fresh_bootstrap_without_writing_the_moved_archive(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """The plan enters backup attestation and immutable archive inspection, never a write route."""
    old_root = workspace_env["archive_root"]
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    new_root = tmp_path / "moved-archive"
    os.rename(old_root, new_root)
    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }

    with pytest.raises(ArchiveRootRelocationError, match="fresh-bootstrap"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            stopped_daemon_evidence_ref="proof:test-daemon-stopped",
            single_writer_evidence_ref="proof:test-writer-lock",
        )

    after = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    assert after == before
