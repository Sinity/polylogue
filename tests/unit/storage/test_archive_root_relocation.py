"""Regression coverage for the offline inode-preserving archive-root move."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.backup import backup_archive
from polylogue.operations.archive_root_relocation import ArchiveRootRelocationError, prepare_archive_root_relocation
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import rebind_released_source_train_archive_identity
from polylogue.storage.sqlite.migration_runner import (
    apply_durable_change_train,
    capture_durable_restart_convergence,
    prove_durable_change_train,
    record_durable_writer_release,
    release_durable_change_train,
)


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


def test_rebind_rewrites_only_the_released_source_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real durable-train lifecycle, then its relocation revision helper."""
    from tests.unit.storage import test_durable_change_train as trains

    database = tmp_path / "source.db"
    trains._create_current_database(database)
    trains._install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
    train = trains._admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(database) as connection:
        train = trains._reserve_and_authorize(connection, train, archive_root=tmp_path)
        train = apply_durable_change_train(connection, train)
    train = record_durable_writer_release(train, evidence_ref="proof:release")
    with sqlite3.connect(database) as connection:
        restart = capture_durable_restart_convergence(
            connection,
            train,
            runtime_consumers=trains._runtime_results(),
            evidence_ref="proof:restart",
        )
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=trains._parity(ArchiveTier.SOURCE),
        runtime_consumers=trains._runtime_results(),
        restart_convergence=restart,
    )
    released = release_durable_change_train(train, evidence_ref="proof:released")
    assert released.apply_evidence is not None
    before = released
    before_evidence = before.apply_evidence
    assert before_evidence is not None
    updated = rebind_released_source_train_archive_identity(
        before,
        archive_identity_digest="a" * 64,
        proof_ref="proof:archive-root-relocation:receipt",
    )

    assert updated.revision == before.revision + 1
    assert updated.apply_evidence == replace(
        before_evidence,
        post=replace(before_evidence.post, archive_identity_digest="a" * 64),
    )
    assert updated.proof_refs == (*before.proof_refs, "proof:archive-root-relocation:receipt")
