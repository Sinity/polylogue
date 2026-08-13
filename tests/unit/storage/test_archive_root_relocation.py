"""Regression coverage for the offline inode-preserving archive-root move."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.backup import backup_archive
from polylogue.operations.archive_root_relocation import (
    ArchiveRootRelocationError,
    apply_archive_root_relocation,
    prepare_archive_root_relocation,
)
from polylogue.operations.historical_source_continuity_recovery import (
    HistoricalSourceContinuityRecoveryError,
    _assert_complete_source_semantic_delta,
    _assert_exact_liveness_delta,
    _table_content_digest,
    _write_refresh_receipt,
    apply_historical_source_continuity_recovery,
    load_historical_source_continuity_recovery_plan,
)
from polylogue.storage.blob_ref_liveness import BlobRefLivenessCandidate
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    load_durable_change_train_manifest,
    rebind_released_source_train_archive_identity,
)
from polylogue.storage.sqlite.migration_runner import (
    apply_durable_change_train,
    capture_durable_restart_convergence,
    prove_durable_change_train,
    record_durable_writer_release,
    release_durable_change_train,
    write_durable_change_train_manifest,
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
    nested = CliRunner().invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-root-relocation", "plan", "--help"],
        catch_exceptions=False,
    )
    assert nested.exit_code == 0, nested.output
    assert "--old-root" in nested.output


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
    assert before.released_at_ms is not None
    current_authority = replace(
        before,
        source_continuity_evidence=replace(before_evidence.post, observed_at_ms=before.released_at_ms + 1),
        proof_refs=(*before.proof_refs, "proof:source-continuity-refresh:" + "d" * 64),
    )
    rebound_current_authority = rebind_released_source_train_archive_identity(
        current_authority,
        archive_identity_digest="c" * 64,
        proof_ref="proof:archive-root-relocation:receipt-current",
    )
    assert current_authority.source_continuity_evidence is not None
    assert rebound_current_authority.source_continuity_evidence == replace(
        current_authority.source_continuity_evidence,
        archive_identity_digest="c" * 64,
    )


def _released_moved_source_train(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a real released source train over a temporary SQLite source tier."""
    from tests.unit.storage import test_durable_change_train as trains

    source = root / "source.db"
    source.unlink()
    trains._create_current_database(source)
    trains._install_synthetic_migration(root.parent, monkeypatch, ArchiveTier.SOURCE)
    train = trains._admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(source) as connection:
        train = trains._reserve_and_authorize(connection, train, archive_root=root)
        train = apply_durable_change_train(connection, train)
    train = record_durable_writer_release(train, evidence_ref="proof:writer-release")
    with sqlite3.connect(source) as connection:
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
    historical = replace(
        released,
        apply_evidence=replace(
            released.apply_evidence,
            post=replace(released.apply_evidence.post, archive_identity_digest="b" * 64),
        ),
    )
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    (manifest_root / ".bootstrap").unlink()
    manifest = manifest_root / "source-002.json"
    write_durable_change_train_manifest(manifest, historical, expected_revision=-1)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 1)
    return manifest


def _legacy_zero_candidate_receipt(path: Path, *, old_root: Path, pre_manifest: Path) -> None:
    """Encode the exact pre-#3868 shape: no backup digest or postcondition field."""
    path.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(old_root / "source.db"),
                "backup_manifest": str(pre_manifest),
                "candidate_count": 0,
                "candidate_digest": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "committed",
                "deleted_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_liveness_delta_database(path: Path, *, keep_body: str = "kept", include_candidate: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE unrelated_authority (id TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL, ref_type TEXT NOT NULL, ref_id TEXT NOT NULL,
                source_path TEXT, size_bytes INTEGER NOT NULL, acquired_at_ms INTEGER NOT NULL,
                PRIMARY KEY (blob_hash, ref_type, ref_id)
            ) STRICT;
            """
        )
        connection.execute("INSERT INTO raw_sessions VALUES ('live', ?)", (keep_body,))
        connection.execute("INSERT INTO unrelated_authority VALUES ('stable', 'unchanged')")
        connection.execute("INSERT INTO blob_refs VALUES (X'01', 'attachment', 'live', NULL, 1, 1)")
        if include_candidate:
            connection.execute("INSERT INTO blob_refs VALUES (X'02', 'attachment', 'deleted', NULL, 2, 2)")


def test_historical_liveness_delta_requires_exact_deletion_and_no_other_source_mutation(tmp_path: Path) -> None:
    """The bridge permits one enumerated orphan deletion, not a broad backup-to-backup rewrite."""
    pre = tmp_path / "pre.db"
    post = tmp_path / "post.db"
    _write_liveness_delta_database(pre)
    _write_liveness_delta_database(post, include_candidate=False)
    candidate = BlobRefLivenessCandidate(
        blob_hash="02",
        ref_type="attachment",
        ref_id="deleted",
        source_path=None,
        size_bytes=2,
        acquired_at_ms=2,
        referent_table="raw_sessions",
        referent_column="raw_id",
    )
    _assert_exact_liveness_delta(pre, post, (candidate,))
    _assert_complete_source_semantic_delta(pre, post)

    changed_table = tmp_path / "changed-table.db"
    _write_liveness_delta_database(changed_table, keep_body="tampered", include_candidate=False)
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="non-blob-ref"):
        _assert_complete_source_semantic_delta(pre, changed_table)

    wrong_blob_set = tmp_path / "wrong-blob-set.db"
    _write_liveness_delta_database(wrong_blob_set, include_candidate=False)
    with sqlite3.connect(wrong_blob_set) as connection:
        connection.execute("INSERT INTO blob_refs VALUES (X'03', 'attachment', 'extra', NULL, 3, 3)")
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="beyond the historical candidates"):
        _assert_exact_liveness_delta(pre, wrong_blob_set, (candidate,))


def test_historical_source_delta_tags_sqlite_storage_classes_and_rejects_refresh_symlinks(tmp_path: Path) -> None:
    """A non-STRICT BLOB/TEXT swap and a symlinked receipt directory are both unsafe."""
    typed = tmp_path / "typed.db"
    with sqlite3.connect(typed) as connection:
        connection.execute("CREATE TABLE values_table (value)")
        connection.execute("INSERT INTO values_table VALUES (?)", ("01",))
        text_digest = _table_content_digest(connection, "values_table")
        connection.execute("UPDATE values_table SET value = X'01'")
        blob_digest = _table_content_digest(connection, "values_table")
    assert text_digest != blob_digest

    root = tmp_path / "archive"
    state = root / ".maintenance-state"
    state.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    (state / "source-continuity-refreshes").symlink_to(target, target_is_directory=True)
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="not a real directory"):
        _write_refresh_receipt(
            state / "source-continuity-refreshes" / ("a" * 64 + ".json"),
            {"refresh_sha256": "a" * 64},
        )


def test_historical_continuity_recovery_is_a_real_cli_route_and_resumes(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise old-path HMACs, backup bytes, train CAS, census, and ordinary verification.

    This is deliberately file-backed: deleting the recovery identity rewrite,
    swapping either backup, or changing the receipt's old path makes the real
    plan/apply route fail before the train manifest can be written.
    """
    from polylogue.storage.sqlite import durable_change_train as trains

    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    pre_backup = backup_archive(output_dir=tmp_path / "pre", profile="rebuildable_cache_exclude", verify=True)
    post_backup = backup_archive(output_dir=tmp_path / "post", profile="rebuildable_cache_exclude", verify=True)
    assert pre_backup.ok and pre_backup.output_path is not None
    assert post_backup.ok and post_backup.output_path is not None
    pre_manifest = Path(pre_backup.output_path) / "manifest.json"
    post_manifest = Path(post_backup.output_path) / "manifest.json"
    legacy_receipt = tmp_path / "legacy-liveness.jsonl"
    _legacy_zero_candidate_receipt(legacy_receipt, old_root=old_root, pre_manifest=pre_manifest)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    moved_manifest = new_root / manifest.relative_to(old_root)
    database_before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }

    plan_path = tmp_path / "continuity-plan.json"
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(new_root)}
    plan_result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "source-continuity-recovery",
            "plan",
            "--old-root",
            str(old_root),
            "--mutation-receipt",
            str(legacy_receipt),
            "--pre-backup-manifest",
            str(pre_manifest),
            "--post-backup-manifest",
            str(post_manifest),
            "--output",
            str(plan_path),
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )
    assert plan_result.exit_code == 0, plan_result.output
    plan = load_historical_source_continuity_recovery_plan(plan_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "polylogue.operations.historical_source_continuity_recovery.running_daemon_pid",
            lambda _config: 4242,
        )
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="daemon to be stopped"):
            apply_historical_source_continuity_recovery(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
                stopped_daemon_evidence_ref="proof:daemon-stopped",
                single_writer_evidence_ref="proof:archive-ownership-lock",
            )
    assert not (new_root / ".maintenance-state" / "historical-source-continuity-recoveries").exists()
    assert database_before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    with sqlite3.connect(new_root / "source.db") as connection:
        with pytest.raises(Exception, match="continuity proof failed"):
            trains._verify_released_train_live_tier(
                new_root,
                connection,
                trains.load_durable_change_train_manifest(moved_manifest),
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            "polylogue.operations.historical_source_continuity_recovery.recover_released_source_train_continuity",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("crash after prepared receipt")),
        )
        with pytest.raises(RuntimeError, match="crash after prepared"):
            apply_historical_source_continuity_recovery(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
                stopped_daemon_evidence_ref="proof:daemon-stopped",
                single_writer_evidence_ref="proof:archive-ownership-lock",
            )
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="prepared but incomplete"):
        trains.reconcile_durable_change_train_startup(new_root)

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "source-continuity-recovery",
            "apply",
            "--plan",
            str(plan_path),
            "--authorize",
            plan.plan_sha256,
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "committed"
    # The second apply is the crash-recovery/idempotency path, not a second revision.
    replay = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "source-continuity-recovery",
            "apply",
            "--plan",
            str(plan_path),
            "--authorize",
            plan.plan_sha256,
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.output)["state"] == "committed"
    with sqlite3.connect(new_root / "source.db") as connection:
        assert (
            trains._verify_released_train_live_tier(
                new_root,
                connection,
                trains.load_durable_change_train_manifest(moved_manifest),
            )
            is None
        )


def test_prepare_apply_rebinds_a_real_released_train_and_resumes_after_prepared_crash(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use production backup, train CAS, and ordinary verifier across a moved temporary archive."""
    from polylogue.storage.sqlite import durable_change_train as trains

    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    moved_manifest = new_root / manifest.relative_to(old_root)
    with sqlite3.connect(new_root / "source.db") as connection:
        with pytest.raises(Exception, match="continuity proof failed"):
            trains._verify_released_train_live_tier(
                new_root,
                connection,
                trains.load_durable_change_train_manifest(moved_manifest),
            )
    database_before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert database_before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "polylogue.operations.archive_root_relocation.rebind_released_source_train_archive_identity",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        with pytest.raises(RuntimeError, match="crash"):
            apply_archive_root_relocation(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
                stopped_daemon_evidence_ref="proof:daemon-stopped",
                single_writer_evidence_ref="proof:archive-ownership-lock",
            )
    result = apply_archive_root_relocation(
        root=new_root,
        plan=plan,
        authorization=plan.plan_sha256,
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert result.state == "committed"
    assert (
        apply_archive_root_relocation(
            root=new_root,
            plan=plan,
            authorization=plan.plan_sha256,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        ).state
        == "committed"
    )
    assert database_before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    with sqlite3.connect(new_root / "source.db") as connection:
        assert (
            trains._verify_released_train_live_tier(
                new_root,
                connection,
                trains.load_durable_change_train_manifest(moved_manifest),
            )
            is None
        )


def test_plan_rejects_the_real_stale_source_train_shape_before_receipt_write(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-liveness current source needs the existing typed continuity receipt."""
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    released = load_durable_change_train_manifest(manifest)
    assert released.apply_evidence is not None
    stale = replace(
        released,
        revision=released.revision + 1,
        apply_evidence=replace(
            released.apply_evidence,
            post=replace(released.apply_evidence.post, content_sha256="f" * 64),
        ),
    )
    write_durable_change_train_manifest(manifest, stale, expected_revision=released.revision)
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    manifest_before = (new_root / manifest.relative_to(old_root)).read_bytes()

    with pytest.raises(ArchiveRootRelocationError, match="typed source-continuity refresh"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    assert (new_root / manifest.relative_to(old_root)).read_bytes() == manifest_before
    assert not (new_root / ".maintenance-state" / "archive-root-relocations").exists()
