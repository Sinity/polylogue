"""Regression coverage for the offline inode-preserving archive-root move."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.backup import backup_archive
from polylogue.operations.archive_root_relocation import (
    ArchiveRootRelocationError,
    ArchiveRootRelocationPlan,
    RelocationActiveIndexPointer,
    RelocationIndexGeneration,
    RelocationPostMoveWitness,
    RelocationTierEvidence,
    _check_backup_against_live,
    _index_generation_evidence,
    apply_archive_root_relocation,
    assert_no_prepared_archive_root_relocation,
    load_archive_root_relocation_plan,
    prepare_archive_root_relocation,
)
from polylogue.operations.archive_root_relocation import (
    _sealed_plan as _sealed_relocation_plan,
)
from polylogue.operations.archive_root_relocation import (
    _sealed_receipt as _sealed_relocation_receipt,
)
from polylogue.operations.archive_root_relocation import (
    _write_receipt as _write_relocation_receipt,
)
from polylogue.operations.historical_source_continuity_recovery import (
    HistoricalSourceContinuityRecoveryError,
    HistoricalSourceContinuityRecoveryReceipt,
    _assert_complete_source_semantic_delta,
    _assert_exact_liveness_delta,
    _current_evidence,
    _immutable_read_connection,
    _sha256,
    _table_content_digest,
    _verify_historical_operation_evidence,
    _write_refresh_receipt,
    apply_historical_source_continuity_recovery,
    assert_no_prepared_historical_source_continuity_recovery,
    load_historical_source_continuity_recovery_plan,
    load_historical_source_continuity_recovery_receipt,
    prepare_historical_source_continuity_recovery,
)
from polylogue.operations.historical_source_continuity_recovery import (
    _legacy_liveness_receipt as _validate_legacy_liveness_receipt,
)
from polylogue.operations.historical_source_continuity_recovery import (
    _sealed_receipt as _sealed_continuity_receipt,
)
from polylogue.operations.historical_source_continuity_recovery import (
    _write_receipt as _write_continuity_receipt,
)
from polylogue.storage.archive_identity import (
    ArchiveIdentity,
    ArchiveLocation,
    ArchiveOwnershipError,
    OwnedArchiveLocation,
)
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessCandidate,
    BlobRefLivenessCandidateDigest,
    classify_blob_ref_liveness,
)
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    DurableChangeTrain,
    DurableChangeTrainError,
    assert_source_continuity_apply_allowed,
    load_durable_change_train_manifest,
    rebind_released_durable_train_archive_identity,
    recover_released_source_train_continuity,
    refresh_released_source_train_continuity,
)
from polylogue.storage.sqlite.migration_runner import (
    DurableDatabaseEvidence,
    _canonical_json_sha256,
    apply_durable_change_train,
    capture_durable_database_evidence,
    capture_durable_restart_convergence,
    capture_durable_schema_inventory,
    durable_change_train_to_payload,
    prove_durable_change_train,
    record_durable_writer_release,
    release_durable_change_train,
    write_durable_change_train_manifest,
)


def test_relocation_names_generation_directory_missing_metadata(tmp_path: Path) -> None:
    """A metadata-less generation is reported as an orphan with its path."""
    generation_root = tmp_path / ".index-generations" / "gen-1755000000000-deadbeef"
    generation_root.mkdir(parents=True)

    with pytest.raises(ArchiveRootRelocationError, match="orphan.*gen-1755000000000-deadbeef"):
        _index_generation_evidence(old_root=tmp_path, new_root=tmp_path, active_index_pointer=None)


def test_historical_evidence_reads_keep_sqlite_temp_btrees_off_the_filesystem(tmp_path: Path) -> None:
    """Read-only continuity evidence remains executable on constrained roots."""
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                ref_type TEXT NOT NULL,
                ref_id TEXT NOT NULL
            );
            INSERT INTO blob_refs VALUES (X'01', 'raw_payload', 'raw-1');
            INSERT INTO blob_refs VALUES (X'02', 'raw_payload', 'raw-2');
            INSERT INTO blob_refs VALUES (X'03', 'attachment', 'att-1');
            """
        )

    with _immutable_read_connection(source) as connection:
        assert connection.execute("PRAGMA temp_store").fetchone() == (2,)
        assert connection.execute(
            "SELECT ref_type, COUNT(*) FROM blob_refs GROUP BY ref_type ORDER BY ref_type"
        ).fetchall() == [("attachment", 1), ("raw_payload", 2)]


@contextmanager
def _test_historical_operation_evidence_resource(path: Path) -> Iterator[None]:
    """Patch the packaged descriptor reader only within a synthetic test scope."""
    with patch(
        "polylogue.operations.historical_source_continuity_recovery._historical_operation_evidence_bytes",
        side_effect=lambda: path.read_bytes(),
    ):
        yield


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
    apply_help = CliRunner().invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-root-relocation", "apply", "--help"],
        catch_exceptions=False,
    )
    assert apply_help.exit_code == 0, apply_help.output
    assert "durable trains and sealed index-generation topology" in apply_help.output


def test_recovery_cli_reports_archive_ownership_conflicts(
    cli_workspace: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Maintenance lock contention is a public CLI error, never an internal traceback."""
    placeholders = {
        name: tmp_path / name
        for name in ("mutation.jsonl", "pre-manifest.json", "post-manifest.json", "sealed-plan.json")
    }
    for path in placeholders.values():
        path.write_text("{}", encoding="utf-8")

    def reject_ownership(*_args: object, **_kwargs: object) -> None:
        raise ArchiveOwnershipError("archive already owned")

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._source_continuity_recovery.acquire_durable_archive_ownership",
        reject_ownership,
    )
    plan_result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "source-continuity-recovery",
            "plan",
            "--old-root",
            str(tmp_path / "old"),
            "--mutation-receipt",
            str(placeholders["mutation.jsonl"]),
            "--pre-backup-manifest",
            str(placeholders["pre-manifest.json"]),
            "--post-backup-manifest",
            str(placeholders["post-manifest.json"]),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert plan_result.exit_code == 1
    assert "archive already owned" in plan_result.output

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._source_continuity_recovery.load_historical_source_continuity_recovery_plan",
        lambda _path: object(),
    )
    apply_result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "source-continuity-recovery",
            "apply",
            "--plan",
            str(placeholders["sealed-plan.json"]),
            "--authorize",
            "a" * 64,
        ],
    )
    assert apply_result.exit_code == 1
    assert "archive already owned" in apply_result.output


def test_relocation_nested_dispatch_keeps_analyze_facets_on_the_real_action(cli_workspace: dict[str, object]) -> None:
    """Nested maintenance routing must not turn the existing aggregate action into a silent no-op."""
    archive_root = cli_workspace["archive_root"]
    assert isinstance(archive_root, Path)
    result = CliRunner().invoke(
        cli,
        ["--plain", "analyze", "--facets"],
        env={"POLYLOGUE_ARCHIVE_ROOT": str(archive_root)},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Facets (global)" in result.output


def test_plan_refuses_fresh_bootstrap_without_writing_the_moved_archive(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan enters backup attestation and immutable archive inspection, never a write route."""
    old_root = workspace_env["archive_root"]
    new_root = tmp_path / "moved-archive"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
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


def test_plan_rejects_mutated_manifest_and_stale_authenticated_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old-path HMAC cannot bypass manifest-byte or closed-package binding."""
    old_root = workspace_env["archive_root"]
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    first = backup_archive(output_dir=tmp_path / "first", profile="full_evidence", verify=True)
    second = backup_archive(output_dir=tmp_path / "second", profile="full_evidence", verify=True)
    assert first.ok and first.output_path is not None
    assert second.ok and second.output_path is not None
    first_manifest = Path(first.output_path) / "manifest.json"
    first_receipt = Path(first.output_path) / "verification-receipt.json"
    second_receipt = Path(second.output_path) / "verification-receipt.json"
    original_manifest = first_manifest.read_bytes()
    original_receipt = first_receipt.read_bytes()

    first_manifest.write_bytes(original_manifest + b"\n")
    with pytest.raises(ArchiveRootRelocationError, match="does not match manifest"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=first_manifest,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    first_manifest.write_bytes(original_manifest)
    first_receipt.write_bytes(second_receipt.read_bytes())
    with pytest.raises(ArchiveRootRelocationError, match="does not match manifest"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=first_manifest,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    first_receipt.write_bytes(original_receipt)
    assert not (new_root / ".maintenance-state" / "archive-root-relocations").exists()


def test_plan_rejects_byte_identical_copied_archive_with_new_inodes(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authenticated pre-move inode facts distinguish a move from copytree bytes."""
    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch)
    new_root = tmp_path / "copied"
    shutil.copytree(old_root, new_root, symlinks=True)
    assert (old_root / "source.db").read_bytes() == (new_root / "source.db").read_bytes()
    assert (old_root / "source.db").stat().st_ino != (new_root / "source.db").stat().st_ino
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None

    with pytest.raises(ArchiveRootRelocationError, match="does not authenticate the moved tier identity"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    assert not (new_root / ".maintenance-state" / "archive-root-relocations").exists()


def test_tier_identity_rejects_a_changed_device_with_a_coincident_inode(tmp_path: Path) -> None:
    """Tier continuity is the full device/inode pair, not an inode alone."""
    snapshot = RelocationTierEvidence(
        tier="source",
        configured_path=str(tmp_path / "source.db"),
        resolved_path=str(tmp_path / "source.db"),
        backup_device=41,
        backup_inode=99,
        device=42,
        inode=99,
        size_bytes=1,
        sha256="a" * 64,
        user_version=0,
        schema_inventory_sha256="b" * 64,
        content_sha256="c" * 64,
        quick_check=("ok",),
    )
    fingerprint = {
        "device": snapshot.backup_device,
        "inode": snapshot.backup_inode,
        "size_bytes": snapshot.size_bytes,
        "sha256": snapshot.sha256,
        "user_version": snapshot.user_version,
    }
    with pytest.raises(ArchiveRootRelocationError, match="device/inode continuity"):
        _check_backup_against_live(
            tmp_path,
            manifest={"tier_source_fingerprints": {"source.db": fingerprint}},
            receipt={"tier_artifacts": [{"tier": "source", "source_fingerprint": fingerprint}]},
            snapshots=(snapshot,),
        )


def test_plan_rejects_root_device_change_with_a_coincident_inode(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full prepare route checks root device/inode continuity from authenticated evidence."""
    from polylogue.operations import archive_root_relocation as relocation

    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    real_authenticated_identity = relocation._authenticated_identity

    def changed_root_device(payload: object, *, label: str) -> tuple[int, int]:
        device, inode = real_authenticated_identity(payload, label=label)
        return (device + 1, inode) if label == "archive root" else (device, inode)

    monkeypatch.setattr(relocation, "_authenticated_identity", changed_root_device)
    with pytest.raises(ArchiveRootRelocationError, match="root device/inode continuity"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )


@pytest.mark.parametrize("leaf_kind", ["symlink", "directory", "hardlink"])
def test_current_source_evidence_rejects_unverified_live_leaves_before_sqlite_read(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leaf_kind: str
) -> None:
    """Continuity recovery validates the live source leaf before evidence collection."""
    root = tmp_path / leaf_kind
    root.mkdir()
    source = workspace_env["archive_root"] / "source.db"
    target = root / "source.db"
    if leaf_kind == "symlink":
        target.symlink_to(source)
    elif leaf_kind == "directory":
        target.mkdir()
    else:
        os.link(source, target)
    monkeypatch.setattr(
        "polylogue.operations.historical_source_continuity_recovery.capture_durable_database_evidence",
        lambda *_args: pytest.fail("live source evidence was read before leaf validation"),
    )
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="real single-linked file"):
        _current_evidence(root)


def test_historical_receipt_rejects_a_one_row_substitute_for_the_bound_operation(tmp_path: Path) -> None:
    """A small synthetic receipt cannot stand in for the 69,340-row offline operation."""
    receipt = tmp_path / "one-row.jsonl"
    old_root = tmp_path / "old"
    old_root.mkdir()
    pre_manifest = tmp_path / "pre-manifest.json"
    pre_manifest.write_text("{}", encoding="utf-8")
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
    _legacy_liveness_receipt(
        receipt,
        old_root=old_root,
        pre_manifest=pre_manifest,
        candidates=(candidate,),
    )
    digest = BlobRefLivenessCandidateDigest()
    digest.update(candidate)
    assert _validate_legacy_liveness_receipt(
        receipt, old_source_path=old_root / "source.db", pre_manifest=pre_manifest
    ) == (1, digest.hexdigest())
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="immutable offline evidence"):
        _verify_historical_operation_evidence(
            mutation_receipt=receipt,
            candidates=1,
            candidate_digest=digest.hexdigest(),
            pre_manifest=pre_manifest,
            pre_receipt=pre_manifest,
            pre_source=pre_manifest,
            post_manifest=pre_manifest,
            post_receipt=pre_manifest,
            post_source=pre_manifest,
        )


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
    updated = rebind_released_durable_train_archive_identity(
        before,
        archive_identity_digest="a" * 64,
        proof_refs=("proof:archive-root-relocation:receipt",),
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
    rebound_current_authority = rebind_released_durable_train_archive_identity(
        current_authority,
        archive_identity_digest="c" * 64,
        proof_refs=(
            "proof:archive-root-relocation:receipt-current",
            "proof:source-continuity-relocation:" + "e" * 64,
        ),
    )
    assert current_authority.source_continuity_evidence is not None
    assert rebound_current_authority.source_continuity_evidence == replace(
        current_authority.source_continuity_evidence,
        archive_identity_digest="c" * 64,
    )


def _attach_retained_source_continuity(root: Path, manifest: Path) -> None:
    """Create the exact ordinary refresh artifact retained by a recovered train."""
    train = load_durable_change_train_manifest(manifest)
    assert train.apply_evidence is not None
    with sqlite3.connect(root / "source.db") as connection:
        current = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    legacy_identity_digest = ArchiveIdentity.resolve(root).authority_identity_digest
    retained_current = replace(current, archive_identity_digest=legacy_identity_digest)
    recovered_apply_evidence = replace(
        train.apply_evidence,
        post=replace(train.apply_evidence.post, archive_identity_digest=legacy_identity_digest),
    )
    payload = {
        "format": "polylogue.source-continuity-refresh.v1",
        "operation_id": "historical-recovery",
        "evidence_ref": "proof:historical-source-continuity-recovery",
        "backup_manifest": "/authenticated/pre/manifest.json",
        "backup_manifest_sha256": "a" * 64,
        "mutation_receipt": "/authenticated/liveness.jsonl",
        "mutation_receipt_sha256": "b" * 64,
        "train_id": train.train_id,
        "source_before": _evidence_payload(recovered_apply_evidence.post),
        "source_after": _evidence_payload(retained_current),
        "refreshed_at_ms": retained_current.observed_at_ms,
    }
    digest = _canonical_json_sha256(payload)
    _write_refresh_receipt(
        root / ".maintenance-state" / "source-continuity-refreshes" / f"{digest}.json",
        {**payload, "refresh_sha256": digest},
    )
    recovered = replace(
        train,
        revision=train.revision + 1,
        apply_evidence=recovered_apply_evidence,
        source_continuity_evidence=retained_current,
        proof_refs=(*train.proof_refs, f"proof:source-continuity-refresh:{digest}"),
    )
    write_durable_change_train_manifest(manifest, recovered, expected_revision=train.revision)


def _refresh_source_continuity_without_content_change(root: Path, evidence_root: Path) -> Path:
    """Exercise the ordinary authenticated refresh route after a relocation."""
    evidence_root.mkdir(parents=True)
    with sqlite3.connect(root / "source.db") as connection:
        before = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    backup_manifest = evidence_root / "refresh-backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    operation_id = BlobRefLivenessCandidateDigest().hexdigest()
    mutation_receipt = evidence_root / "refresh-mutation-receipt.jsonl"
    mutation_receipt.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(root / "source.db"),
                "backup_manifest": str(backup_manifest),
                "candidate_count": 0,
                "candidate_digest": operation_id,
                "backup_manifest_sha256": hashlib.sha256(backup_manifest.read_bytes()).hexdigest(),
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "committed",
                "deleted_count": 0,
                "post_orphaned_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return refresh_released_source_train_continuity(
        root,
        mutation_receipt=mutation_receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id=operation_id,
        evidence_ref="proof:post-relocation-source-maintenance",
    )


def _substitute_foreign_historical_refresh(
    root: Path,
    *,
    plan: dict[str, object],
    mutation_receipt: Path,
    backup_manifest: Path,
) -> Path:
    """Install a valid but non-plan-owned continuity revision for rejection tests."""
    train_path = Path(str(plan["source_train_path"]))
    train = load_durable_change_train_manifest(train_path)
    source_before = plan["source_before"]
    source_after = plan["source_after"]
    assert isinstance(source_before, dict) and isinstance(source_after, dict)
    observed_at_ms = source_after.get("observed_at_ms")
    assert type(observed_at_ms) is int
    payload = {
        "format": "polylogue.source-continuity-refresh.v1",
        "operation_id": "foreign",
        "evidence_ref": "proof:foreign-continuity",
        "backup_manifest": str(backup_manifest),
        "backup_manifest_sha256": _sha256(backup_manifest),
        "mutation_receipt": str(mutation_receipt),
        "mutation_receipt_sha256": _sha256(mutation_receipt),
        "train_id": train.train_id,
        "source_before": source_before,
        "source_after": source_after,
        "refreshed_at_ms": observed_at_ms,
    }
    digest = _canonical_json_sha256(payload)
    _write_refresh_receipt(
        root / ".maintenance-state" / "source-continuity-refreshes" / f"{digest}.json",
        {**payload, "refresh_sha256": digest},
    )
    substituted = recover_released_source_train_continuity(
        train,
        current_evidence=_evidence_from_payload(source_after),
        proof_ref=f"proof:source-continuity-refresh:{digest}",
    )
    write_durable_change_train_manifest(train_path, substituted, expected_revision=train.revision)
    return train_path


def _evidence_from_payload(payload: dict[str, object]) -> DurableDatabaseEvidence:
    from polylogue.operations.historical_source_continuity_recovery import _evidence_from_plan

    return _evidence_from_plan(payload)


def _evidence_payload(evidence: object) -> dict[str, object]:
    from polylogue.operations.historical_source_continuity_recovery import _evidence_payload as render

    return render(evidence)  # type: ignore[arg-type]


def _released_moved_source_train(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, include_orphan_blob_ref: bool = False
) -> Path:
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
    assert released.proof is not None
    source_post = released.apply_evidence.post
    source_proof = released.proof
    if include_orphan_blob_ref:
        with sqlite3.connect(source) as connection:
            connection.executescript(
                """
                CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB) STRICT;
                CREATE TABLE blob_refs (
                    blob_hash BLOB NOT NULL,
                    ref_type TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    source_path TEXT,
                    size_bytes INTEGER NOT NULL,
                    acquired_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (blob_hash, ref_type, ref_id)
                ) STRICT;
                INSERT INTO blob_refs VALUES (X'02', 'attachment', 'deleted', NULL, 2, 2);
                """
            )
            source_post = replace(
                capture_durable_database_evidence(connection, ArchiveTier.SOURCE),
                observed_at_ms=released.apply_evidence.post.observed_at_ms,
            )
        source_proof = replace(
            released.proof,
            fresh_ddl_parity=replace(
                released.proof.fresh_ddl_parity,
                migrated_inventory_sha256=source_post.schema_inventory_sha256,
            ),
            restart_convergence=replace(
                released.proof.restart_convergence,
                observed_schema_inventory_sha256=source_post.schema_inventory_sha256,
            ),
        )
    historical = replace(
        released,
        apply_evidence=replace(
            released.apply_evidence,
            post=replace(
                source_post,
                archive_identity_digest=ArchiveIdentity.resolve(root).authority_identity_digest,
            ),
        ),
        proof=source_proof,
    )
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    (manifest_root / ".bootstrap").unlink()
    manifest = manifest_root / "source-002.json"
    write_durable_change_train_manifest(manifest, historical, expected_revision=-1)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 1)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.AUDIT, 10_000)
    return manifest


def _released_moved_durable_train(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
) -> Path:
    """Build one real released non-source train for complete relocation coverage."""
    from tests.unit.storage import test_durable_change_train as trains

    database = root / f"{tier.value}.db"
    database.unlink()
    trains._create_current_database(database)
    trains._install_synthetic_migration(root.parent, monkeypatch, tier)
    train = trains._admitted(tier)
    with sqlite3.connect(database) as connection:
        train = trains._reserve_and_authorize(connection, train, archive_root=root)
        train = apply_durable_change_train(connection, train)
    train = record_durable_writer_release(train, evidence_ref=f"proof:{tier.value}-writer-release")
    with sqlite3.connect(database) as connection:
        restart = capture_durable_restart_convergence(
            connection,
            train,
            runtime_consumers=trains._runtime_results(),
            evidence_ref=f"proof:{tier.value}-restart",
        )
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=trains._parity(tier),
        runtime_consumers=trains._runtime_results(),
        restart_convergence=restart,
    )
    released = release_durable_change_train(train, evidence_ref=f"proof:{tier.value}-released")
    manifest = root / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-002.json"
    write_durable_change_train_manifest(manifest, released, expected_revision=-1)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, tier, 1)
    return manifest


def _clone_released_durable_train_for_tier(
    root: Path,
    source_manifest: Path,
    tier: ArchiveTier,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ...]:
    """Retarget one validated released fixture train across a tier's whole chain.

    Relocation requires a released manifest for *every* target in
    ``range(adoption_floor + 1, live_user_version + 1)``. Both bounds are read
    from production here -- the floor from its declared table, the head from
    the tier database's own ``PRAGMA user_version`` -- because a literal is
    what broke this fixture when audit gained migration 003: it kept emitting
    only ``audit-002.json`` while relocation had started expecting 2 *and* 3.
    """
    source = load_durable_change_train_manifest(source_manifest)
    assert source.fresh_ddl_parity is not None
    assert source.reservation is not None
    assert source.backup_authorization is not None
    assert source.pre_apply_evidence is not None
    assert source.apply_evidence is not None
    assert source.proof is not None

    # The cloned train begins at v1, so that is this fixture's chain floor --
    # declared once here and then read back, so the targets below are the same
    # set relocation itself computes from this table.
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, tier, 1)
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
    tier_path = root / f"{tier.value}.db"
    with closing(sqlite3.connect(tier_path)) as connection:
        live_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    targets = range(floor + 1, live_version + 1)
    assert targets, f"no durable train targets for {tier.value} at v{live_version} over floor {floor}"

    def _parity(parity: Any, version: int) -> Any:
        return replace(
            parity,
            tier=tier,
            target_version=version,
            migrated_version=version,
            fresh_version=version,
        )

    written: list[Path] = []
    for target in targets:
        pre = replace(source.pre_apply_evidence, tier=tier, user_version=target - 1)
        cloned = replace(
            source,
            train_id=f"train:{tier.value}:v{target}",
            tier=tier,
            current_version=target - 1,
            target_version=target,
            slot=target,
            migration=replace(source.migration, tier=tier, target_version=target, slot=target),
            fresh_ddl_parity=_parity(source.fresh_ddl_parity, target),
            reservation=replace(source.reservation, tier_path=str(tier_path)),
            backup_authorization=replace(
                source.backup_authorization,
                live_tier_path=str(tier_path),
                live_user_version=target - 1,
            ),
            pre_apply_evidence=pre,
            # apply_evidence.pre must be identical to pre_apply_evidence.
            apply_evidence=replace(
                source.apply_evidence,
                pre=pre,
                post=replace(source.apply_evidence.post, tier=tier, user_version=target),
                migration_result=replace(
                    source.apply_evidence.migration_result,
                    tier=tier,
                    from_version=target - 1,
                    to_version=target,
                    applied_versions=(target,),
                ),
            ),
            proof=replace(
                source.proof,
                fresh_ddl_parity=_parity(source.proof.fresh_ddl_parity, target),
                # The restart observation must report the version the train
                # lands on (migration_runner.py:3610), not the source's.
                restart_convergence=replace(source.proof.restart_convergence, observed_user_version=target),
            ),
        )
        manifest = root / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-{target:03d}.json"
        write_durable_change_train_manifest(manifest, cloned, expected_revision=-1)
        written.append(manifest)
    return tuple(written)


def _activate_movable_index_generation(root: Path) -> Path:
    """Promote a real generation using the production absolute symlink layout."""
    store = IndexGenerationStore.for_archive_root(root)
    generation = store.create(owner_id="relocation-test", source_snapshot="snapshot")
    store.promote(generation)
    return Path(generation.index_path).resolve(strict=True)


def _prepare_moved_root_relocation_with_generation(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ArchiveRootRelocationPlan]:
    """Prepare the public moved-root relocation sequence with a retained generation."""
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    _activate_movable_index_generation(old_root)
    _attach_retained_source_continuity(old_root, manifest)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert plan.index_generations
    return new_root, plan


def _legacy_liveness_receipt(
    path: Path,
    *,
    old_root: Path,
    pre_manifest: Path,
    candidates: tuple[BlobRefLivenessCandidate, ...],
) -> None:
    """Encode the exact pre-#3868 shape: no backup digest or postcondition field."""
    digest = BlobRefLivenessCandidateDigest()
    for candidate in candidates:
        digest.update(candidate)
    records = [
        {
            "kind": "blob_ref_liveness_reconciliation",
            "phase": "prepared",
            "source_db": str(old_root / "source.db"),
            "backup_manifest": str(pre_manifest),
            "candidate_count": len(candidates),
            "candidate_digest": digest.hexdigest(),
        },
        *({"kind": "candidate", **candidate.to_dict()} for candidate in candidates),
        {
            "kind": "blob_ref_liveness_reconciliation",
            "phase": "committed",
            "deleted_count": len(candidates),
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _pinned_historical_operation_evidence(
    path: Path,
    *,
    mutation_receipt: Path,
    candidates: tuple[BlobRefLivenessCandidate, ...],
    pre_manifest: Path,
    post_manifest: Path,
) -> None:
    """Write the fixture's immutable-shaped descriptor from independently produced artifacts."""
    digest = BlobRefLivenessCandidateDigest()
    for candidate in candidates:
        digest.update(candidate)
    payload = {
        "format": "polylogue.historical-source-continuity-operation-evidence.v1",
        "operation": "blob-ref-liveness-reconciliation-20260807",
        "mutation_receipt_sha256": _sha256(mutation_receipt),
        "candidate_count": len(candidates),
        "candidate_digest": digest.hexdigest(),
        "pre_backup_manifest_sha256": _sha256(pre_manifest),
        "pre_backup_receipt_sha256": _sha256(pre_manifest.parent / "verification-receipt.json"),
        "pre_source_sha256": _sha256(pre_manifest.parent / "source.db"),
        "post_backup_manifest_sha256": _sha256(post_manifest),
        "post_backup_receipt_sha256": _sha256(post_manifest.parent / "verification-receipt.json"),
        "post_source_sha256": _sha256(post_manifest.parent / "source.db"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _historical_continuity_fixture(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    """Build real backups, a legacy receipt, a released train, and the pinned fixture descriptor."""
    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch, include_orphan_blob_ref=True)
    pre_backup = backup_archive(output_dir=tmp_path / "pre", profile="rebuildable_cache_exclude", verify=True)
    assert pre_backup.ok and pre_backup.output_path is not None
    pre_manifest = Path(pre_backup.output_path) / "manifest.json"
    with sqlite3.connect(f"file:{old_root / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
        prior = classify_blob_ref_liveness(connection)
    assert prior.orphaned_count == 1
    mutation_receipt = tmp_path / "legacy-liveness.jsonl"
    _legacy_liveness_receipt(
        mutation_receipt,
        old_root=old_root,
        pre_manifest=pre_manifest,
        candidates=prior.candidates,
    )
    with sqlite3.connect(old_root / "source.db") as connection:
        connection.execute("DELETE FROM blob_refs WHERE ref_id = 'deleted'")
    post_backup = backup_archive(output_dir=tmp_path / "post", profile="rebuildable_cache_exclude", verify=True)
    assert post_backup.ok and post_backup.output_path is not None
    post_manifest = Path(post_backup.output_path) / "manifest.json"
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    evidence = tmp_path / "pinned-historical-evidence.json"
    _pinned_historical_operation_evidence(
        evidence,
        mutation_receipt=mutation_receipt,
        candidates=prior.candidates,
        pre_manifest=pre_manifest,
        post_manifest=post_manifest,
    )
    return new_root, mutation_receipt, pre_manifest, post_manifest, evidence


def _downgrade_historical_backup_source_identity(manifest_path: Path, *, old_root: Path) -> None:
    """Render a verified pre-inode backup shape using the original local keys."""
    from polylogue.storage.backup_attestation import sign_verification_receipt

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = manifest_path.with_name("verification-receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    fingerprint = manifest["tier_source_fingerprints"]["source.db"]
    fingerprint.pop("device")
    fingerprint.pop("inode")
    for artifact in receipt["tier_artifacts"]:
        if artifact["tier"] == "source":
            artifact["source_fingerprint"] = fingerprint
    manifest_encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_encoded)
    receipt["manifest_size_bytes"] = len(manifest_encoded)
    receipt["manifest_sha256"] = hashlib.sha256(manifest_encoded).hexdigest()
    for item in receipt["artifact_inventory"]:
        if item.get("path") == "manifest.json":
            item["size_bytes"] = len(manifest_encoded)
            item["sha256"] = receipt["manifest_sha256"]
    unsigned = {key: value for key, value in receipt.items() if key != "attestations"}
    receipt_path.write_text(
        json.dumps(
            sign_verification_receipt(
                unsigned,
                authority_paths={
                    "source": old_root / "source.db",
                    "user": old_root / "user.db",
                    "audit": old_root / "audit.db",
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_historical_recovery_consumes_verified_pre_inode_backup_manifests(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery planner supports issued backups that predate inode fields.

    Anti-vacuity: both real backup manifests retain their successful HMAC
    receipts and exact package bytes, but omit only the fields introduced by
    this PR. Requiring the new fields makes an already-completed historical
    mutation impossible to recover.
    """
    new_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    old_root = workspace_env["archive_root"]
    _downgrade_historical_backup_source_identity(pre_manifest, old_root=old_root)
    _downgrade_historical_backup_source_identity(post_manifest, old_root=old_root)
    with sqlite3.connect(f"file:{pre_manifest.parent / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
        candidates = classify_blob_ref_liveness(connection).candidates
    _pinned_historical_operation_evidence(
        evidence,
        mutation_receipt=mutation_receipt,
        candidates=candidates,
        pre_manifest=pre_manifest,
        post_manifest=post_manifest,
    )

    with _test_historical_operation_evidence_resource(evidence):
        plan = prepare_historical_source_continuity_recovery(
            old_root=old_root,
            new_root=new_root,
            mutation_receipt=mutation_receipt,
            pre_backup_manifest=pre_manifest,
            post_backup_manifest=post_manifest,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )
        result = apply_historical_source_continuity_recovery(
            root=new_root,
            plan=plan,
            authorization=plan.plan_sha256,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    assert plan.pre_backup_source_device is None
    assert plan.post_backup_source_inode is None
    assert result.state == "committed"


def test_historical_recovery_loads_non_ascii_legacy_v2_artifacts(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V2 plan and receipt checksums retain the original ASCII-escaped JSON identity.

    Anti-vacuity: paths and the resume command contain non-ASCII text and the
    sealed checksums are computed with the historical ``ensure_ascii=True``
    encoding. Using the migration-runner canonicalizer rejects both retained
    artifacts before an interrupted recovery can resume.
    """
    new_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    old_root = workspace_env["archive_root"]
    with sqlite3.connect(f"file:{pre_manifest.parent / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
        candidates = classify_blob_ref_liveness(connection).candidates
    _pinned_historical_operation_evidence(
        evidence,
        mutation_receipt=mutation_receipt,
        candidates=candidates,
        pre_manifest=pre_manifest,
        post_manifest=post_manifest,
    )
    with _test_historical_operation_evidence_resource(evidence):
        plan = prepare_historical_source_continuity_recovery(
            old_root=old_root,
            new_root=new_root,
            mutation_receipt=mutation_receipt,
            pre_backup_manifest=pre_manifest,
            post_backup_manifest=post_manifest,
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )

    legacy_payload = plan.model_dump(mode="json", exclude={"plan_sha256"})
    legacy_payload["old_configured_root"] = str(tmp_path / "źródło")
    plan_sha256 = hashlib.sha256(
        json.dumps(legacy_payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    retained_plan = tmp_path / "plan-źródło.json"
    retained_plan.write_text(
        json.dumps({**legacy_payload, "plan_sha256": plan_sha256}, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded_plan = load_historical_source_continuity_recovery_plan(retained_plan)
    assert loaded_plan.plan_sha256 == plan_sha256

    receipt_payload = {
        "state": "prepared",
        "revision": 0,
        "plan_sha256": plan_sha256,
        "authorization": plan_sha256,
        "train_before_sha256": plan.source_train_sha256,
        "train_after_sha256": plan.source_train_after_sha256,
        "refresh_receipt_sha256": plan.refresh_receipt_sha256,
        "resume_command": "polylogue ops maintenance source-continuity-recovery apply --plan /tmp/źródło.json",
    }
    receipt_sha256 = hashlib.sha256(
        json.dumps(
            {"format": "polylogue.historical-source-continuity-recovery-receipt.v1", **receipt_payload},
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt = HistoricalSourceContinuityRecoveryReceipt.model_validate(
        {
            "format": "polylogue.historical-source-continuity-recovery-receipt.v1",
            **receipt_payload,
            "receipt_sha256": receipt_sha256,
        }
    )
    receipt_path = new_root / ".maintenance-state" / "historical-source-continuity-recoveries" / f"{plan_sha256}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    assert load_historical_source_continuity_recovery_receipt(receipt_path) == receipt


def _maintenance_json_output(output: str) -> dict[str, object]:
    """Maintenance commands retain the root-provenance line before JSON output."""
    _provenance, separator, payload = output.partition("\n")
    assert separator and payload.startswith("{")
    decoded = json.loads(payload)
    assert isinstance(decoded, dict)
    return decoded


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
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="unsafe"):
        _write_refresh_receipt(
            state / "source-continuity-refreshes" / ("a" * 64 + ".json"),
            {"refresh_sha256": "a" * 64},
        )
    assert not tuple(target.iterdir())


def test_ordinary_source_continuity_refresh_rejects_symlink_receipt_directory(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary refresh writer cannot publish through a foreign directory.

    Anti-vacuity: the real post-maintenance refresh route reaches receipt
    publication with a released source train. A path-based mkdir/replace
    implementation writes the new v2 authority into ``outside``.
    """
    root = workspace_env["archive_root"]
    _released_moved_source_train(root, monkeypatch)
    outside = tmp_path / "outside-refreshes"
    outside.mkdir()
    refresh_root = root / ".maintenance-state" / "source-continuity-refreshes"
    refresh_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DurableChangeTrainError, match="cannot persist source continuity refresh receipt"):
        _refresh_source_continuity_without_content_change(root, tmp_path / "refresh-evidence")

    assert not tuple(outside.iterdir())


def test_refresh_only_authority_chain_requires_exact_manifest_predecessors(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two retained v2 refreshes must form one exact manifest-hash chain.

    Anti-vacuity: source-mutation admission invokes the production released
    train validator. Without refresh-only transition validation, a terminal
    receipt can preserve all evidence and final fields while substituting an
    unrelated, well-formed predecessor manifest hash.
    """
    from polylogue.storage.sqlite import durable_change_train as trains

    root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(root, monkeypatch)
    _refresh_source_continuity_without_content_change(root, tmp_path / "first-refresh")
    terminal_path = _refresh_source_continuity_without_content_change(root, tmp_path / "second-refresh")
    train = load_durable_change_train_manifest(manifest)

    terminal_receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
    old_digest = terminal_receipt.pop("refresh_sha256")
    assert isinstance(old_digest, str)
    terminal_receipt["train_before_sha256"] = "f" * 64
    substituted_digest = _canonical_json_sha256(terminal_receipt)
    substituted_receipt = {**terminal_receipt, "refresh_sha256": substituted_digest}
    substituted_path = terminal_path.with_name(f"{substituted_digest}.json")
    _write_refresh_receipt(substituted_path, substituted_receipt)

    intent = trains._source_continuity_refresh_intent(terminal_receipt, train_id=train.train_id)
    substituted_train = trains._finalize_source_continuity_refresh_intent(
        intent,
        refresh_digest=substituted_digest,
    )
    manifest.write_text(
        json.dumps(durable_change_train_to_payload(substituted_train), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DurableChangeTrainError, match="proof chain has no unique predecessor"):
        assert_source_continuity_apply_allowed(root)


def test_source_continuity_rejects_a_disconnected_legacy_authority_component(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every retained V1/V2/relocation authority must belong to one chain.

    Anti-vacuity: source-mutation admission calls the production continuity
    graph validator. The older terminal-only check accepted an unrelated V1
    root whenever the legitimate V1 authority alone matched current evidence.
    """
    root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(root, monkeypatch)
    _attach_retained_source_continuity(root, manifest)
    train = load_durable_change_train_manifest(manifest)
    retained_digest = next(
        ref.rsplit(":", 1)[-1] for ref in train.proof_refs if ref.startswith("proof:source-continuity-refresh:")
    )
    retained_path = root / ".maintenance-state" / "source-continuity-refreshes" / f"{retained_digest}.json"
    foreign_payload = json.loads(retained_path.read_text(encoding="utf-8"))
    foreign_payload.pop("refresh_sha256")
    foreign_payload["operation_id"] = "disconnected-legacy-authority"
    foreign_after = dict(foreign_payload["source_after"])
    foreign_after["content_sha256"] = "f" * 64
    foreign_payload["source_after"] = foreign_after
    foreign_digest = _canonical_json_sha256(foreign_payload)
    _write_refresh_receipt(
        retained_path.with_name(f"{foreign_digest}.json"),
        {**foreign_payload, "refresh_sha256": foreign_digest},
    )
    disconnected = replace(
        train,
        revision=train.revision + 1,
        proof_refs=(*train.proof_refs, f"proof:source-continuity-refresh:{foreign_digest}"),
    )
    write_durable_change_train_manifest(manifest, disconnected, expected_revision=train.revision)

    with pytest.raises(DurableChangeTrainError, match="one connected authority chain"):
        assert_source_continuity_apply_allowed(root)


def test_source_continuity_admits_connected_multi_refresh_v1_history(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy V1 refreshes remain usable when sealed evidence forms one chain.

    Anti-vacuity: the production admission route validates every retained V1
    receipt. Removing V1 predecessor inference makes these two independently
    sealed, sequential refreshes look like disconnected roots and rejects the
    released train.
    """
    root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(root, monkeypatch)
    _attach_retained_source_continuity(root, manifest)
    first = load_durable_change_train_manifest(manifest)
    assert first.source_continuity_evidence is not None
    with sqlite3.connect(root / "source.db") as connection:
        observed = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    second = replace(
        observed,
        archive_identity_digest=first.source_continuity_evidence.archive_identity_digest,
        observed_at_ms=first.source_continuity_evidence.observed_at_ms + 3,
    )
    second_payload = {
        "format": "polylogue.source-continuity-refresh.v1",
        "operation_id": "legacy-second-refresh",
        "evidence_ref": "proof:legacy-second-refresh",
        "backup_manifest": "/authenticated/second/manifest.json",
        "backup_manifest_sha256": "c" * 64,
        "mutation_receipt": "/authenticated/second.jsonl",
        "mutation_receipt_sha256": "d" * 64,
        "train_id": first.train_id,
        "source_before": _evidence_payload(
            replace(
                first.source_continuity_evidence, observed_at_ms=first.source_continuity_evidence.observed_at_ms + 2
            )
        ),
        "source_after": _evidence_payload(second),
        "refreshed_at_ms": second.observed_at_ms,
    }
    second_digest = _canonical_json_sha256(second_payload)
    _write_refresh_receipt(
        root / ".maintenance-state" / "source-continuity-refreshes" / f"{second_digest}.json",
        {**second_payload, "refresh_sha256": second_digest},
    )
    updated = replace(
        first,
        revision=first.revision + 1,
        source_continuity_evidence=second,
        proof_refs=(*first.proof_refs, f"proof:source-continuity-refresh:{second_digest}"),
    )
    write_durable_change_train_manifest(manifest, updated, expected_revision=first.revision)

    third = replace(second, observed_at_ms=second.observed_at_ms + 3)
    third_payload = {
        **second_payload,
        "operation_id": "legacy-third-refresh",
        "evidence_ref": "proof:legacy-third-refresh",
        "source_before": _evidence_payload(replace(second, observed_at_ms=second.observed_at_ms + 2)),
        "source_after": _evidence_payload(third),
        "refreshed_at_ms": third.observed_at_ms,
    }
    third_digest = _canonical_json_sha256(third_payload)
    _write_refresh_receipt(
        root / ".maintenance-state" / "source-continuity-refreshes" / f"{third_digest}.json",
        {**third_payload, "refresh_sha256": third_digest},
    )
    terminal = replace(
        updated,
        revision=updated.revision + 1,
        source_continuity_evidence=third,
        proof_refs=(*updated.proof_refs, f"proof:source-continuity-refresh:{third_digest}"),
    )
    write_durable_change_train_manifest(manifest, terminal, expected_revision=updated.revision)

    assert_source_continuity_apply_allowed(root)


def test_relocation_revalidation_rejects_an_unbound_prepared_receipt_for_post_state(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-state leaves need the exact retained preparation receipt, not a shaped receipt.

    Anti-vacuity: the generation metadata is atomically replaced with the
    planned after bytes and a revision-1 receipt has the expected structural
    fields but a foreign preparation digest. The old structural gate admitted
    it before the first publication attempt.
    """
    from polylogue.operations import archive_root_relocation as relocation

    new_root, plan = _prepare_moved_root_relocation_with_generation(workspace_env, tmp_path, monkeypatch)
    generation = plan.index_generations[0]
    metadata_path = Path(generation.metadata_path)
    before = metadata_path.read_bytes()
    payload = relocation._index_generation_payload_for_state(generation, after=False, encoded=before)
    after = relocation._index_generation_metadata_bytes(
        {**payload, "archive_root": generation.after_archive_root, "index_path": generation.after_index_path}
    )
    replacement = metadata_path.with_name(".generation.json.after")
    replacement.write_bytes(after)
    os.replace(replacement, metadata_path)

    pointer_fields = relocation._pointer_receipt_fields(plan.active_index_pointer)
    unbound = _sealed_relocation_receipt(
        state="prepared",
        revision=1,
        plan_sha256=plan.plan_sha256,
        authorization=plan.plan_sha256,
        manifest_before_sha256=tuple(item.before_manifest_sha256 for item in plan.durable_trains),
        manifest_after_sha256=tuple("0" * 64 for _item in plan.durable_trains),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command="polylogue ops maintenance archive-root-relocation apply",
        prepared_receipt_sha256="f" * 64,
    )
    _write_relocation_receipt(relocation._receipt_path(new_root, plan), unbound, expected=None)

    with pytest.raises(ArchiveRootRelocationError, match="post-publication state without a prepared receipt"):
        relocation._revalidate_plan_live_state(new_root, plan)


def test_receipt_directory_swap_cannot_redirect_either_operation_outside_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child swapped to a symlink after mkdir is rejected before external writes."""
    root = tmp_path / "archive"
    state = root / ".maintenance-state"
    state.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdir = os.mkdir

    def swapped_mkdir(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real_mkdir(path, mode=mode, dir_fd=dir_fd)
        os.rmdir(path, dir_fd=dir_fd)
        os.symlink(outside, path, target_is_directory=True, dir_fd=dir_fd)

    relocation = _sealed_relocation_receipt(
        state="prepared",
        revision=0,
        plan_sha256="a" * 64,
        authorization="a" * 64,
        manifest_before_sha256=(),
        manifest_after_sha256=(),
        resume_command="resume relocation",
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "mkdir", swapped_mkdir)
        with pytest.raises(ArchiveRootRelocationError, match="unsafe"):
            _write_relocation_receipt(
                state / "archive-root-relocations" / ("a" * 64 + ".json"),
                relocation,
                expected=None,
            )
    (state / "archive-root-relocations").unlink()

    continuity = _sealed_continuity_receipt(
        state="prepared",
        revision=0,
        plan_sha256="b" * 64,
        authorization="b" * 64,
        train_before_sha256="c" * 64,
        train_after_sha256="e" * 64,
        refresh_receipt_sha256="d" * 64,
        resume_command="resume continuity",
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "mkdir", swapped_mkdir)
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="unsafe"):
            _write_continuity_receipt(
                state / "historical-source-continuity-recoveries" / ("b" * 64 + ".json"),
                continuity,
                expected=None,
            )

    assert not tuple(outside.iterdir())


def test_receipt_writers_never_create_through_a_symlinked_maintenance_state(tmp_path: Path) -> None:
    """Missing receipt children cannot make ``mkdir`` traverse an external state target."""
    outside = tmp_path / "outside-state"
    outside.mkdir()
    relocation_root = tmp_path / "relocation-archive"
    relocation_root.mkdir()
    (relocation_root / ".maintenance-state").symlink_to(outside, target_is_directory=True)
    relocation = _sealed_relocation_receipt(
        state="prepared",
        revision=0,
        plan_sha256="e" * 64,
        authorization="e" * 64,
        manifest_before_sha256=(),
        manifest_after_sha256=(),
        resume_command="resume relocation",
    )
    with pytest.raises(ArchiveRootRelocationError, match="unsafe"):
        _write_relocation_receipt(
            relocation_root / ".maintenance-state" / "archive-root-relocations" / ("e" * 64 + ".json"),
            relocation,
            expected=None,
        )

    continuity_root = tmp_path / "continuity-archive"
    continuity_root.mkdir()
    (continuity_root / ".maintenance-state").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="unsafe"):
        _write_refresh_receipt(
            continuity_root / ".maintenance-state" / "source-continuity-refreshes" / ("f" * 64 + ".json"),
            {"refresh_sha256": "f" * 64},
        )
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="unsafe"):
        _write_continuity_receipt(
            continuity_root / ".maintenance-state" / "historical-source-continuity-recoveries" / ("f" * 64 + ".json"),
            _sealed_continuity_receipt(
                state="prepared",
                revision=0,
                plan_sha256="f" * 64,
                authorization="f" * 64,
                train_before_sha256="0" * 64,
                train_after_sha256="2" * 64,
                refresh_receipt_sha256="1" * 64,
                resume_command="resume continuity",
            ),
            expected=None,
        )

    assert not tuple(outside.iterdir())


def test_relocation_startup_reader_rejects_receipt_swapped_after_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon preflight reads the enumerated relocation receipt descriptor, not a replacement pathname."""
    root = tmp_path / "archive"
    state = root / ".maintenance-state" / "archive-root-relocations"
    state.mkdir(parents=True)
    receipt = state / ("a" * 64 + ".json")
    substitute = state / "replacement.json"
    _write_relocation_receipt(
        receipt,
        _sealed_relocation_receipt(
            state="prepared",
            revision=0,
            plan_sha256="a" * 64,
            authorization="a" * 64,
            manifest_before_sha256=(),
            manifest_after_sha256=(),
            resume_command="resume relocation",
        ),
        expected=None,
    )
    _write_relocation_receipt(
        substitute,
        _sealed_relocation_receipt(
            state="committed",
            revision=1,
            plan_sha256="b" * 64,
            authorization="b" * 64,
            manifest_before_sha256=(),
            manifest_after_sha256=(),
            resume_command="replacement",
        ),
        expected=None,
    )
    real_open = os.open
    swapped = False

    def swap_after_enumeration(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == receipt.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            os.replace(substitute, receipt)
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", swap_after_enumeration)
    with pytest.raises(ArchiveRootRelocationError, match="changed during pinned enumeration"):
        assert_no_prepared_archive_root_relocation(root)
    assert swapped


def test_historical_startup_reader_rejects_receipt_swapped_after_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical recovery preflight also pins the receipt it enumerated."""
    root = tmp_path / "archive"
    state = root / ".maintenance-state" / "historical-source-continuity-recoveries"
    state.mkdir(parents=True)
    receipt = state / ("c" * 64 + ".json")
    substitute = state / "replacement.json"
    _write_continuity_receipt(
        receipt,
        _sealed_continuity_receipt(
            state="prepared",
            revision=0,
            plan_sha256="c" * 64,
            authorization="c" * 64,
            train_before_sha256="d" * 64,
            train_after_sha256="f" * 64,
            refresh_receipt_sha256="e" * 64,
            resume_command="resume continuity",
        ),
        expected=None,
    )
    _write_continuity_receipt(
        substitute,
        _sealed_continuity_receipt(
            state="committed",
            revision=1,
            plan_sha256="f" * 64,
            authorization="f" * 64,
            train_before_sha256="0" * 64,
            train_after_sha256="1" * 64,
            refresh_receipt_sha256="2" * 64,
            resume_command="replacement",
        ),
        expected=None,
    )
    real_open = os.open
    swapped = False

    def swap_after_enumeration(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == receipt.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            os.replace(substitute, receipt)
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", swap_after_enumeration)
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="changed during pinned enumeration"):
        assert_no_prepared_historical_source_continuity_recovery(root)
    assert swapped


def test_source_mutation_admission_fences_each_prepared_maintenance_operation(workspace_env: dict[str, Path]) -> None:
    """The shared source-mutation boundary cannot invalidate either resume path.

    Anti-vacuity: both receipts are production-sealed files under the archive's
    maintenance root. Calling the shared admission function proves the same
    guard used by offline source mutations rejects them before SQLite work.
    """
    root = workspace_env["archive_root"]
    relocation_path = root / ".maintenance-state" / "archive-root-relocations" / ("a" * 64 + ".json")
    _write_relocation_receipt(
        relocation_path,
        _sealed_relocation_receipt(
            state="prepared",
            revision=0,
            plan_sha256="a" * 64,
            authorization="a" * 64,
            manifest_before_sha256=(),
            manifest_after_sha256=(),
            resume_command="resume relocation",
        ),
        expected=None,
    )
    with pytest.raises(ArchiveRootRelocationError, match="prepared but incomplete"):
        assert_source_continuity_apply_allowed(root)

    relocation_path.unlink()
    recovery_path = root / ".maintenance-state" / "historical-source-continuity-recoveries" / ("b" * 64 + ".json")
    _write_continuity_receipt(
        recovery_path,
        _sealed_continuity_receipt(
            state="prepared",
            revision=0,
            plan_sha256="b" * 64,
            authorization="b" * 64,
            train_before_sha256="c" * 64,
            train_after_sha256="d" * 64,
            refresh_receipt_sha256="e" * 64,
            resume_command="resume continuity",
        ),
        expected=None,
    )
    with pytest.raises(HistoricalSourceContinuityRecoveryError, match="prepared but incomplete"):
        assert_source_continuity_apply_allowed(root)


def test_historical_continuity_recovery_cli_rejects_an_unbound_synthetic_operation(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production CLI refuses a file-backed substitute for the attested operation."""

    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch, include_orphan_blob_ref=True)
    pre_backup = backup_archive(output_dir=tmp_path / "pre", profile="rebuildable_cache_exclude", verify=True)
    assert pre_backup.ok and pre_backup.output_path is not None
    with sqlite3.connect(f"file:{old_root / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
        prior = classify_blob_ref_liveness(connection)
    assert prior.orphaned_count == 1
    assert prior.candidates[0].ref_id == "deleted"
    pre_manifest = Path(pre_backup.output_path) / "manifest.json"
    legacy_receipt = tmp_path / "legacy-liveness.jsonl"
    _legacy_liveness_receipt(
        legacy_receipt,
        old_root=old_root,
        pre_manifest=pre_manifest,
        candidates=prior.candidates,
    )
    post_backup = backup_archive(output_dir=tmp_path / "post", profile="rebuildable_cache_exclude", verify=True)
    assert post_backup.ok and post_backup.output_path is not None
    post_manifest = Path(post_backup.output_path) / "manifest.json"
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)

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
    assert plan_result.exit_code != 0
    assert "immutable offline evidence" in plan_result.output


def test_historical_continuity_recovery_cli_recovers_pinned_fixture_and_resumes_crashes(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real CLI bridge through prepared and refresh-publication interruptions.

    Anti-vacuity: the fixture's descriptor only authorizes independently made
    backup, receipt, and SQLite artifacts.  The test invokes the public plan
    and apply routes, then inspects the production refresh receipt and durable
    train CAS result.  Removing either route's operation wiring leaves no plan,
    no prepared admission block, or no train revision.
    """
    from polylogue.operations import historical_source_continuity_recovery as recovery

    new_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    shell_sensitive_root = tmp_path / "moved root;still-one-argument"
    os.rename(new_root, shell_sensitive_root)
    new_root = shell_sensitive_root
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(new_root)}
    plan_path = tmp_path / "continuity-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
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
        assert planned.exit_code == 0, planned.output
        plan_payload = _maintenance_json_output(planned.output)
        plan_sha256 = str(plan_payload["plan_sha256"])
        plan_train = Path(str(plan_payload["source_train_path"]))
        train_before = load_durable_change_train_manifest(plan_train)
        real_write_refresh = recovery._write_refresh_receipt

        def crash_before_refresh(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("crash after prepared receipt")

        monkeypatch.setattr(recovery, "_write_refresh_receipt", crash_before_refresh)
        with pytest.raises(RuntimeError, match="crash after prepared receipt"):
            CliRunner().invoke(
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
                    plan_sha256,
                    "--output-format",
                    "json",
                ],
                env=command_env,
                catch_exceptions=False,
            )
        retained_plan = (
            new_root / ".maintenance-state" / "historical-source-continuity-recovery-plans" / f"{plan_sha256}.json"
        )
        assert retained_plan.read_bytes() == plan_path.read_bytes()
        prepared_receipt = json.loads(
            (
                new_root / ".maintenance-state" / "historical-source-continuity-recoveries" / f"{plan_sha256}.json"
            ).read_text(encoding="utf-8")
        )
        assert f"POLYLOGUE_ARCHIVE_ROOT={shlex.quote(str(new_root))}" in prepared_receipt["resume_command"]
        assert f"--plan {shlex.quote(str(retained_plan))}" in prepared_receipt["resume_command"]
        plan_path.unlink()
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="prepared but incomplete"):
            assert_no_prepared_historical_source_continuity_recovery(new_root)
        from polylogue.daemon import cli as daemon_cli

        blocked_components = Mock()
        monkeypatch.setattr("polylogue.paths.archive_root", lambda: new_root)
        monkeypatch.setattr("polylogue.daemon.status_snapshot.configure_runtime_components", blocked_components)
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="prepared but incomplete"):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    debounce_s=1.0,
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                )
            )
        blocked_components.assert_not_called()

        def crash_after_refresh(path: Path, payload: dict[str, object]) -> None:
            real_write_refresh(path, payload)
            raise RuntimeError("crash after refresh receipt")

        monkeypatch.setattr(recovery, "_write_refresh_receipt", crash_after_refresh)
        with pytest.raises(RuntimeError, match="crash after refresh receipt"):
            CliRunner().invoke(
                cli,
                [
                    "--plain",
                    "ops",
                    "maintenance",
                    "source-continuity-recovery",
                    "apply",
                    "--plan",
                    str(retained_plan),
                    "--authorize",
                    plan_sha256,
                    "--output-format",
                    "json",
                ],
                env=command_env,
                catch_exceptions=False,
            )
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="prepared but incomplete"):
            assert_no_prepared_historical_source_continuity_recovery(new_root)

        monkeypatch.setattr(recovery, "_write_refresh_receipt", real_write_refresh)
        applied = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "apply",
                "--plan",
                str(retained_plan),
                "--authorize",
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
        assert applied.exit_code == 0, applied.output
        result = _maintenance_json_output(applied.output)
        assert result["state"] == "committed"
        refresh_path = Path(str(result["refresh_receipt_path"]))
        refresh_payload = json.loads(refresh_path.read_text(encoding="utf-8"))
        assert refresh_payload["refresh_sha256"] == _canonical_json_sha256(
            {key: value for key, value in refresh_payload.items() if key != "refresh_sha256"}
        )
        train_after = load_durable_change_train_manifest(plan_train)
        assert train_after.revision == train_before.revision + 1
        assert train_after.source_continuity_evidence is not None
        assert_no_prepared_historical_source_continuity_recovery(new_root)
        admitted_components = Mock(side_effect=RuntimeError("daemon admission reached"))
        monkeypatch.setattr("polylogue.daemon.status_snapshot.configure_runtime_components", admitted_components)
        with pytest.raises(RuntimeError, match="daemon admission reached"):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    debounce_s=1.0,
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                )
            )
        admitted_components.assert_called_once()
        foreign_refresh = tmp_path / "foreign-recovery-refresh.json"
        shutil.copyfile(refresh_path, foreign_refresh)
        refresh_path.unlink()
        refresh_path.symlink_to(foreign_refresh)
        train_before_rejected_resume = plan_train.read_bytes()
        with pytest.raises(DurableChangeTrainError, match="refresh receipt is unreadable"):
            CliRunner().invoke(
                cli,
                [
                    "--plain",
                    "ops",
                    "maintenance",
                    "source-continuity-recovery",
                    "apply",
                    "--plan",
                    str(retained_plan),
                    "--authorize",
                    plan_sha256,
                    "--output-format",
                    "json",
                ],
                env=command_env,
                catch_exceptions=False,
            )
        assert plan_train.read_bytes() == train_before_rejected_resume
        refresh_path.unlink()
        shutil.copyfile(foreign_refresh, refresh_path)
        rerun = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "apply",
                "--plan",
                str(retained_plan),
                "--authorize",
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
        assert rerun.exit_code == 0, rerun.output
        assert _maintenance_json_output(rerun.output)["state"] == "committed"


def test_historical_continuity_recovery_apply_rechecks_the_pinned_evidence_binding(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sealed plan cannot outlive the exact historical-evidence descriptor it authenticated."""
    new_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(new_root)}
    plan_path = tmp_path / "continuity-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
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
        assert planned.exit_code == 0, planned.output
        plan_sha256 = str(_maintenance_json_output(planned.output)["plan_sha256"])
        evidence.write_bytes(evidence.read_bytes() + b"\n")
        applied = CliRunner().invoke(
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
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
    assert applied.exit_code != 0
    assert "evidence binding changed" in applied.output


def test_prepare_apply_rebinds_a_real_released_train_and_resumes_after_prepared_crash(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use production backup, train CAS, and ordinary verifier across a moved temporary archive."""
    from polylogue.storage.sqlite import durable_change_train as trains

    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    _attach_retained_source_continuity(old_root, manifest)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    moved_manifest = new_root / manifest.relative_to(old_root)
    with sqlite3.connect(new_root / "source.db") as connection:
        with pytest.raises(DurableChangeTrainError, match="continuity proof failed"):
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
    assert plan.backup_root_inode == plan.new_root_inode
    assert all(item.backup_inode == item.inode for item in plan.tiers)
    assert database_before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in new_root.glob("*.db")
    }
    with OwnedArchiveLocation.acquire(ArchiveLocation.resolve(new_root), owner_id="held-by-another-operation"):
        with pytest.raises(ArchiveRootRelocationError, match="exclusive archive ownership"):
            apply_archive_root_relocation(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
            )
    with monkeypatch.context() as scoped:
        scoped.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 4242)
        with pytest.raises(ArchiveRootRelocationError, match="daemon to be stopped"):
            apply_archive_root_relocation(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
            )
    assert not (new_root / ".maintenance-state" / "archive-root-relocations").exists()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "polylogue.operations.archive_root_relocation.rebind_released_durable_train_archive_identity",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        with pytest.raises(RuntimeError, match="crash"):
            apply_archive_root_relocation(
                root=new_root,
                plan=plan,
                authorization=plan.plan_sha256,
            )
    result = apply_archive_root_relocation(
        root=new_root,
        plan=plan,
        authorization=plan.plan_sha256,
    )
    assert result.state == "committed"
    assert (
        apply_archive_root_relocation(
            root=new_root,
            plan=plan,
            authorization=plan.plan_sha256,
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
        # This is the same verifier branch that rejected the deployed v27
        # manifest after later source trains had advanced the archive.  The
        # real SQLite tier advances here; the fixture supplies its matching
        # canonical inventory because it deliberately has no synthetic v3 DDL.
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
        advanced = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
        live_inventory = capture_durable_schema_inventory(connection)
        forward = trains._verify_released_train_live_tier(
            new_root,
            connection,
            trains.load_durable_change_train_manifest(moved_manifest),
            current_target_version=advanced.user_version,
            actual_evidence=advanced,
            live_inventory=live_inventory,
            canonical_inventory=live_inventory,
        )
    assert forward is not None
    assert forward.historical_target_version == 2
    assert forward.observed_live_version == 3


def test_relocation_remaps_an_active_generation_pointer_and_resumes_after_publication_crash(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real relocation route must move the active generation pointer with its root.

    Anti-vacuity: this uses the production index-generation promotion and the
    real relocation prepare/apply functions.  Before the repair, preparation
    follows the stale absolute pointer beneath ``old_root`` and rejects the
    otherwise valid moved archive before any receipt can be written.
    """
    from polylogue.operations import archive_root_relocation as relocation

    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    old_active_target = _activate_movable_index_generation(old_root)
    active_generation_id = old_active_target.parent.name
    old_store = IndexGenerationStore.for_archive_root(old_root)
    inactive_generation = old_store.create(owner_id="relocation-paused", source_snapshot="paused-snapshot")
    _attach_retained_source_continuity(old_root, manifest)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None

    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )

    pointer = plan.active_index_pointer
    assert pointer is not None
    assert pointer.old_target == str(old_root / "index.db")
    assert pointer.new_target == str(new_root / "index.db")
    assert pointer.old_resolved_target == str(old_active_target)
    assert pointer.conventional_symlink_old_target == str(old_active_target)
    assert pointer.conventional_symlink_new_target == str(new_root / old_active_target.relative_to(old_root))
    assert {item.generation_id for item in plan.index_generations} == {
        active_generation_id,
        inactive_generation.generation_id,
    }
    real_publish = relocation._publish_active_index_pointer
    real_write = os.write
    short_pointer_write = False

    def write_pointer_in_two_calls(descriptor: int, payload: bytes) -> int:
        nonlocal short_pointer_write
        expected = (pointer.new_target + "\n").encode("utf-8")
        if not short_pointer_write and payload == expected:
            short_pointer_write = True
            partial = len(payload) - 1
            assert real_write(descriptor, payload[:partial]) == partial
            return partial
        return real_write(descriptor, payload)

    def crash_after_pointer_publication(root: Path, pointer: RelocationActiveIndexPointer | None) -> None:
        real_publish(root, pointer)
        raise RuntimeError("crash after active pointer publication")

    monkeypatch.setattr(os, "write", write_pointer_in_two_calls)
    monkeypatch.setattr(relocation, "_publish_active_index_pointer", crash_after_pointer_publication)
    with pytest.raises(RuntimeError, match="crash after active pointer publication"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert short_pointer_write
    assert (new_root / ".index-active-pointer").read_text(encoding="utf-8").strip() == pointer.new_target
    crashed_store = IndexGenerationStore.for_archive_root(new_root)
    for generation_id in (active_generation_id, inactive_generation.generation_id):
        generation = crashed_store.load(generation_id)
        assert generation.archive_root == str(new_root)
        assert generation.index_path == str(new_root / ".index-generations" / generation_id / "index.db")
        generation_root = new_root / ".index-generations" / generation_id
        for filename in ("source.db", "user.db", "embeddings.db", "ops.db", "blob"):
            link = generation_root / filename
            if link.is_symlink():
                assert os.readlink(link) == str(new_root / filename)
    with pytest.raises(ArchiveRootRelocationError, match="prepared but incomplete"):
        assert_no_prepared_archive_root_relocation(new_root)

    monkeypatch.setattr(relocation, "_publish_active_index_pointer", real_publish)
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert result.state == "committed"
    relocated_location = ArchiveLocation.resolve(new_root)
    assert relocated_location.active_index_path == Path(pointer.new_target)
    assert relocated_location.active_index.resolved_path == Path(pointer.new_resolved_target)
    assert apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256).state == "committed"
    relocated_store = IndexGenerationStore.for_archive_root(new_root)
    promoted = relocated_store.promote(relocated_store.load(inactive_generation.generation_id))
    assert promoted.state == "active"
    assert (new_root / "index.db").resolve(strict=True) == Path(promoted.index_path)


def test_relocation_generation_publication_rejects_a_post_validation_directory_swap(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation publication pins the planned directory before every write.

    Anti-vacuity: the swap occurs after the production validator returns and
    before publication. Path-based temporary files and replaces mutate the
    byte-identical foreign generation; descriptor-pinned publication rejects
    its different directory identity without changing any foreign artifact.
    """
    from polylogue.operations import archive_root_relocation as relocation

    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    _activate_movable_index_generation(old_root)
    _attach_retained_source_continuity(old_root, manifest)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert plan.index_generations
    generation_root = Path(plan.index_generations[0].metadata_path).parent
    outside = tmp_path / "foreign-generation"
    shutil.copytree(generation_root, outside, symlinks=True)
    outside_metadata_before = (outside / "generation.json").read_bytes()
    outside_links_before = {path.name: os.readlink(path) for path in outside.iterdir() if path.is_symlink()}
    detached = tmp_path / "detached-authoritative-generation"
    real_validate = relocation._validate_index_generation_state
    validation_calls = 0

    def swap_after_publication_preflight(
        root: Path,
        items: tuple[RelocationIndexGeneration, ...],
        pointer: RelocationActiveIndexPointer | None,
        *,
        allow_post_publication: bool,
    ) -> None:
        nonlocal validation_calls
        real_validate(root, items, pointer, allow_post_publication=allow_post_publication)
        validation_calls += 1
        if validation_calls == 2:
            os.rename(generation_root, detached)
            generation_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(relocation, "_validate_index_generation_state", swap_after_publication_preflight)
    with pytest.raises(ArchiveRootRelocationError, match="cannot pin.*index generation directory"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert validation_calls == 2
    assert (outside / "generation.json").read_bytes() == outside_metadata_before
    assert {path.name: os.readlink(path) for path in outside.iterdir() if path.is_symlink()} == outside_links_before


def test_relocation_apply_rejects_byte_identical_generation_metadata_substitution(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public apply authenticates the planned generation metadata object.

    Anti-vacuity: a same-directory atomic replacement preserves the exact
    metadata bytes and parent directory while changing only the leaf inode.
    The production apply route must reject it before retaining a plan, writing
    a receipt, publishing generation state, or rebinding a durable train.
    """
    new_root, plan = _prepare_moved_root_relocation_with_generation(workspace_env, tmp_path, monkeypatch)
    generation = plan.index_generations[0]
    metadata_path = Path(generation.metadata_path)
    encoded = metadata_path.read_bytes()
    planned_identity = (metadata_path.lstat().st_dev, metadata_path.lstat().st_ino)
    assert (generation.metadata_before_device, generation.metadata_before_inode) == planned_identity
    substitute = metadata_path.parent / ".generation.json.substitute"
    substitute.write_bytes(encoded)
    os.replace(substitute, metadata_path)
    substituted_identity = (metadata_path.lstat().st_dev, metadata_path.lstat().st_ino)
    assert substituted_identity != planned_identity
    pointer_path = new_root / ".index-active-pointer"
    pointer_before = pointer_path.read_bytes()
    manifests_before = {item.path: Path(item.path).read_bytes() for item in plan.durable_trains}

    with pytest.raises(ArchiveRootRelocationError, match="generation metadata identity changed"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert metadata_path.read_bytes() == encoded
    assert (metadata_path.lstat().st_dev, metadata_path.lstat().st_ino) == substituted_identity
    assert pointer_path.read_bytes() == pointer_before
    assert {path: Path(path).read_bytes() for path in manifests_before} == manifests_before
    assert not (new_root / ".maintenance-state" / "archive-root-relocation-plans" / f"{plan.plan_sha256}.json").exists()
    assert not (new_root / ".maintenance-state" / "archive-root-relocations" / f"{plan.plan_sha256}.json").exists()


def test_relocation_apply_rejects_equivalent_generation_tier_symlink_substitution(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public apply authenticates each planned generation tier-link object.

    Anti-vacuity: a fresh symlink with the same absolute target preserves the
    planned directory, tier inventory, and link value while changing only the
    leaf inode. The production apply route must reject it before any relocation
    state is published.
    """
    new_root, plan = _prepare_moved_root_relocation_with_generation(workspace_env, tmp_path, monkeypatch)
    generation = plan.index_generations[0]
    link = next(item for item in generation.tier_symlinks if item.old_target != item.new_target)
    link_path = Path(link.path)
    planned_identity = (link_path.lstat().st_dev, link_path.lstat().st_ino)
    assert (link.before_device, link.before_inode) == planned_identity
    substitute = link_path.parent / f".{link_path.name}.substitute"
    os.symlink(link.old_target, substitute)
    os.replace(substitute, link_path)
    substituted_identity = (link_path.lstat().st_dev, link_path.lstat().st_ino)
    assert substituted_identity != planned_identity
    pointer_path = new_root / ".index-active-pointer"
    pointer_before = pointer_path.read_bytes()
    metadata_before = Path(generation.metadata_path).read_bytes()
    manifests_before = {item.path: Path(item.path).read_bytes() for item in plan.durable_trains}

    with pytest.raises(ArchiveRootRelocationError, match="generation tier link identity changed"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert os.readlink(link_path) == link.old_target
    assert (link_path.lstat().st_dev, link_path.lstat().st_ino) == substituted_identity
    assert Path(generation.metadata_path).read_bytes() == metadata_before
    assert pointer_path.read_bytes() == pointer_before
    assert {path: Path(path).read_bytes() for path in manifests_before} == manifests_before
    assert not (new_root / ".maintenance-state" / "archive-root-relocation-plans" / f"{plan.plan_sha256}.json").exists()
    assert not (new_root / ".maintenance-state" / "archive-root-relocations" / f"{plan.plan_sha256}.json").exists()


def test_relocation_v3_prepared_resume_rejects_generation_metadata_post_state(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first apply cannot treat an after-state leaf as proof of publication.

    Anti-vacuity: the production plan carries only its revision-0 receipt,
    which has no bound manifest transition, while generation metadata is
    atomically changed to its exact planned after bytes. Removing the
    publication-begun gate accepts the substituted state.
    """
    from polylogue.operations import archive_root_relocation as relocation

    new_root, current_plan = _prepare_moved_root_relocation_with_generation(workspace_env, tmp_path, monkeypatch)
    legacy_payload = current_plan.model_dump(mode="json")
    legacy_payload["format"] = "polylogue.archive-root-relocation-plan.v3"
    legacy_payload.pop("plan_sha256")
    for generation_payload in legacy_payload["index_generations"]:
        generation_payload.pop("metadata_before_device")
        generation_payload.pop("metadata_before_inode")
        for link_payload in generation_payload["tier_symlinks"]:
            link_payload.pop("before_device")
            link_payload.pop("before_inode")
    plan = _sealed_relocation_plan(**legacy_payload)
    generation = plan.index_generations[0]
    metadata_path = Path(generation.metadata_path)
    before = metadata_path.read_bytes()
    payload = relocation._index_generation_payload_for_state(generation, after=False, encoded=before)
    after = relocation._index_generation_metadata_bytes(
        {**payload, "archive_root": generation.after_archive_root, "index_path": generation.after_index_path}
    )
    assert hashlib.sha256(after).hexdigest() == generation.after_sha256
    replacement = metadata_path.with_name(".generation.json.after")
    replacement.write_bytes(after)
    os.replace(replacement, metadata_path)

    pointer_fields = relocation._pointer_receipt_fields(plan.active_index_pointer)
    initial = _sealed_relocation_receipt(
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=plan.plan_sha256,
        manifest_before_sha256=tuple(item.before_manifest_sha256 for item in plan.durable_trains),
        manifest_after_sha256=(),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command="polylogue ops maintenance archive-root-relocation apply",
    )
    _write_relocation_receipt(relocation._receipt_path(new_root, plan), initial, expected=None)

    with pytest.raises(ArchiveRootRelocationError, match="post-publication state without a prepared receipt"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert metadata_path.read_bytes() == after
    retained_initial = json.loads(
        (new_root / ".maintenance-state" / "archive-root-relocations" / f"{plan.plan_sha256}.json").read_text(
            encoding="utf-8"
        )
    )
    assert retained_initial["revision"] == 0
    assert retained_initial["manifest_after_sha256"] == []


def test_relocation_v3_plan_decodes_but_requires_a_prepared_resume_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-leaf-sealing v3 plans refuse new publication instead of stranding resume.

    Anti-vacuity: the retained plan omits exactly the v4 leaf identities. The
    public loader must still decode it so a prepared receipt can be inspected,
    while the public apply route rejects a first publication that cannot prove
    its original leaves.
    """
    new_root, plan = _prepare_moved_root_relocation_with_generation(workspace_env, tmp_path, monkeypatch)
    legacy_payload = plan.model_dump(mode="json")
    legacy_payload["format"] = "polylogue.archive-root-relocation-plan.v3"
    legacy_payload.pop("plan_sha256")
    for generation in legacy_payload["index_generations"]:
        generation.pop("metadata_before_device")
        generation.pop("metadata_before_inode")
        for link in generation["tier_symlinks"]:
            link.pop("before_device")
            link.pop("before_inode")
    legacy = _sealed_relocation_plan(**legacy_payload)
    retained = tmp_path / "retained-v3-plan.json"
    retained.write_text(
        json.dumps(legacy.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_archive_root_relocation_plan(retained)
    assert loaded.format == "polylogue.archive-root-relocation-plan.v3"
    assert loaded.index_generations[0].metadata_before_device is None
    with pytest.raises(ArchiveRootRelocationError, match="create a v4 plan"):
        apply_archive_root_relocation(root=new_root, plan=loaded, authorization=loaded.plan_sha256)

    from polylogue.operations import archive_root_relocation as relocation

    legacy_receipt = {
        "format": "polylogue.archive-root-relocation-receipt.v1",
        "state": "prepared",
        "revision": 0,
        "plan_sha256": loaded.plan_sha256,
        "authorization": loaded.plan_sha256,
        "manifest_before_sha256": [item.before_manifest_sha256 for item in loaded.durable_trains],
        "manifest_after_sha256": [],
        "resume_command": "polylogue ops maintenance archive-root-relocation apply",
    }
    legacy_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(legacy_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    receipt_path = relocation._receipt_path(new_root, loaded)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(legacy_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resumed = apply_archive_root_relocation(root=new_root, plan=loaded, authorization=loaded.plan_sha256)
    assert resumed.state == "committed"


def test_relocation_rejects_backup_root_nested_under_moved_archive(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relocation authority cannot be sourced from a package under the live root.

    Anti-vacuity: this uses a real verified backup package copied below the
    moved archive. Removing backup-root separation lets the public planner
    accept evidence that the operation can overwrite or recursively include.
    """
    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    nested = new_root / "retained-backup"
    shutil.copytree(Path(backup.output_path), nested)

    with pytest.raises(ArchiveRootRelocationError, match="backup root must be separate"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=nested / "manifest.json",
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
        )


def test_relocation_backup_validation_checks_every_live_tier_alias(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full-evidence relocation validation gives every artifact its live tier.

    Anti-vacuity: a real full backup is revalidated through the production
    validator. Removing the per-tier mapping passes ``None`` for every
    artifact and this capture no longer observes the live alias boundary.
    """
    from polylogue.storage.sqlite import migration_runner

    root = workspace_env["archive_root"]
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    observed: dict[str, Path | None] = {}
    real_validate = migration_runner._validate_tier_artifact

    def capture_live_alias(
        backup_root: Path,
        artifact: dict[str, object],
        *,
        file_evidence: dict[str, dict[str, object]],
        live_tier_path: Path | None,
    ) -> None:
        observed[str(artifact["tier"])] = live_tier_path
        real_validate(
            backup_root,
            artifact,
            file_evidence=file_evidence,
            live_tier_path=live_tier_path,
        )

    monkeypatch.setattr(migration_runner, "_validate_tier_artifact", capture_live_alias)
    migration_runner.validate_full_evidence_backup_for_archive_root_relocation(
        Path(backup.output_path) / "manifest.json",
        backup_configured_root=root,
        backup_archive_root=root,
    )

    assert observed == {tier.value: root / f"{tier.value}.db" for tier in ArchiveTier}


def test_relocation_remaps_generations_beside_a_nested_active_index(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relocation follows the canonical index parent to its generation store.

    Anti-vacuity: the production ``IndexGenerationStore`` derives its retained
    generation directory from the active pointer's parent.  Root-only
    inventory leaves both generation metadata and tier links carrying the
    retired root after this public plan/apply sequence.
    """
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    nested_index = old_root / "nested" / "index.db"
    nested_index.parent.mkdir()
    (old_root / ".index-active-pointer").write_text(str(nested_index), encoding="utf-8")
    store = IndexGenerationStore.for_archive_root(old_root)
    active = store.create(owner_id="nested-active", source_snapshot="active-snapshot")
    store.promote(active)
    inactive = store.create(owner_id="nested-inactive", source_snapshot="inactive-snapshot")
    _attach_retained_source_continuity(old_root, manifest)

    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )

    expected_root = new_root / "nested" / ".index-generations"
    assert {Path(item.metadata_path).parent.parent for item in plan.index_generations} == {expected_root}
    assert {item.generation_id for item in plan.index_generations} == {
        active.generation_id,
        inactive.generation_id,
    }
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert result.state == "committed"
    relocated_store = IndexGenerationStore.for_archive_root(new_root)
    assert relocated_store.generations_root == expected_root
    for generation_id in (active.generation_id, inactive.generation_id):
        generation = relocated_store.load(generation_id)
        assert generation.archive_root == str(new_root)
        assert generation.index_path == str(expected_root / generation_id / "index.db")
        generation_root = expected_root / generation_id
        for filename in ("source.db", "user.db", "embeddings.db", "ops.db", "blob"):
            link = generation_root / filename
            if link.is_symlink():
                assert os.readlink(link) == str(new_root / filename)
    promoted = relocated_store.promote(relocated_store.load(inactive.generation_id))
    assert promoted.state == "active"
    assert (new_root / "nested" / "index.db").resolve(strict=True) == Path(promoted.index_path)


@pytest.mark.parametrize("pointer_kind", ["regular", "symlink"])
def test_relocation_backup_maps_a_nested_regular_active_index(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer_kind: str,
) -> None:
    """The moved-root backup follows a stale pointer to a regular in-root index.

    Anti-vacuity: the production backup must fingerprint the nested active
    inode, not the regular shadow at ``root/index.db``.  Relocation's real
    backup identity comparison rejects the shadow inode before planning.
    """
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    nested_index = old_root / "nested" / "index.db"
    nested_index.parent.mkdir()
    shutil.copy2(old_root / "index.db", nested_index)
    pointer = old_root / ".index-active-pointer"
    if pointer_kind == "regular":
        pointer.write_text(str(nested_index), encoding="utf-8")
    else:
        pointer.symlink_to(nested_index)
    _attach_retained_source_continuity(old_root, manifest)

    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    backup_manifest = Path(backup.output_path) / "manifest.json"
    backup_payload = json.loads(backup_manifest.read_text(encoding="utf-8"))
    index_fingerprint = backup_payload["tier_source_fingerprints"]["index.db"]
    moved_nested_index = new_root / "nested" / "index.db"
    assert (index_fingerprint["device"], index_fingerprint["inode"]) == (
        moved_nested_index.stat().st_dev,
        moved_nested_index.stat().st_ino,
    )
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=backup_manifest,
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )

    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)

    assert result.state == "committed"
    assert plan.active_index_pointer is not None
    assert plan.active_index_pointer.new_target == str(moved_nested_index)
    assert ArchiveLocation.resolve(new_root).active_index.resolved_path == moved_nested_index


def test_active_index_publication_updates_the_bound_nested_conventional_symlink(tmp_path: Path) -> None:
    """Pointer publication updates the exact conventional path sealed by the plan.

    Anti-vacuity: ``_publish_active_index_pointer`` is the production apply
    helper.  Replacing a hard-coded ``<root>/index.db`` leaves this nested
    conventional symlink stale while publishing a pointer that selects it.
    """
    from polylogue.operations import archive_root_relocation as relocation

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    conventional = new_root / "nested" / "index.db"
    resolved = new_root / ".index-generations" / "gen-1" / "index.db"
    conventional.parent.mkdir(parents=True)
    resolved.parent.mkdir(parents=True)
    resolved.write_bytes(b"index generation")
    old_resolved = old_root / resolved.relative_to(new_root)
    conventional.symlink_to(old_resolved)
    old_conventional = old_root / conventional.relative_to(new_root)
    (new_root / ".index-active-pointer").write_text(str(old_conventional), encoding="utf-8")
    metadata = resolved.stat()
    pointer = RelocationActiveIndexPointer(
        old_target=str(old_conventional),
        new_target=str(conventional),
        old_resolved_target=str(old_resolved),
        new_resolved_target=str(resolved),
        conventional_symlink_old_target=str(old_resolved),
        conventional_symlink_new_target=str(resolved),
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )

    relocation._publish_active_index_pointer(new_root, pointer)

    assert os.readlink(conventional) == str(resolved)
    assert not (new_root / "index.db").exists()
    assert (new_root / ".index-active-pointer").read_text(encoding="utf-8").strip() == str(conventional)


def test_relocation_accepts_a_modern_no_rebind_train_without_rewriting_it(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inode-preserving move leaves modern tier identity authority unchanged."""
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    train = load_durable_change_train_manifest(manifest)
    assert train.apply_evidence is not None
    identity = ArchiveIdentity.resolve(old_root).tier("source").stable_id
    modern = replace(
        train,
        revision=train.revision + 1,
        apply_evidence=replace(
            train.apply_evidence,
            post=replace(
                train.apply_evidence.post, archive_identity_digest=hashlib.sha256(identity.encode()).hexdigest()
            ),
        ),
    )
    write_durable_change_train_manifest(manifest, modern, expected_revision=train.revision)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert len(plan.durable_trains) == 1
    assert plan.durable_trains[0].requires_rebind is False
    moved_manifest = Path(plan.durable_trains[0].path)
    before = moved_manifest.read_bytes()
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert result.state == "committed"
    assert result.changed_manifests == ()
    assert moved_manifest.read_bytes() == before
    repeated = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert repeated.changed_manifests == ()


def test_relocation_resume_rejects_a_same_revision_manifest_substituted_after_cas(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepared recovery accepts only the exact post-CAS bytes bound before mutation."""
    from polylogue.operations import archive_root_relocation as relocation

    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch)
    new_root = tmp_path / "moved root;still-one-argument"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    real_write = write_durable_change_train_manifest

    def crash_after_cas(path: Path, train: DurableChangeTrain, *, expected_revision: int) -> None:
        real_write(path, train, expected_revision=expected_revision)
        raise RuntimeError("crash after relocation manifest CAS")

    monkeypatch.setattr(relocation, "write_durable_change_train_manifest", crash_after_cas)
    with pytest.raises(RuntimeError, match="crash after relocation manifest CAS"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    monkeypatch.setattr(relocation, "write_durable_change_train_manifest", real_write)
    retained_plan = new_root / ".maintenance-state" / "archive-root-relocation-plans" / f"{plan.plan_sha256}.json"
    assert retained_plan.is_file()
    prepared_receipt = json.loads(
        (new_root / ".maintenance-state" / "archive-root-relocations" / f"{plan.plan_sha256}.json").read_text(
            encoding="utf-8"
        )
    )
    assert f"POLYLOGUE_ARCHIVE_ROOT={shlex.quote(str(new_root))}" in prepared_receipt["resume_command"]
    assert f"--plan {shlex.quote(str(retained_plan))}" in prepared_receipt["resume_command"]
    train_path = Path(plan.durable_trains[0].path)
    relocated = load_durable_change_train_manifest(train_path)
    substituted = replace(relocated, proof_refs=(*relocated.proof_refs, "proof:foreign-substitution"))
    payload = durable_change_train_to_payload(substituted)
    train_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArchiveRootRelocationError, match="manifest changed"):
        apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)


def test_continuity_free_rebind_requires_its_retained_relocation_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon admission resolves relocation authority even without source refresh evidence."""
    from polylogue.storage.sqlite import durable_change_train as trains

    old_root = workspace_env["archive_root"]
    _released_moved_source_train(old_root, monkeypatch)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert result.state == "committed"
    Path(result.receipt_path or "").unlink()
    train = load_durable_change_train_manifest(Path(plan.durable_trains[0].path))
    with sqlite3.connect(new_root / "source.db") as connection:
        with pytest.raises(DurableChangeTrainError, match="committed receipt"):
            trains._verify_released_train_live_tier(new_root, connection, train)


def test_relocation_rebinds_released_trains_for_every_durable_tier(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy source, user, and audit trains move under one exact CAS plan."""
    old_root = workspace_env["archive_root"]
    source_manifest = _released_moved_source_train(old_root, monkeypatch)
    user_manifest = _released_moved_durable_train(old_root, monkeypatch, ArchiveTier.USER)
    manifests = [
        source_manifest,
        user_manifest,
        *_clone_released_durable_train_for_tier(old_root, user_manifest, ArchiveTier.AUDIT, monkeypatch),
    ]
    legacy_identity = ArchiveIdentity.resolve(old_root).authority_identity_digest
    for manifest in manifests:
        train = load_durable_change_train_manifest(manifest)
        assert train.apply_evidence is not None
        rebound = replace(
            train,
            revision=train.revision + 1,
            apply_evidence=replace(
                train.apply_evidence,
                post=replace(train.apply_evidence.post, archive_identity_digest=legacy_identity),
            ),
        )
        write_durable_change_train_manifest(manifest, rebound, expected_revision=train.revision)
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    assert {item.tier for item in plan.durable_trains} == {"source", "user", "audit"}
    assert all(item.requires_rebind for item in plan.durable_trains)
    assert apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256).state == "committed"
    identity = ArchiveIdentity.resolve(new_root)
    for item in plan.durable_trains:
        train = load_durable_change_train_manifest(Path(item.path))
        assert train.apply_evidence is not None
        expected = hashlib.sha256(identity.tier(item.tier).stable_id.encode()).hexdigest()
        assert train.apply_evidence.post.archive_identity_digest == expected


def test_relocation_rejects_an_active_pointer_not_owned_by_the_old_root(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remap boundary cannot turn an arbitrary external index into authority."""
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    _attach_retained_source_continuity(old_root, manifest)
    foreign = tmp_path / "foreign" / "index.db"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign")
    (old_root / ".index-active-pointer").write_text(str(foreign), encoding="utf-8")
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None

    with pytest.raises(ArchiveRootRelocationError, match="not owned by the old root"):
        prepare_archive_root_relocation(
            old_root=old_root,
            new_root=new_root,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            stopped_daemon_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
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
    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
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


def test_relocation_accepts_a_mover_rewritten_pointer_only_with_a_bound_witness(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mover may rewrite absolute pointers before Polylogue plans the move."""
    old_root = workspace_env["archive_root"]
    manifest = _released_moved_source_train(old_root, monkeypatch)
    _activate_movable_index_generation(old_root)
    _attach_retained_source_continuity(old_root, manifest)
    new_root = tmp_path / "moved"
    source_inode = (old_root / "source.db").stat().st_ino
    legacy_device = (old_root / "source.db").stat().st_dev
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    (new_root / ".index-active-pointer").write_text(str(new_root / "index.db"), encoding="utf-8")
    witness = RelocationPostMoveWitness(
        format="polylogue.archive-root-relocation-post-move-witness.v1",
        old_configured_root=str(old_root.absolute()),
        old_resolved_root=str(old_root.absolute()),
        new_configured_root=str(new_root.absolute()),
        new_resolved_root=str(new_root.resolve()),
        legacy_device=legacy_device,
        source_inode=source_inode,
        evidence_ref="test:mover-rewritten-pointer",
    )

    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        post_move_witness=witness,
    )

    assert plan.post_move_witness == witness
    assert plan.active_index_pointer is not None
    assert plan.active_index_pointer.old_target == str(old_root / "index.db")


def test_historical_continuity_recovery_cli_rejects_a_byte_identical_copied_archive(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The historical old-path attestation must not authorize a copied destination file."""
    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    copied_root = tmp_path / "copied"
    shutil.copytree(moved_root, copied_root, symlinks=True)
    plan_path = tmp_path / "copied-continuity-plan.json"

    with _test_historical_operation_evidence_resource(evidence):
        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
                "--pre-backup-manifest",
                str(pre_manifest),
                "--post-backup-manifest",
                str(post_manifest),
                "--output",
                str(plan_path),
                "--output-format",
                "json",
            ],
            env={"POLYLOGUE_ARCHIVE_ROOT": str(copied_root)},
            catch_exceptions=False,
        )

    assert result.exit_code != 0
    assert "device/inode continuity" in result.output


def test_historical_recovery_rejects_a_copied_legacy_backup_destination(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy tier fingerprints still bind the moved source-file identity.

    Anti-vacuity: the two authentic backup packages omit only their original
    source-file device/inode fields, then a byte-identical archive copy is
    presented to the real planning API. Accepting a self-observed destination
    identity would let that copy receive fresh continuity authority.
    """
    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    old_root = workspace_env["archive_root"]
    _downgrade_historical_backup_source_identity(pre_manifest, old_root=old_root)
    _downgrade_historical_backup_source_identity(post_manifest, old_root=old_root)
    with sqlite3.connect(f"file:{pre_manifest.parent / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
        candidates = classify_blob_ref_liveness(connection).candidates
    _pinned_historical_operation_evidence(
        evidence,
        mutation_receipt=mutation_receipt,
        candidates=candidates,
        pre_manifest=pre_manifest,
        post_manifest=post_manifest,
    )
    copied_root = tmp_path / "copied-legacy"
    shutil.copytree(moved_root, copied_root, symlinks=True)

    with _test_historical_operation_evidence_resource(evidence):
        with pytest.raises(HistoricalSourceContinuityRecoveryError, match="source.db device/inode continuity"):
            prepare_historical_source_continuity_recovery(
                old_root=old_root,
                new_root=copied_root,
                mutation_receipt=mutation_receipt,
                pre_backup_manifest=pre_manifest,
                post_backup_manifest=post_manifest,
                stopped_daemon_evidence_ref="proof:daemon-stopped",
                single_writer_evidence_ref="proof:archive-ownership-lock",
            )


def test_cli_runs_historical_recovery_then_uses_a_fresh_moved_root_backup_for_relocation(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented recovery, fresh backup, then relocation sequence uses public commands."""
    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(moved_root)}
    continuity_plan = tmp_path / "continuity-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
                "--pre-backup-manifest",
                str(pre_manifest),
                "--post-backup-manifest",
                str(post_manifest),
                "--output",
                str(continuity_plan),
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
        assert planned.exit_code == 0, planned.output
        continuity_digest = str(_maintenance_json_output(planned.output)["plan_sha256"])
        recovered = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "apply",
                "--plan",
                str(continuity_plan),
                "--authorize",
                continuity_digest,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
    assert recovered.exit_code == 0, recovered.output

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(moved_root))
    backup = backup_archive(output_dir=tmp_path / "moved-backup", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None
    relocation_plan = tmp_path / "relocation-plan.json"
    relocated = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "plan",
            "--old-root",
            str(workspace_env["archive_root"]),
            "--backup-manifest",
            str(Path(backup.output_path) / "manifest.json"),
            "--output",
            str(relocation_plan),
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )

    assert relocated.exit_code == 0, relocated.output
    relocation_payload = _maintenance_json_output(relocated.output)
    relocation_digest = str(relocation_payload["plan_sha256"])
    relocation_plan_payload = json.loads(relocation_plan.read_text(encoding="utf-8"))
    source_trains = relocation_plan_payload["durable_trains"]
    assert isinstance(source_trains, list) and len(source_trains) == 1
    assert source_trains[0]["requires_rebind"] is False

    refresh_digests = source_trains[0]["continuity_receipt_digests"]
    assert isinstance(refresh_digests, list) and len(refresh_digests) == 1
    refresh_path = moved_root / ".maintenance-state" / "source-continuity-refreshes" / f"{refresh_digests[0]}.json"
    foreign_refresh = tmp_path / "foreign-refresh.json"
    shutil.copyfile(refresh_path, foreign_refresh)
    refresh_path.unlink()
    refresh_path.symlink_to(foreign_refresh)
    train_path = Path(str(source_trains[0]["path"]))
    protected_paths = (*sorted(moved_root.glob("*.db")), train_path)
    before_rejections = {
        path: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in protected_paths
    }
    relocation_receipt = moved_root / ".maintenance-state" / "archive-root-relocations" / f"{relocation_digest}.json"

    rejected_plan = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "plan",
            "--old-root",
            str(workspace_env["archive_root"]),
            "--backup-manifest",
            str(Path(backup.output_path) / "manifest.json"),
            "--output",
            str(tmp_path / "rejected-relocation-plan.json"),
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )
    assert rejected_plan.exit_code != 0
    assert "relocation authority is invalid" in rejected_plan.output

    rejected_apply = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "apply",
            "--plan",
            str(relocation_plan),
            "--authorize",
            relocation_digest,
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )
    assert rejected_apply.exit_code != 0
    assert "retained train authority is invalid" in rejected_apply.output

    from polylogue.daemon import cli as daemon_cli
    from polylogue.operations import durable_change_train as durable_operations

    configure = Mock()
    admission = Mock(wraps=durable_operations.reconcile_durable_change_trains_on_startup)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.AUDIT, 10_000)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: moved_root)
    monkeypatch.setattr(
        "polylogue.operations.durable_change_train.reconcile_durable_change_trains_on_startup",
        admission,
    )
    monkeypatch.setattr("polylogue.daemon.status_snapshot.configure_runtime_components", configure)
    with pytest.raises(DurableChangeTrainError, match="refresh receipt is unreadable"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )
    admission.assert_called_once_with(moved_root)
    configure.assert_called_once()
    assert not relocation_receipt.exists()
    assert {
        path: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in protected_paths
    } == before_rejections

    refresh_path.unlink()
    shutil.copyfile(foreign_refresh, refresh_path)

    applied = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "apply",
            "--plan",
            str(relocation_plan),
            "--authorize",
            relocation_digest,
            "--output-format",
            "json",
        ],
        env=command_env,
        catch_exceptions=False,
    )

    assert applied.exit_code == 0, applied.output
    applied_payload = _maintenance_json_output(applied.output)
    assert applied_payload["state"] == "committed"
    train = load_durable_change_train_manifest(Path(source_trains[0]["path"]))
    relocation_refs = tuple(ref for ref in train.proof_refs if ref.startswith("proof:archive-root-relocation:"))
    transition_refs = tuple(ref for ref in train.proof_refs if ref.startswith("proof:source-continuity-relocation:"))
    assert len(relocation_refs) == 1
    assert len(transition_refs) == 1

    refresh_path = _refresh_source_continuity_without_content_change(moved_root, tmp_path / "post-relocation-refresh")
    refresh_payload = json.loads(refresh_path.read_text(encoding="utf-8"))
    assert refresh_payload["format"] == "polylogue.source-continuity-refresh.v2"
    refreshed_train = load_durable_change_train_manifest(Path(source_trains[0]["path"]))
    from polylogue.storage.sqlite import durable_change_train as trains

    with sqlite3.connect(moved_root / "source.db") as connection:
        assert trains._verify_released_train_live_tier(moved_root, connection, refreshed_train) is None
    exact_refresh_bytes = refresh_path.read_bytes()
    substituted_refresh = json.loads(exact_refresh_bytes)
    intent_payload = substituted_refresh["train_after_without_receipt"]
    assert isinstance(intent_payload, dict)
    intent_refs = intent_payload["proof_refs"]
    assert isinstance(intent_refs, list)
    intent_refs.append("proof:foreign-same-evidence-substitution")
    refresh_path.write_text(json.dumps(substituted_refresh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with sqlite3.connect(moved_root / "source.db") as connection:
        with pytest.raises(DurableChangeTrainError, match="refresh receipt checksum mismatch"):
            trains._verify_released_train_live_tier(moved_root, connection, refreshed_train)
    refresh_path.write_bytes(exact_refresh_bytes)

    second_root = tmp_path / "moved-again"
    os.rename(moved_root, second_root)
    second_env = {"POLYLOGUE_ARCHIVE_ROOT": str(second_root)}
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(second_root))
    second_backup = backup_archive(output_dir=tmp_path / "second-moved-backup", profile="full_evidence", verify=True)
    assert second_backup.ok and second_backup.output_path is not None
    second_plan = tmp_path / "second-relocation-plan.json"
    second_planned = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "plan",
            "--old-root",
            str(moved_root),
            "--backup-manifest",
            str(Path(second_backup.output_path) / "manifest.json"),
            "--output",
            str(second_plan),
            "--output-format",
            "json",
        ],
        env=second_env,
        catch_exceptions=False,
    )
    assert second_planned.exit_code == 0, second_planned.output
    second_digest = str(_maintenance_json_output(second_planned.output)["plan_sha256"])
    second_applied = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-root-relocation",
            "apply",
            "--plan",
            str(second_plan),
            "--authorize",
            second_digest,
            "--output-format",
            "json",
        ],
        env=second_env,
        catch_exceptions=False,
    )
    assert second_applied.exit_code == 0, second_applied.output
    second_payload = json.loads(second_plan.read_text(encoding="utf-8"))
    second_train_path = Path(str(second_payload["durable_trains"][0]["path"]))
    second_train = load_durable_change_train_manifest(second_train_path)
    second_transition_refs = tuple(
        ref for ref in second_train.proof_refs if ref.startswith("proof:source-continuity-relocation:")
    )
    assert len(second_transition_refs) == 2
    latest_transition = json.loads(
        (
            second_root
            / ".maintenance-state"
            / "source-continuity-relocations"
            / f"{second_transition_refs[-1].rsplit(':', 1)[-1]}.json"
        ).read_text(encoding="utf-8")
    )
    assert latest_transition["predecessor_authority"] == {
        "kind": "refresh",
        "sha256": refresh_path.stem,
    }

    with sqlite3.connect(second_root / "source.db") as connection:
        assert trains._verify_released_train_live_tier(second_root, connection, second_train) is None


def test_historical_continuity_recovery_resume_rejects_a_foreign_same_evidence_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared recovery may resume only with its sealed refresh receipt and CAS revision."""
    from polylogue.operations import historical_source_continuity_recovery as recovery

    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(moved_root)}
    plan_path = tmp_path / "continuity-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
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
        assert planned.exit_code == 0, planned.output
        plan = _maintenance_json_output(planned.output)
        plan_sha256 = str(plan["plan_sha256"])
        source_before = plan.get("source_before")
        source_after = plan.get("source_after")
        assert isinstance(source_before, dict) and isinstance(source_after, dict)
        observed_at_ms = source_after.get("observed_at_ms")
        assert type(observed_at_ms) is int
        real_write_refresh = recovery._write_refresh_receipt
        monkeypatch.setattr(
            recovery, "_write_refresh_receipt", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("crash"))
        )
        with pytest.raises(RuntimeError, match="crash"):
            CliRunner().invoke(
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
                    plan_sha256,
                    "--output-format",
                    "json",
                ],
                env=command_env,
                catch_exceptions=False,
            )
        monkeypatch.setattr(recovery, "_write_refresh_receipt", real_write_refresh)
        train_path = Path(str(plan["source_train_path"]))
        train = load_durable_change_train_manifest(train_path)
        foreign_payload = {
            "format": "polylogue.source-continuity-refresh.v1",
            "operation_id": "foreign",
            "evidence_ref": "proof:foreign-continuity",
            "backup_manifest": str(pre_manifest),
            "backup_manifest_sha256": _sha256(pre_manifest),
            "mutation_receipt": str(mutation_receipt),
            "mutation_receipt_sha256": _sha256(mutation_receipt),
            "train_id": train.train_id,
            "source_before": source_before,
            "source_after": source_after,
            "refreshed_at_ms": observed_at_ms,
        }
        foreign_digest = _canonical_json_sha256(foreign_payload)
        real_write_refresh(
            moved_root / ".maintenance-state" / "source-continuity-refreshes" / f"{foreign_digest}.json",
            {**foreign_payload, "refresh_sha256": foreign_digest},
        )
        substituted = recover_released_source_train_continuity(
            train,
            current_evidence=recovery._evidence_from_plan(source_after),
            proof_ref=f"proof:source-continuity-refresh:{foreign_digest}",
        )
        write_durable_change_train_manifest(train_path, substituted, expected_revision=train.revision)
        resumed = CliRunner().invoke(
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
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )

    assert resumed.exit_code != 0
    assert "neither the sealed pre-CAS nor post-CAS manifest" in resumed.output


def test_historical_recovery_resume_and_committed_return_require_exact_post_cas_manifest(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepared and committed recovery accept only the plan-sealed post-CAS bytes.

    Anti-vacuity: the public apply route crashes immediately after the real
    manifest CAS. A different checksummed released manifest retaining the
    exact V1 recovery evidence plus one foreign proof ref used to resume and
    later return as committed.
    """
    from polylogue.operations import historical_source_continuity_recovery as recovery

    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(moved_root)}
    plan_path = tmp_path / "exact-post-cas-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
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
        assert planned.exit_code == 0, planned.output
        plan = _maintenance_json_output(planned.output)
        plan_sha256 = str(plan["plan_sha256"])
        train_path = Path(str(plan["source_train_path"]))
        real_write = write_durable_change_train_manifest

        def crash_after_manifest_cas(path: Path, train: DurableChangeTrain, *, expected_revision: int) -> None:
            real_write(path, train, expected_revision=expected_revision)
            raise RuntimeError("crash after historical recovery manifest CAS")

        monkeypatch.setattr(recovery, "write_durable_change_train_manifest", crash_after_manifest_cas)
        with pytest.raises(RuntimeError, match="crash after historical recovery manifest CAS"):
            CliRunner().invoke(
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
                    plan_sha256,
                    "--output-format",
                    "json",
                ],
                env=command_env,
                catch_exceptions=False,
            )
        monkeypatch.setattr(recovery, "write_durable_change_train_manifest", real_write)
        exact_post_cas = train_path.read_bytes()
        exact_train = load_durable_change_train_manifest(train_path)
        substituted = replace(
            exact_train,
            proof_refs=(*exact_train.proof_refs, "proof:foreign-post-cas-substitution"),
        )
        train_path.write_text(
            json.dumps(durable_change_train_to_payload(substituted), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_prepared = CliRunner().invoke(
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
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
        assert rejected_prepared.exit_code != 0
        assert "neither the sealed pre-CAS nor post-CAS manifest" in rejected_prepared.output

        train_path.write_bytes(exact_post_cas)
        committed = CliRunner().invoke(
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
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )
        assert committed.exit_code == 0, committed.output
        train_path.write_text(
            json.dumps(durable_change_train_to_payload(substituted), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_committed = CliRunner().invoke(
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
                plan_sha256,
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )

    assert rejected_committed.exit_code != 0
    assert "neither the sealed pre-CAS nor post-CAS manifest" in rejected_committed.output


def test_historical_recovery_rejects_foreign_train_authority_before_preparing(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign valid refresh cannot make this recovery operation prepared."""
    moved_root, mutation_receipt, pre_manifest, post_manifest, evidence = _historical_continuity_fixture(
        workspace_env, tmp_path, monkeypatch
    )
    command_env = {"POLYLOGUE_ARCHIVE_ROOT": str(moved_root)}
    plan_path = tmp_path / "foreign-pre-prepare-plan.json"
    with _test_historical_operation_evidence_resource(evidence):
        planned = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "source-continuity-recovery",
                "plan",
                "--old-root",
                str(workspace_env["archive_root"]),
                "--mutation-receipt",
                str(mutation_receipt),
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
        assert planned.exit_code == 0, planned.output
        plan = _maintenance_json_output(planned.output)
        _substitute_foreign_historical_refresh(
            moved_root,
            plan=plan,
            mutation_receipt=mutation_receipt,
            backup_manifest=pre_manifest,
        )
        applied = CliRunner().invoke(
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
                str(plan["plan_sha256"]),
                "--output-format",
                "json",
            ],
            env=command_env,
            catch_exceptions=False,
        )

    assert applied.exit_code != 0
    assert "neither the sealed pre-CAS nor post-CAS manifest" in applied.output
    plan_sha256 = str(plan["plan_sha256"])
    assert not (
        moved_root / ".maintenance-state" / "historical-source-continuity-recoveries" / f"{plan_sha256}.json"
    ).exists()
    assert not (
        moved_root / ".maintenance-state" / "historical-source-continuity-recovery-plans" / f"{plan_sha256}.json"
    ).exists()


def _adopted_audit_archive_with_later_train(
    archive_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_count: int = 1,
) -> tuple[int, tuple[int, ...]]:
    """Build the adoption-era audit shape: adopted at vN, then migrated to vN+1.

    This is the shape an established archive reaches through the supported
    route -- ``migrate-tier audit --adopt-established-audit`` publishes a
    canonical image and its receipt, and a later ``migrate-tier audit``
    publishes one train for the next version only.  No train manifest exists
    at or below the adopted version, because no migration ever produced it.
    """
    from polylogue.operations.durable_change_train import (
        acquire_durable_archive_ownership,
        adopt_missing_audit_tier,
        execute_durable_change_train,
    )
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import (
        DurableChangeRider,
        DurableRuntimeConsumer,
        declare_durable_change_train,
        durable_change_train_to_payload,
        durable_migration_claim_for_sql,
    )

    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=tmp_path / "adoption-backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    backup_manifest = Path(backup.output_path) / "manifest.json"
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption") as owner:
        adopted_version, _receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=backup_manifest,
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    assert adopted_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]

    target_versions = tuple(adopted_version + step for step in range(1, train_count + 1))
    # Unique per test: the fixture package is imported and cached in sys.modules.
    package_name = f"fixture_migrations_adopted_audit_{tmp_path.name.replace('-', '_')}"
    package_root = tmp_path / package_name
    tier_package = package_root / ArchiveTier.AUDIT.value
    # Sidecar discovery requires a contiguous slot run, so the fixture package
    # carries the real audit migrations plus the synthetic slots above them.
    real_audit_migrations = Path(str(migration_runner.__file__)).parent / "migrations" / ArchiveTier.AUDIT.value
    shutil.copytree(real_audit_migrations, tier_package)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    statements: list[str] = []
    for target_version in target_versions:
        slot = f"{target_version:03d}"
        sql = (
            "-- migration-safety: additive-no-backup\n"
            f"CREATE TABLE adopted_audit_probe_{target_version} (id INTEGER PRIMARY KEY) STRICT;\n"
        )
        statements.append(sql)
        migration_name = f"{slot}_adopted_audit_probe.sql"
        (tier_package / migration_name).write_text(sql, encoding="utf-8")
        claim = durable_migration_claim_for_sql(
            ArchiveTier.AUDIT,
            migration_name,
            sql,
            owner_ref="owner:adopted-audit",
        )
        rider = DurableChangeRider(
            rider_id=f"rider:adopted-audit:{target_version}",
            owner_ref="owner:adopted-audit",
            schema_objects=(f"table:adopted_audit_probe_{target_version}",),
            runtime_consumers=(
                DurableRuntimeConsumer(
                    "bootstrap",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                    "proof:bootstrap",
                    ("write",),
                ),
                DurableRuntimeConsumer(
                    "daemon-health",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                    "proof:daemon-health",
                    ("read",),
                ),
            ),
            behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
        )
        declared = declare_durable_change_train(
            train_id=f"train:audit:adopted-v{target_version}",
            tier=ArchiveTier.AUDIT,
            current_version=target_version - 1,
            target_version=target_version,
            slot=target_version,
            owner_ref="owner:adopted-audit",
            migration=claim,
            riders=(rider,),
            declared_at_ms=1,
        )
        (tier_package / f"{slot}.train.json").write_text(
            json.dumps(durable_change_train_to_payload(declared)), encoding="utf-8"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        migration_runner, "_migration_package", lambda _tier: f"{package_name}.{ArchiveTier.AUDIT.value}"
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: f"{package_name}.{ArchiveTier.AUDIT.value}",
    )
    for applied_version in target_versions:
        # Each train carries exactly one slot, so the runtime version advances
        # one step at a time, as the shipped migration route does.
        versions = dict(ARCHIVE_VERSION_BY_TIER)
        versions[ArchiveTier.AUDIT] = applied_version
        monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
        monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
        applied_statements = statements[: applied_version - adopted_version]
        ddl = dict(ARCHIVE_DDL_BY_TIER)
        ddl[ArchiveTier.AUDIT] = "\n".join((ARCHIVE_DDL_BY_TIER[ArchiveTier.AUDIT], *applied_statements))
        monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
        monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-migrate") as owner:
            execution = execute_durable_change_train(
                archive_root,
                ArchiveTier.AUDIT,
                backup_manifest=None,
                daemon_stopped_evidence_ref="proof:test-daemon-stopped",
                single_writer_evidence_ref="proof:archive-ownership-lock",
                release_archive_ownership=owner.release,
            )
        assert execution.manifest_path is not None

    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    train_manifests = sorted(
        path.name for path in manifest_root.glob("audit-*.json") if re.fullmatch(r"audit-\d{3,}\.json", path.name)
    )
    # The exact live shape: trains only above the adopted version, and no
    # manifest at or below it.
    assert train_manifests == [f"audit-{version:03d}.json" for version in target_versions]
    assert (manifest_root / "audit-adoption.json").is_file()
    with sqlite3.connect(audit_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (target_versions[-1],)
    return adopted_version, target_versions


def test_relocation_accepts_an_audit_tier_adopted_below_its_only_train(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adopted audit tier carries its own evidence for versions with no train.

    Anti-vacuity: computing ``expected_targets`` from the raw
    ``DURABLE_MIGRATION_ADOPTION_FLOORS[AUDIT]`` instead of the chain floor
    demands a train manifest for every version between the floor and the
    adopted image, which the archive never performed, and this test goes red
    with "unexpected audit train target".
    """
    from polylogue.operations import archive_root_relocation as relocation

    archive_root = workspace_env["archive_root"]
    adopted_version, target_versions = _adopted_audit_archive_with_later_train(archive_root, tmp_path, monkeypatch)
    target_version = target_versions[-1]
    assert adopted_version >= DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.AUDIT] + 1

    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    bootstrap_marker = manifest_root / ".bootstrap"
    if bootstrap_marker.exists():
        bootstrap_marker.unlink()
    # Only the audit tier is under test; keep the other durable tiers at their
    # own current version so their chain checks are trivially satisfied.
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)

    snapshots = tuple(
        relocation._tier_snapshot(
            archive_root,
            tier,
            backup_device=0,
            backup_inode=0,
        )
        for tier in ArchiveTier
    )
    audit_snapshot = next(item for item in snapshots if item.tier == ArchiveTier.AUDIT.value)
    assert audit_snapshot.user_version == target_version

    trains = relocation._durable_trains(archive_root, old_root=archive_root, snapshots=snapshots)
    audit_trains = [train for train in trains if train.tier == ArchiveTier.AUDIT.value]
    assert [Path(train.path).name for train in audit_trains] == [f"audit-{target_version:03d}.json"]


def test_relocation_still_refuses_a_missing_audit_train_without_adoption_evidence(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the adoption receipt restores the refusal for the same archive."""
    from polylogue.operations import archive_root_relocation as relocation
    from polylogue.operations.durable_change_train import audit_adoption_receipt_path

    archive_root = workspace_env["archive_root"]
    _adopted_audit_archive_with_later_train(archive_root, tmp_path, monkeypatch)

    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    bootstrap_marker = manifest_root / ".bootstrap"
    if bootstrap_marker.exists():
        bootstrap_marker.unlink()
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)
    audit_adoption_receipt_path(archive_root).unlink()

    snapshots = tuple(
        relocation._tier_snapshot(archive_root, tier, backup_device=0, backup_inode=0) for tier in ArchiveTier
    )
    with pytest.raises(ArchiveRootRelocationError, match="unexpected audit train target"):
        relocation._durable_trains(archive_root, old_root=archive_root, snapshots=snapshots)


def test_relocation_accepts_durable_trains_below_the_chain_floor(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical trains at or below the floor are evidence, not a fault.

    This is the source-tier shape of the same bug: an archive that walked the
    numbered train from below its adoption floor keeps those manifests, and a
    set-equality check against the versions above the floor refuses it.

    Anti-vacuity: requiring the manifest set to equal the expected targets
    exactly makes this red with "unexpected audit train target".
    """
    from polylogue.operations import archive_root_relocation as relocation

    archive_root = workspace_env["archive_root"]
    _adopted, target_versions = _adopted_audit_archive_with_later_train(
        archive_root, tmp_path, monkeypatch, train_count=2
    )
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    bootstrap_marker = manifest_root / ".bootstrap"
    if bootstrap_marker.exists():
        bootstrap_marker.unlink()
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)
    # Raise the audit floor above the first train, leaving one released train
    # manifest below the floor exactly as source-027..030 sit below source v37.
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.AUDIT, target_versions[0])

    snapshots = tuple(
        relocation._tier_snapshot(archive_root, tier, backup_device=0, backup_inode=0) for tier in ArchiveTier
    )
    trains = relocation._durable_trains(archive_root, old_root=archive_root, snapshots=snapshots)
    assert sorted(Path(train.path).name for train in trains if train.tier == ArchiveTier.AUDIT.value) == [
        f"audit-{version:03d}.json" for version in target_versions
    ]


def test_relocation_refuses_a_train_above_the_live_durable_version(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A train claiming a version the live tier never reached is still a fault."""
    from polylogue.operations import archive_root_relocation as relocation

    archive_root = workspace_env["archive_root"]
    _adopted, target_versions = _adopted_audit_archive_with_later_train(archive_root, tmp_path, monkeypatch)
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    bootstrap_marker = manifest_root / ".bootstrap"
    if bootstrap_marker.exists():
        bootstrap_marker.unlink()
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, 10_000)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, 10_000)
    with sqlite3.connect(archive_root / "audit.db") as connection:
        connection.execute(f"PRAGMA user_version = {target_versions[-1] - 1}")
        connection.commit()
        # Relocation reads durable tiers immutably, which ignores the WAL.
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    snapshots = tuple(
        relocation._tier_snapshot(archive_root, tier, backup_device=0, backup_inode=0) for tier in ArchiveTier
    )
    audit_snapshot = next(item for item in snapshots if item.tier == ArchiveTier.AUDIT.value)
    assert audit_snapshot.user_version == target_versions[-1] - 1
    with pytest.raises(ArchiveRootRelocationError, match="unexpected audit train target"):
        relocation._durable_trains(archive_root, old_root=archive_root, snapshots=snapshots)


def _quiesce_archive_sidecars(root: Path) -> None:
    """Drop every SQLite sidecar, as the operator does before an offline move.

    Relocation refuses to move a tier that still has a WAL beside it, so each
    tier is checkpointed and left in rollback-journal mode; a later read-only
    open then cannot recreate one.
    """
    for tier in ArchiveTier:
        tier_path = root / f"{tier.value}.db"
        if tier_path.is_file():
            with closing(sqlite3.connect(tier_path)) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                connection.execute("PRAGMA journal_mode=DELETE")
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = root / f"{tier.value}.db{suffix}"
            if sidecar.exists():
                sidecar.unlink()


def _live_shaped_durable_archive(
    archive_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Build one archive carrying the live archive's whole durable chain shape.

    ``/realm/state/polylogue`` (measured read-only) is a single archive that
    holds *both* halves of the relocation refusal at once:

    * ``audit.db`` reached its version by **adoption** -- ``audit-adoption.json``
      and no ``audit-0NN.json`` train manifest at or below the adopted version.
      Migrating the tier forward then publishes exactly one train, above the
      adopted image.
    * ``source.db`` carries released train manifests (``source-027..030``) whose
      targets sit **below** the source adoption floor (37), while the versions
      above that floor each still have their own released train.
    * ``user.db`` carries its one train above the user floor.

    Reproducing that as two separate single-property fixtures is weaker
    evidence than the criterion asks for, so this builder composes all of it
    into one archive, through the real routes: ``adopt_missing_audit_tier`` for
    the adoption receipt and ``execute_durable_change_train`` for every
    manifest. Nothing here hand-writes a train manifest or a receipt.

    The live slot numbers cannot be replayed literally -- a fresh archive is
    bootstrapped at the current runtime DDL, and the shipped ``source-031..044``
    migrations cannot be re-applied to it -- so the fixture reproduces the
    *shape* with synthetic slots above the runtime versions and the adoption
    floors moved to the matching relative positions.
    """
    from polylogue.operations.durable_change_train import (
        acquire_durable_archive_ownership,
        adopt_missing_audit_tier,
        execute_durable_change_train,
    )
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import (
        DurableChangeRider,
        DurableRuntimeConsumer,
        declare_durable_change_train,
        durable_change_train_to_payload,
        durable_migration_claim_for_sql,
    )

    initialize_active_archive_root(archive_root)

    # --- audit tier: adopted, never walked ------------------------------------
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=tmp_path / "adoption-backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:live-shape-adoption") as owner:
        adopted_audit_version, _receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    assert adopted_audit_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]

    # --- the synthetic slots each durable tier will walk -----------------------
    base_versions = {tier: ARCHIVE_VERSION_BY_TIER[tier] for tier in DURABLE_MIGRATION_ADOPTION_FLOORS}
    # source walks four slots; the floor will later sit after the first two, so
    # two released manifests land below it and two above -- the source-027..030
    # (below) plus source-038..044 (above) shape in one tier.
    slot_plan = {
        ArchiveTier.AUDIT: (base_versions[ArchiveTier.AUDIT] + 1,),
        ArchiveTier.SOURCE: tuple(base_versions[ArchiveTier.SOURCE] + step for step in range(1, 5)),
        ArchiveTier.USER: (base_versions[ArchiveTier.USER] + 1,),
    }

    package_name = f"fixture_migrations_live_shape_{tmp_path.name.replace('-', '_')}"
    package_root = tmp_path / package_name
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    real_migration_root = Path(str(migration_runner.__file__)).parent / "migrations"
    statements_by_tier: dict[ArchiveTier, list[str]] = {}
    for tier, slots in slot_plan.items():
        tier_package = package_root / tier.value
        shutil.copytree(real_migration_root / tier.value, tier_package)
        statements: list[str] = []
        for target_version in slots:
            slot = f"{target_version:03d}"
            table = f"live_shape_probe_{tier.value}_{target_version}"
            sql = f"-- migration-safety: additive-no-backup\nCREATE TABLE {table} (id INTEGER PRIMARY KEY) STRICT;\n"
            statements.append(sql)
            migration_name = f"{slot}_live_shape_probe.sql"
            (tier_package / migration_name).write_text(sql, encoding="utf-8")
            claim = durable_migration_claim_for_sql(tier, migration_name, sql, owner_ref="owner:live-shape")
            rider = DurableChangeRider(
                rider_id=f"rider:live-shape:{tier.value}:{target_version}",
                owner_ref="owner:live-shape",
                schema_objects=(f"table:{table}",),
                runtime_consumers=(
                    DurableRuntimeConsumer(
                        "bootstrap",
                        "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                        "proof:bootstrap",
                        ("write",),
                    ),
                    DurableRuntimeConsumer(
                        "daemon-health",
                        "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                        "proof:daemon-health",
                        ("read",),
                    ),
                ),
                behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
            )
            declared = declare_durable_change_train(
                train_id=f"train:{tier.value}:live-shape-v{target_version}",
                tier=tier,
                current_version=target_version - 1,
                target_version=target_version,
                slot=target_version,
                owner_ref="owner:live-shape",
                migration=claim,
                riders=(rider,),
                declared_at_ms=1,
            )
            (tier_package / f"{slot}.train.json").write_text(
                json.dumps(durable_change_train_to_payload(declared)), encoding="utf-8"
            )
        statements_by_tier[tier] = statements

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda tier: f"{package_name}.{tier.value}")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda tier: f"{package_name}.{tier.value}",
    )

    # --- walk every tier through the real train route -------------------------
    # Mutate the canonical mappings in place: every reader -- the migration
    # runner, bootstrap, and the read-only connection profile's schema-skew
    # guard -- resolves them from this one dict, so rebinding a copy in two
    # modules would leave the rest of the runtime refusing the walked tier.
    base_ddl = {tier: ARCHIVE_DDL_BY_TIER[tier] for tier in slot_plan}
    applied_by_tier: dict[ArchiveTier, list[str]] = {tier: [] for tier in slot_plan}
    for tier in (ArchiveTier.AUDIT, ArchiveTier.SOURCE, ArchiveTier.USER):
        for target_version in slot_plan[tier]:
            applied_by_tier[tier].append(statements_by_tier[tier][len(applied_by_tier[tier])])
            monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, tier, target_version)
            monkeypatch.setitem(ARCHIVE_DDL_BY_TIER, tier, "\n".join((base_ddl[tier], *applied_by_tier[tier])))
            with acquire_durable_archive_ownership(archive_root, owner_id=f"test:live-shape:{tier.value}") as owner:
                execution = execute_durable_change_train(
                    archive_root,
                    tier,
                    backup_manifest=None,
                    daemon_stopped_evidence_ref="proof:test-daemon-stopped",
                    single_writer_evidence_ref="proof:archive-ownership-lock",
                    release_archive_ownership=owner.release,
                )
            assert execution.manifest_path is not None

    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    bootstrap_marker = manifest_root / ".bootstrap"
    # The live archive predates the fresh-bootstrap marker and carries none, so
    # every tier falls back to its declared adoption floor.
    if bootstrap_marker.exists():
        bootstrap_marker.unlink()
    # Place the source floor where the live floor sits: after the first two
    # released trains, so those two manifests are history below the floor and
    # the rest are the versions that still require their train.
    source_floor = slot_plan[ArchiveTier.SOURCE][1]
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.SOURCE, source_floor)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, ArchiveTier.USER, base_versions[ArchiveTier.USER])

    # --- assert the archive really carries the measured live shape ------------
    def _manifest_targets(tier: ArchiveTier) -> list[int]:
        return sorted(
            int(path.stem.split("-")[-1])
            for path in manifest_root.glob(f"{tier.value}-*.json")
            if re.fullmatch(rf"{tier.value}-\d{{3,}}\.json", path.name)
        )

    assert (manifest_root / "audit-adoption.json").is_file()
    # audit: adopted image, one train above it, nothing at or below it.
    assert _manifest_targets(ArchiveTier.AUDIT) == [adopted_audit_version + 1]
    # source: released manifests below the floor and above it, in one tier.
    source_targets = _manifest_targets(ArchiveTier.SOURCE)
    assert source_targets == list(slot_plan[ArchiveTier.SOURCE])
    assert [target for target in source_targets if target <= source_floor] == list(slot_plan[ArchiveTier.SOURCE][:2])
    assert [target for target in source_targets if target > source_floor] == list(slot_plan[ArchiveTier.SOURCE][2:])
    assert _manifest_targets(ArchiveTier.USER) == [base_versions[ArchiveTier.USER] + 1]
    for tier in slot_plan:
        with closing(sqlite3.connect(archive_root / f"{tier.value}.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (slot_plan[tier][-1],)

    _quiesce_archive_sidecars(archive_root)

    return {
        "adopted_audit_version": adopted_audit_version,
        "slot_plan": slot_plan,
        "source_floor": source_floor,
    }


def test_live_shaped_archive_relocates_end_to_end(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live archive's whole durable shape relocates, in one archive.

    This is the acceptance case for the operator's archive: an adopted audit
    tier with no train at or below its adopted image, *and* a source tier whose
    released train history reaches below its adoption floor, carried by one
    archive and moved through the real ``prepare``/``apply`` relocation route.

    Anti-vacuity, demonstrated by reverting each half of the fix in turn:

    * computing ``chain_floor`` in ``_durable_trains`` from the raw
      ``DURABLE_MIGRATION_ADOPTION_FLOORS[tier]`` instead of ``_chain_floor``
      makes this red with "archive-root relocation found an unexpected audit
      train target" -- the versions between the raw floor and the adopted image
      have no manifest the archive was ever supposed to produce;
    * restoring the set-equality manifest check (``set(manifests) !=
      expected_targets``) makes this red with "... unexpected source train
      target" -- the released trains below the source floor are history, not a
      fault.

    Both are the refusal measured against the operator's live archive.
    """
    old_root = workspace_env["archive_root"]
    shape = _live_shaped_durable_archive(old_root, tmp_path, monkeypatch)
    slot_plan: dict[ArchiveTier, tuple[int, ...]] = shape["slot_plan"]  # type: ignore[assignment]

    new_root = tmp_path / "moved"
    os.rename(old_root, new_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(new_root))
    backup = backup_archive(output_dir=tmp_path / "relocation-backup", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None

    plan = prepare_archive_root_relocation(
        old_root=old_root,
        new_root=new_root,
        backup_manifest=Path(backup.output_path) / "manifest.json",
        stopped_daemon_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
    )
    planned = {(item.tier, Path(item.path).name) for item in plan.durable_trains}
    assert planned == {
        (tier.value, f"{tier.value}-{target:03d}.json") for tier, slots in slot_plan.items() for target in slots
    }
    # Planning opens every tier read-only, which can leave a WAL behind.
    _quiesce_archive_sidecars(new_root)
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert result.state == "committed"

    identity = ArchiveIdentity.resolve(new_root)
    for item in plan.durable_trains:
        train = load_durable_change_train_manifest(Path(item.path))
        assert train.apply_evidence is not None
        assert (
            train.apply_evidence.post.archive_identity_digest
            == hashlib.sha256(identity.tier(item.tier).stable_id.encode()).hexdigest()
        )
