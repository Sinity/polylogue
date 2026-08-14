"""Regression coverage for the offline inode-preserving archive-root move."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.backup import backup_archive
from polylogue.operations.archive_root_relocation import (
    ArchiveRootRelocationError,
    RelocationActiveIndexPointer,
    RelocationTierEvidence,
    _check_backup_against_live,
    apply_archive_root_relocation,
    assert_no_prepared_archive_root_relocation,
    prepare_archive_root_relocation,
)
from polylogue.operations.archive_root_relocation import (
    _sealed_receipt as _sealed_relocation_receipt,
)
from polylogue.operations.archive_root_relocation import (
    _write_receipt as _write_relocation_receipt,
)
from polylogue.operations.historical_source_continuity_recovery import (
    HistoricalSourceContinuityRecoveryError,
    _assert_complete_source_semantic_delta,
    _assert_exact_liveness_delta,
    _current_evidence,
    _sha256,
    _table_content_digest,
    _verify_historical_operation_evidence,
    _write_refresh_receipt,
    assert_no_prepared_historical_source_continuity_recovery,
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
    load_durable_change_train_manifest,
    rebind_released_durable_train_archive_identity,
    recover_released_source_train_continuity,
)
from polylogue.storage.sqlite.migration_runner import (
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
) -> Path:
    """Retarget one validated released fixture train to another durable tier."""
    source = load_durable_change_train_manifest(source_manifest)
    assert source.fresh_ddl_parity is not None
    assert source.reservation is not None
    assert source.backup_authorization is not None
    assert source.pre_apply_evidence is not None
    assert source.apply_evidence is not None
    assert source.proof is not None
    cloned = replace(
        source,
        train_id=f"train:{tier.value}:v{source.target_version}",
        tier=tier,
        migration=replace(source.migration, tier=tier),
        fresh_ddl_parity=replace(source.fresh_ddl_parity, tier=tier),
        reservation=replace(source.reservation, tier_path=str(root / f"{tier.value}.db")),
        backup_authorization=replace(
            source.backup_authorization,
            live_tier_path=str(root / f"{tier.value}.db"),
        ),
        pre_apply_evidence=replace(source.pre_apply_evidence, tier=tier),
        apply_evidence=replace(
            source.apply_evidence,
            pre=replace(source.apply_evidence.pre, tier=tier),
            post=replace(source.apply_evidence.post, tier=tier),
            migration_result=replace(source.apply_evidence.migration_result, tier=tier),
        ),
        proof=replace(
            source.proof,
            fresh_ddl_parity=replace(source.proof.fresh_ddl_parity, tier=tier),
        ),
    )
    manifest = root / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-002.json"
    write_durable_change_train_manifest(manifest, cloned, expected_revision=-1)
    monkeypatch.setitem(DURABLE_MIGRATION_ADOPTION_FLOORS, tier, 1)
    return manifest


def _activate_movable_index_generation(root: Path) -> Path:
    """Promote a real generation using the production absolute symlink layout."""
    store = IndexGenerationStore.for_archive_root(root)
    generation = store.create(owner_id="relocation-test", source_snapshot="snapshot")
    store.promote(generation)
    return Path(generation.index_path).resolve(strict=True)


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
        train_after_sha256=None,
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
                train_after_sha256=None,
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
            train_after_sha256=None,
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
        assert f"--plan {retained_plan}" in prepared_receipt["resume_command"]
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
    with pytest.raises(ArchiveRootRelocationError, match="prepared but incomplete"):
        assert_no_prepared_archive_root_relocation(new_root)

    monkeypatch.setattr(relocation, "_publish_active_index_pointer", real_publish)
    result = apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256)
    assert result.state == "committed"
    relocated_location = ArchiveLocation.resolve(new_root)
    assert relocated_location.active_index_path == Path(pointer.new_target)
    assert relocated_location.active_index.resolved_path == Path(pointer.new_resolved_target)
    assert apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256).state == "committed"


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
    assert apply_archive_root_relocation(root=new_root, plan=plan, authorization=plan.plan_sha256).state == "committed"
    assert moved_manifest.read_bytes() == before


def test_relocation_resume_rejects_a_same_revision_manifest_substituted_after_cas(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepared recovery accepts only the exact post-CAS bytes bound before mutation."""
    from polylogue.operations import archive_root_relocation as relocation

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
    assert f"--plan {retained_plan}" in prepared_receipt["resume_command"]
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
        _clone_released_durable_train_for_tier(old_root, user_manifest, ArchiveTier.AUDIT, monkeypatch),
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
    assert "source continuity authority is invalid" in rejected_plan.output

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
    assert "continuity receipt is invalid" in rejected_apply.output

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
        "kind": "relocation",
        "sha256": second_transition_refs[-2].rsplit(":", 1)[-1],
    }
    from polylogue.storage.sqlite import durable_change_train as trains

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
    assert "exact refresh proof" in resumed.output
