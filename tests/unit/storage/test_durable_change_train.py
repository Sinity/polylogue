"""Lifecycle contracts for durable source/user schema-change authority."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import polylogue.storage.sqlite.durable_change_train as durable_change_train_module
from polylogue.daemon.backup import backup_archive
from polylogue.operations.durable_change_train import (
    _audit_live_metadata,
    _write_immutable_audit_adoption_receipt,
    acquire_durable_archive_ownership,
    adopt_missing_audit_tier,
    audit_adoption_receipt_path,
    restore_adopted_audit_tier,
)
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    DurableSourceContinuitySemanticError,
    DurableSourceTrainMissingError,
    _runtime_consumer_results,
    assert_source_continuity_apply_allowed,
    durable_change_train_manifest_path,
    durable_change_train_policy_report,
    durable_migration_sidecar_for_slot,
    execute_durable_change_train,
    mark_source_continuity_pending_intent_terminal,
    reconcile_durable_change_train_startup,
    refresh_released_source_train_continuity,
    validate_durable_migration_sidecars,
    write_source_continuity_pending_intent,
)
from polylogue.storage.sqlite.migration_runner import (
    DurableChangeRider,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableFailureClassification,
    DurableFreshDDLParityProof,
    DurableMigrationClaim,
    DurableRuntimeConsumer,
    DurableRuntimeConsumerResult,
    MigrationError,
    add_durable_change_train_rider,
    admit_durable_change_train,
    apply_durable_change_train,
    authorize_durable_change_train_backup,
    capture_durable_restart_convergence,
    declare_durable_change_train,
    durable_migration_claim_for_sql,
    durable_migration_collision_report,
    load_durable_change_train_manifest,
    migrate_archive_tier,
    prove_durable_change_train,
    prove_durable_fresh_ddl_parity,
    reconcile_interrupted_durable_change_train,
    record_durable_writer_release,
    recover_durable_change_train,
    release_durable_change_train,
    reserve_durable_change_train,
    write_durable_change_train_manifest,
)

_CURRENT_VERSION = 1
_TARGET_VERSION = 2
_EMPTY_LIVENESS_DIGEST = hashlib.sha256(b"[]").hexdigest()
_ADDITIVE_SQL = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (
    item_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
) STRICT;
"""


@contextmanager
def _memory_target(*, include_durable_items: bool = True) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        if include_durable_items:
            conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
        yield conn
    finally:
        conn.close()


def _create_current_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute("INSERT INTO base_items VALUES ('base-1', 'preserve-me')")
        conn.execute(f"PRAGMA user_version = {_CURRENT_VERSION}")
        conn.commit()


def _claim(tier: ArchiveTier, sql: str = _ADDITIVE_SQL) -> DurableMigrationClaim:
    return durable_migration_claim_for_sql(
        tier,
        "002_durable_items.sql",
        sql,
        owner_ref=f"owner:migration:{tier.value}:002",
    )


def _rider(*, consumer_count: int = 2, trust_floor_exception_ref: str | None = None) -> DurableChangeRider:
    consumers = tuple(
        DurableRuntimeConsumer(
            consumer_id=f"consumer-{index}",
            production_ref=f"polylogue/storage/{'writer' if index == 0 else 'reader'}_{index}.py:consume",
            behavior_proof_ref=f"proof:behavior:{index}",
            roles=("write" if index == 0 else "read",),
        )
        for index in range(consumer_count)
    )
    return DurableChangeRider(
        rider_id="rider:durable-items",
        owner_ref="owner:rider",
        schema_objects=("table:durable_items",),
        runtime_consumers=consumers,
        behavior_proof_refs=tuple(consumer.behavior_proof_ref for consumer in consumers),
        trust_floor_exception_ref=trust_floor_exception_ref,
    )


def _production_rider() -> DurableChangeRider:
    return DurableChangeRider(
        rider_id="rider:startup",
        owner_ref="owner:startup-rider",
        schema_objects=("table:durable_items",),
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


def _source_hook_event_production_rider() -> DurableChangeRider:
    return DurableChangeRider(
        rider_id="rider:source-hook-event",
        owner_ref="owner:source-hook-event",
        schema_objects=("table:raw_hook_events",),
        runtime_consumers=(
            DurableRuntimeConsumer(
                "source-hook-event-writer",
                "polylogue/storage/sqlite/archive_tiers/source_write.py:write_source_hook_event",
                "proof:source-v27:raw-hook-events-origin-repair",
                ("write",),
            ),
            DurableRuntimeConsumer(
                "source-hook-event-bootstrap",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                "proof:source-v27:raw-hook-events-origin-repair",
                ("read",),
            ),
        ),
        behavior_proof_refs=("proof:source-v27:raw-hook-events-origin-repair",),
    )


def _parity(tier: ArchiveTier, *, include_durable_items: bool = True) -> DurableFreshDDLParityProof:
    with _memory_target(include_durable_items=include_durable_items) as migrated:
        with _memory_target() as fresh:
            return prove_durable_fresh_ddl_parity(
                tier,
                _TARGET_VERSION,
                migrated_connection=migrated,
                fresh_connection=fresh,
                evidence_ref=f"proof:fresh-ddl:{tier.value}",
            )


def _declared(
    tier: ArchiveTier,
    *,
    claim: DurableMigrationClaim | None = None,
    rider: DurableChangeRider | None = None,
    owner_ref: str = "owner:train",
    backup_plan_ref: str | None = None,
) -> DurableChangeTrain:
    migration = claim or _claim(tier)
    return declare_durable_change_train(
        train_id=f"train:{tier.value}:v{_TARGET_VERSION}",
        tier=tier,
        current_version=_CURRENT_VERSION,
        target_version=_TARGET_VERSION,
        slot=_TARGET_VERSION,
        owner_ref=owner_ref,
        migration=migration,
        riders=((rider or _rider()),),
        backup_plan_ref=backup_plan_ref,
        declared_at_ms=1,
    )


def _admitted(
    tier: ArchiveTier,
    *,
    claim: DurableMigrationClaim | None = None,
    rider: DurableChangeRider | None = None,
    owner_ref: str = "owner:train",
    backup_plan_ref: str | None = None,
    active_trains: tuple[DurableChangeTrain, ...] = (),
) -> DurableChangeTrain:
    migration = claim or _claim(tier)
    return admit_durable_change_train(
        _declared(
            tier,
            claim=migration,
            rider=rider,
            owner_ref=owner_ref,
            backup_plan_ref=backup_plan_ref,
        ),
        observed_current_version=_CURRENT_VERSION,
        fresh_ddl_parity=_parity(tier),
        admission_evidence_ref=f"proof:admit:{tier.value}",
        active_trains=active_trains,
        migration_claims=(migration,),
        canonical_target_version=_TARGET_VERSION,
        admitted_at_ms=2,
    )


def _install_synthetic_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    *,
    sql: str = _ADDITIVE_SQL,
) -> None:
    package_name = f"fixture_migrations_{tier.value}_{tmp_path.name.replace('-', '_')}"
    package_root = tmp_path / package_name
    tier_package = package_root / tier.value
    tier_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "002_durable_items.sql").write_text(sql, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[tier] = _TARGET_VERSION
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(
        migration_runner, "_migration_package", lambda observed_tier: f"{package_name}.{observed_tier.value}"
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda observed_tier: f"{package_name}.{observed_tier.value}",
    )


def _reserve_and_authorize(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    archive_root: Path,
) -> DurableChangeTrain:
    reserved = reserve_durable_change_train(
        train,
        reservation_id="lease:archive-root",
        reservation_owner_ref=train.owner_ref,
        archive_root=archive_root,
        tier_path=archive_root / f"{train.tier.value}.db",
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:rebuild-lease",
        reserved_at_ms=3,
    )
    return authorize_durable_change_train_backup(
        conn,
        reserved,
        backup_manifest=None,
        evidence_ref="proof:additive-no-backup",
        authorized_at_ms=4,
    )


def _runtime_results() -> tuple[DurableRuntimeConsumerResult, ...]:
    return (
        DurableRuntimeConsumerResult("consumer-0", "proof:behavior:0", True),
        DurableRuntimeConsumerResult("consumer-1", "proof:behavior:1", True),
    )


def test_source_v27_sidecar_proves_the_real_hook_event_writer_against_fresh_schema(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    sidecar = durable_migration_sidecar_for_slot(ArchiveTier.SOURCE, 27)
    assert sidecar is not None
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)

    results = _runtime_consumer_results(sidecar.train, tmp_path)

    hook_writer = next(result for result in results if result.consumer_id == "source-hook-event-writer")
    assert hook_writer.passed is True
    assert hook_writer.behavior_proof_ref == "proof:source-v27:raw-hook-events-origin-repair"
    assert hook_writer.detail == "wrote and read back a hook payload in a fresh source tier"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (0,)


def test_source_v29_sidecar_proves_failure_lifecycle_consumers_against_fresh_schema(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    sidecar = durable_migration_sidecar_for_slot(ArchiveTier.SOURCE, 29)
    assert sidecar is not None
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)

    results = _runtime_consumer_results(sidecar.train, tmp_path)

    assert [(result.consumer_id, result.passed) for result in results] == [
        ("raw-failure-lifecycle", True),
        ("historical-disposition-actuator", True),
    ]
    assert results[0].detail == "read raw failure lifecycle state=healthy"
    assert results[1].detail == "validated one raw failure disposition without mutation"


def test_applied_train_release_requires_the_source_hook_event_writer_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
    train = _admitted(ArchiveTier.SOURCE, rider=_source_hook_event_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)

    train = record_durable_writer_release(train, evidence_ref="proof:source-hook-event-writer-release")
    with sqlite3.connect(db_path) as restarted:
        actual_parity = _parity(ArchiveTier.SOURCE)
        runtime_results = _runtime_consumer_results(train, tmp_path)
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:source-hook-event-restart",
        )
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=actual_parity,
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    released = release_durable_change_train(train, evidence_ref="proof:source-hook-event-release")

    assert released.state is DurableChangeTrainState.RELEASED
    assert released.proof is not None
    assert released.apply_evidence is not None
    hook_writer = next(
        result for result in released.proof.runtime_consumers if result.consumer_id == "source-hook-event-writer"
    )
    assert hook_writer.passed is True


def test_released_source_train_can_record_an_authorized_mutation_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.archive_identity import ArchiveIdentity

    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    train = record_durable_writer_release(train, evidence_ref="proof:writer-release")
    with sqlite3.connect(db_path) as conn:
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            conn,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:restart",
        )
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    train = release_durable_change_train(train, evidence_ref="proof:release")
    assert train.apply_evidence is not None
    train = replace(
        train,
        apply_evidence=replace(
            train.apply_evidence,
            post=replace(
                train.apply_evidence.post,
                archive_identity_digest=ArchiveIdentity.resolve(tmp_path).authority_identity_digest,
            ),
        ),
    )
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    manifest.parent.mkdir(parents=True)
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)
    released = load_durable_change_train_manifest(manifest)
    with sqlite3.connect(db_path) as conn:
        before = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        conn.execute("INSERT INTO base_items VALUES ('mutation-1', 'authorized')")
        conn.commit()

    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    mutation_receipt = tmp_path / "mutation-receipt.jsonl"
    mutation_receipt.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(db_path),
                "backup_manifest": str(backup_manifest),
                "candidate_count": 0,
                "candidate_digest": _EMPTY_LIVENESS_DIGEST,
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
    refresh_fsync_calls: list[Path] = []
    real_fsync_manifest_directory = migration_runner._fsync_manifest_directory

    def record_refresh_fsync(path: Path) -> None:
        refresh_fsync_calls.append(path)
        real_fsync_manifest_directory(path)

    monkeypatch.setattr(migration_runner, "_fsync_manifest_directory", record_refresh_fsync)
    refreshed_path = refresh_released_source_train_continuity(
        tmp_path,
        mutation_receipt=mutation_receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id=_EMPTY_LIVENESS_DIGEST,
        evidence_ref="proof:mutation-1",
    )
    assert tmp_path / ".maintenance-state" in refresh_fsync_calls

    refreshed = load_durable_change_train_manifest(manifest)
    assert refreshed.state is DurableChangeTrainState.RELEASED
    assert refreshed.source_continuity_evidence is not None
    assert released.apply_evidence is not None
    assert refreshed.apply_evidence is not None
    assert released.apply_evidence is not None
    assert refreshed.apply_evidence.pre == released.apply_evidence.pre
    assert refreshed.apply_evidence.post.archive_identity_digest != released.apply_evidence.post.archive_identity_digest
    assert (
        refreshed.apply_evidence.post.archive_identity_digest
        == refreshed.source_continuity_evidence.archive_identity_digest
    )
    assert refreshed.proof == released.proof
    assert refreshed.revision == released.revision + 1
    assert any(ref.startswith("proof:source-continuity-refresh:") for ref in refreshed.proof_refs)
    assert refreshed.source_continuity_evidence.content_sha256 != released.apply_evidence.post.content_sha256
    assert refreshed_path.is_file()
    # A pending-intent retry after manifest persistence accepts only the
    # receipt's exact pre/post evidence, rather than bypassing pre-state
    # authentication because the receipt digest already exists.
    assert (
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-1",
        )
        == refreshed_path
    )
    with pytest.raises(DurableSourceContinuitySemanticError, match="unreceipted content drift"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=replace(before, content_sha256="f" * 64),
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-1",
        )
    with pytest.raises(DurableSourceContinuitySemanticError, match="source-tier pre-mutation evidence"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=replace(before, tier=ArchiveTier.USER),
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-wrong-tier",
        )
    operator_cwd = tmp_path / "operator-cwd"
    operator_cwd.mkdir()
    monkeypatch.chdir(tmp_path)
    pending_fsync_start = len(refresh_fsync_calls)
    pending_path = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=Path("mutation-receipt.jsonl"),
        backup_manifest=Path("backup-manifest.json"),
        pre_mutation_evidence=before,
        operation_id=_EMPTY_LIVENESS_DIGEST,
        evidence_ref="proof:mutation-1",
    )
    assert pending_path.is_file()
    assert tmp_path / ".maintenance-state" in refresh_fsync_calls[pending_fsync_start:]
    pending_payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending_payload["mutation_receipt"] == str(mutation_receipt)
    assert pending_payload["backup_manifest"] == str(backup_manifest)
    monkeypatch.chdir(operator_cwd)
    assert reconcile_durable_change_train_startup(tmp_path) == (manifest,)
    assert not pending_path.exists()
    terminal_pending_path = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=mutation_receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id=_EMPTY_LIVENESS_DIGEST,
        evidence_ref="proof:mutation-1",
    )
    mark_source_continuity_pending_intent_terminal(
        terminal_pending_path,
        error=DurableSourceContinuitySemanticError("continuity precondition rejected"),
    )
    assert reconcile_durable_change_train_startup(tmp_path) == (manifest,)
    assert not terminal_pending_path.exists()
    refreshed_path.unlink()
    with pytest.raises(DurableChangeTrainError, match="refresh receipt"):
        reconcile_durable_change_train_startup(tmp_path)
    with pytest.raises(DurableChangeTrainError, match="refresh receipt"):
        assert_source_continuity_apply_allowed(tmp_path)

    mutation_receipt.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(db_path),
                "backup_manifest": str(backup_manifest),
                "candidate_count": 0,
                "candidate_digest": _EMPTY_LIVENESS_DIGEST,
                "backup_manifest_sha256": hashlib.sha256(backup_manifest.read_bytes()).hexdigest(),
            }
        )
        + "\n"
        + json.dumps({"kind": "blob_ref_liveness_reconciliation", "phase": "prepared"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DurableChangeTrainError, match="does not bind"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-invalid-footer",
        )

    mutation_receipt.write_text('{"kind": "blob_ref_liveness_reconciliation"}\n', encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="incomplete"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-incomplete",
        )

    mutation_receipt.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="valid JSONL"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-malformed",
        )

    mutation_receipt.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(db_path),
                "backup_manifest": str(backup_manifest),
                "candidate_count": 0,
                "candidate_digest": "b" * 64,
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
    with pytest.raises(DurableChangeTrainError, match="does not bind"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-wrong-operation",
        )

    manifest.unlink()
    mutation_receipt.write_text(
        json.dumps(
            {
                "kind": "blob_ref_liveness_reconciliation",
                "phase": "prepared",
                "source_db": str(db_path),
                "backup_manifest": str(backup_manifest),
                "candidate_count": 0,
                "candidate_digest": _EMPTY_LIVENESS_DIGEST,
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
    with pytest.raises(DurableSourceTrainMissingError, match="no released source train"):
        refresh_released_source_train_continuity(
            tmp_path,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=before,
            operation_id=_EMPTY_LIVENESS_DIGEST,
            evidence_ref="proof:mutation-no-train",
        )


def test_startup_consumes_an_already_recovered_rollback_intent(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    with sqlite3.connect(db_path) as connection:
        before = migration_runner.capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    receipt = tmp_path / "rolled-back.jsonl"
    receipt.write_text('{"phase": "recovered_rolled_back"}\n', encoding="utf-8")
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    pending_path = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="rolled-back-operation",
        evidence_ref="proof:rolled-back-operation",
    )

    assert reconcile_durable_change_train_startup(tmp_path) == ()
    assert not pending_path.exists()


def test_startup_skips_unfinalized_raw_authority_receipt_and_recovers_other_intents(tmp_path: Path) -> None:
    """A crash before receipt publication leaves the raw recovery intent resumable."""

    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    with sqlite3.connect(db_path) as connection:
        before = migration_runner.capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    missing_pending = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=tmp_path / "raw-authority-receipt.json",
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="raw-authority-recovery",
        evidence_ref="proof:raw-authority-recovery",
        mutation_kind="raw_authority_recovery",
    )
    rolled_back_receipt = tmp_path / "rolled-back.jsonl"
    rolled_back_receipt.write_text('{"phase": "recovered_rolled_back"}\n', encoding="utf-8")
    rolled_back_pending = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=rolled_back_receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="other-recovery",
        evidence_ref="proof:other-recovery",
    )

    assert durable_change_train_module._recover_pending_source_continuity_intents(tmp_path) == frozenset(
        {ArchiveTier.SOURCE}
    )

    assert missing_pending.exists()
    assert not rolled_back_pending.exists()


def test_startup_defers_released_source_validation_for_an_unfinalized_reset_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset awaiting its receipt cannot be rejected by stale released-train evidence."""

    source_db = tmp_path / "source.db"
    _create_current_database(source_db)
    with sqlite3.connect(source_db) as connection:
        before = migration_runner.capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    pending = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=tmp_path / "missing-raw-authority-receipt.json",
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="raw-authority-recovery",
        evidence_ref="proof:raw-authority-recovery",
        mutation_kind="raw_authority_recovery",
    )
    manifest_path = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-001.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.touch()
    released = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=1),
    )

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unfinalized reset receipt must defer released source-train validation")

    monkeypatch.setattr(durable_change_train_module, "DURABLE_MIGRATION_ADOPTION_FLOORS", {ArchiveTier.SOURCE: 0})
    monkeypatch.setattr(durable_change_train_module, "_fresh_durable_bootstrap_versions", lambda *_args: {})
    monkeypatch.setattr(durable_change_train_module, "load_durable_change_train_manifest", lambda _path: released)
    monkeypatch.setattr(durable_change_train_module, "_released_train_manifests_by_target", fail_validation)
    monkeypatch.setattr(durable_change_train_module, "_verify_released_train_live_tier", fail_validation)

    assert durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path) == ()
    assert pending.exists()


def test_startup_rejects_a_missing_liveness_receipt(tmp_path: Path) -> None:
    """Liveness pending evidence is corrupt when its prewritten receipt disappears."""

    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    with sqlite3.connect(db_path) as connection:
        before = migration_runner.capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    pending = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=tmp_path / "missing-liveness-receipt.jsonl",
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="liveness-operation",
        evidence_ref="proof:blob-ref-liveness:liveness-operation",
    )

    with pytest.raises(DurableChangeTrainError, match="liveness receipt is missing"):
        durable_change_train_module._recover_pending_source_continuity_intents(tmp_path)

    assert pending.exists()


@pytest.mark.parametrize("after_counts", ({}, {"raw_authority_censuses": False}))
def test_raw_authority_reset_receipt_requires_nonempty_integer_zero_counts(
    tmp_path: Path, after_counts: dict[str, object]
) -> None:
    """A self-hashed reset receipt must prove each ledger table reached zero."""

    source_path = tmp_path / "source.db"
    _create_current_database(source_path)
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    payload: dict[str, object] = {
        "format": "polylogue.raw-authority-recovery-receipt.v1",
        "operation": "reset_raw_authority_census",
        "operation_id": "raw-authority-reset",
        "archive_root": str(tmp_path),
        "backup_authority": {
            "tier": ArchiveTier.SOURCE.value,
            "manifest_path": str(backup_manifest),
            "manifest_sha256": hashlib.sha256(backup_manifest.read_bytes()).hexdigest(),
        },
        "after_counts": after_counts,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    with pytest.raises(DurableChangeTrainError, match="does not prove"):
        durable_change_train_module._validate_source_mutation_receipt_bytes(
            json.dumps(payload).encode("utf-8"),
            source_path=source_path,
            backup_manifest=backup_manifest,
            operation_id="raw-authority-reset",
        )


def test_postcondition_recovery_rejects_remaining_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    with sqlite3.connect(db_path) as connection:
        before = migration_runner.capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    receipt = tmp_path / "postcondition-failed.jsonl"
    receipt.write_text('{"phase": "postcondition_failed"}\n', encoding="utf-8")
    backup_manifest = tmp_path / "backup-manifest.json"
    backup_manifest.write_text("{}\n", encoding="utf-8")
    pending_path = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=receipt,
        backup_manifest=backup_manifest,
        pre_mutation_evidence=before,
        operation_id="postcondition-failed-operation",
        evidence_ref="proof:postcondition-failed-operation",
    )

    def recover_with_postcondition_check(*_args: object, **kwargs: object) -> str:
        postcondition_check = cast(Callable[[], None], kwargs["postcondition_check"])
        postcondition_check()
        return "recovered_committed"

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation._recover_prepared_receipt",
        recover_with_postcondition_check,
    )
    monkeypatch.setattr(
        "polylogue.storage.blob_ref_liveness.classify_blob_ref_liveness",
        lambda _connection: SimpleNamespace(safe_to_apply=True, orphaned_count=1),
    )

    with pytest.raises(DurableChangeTrainError, match="postcondition remains unsafe"):
        durable_change_train_module._recover_pending_source_continuity_intents(tmp_path)

    assert pending_path.exists()


@pytest.mark.parametrize("tier", (ArchiveTier.SOURCE, ArchiveTier.USER))
def test_synthetic_source_and_user_trains_complete_the_full_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
) -> None:
    """Both durable tiers persist every state and use the shipped migration transaction."""
    db_path = tmp_path / f"{tier.value}.db"
    manifest = tmp_path / f"{tier.value}-train.json"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, tier)

    def persist_and_reload(next_train: DurableChangeTrain, expected_revision: int) -> DurableChangeTrain:
        write_durable_change_train_manifest(
            manifest,
            next_train,
            expected_revision=expected_revision,
        )
        return load_durable_change_train_manifest(manifest)

    claim = _claim(tier)
    train = _declared(tier, claim=claim)
    train = persist_and_reload(train, -1)
    previous_revision = train.revision
    train = admit_durable_change_train(
        train,
        observed_current_version=_CURRENT_VERSION,
        fresh_ddl_parity=_parity(tier),
        admission_evidence_ref=f"proof:admit:{tier.value}",
        migration_claims=(claim,),
        canonical_target_version=_TARGET_VERSION,
        admitted_at_ms=2,
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:archive-root",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:rebuild-lease",
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    with sqlite3.connect(db_path) as conn:
        train = authorize_durable_change_train_backup(
            conn,
            train,
            backup_manifest=None,
            evidence_ref="proof:additive-no-backup",
        )
        train = persist_and_reload(train, previous_revision)
        previous_revision = train.revision
        train = apply_durable_change_train(conn, train)
        train = persist_and_reload(train, previous_revision)
        assert conn.execute("SELECT payload FROM base_items WHERE item_id='base-1'").fetchone() == ("preserve-me",)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='durable_items'").fetchone() == (
            "durable_items",
        )
    assert train.apply_evidence is not None
    assert train.apply_evidence.migration_result.applied_versions == (_TARGET_VERSION,)
    assert train.apply_evidence.row_parity.ok is True

    previous_revision = train.revision
    train = record_durable_writer_release(train, evidence_ref="proof:lease-released")
    train = persist_and_reload(train, previous_revision)
    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            actual_parity = prove_durable_fresh_ddl_parity(
                tier,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref=f"proof:post-apply-fresh:{tier.value}",
            )
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:runtime-restart",
        )
    previous_revision = train.revision
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=actual_parity,
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    train = release_durable_change_train(train, evidence_ref="proof:train-release")
    train = persist_and_reload(train, previous_revision)

    assert train.state is DurableChangeTrainState.RELEASED
    assert train.revision == 7


def test_future_train_sidecar_discovery_uses_real_package_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / "027_future_items.sql"
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        sql_path.name,
        sql,
        owner_ref="owner:future-source",
    )
    train = declare_durable_change_train(
        train_id="train:source:v27",
        tier=ArchiveTier.SOURCE,
        current_version=DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE],
        target_version=27,
        slot=27,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    (source_package / "027.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(train)),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations.source",
    )
    monkeypatch.setattr(
        migration_runner,
        "_migration_package",
        lambda _tier: "fixture_migrations.source",
    )

    observed = validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))
    loaded = migration_runner._load_migrations(ArchiveTier.SOURCE)

    assert [item.resource_name for item in observed] == ["027.train.json"]
    assert observed[0].train.migration.sql_sha256 == claim.sql_sha256
    assert loaded[0].version == 27
    assert "fixture_migrations.source" in sys.modules

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = 27
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = """
    CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
    CREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;
    """
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "real-route-source.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute("PRAGMA user_version = 26")
        conn.commit()
        result = migration_runner.migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)
        assert result.applied_versions == (27,)
        assert conn.execute("PRAGMA user_version").fetchone() == (27,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='future_items'").fetchone() == ("future_items",)


def test_maintenance_route_persists_and_proves_a_future_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "fixture_migrations_maintenance"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    (source_package / "002_future_items.sql").write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        "002_future_items.sql",
        sql,
        owner_ref="owner:maintenance-source",
    )
    rider = DurableChangeRider(
        rider_id="rider:maintenance",
        owner_ref="owner:maintenance-rider",
        schema_objects=("table:future_items",),
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
        train_id="train:source:v2",
        tier=ArchiveTier.SOURCE,
        current_version=1,
        target_version=2,
        slot=2,
        owner_ref="owner:maintenance-source",
        migration=claim,
        riders=(rider,),
        declared_at_ms=1,
    )
    (source_package / "002.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(declared)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_maintenance.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_maintenance.source",
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train.DURABLE_MIGRATION_ADOPTION_FLOORS",
        {ArchiveTier.SOURCE: 1, ArchiveTier.USER: 1},
    )
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = 2
    monkeypatch.setattr("polylogue.storage.sqlite.migration_runner.ARCHIVE_VERSION_BY_TIER", versions)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; CREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)

    released: list[bool] = []
    result = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )

    assert result.train is not None
    assert result.train.state is DurableChangeTrainState.RELEASED
    assert result.migration_result is not None
    assert result.migration_result.applied_versions == (2,)
    assert released == [True]
    manifest_path = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2)
    assert load_durable_change_train_manifest(manifest_path).state is DurableChangeTrainState.RELEASED

    released_bytes = db_path.read_bytes()
    db_path.unlink()
    with pytest.raises(DurableChangeTrainError, match="durable tier is missing"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("missing released tier was released again"),
        )
    assert not db_path.exists()
    db_path.write_bytes(released_bytes)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    with pytest.raises(
        DurableChangeTrainError, match="(?:released source train .* expects live v2|continuity proof failed)"
    ):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("stale released train was released again"),
        )


def test_maintenance_route_replays_historical_sidecars_before_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later shipped slot must not reject an earlier persisted train."""
    package_root = tmp_path / "fixture_migrations_sequential"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")

    rider_template = DurableChangeRider(
        rider_id="rider:maintenance",
        owner_ref="owner:maintenance-rider",
        schema_objects=(),
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
    migrations = (
        (2, "durable_items", "CREATE TABLE durable_items (id INTEGER PRIMARY KEY) STRICT;"),
        (3, "later_items", "CREATE TABLE later_items (id INTEGER PRIMARY KEY) STRICT;"),
    )
    for slot, table_name, statement in migrations:
        sql = f"-- migration-safety: additive-no-backup\n{statement}\n"
        filename = f"{slot:03d}_{table_name}.sql"
        (source_package / filename).write_text(sql, encoding="utf-8")
        claim = durable_migration_claim_for_sql(
            ArchiveTier.SOURCE,
            filename,
            sql,
            owner_ref=f"owner:maintenance-source:{slot}",
        )
        rider = replace(
            rider_template,
            rider_id=f"rider:maintenance:{slot}",
            schema_objects=(f"table:{table_name}",),
        )
        declared = declare_durable_change_train(
            train_id=f"train:source:v{slot}",
            tier=ArchiveTier.SOURCE,
            current_version=slot - 1,
            target_version=slot,
            slot=slot,
            owner_ref=f"owner:maintenance-source:{slot}",
            migration=claim,
            riders=(rider,),
            declared_at_ms=slot,
        )
        (source_package / f"{slot:03d}.train.json").write_text(
            json.dumps(migration_runner.durable_change_train_to_payload(declared)), encoding="utf-8"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_sequential.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_sequential.source",
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train.DURABLE_MIGRATION_ADOPTION_FLOORS",
        {ArchiveTier.SOURCE: 1, ArchiveTier.USER: 1},
    )
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = 3
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
        "CREATE TABLE durable_items (id INTEGER PRIMARY KEY) STRICT;"
        "CREATE TABLE later_items (id INTEGER PRIMARY KEY) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    released: list[bool] = []

    first = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert first.migration_result is not None
    assert first.migration_result.applied_versions == (2,)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)

    second = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert second.migration_result is not None
    assert second.migration_result.applied_versions == (3,)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (3,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='later_items'").fetchone() == ("later_items",)
    assert released == [True, True]
    historical_manifest = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2)
    manifest_v3 = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 3)
    manifest_v3_bytes = manifest_v3.read_bytes()
    manifest_v3.unlink()
    with pytest.raises(DurableChangeTrainError, match=r"versions \[3\]"):
        durable_change_train_module.reconcile_durable_change_train_startup(tmp_path)
    assert released == [True, True]
    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("missing intervening train was admitted"),
        )
    manifest_v3.write_bytes(manifest_v3_bytes)
    evidence_captures = 0
    schema_inventories = 0
    canonical_inventories = 0
    real_capture = migration_runner.capture_durable_database_evidence
    real_schema_inventory = migration_runner.capture_durable_schema_inventory
    real_canonical_inventory = durable_change_train_module._canonical_schema_inventory

    def count_evidence_captures(
        connection: sqlite3.Connection, tier: ArchiveTier
    ) -> migration_runner.DurableDatabaseEvidence:
        nonlocal evidence_captures
        evidence_captures += 1
        return real_capture(connection, tier)

    def count_schema_inventories(
        connection: sqlite3.Connection,
    ) -> migration_runner.DurableSchemaInventory:
        nonlocal schema_inventories
        schema_inventories += 1
        return real_schema_inventory(connection)

    def count_canonical_inventories(tier: ArchiveTier, target_version: int) -> migration_runner.DurableSchemaInventory:
        nonlocal canonical_inventories
        canonical_inventories += 1
        return real_canonical_inventory(tier, target_version)

    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", count_evidence_captures)
    monkeypatch.setattr(migration_runner, "capture_durable_schema_inventory", count_schema_inventories)
    monkeypatch.setattr(durable_change_train_module, "_canonical_schema_inventory", count_canonical_inventories)
    third = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert third.forward_version_receipt is not None
    assert third.forward_version_receipt.historical_target_version == 2
    assert third.forward_version_receipt.observed_live_version == 3
    assert evidence_captures == 1
    # Three inventories: one nested inside the single evidence capture, one
    # live inventory read during startup reconciliation, and one canonical
    # inventory built from the fresh DDL. The no-op receipt reuses all of them.
    assert schema_inventories == 3
    assert canonical_inventories == 1

    historical_train = load_durable_change_train_manifest(historical_manifest)
    with sqlite3.connect(db_path) as conn:
        actual = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        receipt = durable_change_train_module._verify_released_train_live_tier(
            tmp_path,
            conn,
            historical_train,
            current_target_version=3,
            actual_evidence=actual,
        )
    assert receipt is not None
    assert receipt.historical_target_version == 2
    assert receipt.current_target_version == 3
    assert receipt.observed_live_version == 3
    assert historical_train.proof is not None
    assert (
        receipt.historical_schema_inventory_sha256 == historical_train.proof.fresh_ddl_parity.migrated_inventory_sha256
    )
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(DurableChangeTrainError, match="is newer than current target"):
            durable_change_train_module._verify_released_train_live_tier(
                tmp_path,
                conn,
                historical_train,
                current_target_version=2,
                actual_evidence=actual,
            )

    captures = 0
    real_capture = migration_runner.capture_durable_database_evidence

    def count_captures(connection: sqlite3.Connection, tier: ArchiveTier) -> migration_runner.DurableDatabaseEvidence:
        nonlocal captures
        captures += 1
        return real_capture(connection, tier)

    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", count_captures)
    assert reconcile_durable_change_train_startup(tmp_path) == (
        durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2),
        durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 3),
    )
    assert captures == 1

    unrelated_root = tmp_path / "unrelated-archive"
    unrelated_root.mkdir()
    shutil.copy2(db_path, unrelated_root / "source.db")
    unrelated_manifest = durable_change_train_manifest_path(unrelated_root, ArchiveTier.SOURCE, 2)
    unrelated_manifest.parent.mkdir(parents=True)
    shutil.copy2(historical_manifest, unrelated_manifest)
    shutil.copy2(manifest_v3, durable_change_train_manifest_path(unrelated_root, ArchiveTier.SOURCE, 3))
    with sqlite3.connect(unrelated_root / "source.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with pytest.raises(DurableChangeTrainError, match="immutable archive identity differs"):
        reconcile_durable_change_train_startup(unrelated_root)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE base_items")
        conn.commit()
        tampered = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        with pytest.raises(DurableChangeTrainError, match="canonical live version"):
            durable_change_train_module._verify_released_train_live_tier(
                tmp_path,
                conn,
                historical_train,
                current_target_version=3,
                actual_evidence=tampered,
            )


def test_continuity_admits_legacy_full_archive_identity_digest(tmp_path: Path) -> None:
    from polylogue.storage.archive_identity import ArchiveIdentity
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        current = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        legacy = replace(
            current,
            archive_identity_digest=ArchiveIdentity.resolve(tmp_path).authority_identity_digest,
        )
        migration_runner._assert_durable_database_continuity(
            current,
            legacy,
            label="legacy identity compatibility",
            connection=conn,
        )
        with pytest.raises(DurableChangeTrainError, match="continuity proof failed"):
            migration_runner._assert_durable_database_continuity(
                current,
                replace(current, archive_identity_digest="a" * 64),
                label="foreign identity",
                connection=conn,
            )
        with pytest.raises(DurableChangeTrainError, match="continuity proof failed"):
            migration_runner._assert_durable_database_continuity(
                current,
                legacy,
                label="legacy identity without connection",
            )


def test_released_train_chain_is_anchored_at_adoption_floor() -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    released = cast(DurableChangeTrain, SimpleNamespace(state=DurableChangeTrainState.RELEASED))

    with pytest.raises(DurableChangeTrainError, match=rf"versions \[{floor + 1}\]"):
        durable_change_train_module._require_released_train_chain(
            ArchiveTier.SOURCE,
            {
                floor + 2: released,
                floor + 3: released,
            },
            current_version=floor + 3,
        )


def test_released_train_chain_can_start_at_bootstrap_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    current_version = floor + 1
    released = cast(DurableChangeTrain, SimpleNamespace(state=DurableChangeTrainState.RELEASED))
    monkeypatch.setattr(durable_change_train_module, "_historical_schema_evidence", lambda _train: None)

    durable_change_train_module._require_released_train_chain(
        ArchiveTier.SOURCE,
        {current_version: released},
        current_version=current_version,
        floor=floor,
    )


def test_forward_receipt_checks_missing_chain_before_empty_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    current_version = floor + 2
    current_train = cast(
        DurableChangeTrain,
        SimpleNamespace(target_version=current_version, state=DurableChangeTrainState.RELEASED),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_released_train_manifests_by_target",
        lambda _manifest_root, _tier: {current_version: current_train},
    )

    with (
        sqlite3.connect(":memory:") as conn,
        pytest.raises(
            DurableChangeTrainError,
            match=rf"versions \[{floor + 1}\]",
        ),
    ):
        durable_change_train_module._forward_version_receipt_for_current_tier(
            tmp_path,
            conn,
            ArchiveTier.SOURCE,
            current_version=current_version,
            current_target_version=current_version,
        )


def test_forward_receipt_skips_non_train_audit_tier(tmp_path: Path) -> None:
    with sqlite3.connect(":memory:") as conn:
        assert (
            durable_change_train_module._forward_version_receipt_for_current_tier(
                tmp_path,
                conn,
                ArchiveTier.AUDIT,
                current_version=1,
                current_target_version=1,
            )
            is None
        )


def test_startup_recovers_later_train_before_released_chain_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True)
    first_path = manifest_root / "source-027.json"
    later_path = manifest_root / "source-028.json"
    first_path.touch()
    later_path.touch()
    released = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=27),
    )
    later_released = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=28),
    )
    backup_authorized = cast(
        DurableChangeTrain,
        SimpleNamespace(
            state=DurableChangeTrainState.BACKUP_AUTHORIZED,
            tier=ArchiveTier.SOURCE,
            target_version=28,
            train_id="train:source:v28",
            revision=0,
        ),
    )
    states = {first_path: released, later_path: backup_authorized}
    events: list[tuple[str, int]] = []

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            yield connection

    def fake_load(path: Path) -> DurableChangeTrain:
        return states[path]

    def fake_persist(path: Path, train: DurableChangeTrain, *, expected_revision: int) -> DurableChangeTrain:
        states[path] = train
        return train

    def fake_recover(*_args: object, **_kwargs: object) -> DurableChangeTrain:
        events.append(("recover", 28))
        return later_released

    def fake_capture(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(user_version=28)

    monkeypatch.setattr(
        durable_change_train_module,
        "_recover_pending_source_continuity_intents",
        lambda _root: frozenset(),
    )
    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)
    monkeypatch.setattr(durable_change_train_module, "load_durable_change_train_manifest", fake_load)
    monkeypatch.setattr(durable_change_train_module, "_persist_train_transition", fake_persist)
    monkeypatch.setattr(durable_change_train_module, "reconcile_interrupted_durable_change_train", fake_recover)
    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", fake_capture)
    monkeypatch.setattr(durable_change_train_module, "_historical_schema_evidence", lambda _train: None)
    monkeypatch.setattr(
        migration_runner,
        "capture_durable_schema_inventory",
        lambda _connection: SimpleNamespace(sha256="inventory"),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_canonical_schema_inventory",
        lambda _tier, _version: SimpleNamespace(sha256="canonical"),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_verify_released_train_live_tier",
        lambda _root, _connection, train, **_kwargs: events.append(("verify", train.target_version)),
    )

    durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)

    assert events == [("recover", 28), ("verify", 27), ("verify", 28)]


def test_startup_checks_chain_when_only_current_train_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True)
    manifest_path = manifest_root / "source-028.json"
    manifest_path.touch()
    current = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=28),
    )

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            yield connection

    monkeypatch.setattr(
        durable_change_train_module, "_recover_pending_source_continuity_intents", lambda _root: frozenset()
    )
    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)
    monkeypatch.setattr(durable_change_train_module, "load_durable_change_train_manifest", lambda _path: current)
    monkeypatch.setattr(
        durable_change_train_module,
        "capture_durable_database_evidence",
        lambda _connection, _tier: SimpleNamespace(user_version=28),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_released_train_manifests_by_target",
        lambda _root, _tier: {28: current},
    )

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)


def test_startup_checks_chain_when_manifest_directory_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source.db").touch()

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("PRAGMA user_version = 28")
            yield connection

    monkeypatch.setattr(
        durable_change_train_module, "_recover_pending_source_continuity_intents", lambda _root: frozenset()
    )
    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)


def test_fresh_archive_bootstrap_receipt_allows_repeat_startup(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    assert reconcile_durable_change_train_startup(tmp_path) == ()
    initialize_active_archive_root(tmp_path)


def test_audit_adoption_receipt_survives_startup_preflight(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The storage startup route validates the receipt created by the real adopter."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    (archive_root / "audit.db").unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    fsynced_paths: set[Path] = set()
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        try:
            fsynced_paths.add(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", record_fsync)

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption") as owner:
        version, receipt = adopt_missing_audit_tier(
            archive_root / "audit.db",
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    assert receipt == audit_adoption_receipt_path(archive_root)
    assert {
        archive_root,
        archive_root / ".maintenance-state",
        archive_root / ".maintenance-state" / "durable-change-trains",
    }.issubset(fsynced_paths)
    assert reconcile_durable_change_train_startup(archive_root) == ()
    receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(MigrationError, match="invalid audit adoption receipt"):
        reconcile_durable_change_train_startup(archive_root)


def test_audit_adoption_receipt_allows_a_mutated_audit_journal(workspace_env: dict[str, Path]) -> None:
    """Startup accepts an adopted audit tier after normal SQLite journal writes."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-mutable-journal") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("adopted-audit-journal", 1, 1),
        )
        connection.commit()

    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_audit_metadata_read_is_read_only_for_uri_metacharacter_paths(tmp_path: Path) -> None:
    """Archive path punctuation cannot consume SQLite's read-only URI parameter."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = tmp_path / "archive?#uri"
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"

    version, application_id, quick_check = _audit_live_metadata(audit_path)

    assert version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    assert application_id == 0
    assert quick_check == ("ok",)
    assert not audit_path.with_name("audit.db-wal").exists()
    assert not audit_path.with_name("audit.db-shm").exists()


def test_adopted_audit_restore_rebinds_continuity_from_verified_backup(workspace_env: dict[str, Path]) -> None:
    """The real offline restore publishes a new immutable continuity generation."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "pre-adoption", profile="full_evidence", verify=True)
    assert pre_adoption.ok, pre_adoption.error
    assert pre_adoption.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-restore-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "post-adoption", profile="full_evidence", verify=True)
    assert verified.ok, verified.error
    assert verified.output_path is not None
    with closing(sqlite3.connect(archive_root / "index.db")) as connection:
        current_index_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        connection.execute(f"PRAGMA user_version = {current_index_version + 1}")
        connection.commit()
    old_identity = (audit_path.stat().st_dev, audit_path.stat().st_ino)
    audit_path.write_bytes(b"corrupted audit image")

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-restore") as owner:
        receipt = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    with closing(sqlite3.connect(audit_path)) as audit:
        assert audit.execute("PRAGMA user_version").fetchone() == (ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT],)
        assert audit.execute("SELECT generation FROM audit_continuity_head").fetchone() == (2,)
    assert (audit_path.stat().st_dev, audit_path.stat().st_ino) != old_identity
    assert receipt.name.endswith(".committed.json")
    assert receipt.with_name(receipt.name.replace(".committed.json", ".prepared.json")).is_file()
    with (
        closing(sqlite3.connect(archive_root / "source.db")) as source,
        closing(sqlite3.connect(archive_root / "audit.db")) as audit,
    ):
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        )
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_adopted_audit_restore_resumes_an_interrupted_continuity_commit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared record blocks startup but the same verified backup can complete it."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "resume-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "resume-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    source_wal_reader: sqlite3.Connection | None = None
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_rebind_commit(self: AuditContinuityCoordinator, phase: str, mutation: object) -> None:
        nonlocal source_wal_reader
        if getattr(mutation, "mutation_id", "").startswith("audit-restore:"):
            if phase == "before_source_prepare":
                with sqlite3.connect(archive_root / "source.db") as source:
                    assert source.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
                source_wal_reader = sqlite3.connect(archive_root / "source.db")
                source_wal_reader.execute("BEGIN")
                source_wal_reader.execute("SELECT * FROM audit_continuity_control").fetchone()
            if phase == "after_source_prepare":
                raise RuntimeError("simulated continuity prepare interruption")
        original_phase(self, phase, mutation)  # type: ignore[arg-type]

    try:
        with monkeypatch.context() as interrupted:
            interrupted.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_rebind_commit)
            with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-interrupt") as owner:
                with pytest.raises(RuntimeError, match="continuity prepare interruption"):
                    restore_adopted_audit_tier(
                        audit_path,
                        backup_manifest=Path(verified.output_path) / "manifest.json",
                        directory_fd=owner.directory_fd,
                        stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                    )

        # The production source-prepare transition is durable only in a WAL;
        # retry must validate this operation-owned pending command before it
        # can republish/rebind the audit image.
        assert (archive_root / "source.db-wal").stat().st_size > 0
        with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
            assert str(
                source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0]
            ).startswith("audit-restore:")
            assert (
                source.execute(
                    "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
                ).fetchone()
                == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
            )
        with sqlite3.connect(archive_root / "source.db") as source:
            source.execute("CREATE TABLE restore_retry_tamper (value TEXT NOT NULL) STRICT")
            source.commit()
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-tamper") as owner:
            with pytest.raises(MigrationError, match="backup is stale for source.db"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )
        with sqlite3.connect(archive_root / "source.db") as source:
            source.execute("DROP TABLE restore_retry_tamper")
            source.commit()
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume") as owner:
            receipt = restore_adopted_audit_tier(
                audit_path,
                backup_manifest=Path(verified.output_path) / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )
    finally:
        if source_wal_reader is not None:
            source_wal_reader.close()

    assert receipt.name.endswith(".committed.json")
    assert reconcile_durable_change_train_startup(archive_root) == ()


@pytest.mark.parametrize("order", ((ArchiveTier.AUDIT, ArchiveTier.SOURCE), (ArchiveTier.SOURCE, ArchiveTier.AUDIT)))
def test_continuity_migrations_have_a_deployable_cross_tier_compatibility_window(
    workspace_env: dict[str, Path], order: tuple[ArchiveTier, ArchiveTier]
) -> None:
    """Each numbered migration can ship first; coordination activates only after both."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as source:
        source.execute("DROP TABLE audit_continuity_control")
        source.execute("PRAGMA user_version = 31")
        source.commit()
    with sqlite3.connect(archive_root / "audit.db") as audit:
        audit.execute("DROP TABLE audit_continuity_head")
        audit.execute("PRAGMA user_version = 1")
        audit.commit()
    backup = backup_archive(
        output_dir=archive_root.parent / f"continuity-{order[0].value}-first", profile="full_evidence", verify=True
    )
    assert backup.ok and backup.output_path is not None, backup.error
    manifest = Path(backup.output_path) / "manifest.json"

    for position, tier in enumerate(order):
        with sqlite3.connect(archive_root / f"{tier.value}.db") as connection:
            result = migrate_archive_tier(connection, tier, backup_manifest=manifest)
        assert result.applied_versions == (ARCHIVE_VERSION_BY_TIER[tier],)
        sidecar = durable_migration_sidecar_for_slot(tier, ARCHIVE_VERSION_BY_TIER[tier])
        assert sidecar is not None
        train = sidecar.train
        results = _runtime_consumer_results(train, archive_root)
        assert {result.consumer_id for result in results} == {
            consumer.consumer_id for rider in train.riders for consumer in rider.runtime_consumers
        }
        probe = AuditContinuityCoordinator(archive_root)
        if position == 0:
            assert probe.runtime_probe().startswith("standby")
        else:
            assert probe.runtime_probe() == "reconciled matching source/audit continuity heads"


def test_adopted_audit_restore_replaces_stale_operation_staging_after_crash(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retry completes a prepared restore after a crash leaves its private image."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "staging-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "staging-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    real_unlink = os.unlink

    def interrupt_publication(*args: object, **kwargs: object) -> None:
        raise OSError("simulated restore publication crash")

    def leave_staging(name: os.PathLike[str] | str, *, dir_fd: int | None = None) -> None:
        if str(name).startswith(".audit.db.restore-"):
            return
        real_unlink(name, dir_fd=dir_fd)

    with monkeypatch.context() as interrupted:
        interrupted.setattr("polylogue.operations.durable_change_train.os.replace", interrupt_publication)
        interrupted.setattr("polylogue.operations.durable_change_train.os.unlink", leave_staging)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-interrupt") as owner:
            with pytest.raises(OSError, match="simulated restore publication crash"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    stale = tuple(archive_root.glob(".audit.db.restore-*.tmp"))
    assert len(stale) == 1
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-resume") as owner:
        receipt = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert receipt.name.endswith(".committed.json")
    assert not tuple(archive_root.glob(".audit.db.restore-*.tmp"))
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_adopted_audit_restore_record_survives_publication_temp_hardlink(
    workspace_env: dict[str, Path],
) -> None:
    """A crash after immutable publication may leave the valid record with two names."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "hardlink-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-hardlink-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "hardlink-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-hardlink-restore") as owner:
        committed = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    leftover = committed.with_name(f".{committed.name}.publication.tmp")
    os.link(committed, leftover)

    assert committed.stat().st_nlink == 2
    assert reconcile_durable_change_train_startup(archive_root) == ()


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("artifact", "migration backup tier artifact"),
        ("receipt", "adopted-audit restore"),
        ("stale-source", "adopted-audit restore backup is stale for source.db"),
    ],
)
def test_adopted_audit_restore_rejects_untrusted_or_stale_backup(
    workspace_env: dict[str, Path], tamper: str, expected_error: str
) -> None:
    """Mutation: bypassing receipt, artifact, or current-authority checks reaches the real restore call."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(
        output_dir=archive_root.parent / f"pre-{tamper}", profile="full_evidence", verify=True
    )
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id=f"test:audit-restore-adopt-{tamper}") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / f"post-{tamper}", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    backup_root = Path(verified.output_path)
    if tamper == "artifact":
        (backup_root / "audit.db").write_bytes(b"altered backup audit")
    elif tamper == "receipt":
        (backup_root / "verification-receipt.json").write_text("{}", encoding="utf-8")
    else:
        with closing(sqlite3.connect(archive_root / "source.db")) as connection:
            connection.execute("PRAGMA user_version = 999")
            connection.commit()
    audit_path.write_bytes(b"corrupted audit image")

    with acquire_durable_archive_ownership(archive_root, owner_id=f"test:audit-restore-reject-{tamper}") as owner:
        with pytest.raises(MigrationError, match=expected_error):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=backup_root / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == b"corrupted audit image"


def test_adopted_audit_restore_rejects_wrong_archive_application_id(
    workspace_env: dict[str, Path],
) -> None:
    """A valid backup from different audit authority cannot replace the adopted journal."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "app-id-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-app-id-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    with closing(sqlite3.connect(audit_path)) as connection:
        adopted_application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        connection.execute(f"PRAGMA application_id = {adopted_application_id + 1}")
        connection.commit()
    wrong_authority = backup_archive(
        output_dir=archive_root.parent / "app-id-wrong", profile="full_evidence", verify=True
    )
    assert wrong_authority.ok and wrong_authority.output_path is not None, wrong_authority.error
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(f"PRAGMA application_id = {adopted_application_id}")
        connection.commit()
    original = audit_path.read_bytes()

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-app-id-restore") as owner:
        with pytest.raises(MigrationError, match="does not belong to this audit adoption"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=Path(wrong_authority.output_path) / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == original


def test_adopted_audit_restore_rejects_backup_swap_after_validation(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact manifest and verification receipt stay fixed through publication."""
    from polylogue.operations import durable_change_train as operations_durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "swap-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-swap-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "swap-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    backup_root = Path(verified.output_path)
    audit_path.write_bytes(b"corrupt-before-swap-test")
    from polylogue.storage.sqlite.migration_runner import validate_full_evidence_backup_for_adopted_audit_restore

    real_validate = validate_full_evidence_backup_for_adopted_audit_restore
    calls = 0

    def swap_after_validation(path: Path, *, archive_root: Path, **kwargs: object) -> tuple[Path, Path]:
        nonlocal calls
        calls += 1
        manifest, receipt = real_validate(path, archive_root=archive_root, **kwargs)  # type: ignore[arg-type]
        if calls == 2:
            receipt.write_bytes(receipt.read_bytes() + b"\n")
        return manifest, receipt

    monkeypatch.setattr(
        operations_durable_change_train,
        "validate_full_evidence_backup_for_adopted_audit_restore",
        swap_after_validation,
    )
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-swap-restore") as owner:
        with pytest.raises(MigrationError, match="backup changed during the operation"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=backup_root / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == b"corrupt-before-swap-test"


def test_audit_adoption_binds_only_the_source_user_authority(workspace_env: dict[str, Path]) -> None:
    """Routine replacement of rebuildable or disposable tiers leaves adoption valid."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-stable-authority") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    (archive_root / "index.db").unlink()
    (archive_root / "ops.db").unlink()
    initialize_active_archive_root(archive_root)

    assert (archive_root / "index.db").is_file()
    assert (archive_root / "ops.db").is_file()


def test_audit_adoption_receipt_keeps_initial_schema_evidence_after_upgrade(
    workspace_env: dict[str, Path],
) -> None:
    """Receipt validation permits later audit schema versions for normal train handling."""
    from polylogue.operations.durable_change_train import validate_audit_adoption_receipt
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-schema-evidence") as owner:
        _version, receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    initial_schema_digest = json.loads(receipt.read_text(encoding="utf-8"))["audit_schema_inventory_sha256"]

    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute("CREATE TABLE future_audit_schema (value TEXT)")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()

    assert validate_audit_adoption_receipt(archive_root) == receipt
    assert json.loads(receipt.read_text(encoding="utf-8"))["audit_schema_inventory_sha256"] == initial_schema_digest


def test_audit_adoption_rejects_a_stale_audit_file_clone(workspace_env: dict[str, Path]) -> None:
    """The continuity record distinguishes in-place writes from a stale file replacement."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-stale-clone") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    stale_clone = archive_root / "stale-audit.db"
    shutil.copy2(audit_path, stale_clone)
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("live-audit-after-clone", 2, 1),
        )
        connection.commit()
    os.replace(stale_clone, audit_path)

    with pytest.raises(MigrationError, match="continuity"):
        reconcile_durable_change_train_startup(archive_root)


def test_adopted_audit_startup_runs_one_receipt_integrity_check(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bootstrap delegates adopted-tier validation to startup reconciliation once."""
    from polylogue.operations import durable_change_train as operations_durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-single-quick-check") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    calls = 0
    real_validate = operations_durable_change_train.validate_audit_adoption_receipt

    def count_validate(root: Path, *, require_initial_image: bool = False) -> Path | None:
        nonlocal calls
        calls += 1
        return real_validate(root, require_initial_image=require_initial_image)

    monkeypatch.setattr(operations_durable_change_train, "validate_audit_adoption_receipt", count_validate)
    initialize_active_archive_root(archive_root)

    assert calls == 1


def test_audit_adoption_receipt_recovers_interrupted_publication_during_bootstrap(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal bootstrap route completes the receipt-backed publication after a crash."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    monkeypatch.setitem(
        durable_change_train_module.DURABLE_MIGRATION_ADOPTION_FLOORS,
        ArchiveTier.SOURCE,
        ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1,
    )
    monkeypatch.setitem(
        durable_change_train_module.DURABLE_MIGRATION_ADOPTION_FLOORS,
        ArchiveTier.USER,
        ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] - 1,
    )
    real_link = os.link

    def fail_audit_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination).name == "audit.db":
            raise OSError("simulated interruption")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with monkeypatch.context() as failed_publication:
        failed_publication.setattr("polylogue.operations.durable_change_train.os.link", fail_audit_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-interrupted-publication") as owner:
            with pytest.raises(MigrationError):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    initialize_active_archive_root(archive_root)

    assert audit_path.is_file()
    assert (marker_root / ".bootstrap").is_file()
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_audit_adoption_retry_reports_recovered_audit_schema_version(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt-backed retry reports the live audit schema, not a sentinel."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None, backup.error
    manifest = Path(backup.output_path) / "manifest.json"
    real_link = os.link

    def interrupt_audit_publication(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination).name == "audit.db":
            raise OSError("simulated publication interruption")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with monkeypatch.context() as interrupted:
        interrupted.setattr("polylogue.operations.durable_change_train.os.link", interrupt_audit_publication)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption") as owner:
            with pytest.raises(MigrationError, match="anonymous durable publication failed"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=manifest,
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )
    assert not audit_path.exists()

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption-retry") as owner:
        recovered_version, _receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=manifest,
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert recovered_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]


def test_audit_adoption_bootstrap_rejects_stale_replacement_before_recording_continuity(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup does not bless a replaced audit image in the post-link crash window."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    real_link = os.link

    def interrupt_continuity_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination).name == "audit-continuity.json":
            raise OSError("simulated crash after audit link")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with monkeypatch.context() as interrupted_publication:
        interrupted_publication.setattr("polylogue.operations.durable_change_train.os.link", interrupt_continuity_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-continuity-crash") as owner:
            with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
        ).fetchone()
        audit_head = audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
    assert source_head == audit_head
    assert source_head[0] == 1

    stale_clone = archive_root / "stale-audit.db"
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("audit-before-stale-clone", 1, 1),
        )
        connection.commit()
    shutil.copy2(audit_path, stale_clone)
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("audit-after-stale-clone", 2, 1),
        )
        connection.commit()
    os.replace(stale_clone, audit_path)
    stale_image = audit_path.read_bytes()

    with pytest.raises(MigrationError, match="audit tier changed before recording adoption continuity"):
        initialize_active_archive_root(archive_root)

    assert audit_path.read_bytes() == stale_image
    assert not (marker_root / "audit-continuity.json").exists()


def test_audit_adoption_retries_seeded_machine_head_before_continuity_publication(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuity-receipt crash resumes the already-seeded adoption head."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(
        output_dir=archive_root.parent / "seed-before-publish", profile="full_evidence", verify=True
    )
    assert backup.ok and backup.output_path is not None, backup.error
    real_link = os.link

    def interrupt_continuity_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        **kwargs: object,
    ) -> None:
        if Path(destination).name == "audit-continuity.json":
            raise OSError("simulated crash after machine-head seed")
        real_link(source, destination, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as interrupted:
        interrupted.setattr("polylogue.operations.durable_change_train.os.link", interrupt_continuity_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-seed-before-publish") as owner:
            with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
        assert source.execute("SELECT committed_generation FROM audit_continuity_control").fetchone() == (1,)
        assert audit.execute("SELECT generation FROM audit_continuity_head").fetchone() == (1,)
    initialize_active_archive_root(archive_root)
    assert (archive_root / ".maintenance-state" / "durable-change-trains" / "audit-continuity.json").is_file()


def test_adopted_audit_restore_removes_owned_sidecars_before_publication(workspace_env: dict[str, Path]) -> None:
    """The real restore cannot replay stale audit WAL, SHM, or rollback bytes."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "sidecar-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-sidecar-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "sidecar-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt-audit-main")
    for suffix in ("-wal", "-shm", "-journal"):
        audit_path.with_name(f"audit.db{suffix}").write_bytes(b"stale-owned-sidecar")

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-sidecar-restore") as owner:
        restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert not any(audit_path.with_name(f"audit.db{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    assert _audit_live_metadata(audit_path)[2] == ("ok",)


def test_audit_adoption_recovery_preserves_missing_tier_after_continuity(
    workspace_env: dict[str, Path],
) -> None:
    """Recovery requires restore, without recreating audit.db after completed adoption."""
    from polylogue.operations.durable_change_train import recover_pending_audit_adoption
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-missing-after-continuity") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    audit_path.unlink()
    before = {path.relative_to(archive_root): path.read_bytes() for path in archive_root.rglob("*") if path.is_file()}

    with pytest.raises(MigrationError, match="missing after continuity"):
        recover_pending_audit_adoption(archive_root)

    after = {path.relative_to(archive_root): path.read_bytes() for path in archive_root.rglob("*") if path.is_file()}
    assert after == before
    assert not audit_path.exists()


def test_audit_adoption_receipt_is_excluded_from_pre_marker_train_state(workspace_env: dict[str, Path]) -> None:
    """The adoption receipt does not prevent legacy current-schema bootstrap adoption."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker = archive_root / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    marker.unlink()
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-pre-marker") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    initialize_active_archive_root(archive_root)

    assert marker.is_file()


def test_adoption_receipt_short_write_is_removed_for_a_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed receipt write leaves no immutable-looking truncated publication behind."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    receipt_path = audit_adoption_receipt_path(archive_root)
    write_calls = 0

    def short_then_fail(descriptor: int, data: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return min(1, len(data))
        raise OSError("simulated receipt write failure")

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.write", short_then_fail)

    with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
        _write_immutable_audit_adoption_receipt(receipt_path, {"format": "test"}, archive_root=archive_root)

    assert not receipt_path.exists()


def test_adoption_receipt_refuses_a_symlinked_maintenance_parent(tmp_path: Path) -> None:
    """Receipt publication and loading stay beneath the owned archive descriptor."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (archive_root / ".maintenance-state").symlink_to(outside, target_is_directory=True)
    receipt_path = audit_adoption_receipt_path(archive_root)

    with pytest.raises(MigrationError, match="must not traverse outside archive-owned directories"):
        _write_immutable_audit_adoption_receipt(receipt_path, {"format": "test"}, archive_root=archive_root)

    assert not (outside / "durable-change-trains" / "audit-adoption.json").exists()


def test_runtime_bootstrap_refuses_an_established_archive_missing_audit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary writable startup cannot create audit.db without adoption evidence."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    archive_root = workspace_env["archive_root"]
    bootstrap.initialize_active_archive_root(archive_root)
    (archive_root / "audit.db").unlink()
    reconciled = False

    def observe_reconciliation(_root: Path) -> tuple[Path, ...]:
        nonlocal reconciled
        reconciled = True
        return ()

    monkeypatch.setattr(bootstrap, "reconcile_durable_change_trains_on_startup", observe_reconciliation)

    with pytest.raises(RuntimeError, match="adopt-established-audit"):
        bootstrap.initialize_active_archive_root(archive_root)

    assert not reconciled
    assert not (archive_root / "audit.db").exists()


def test_runtime_bootstrap_refuses_source_v31_archive_missing_audit(workspace_env: dict[str, Path]) -> None:
    """Bootstrap evidence remains authoritative before source v32 exists."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as source:
        source.execute("DROP TABLE audit_continuity_control")
        source.execute("PRAGMA user_version = 31")
        source.commit()
    (archive_root / "audit.db").unlink()

    with pytest.raises(RuntimeError, match="adopt-established-audit"):
        initialize_active_archive_root(archive_root)

    assert not (archive_root / "audit.db").exists()


def test_fresh_bootstrap_intent_recovers_after_late_tier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    real_initialize_archive_database = bootstrap.initialize_archive_database
    failed = False

    def fail_embeddings_once(
        path: Path,
        tier: ArchiveTier,
        *,
        allow_create: bool = True,
        expected_version: int | None = None,
    ) -> None:
        nonlocal failed
        if tier is ArchiveTier.EMBEDDINGS and not failed:
            failed = True
            raise RuntimeError("simulated late fresh-bootstrap failure")
        real_initialize_archive_database(path, tier, allow_create=allow_create, expected_version=expected_version)

    monkeypatch.setattr(bootstrap, "initialize_archive_database", fail_embeddings_once)
    with pytest.raises(RuntimeError, match="simulated late fresh-bootstrap failure"):
        bootstrap.initialize_active_archive_root(tmp_path)

    marker_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    assert (marker_root / ".bootstrap.pending").is_file()
    assert not (marker_root / ".bootstrap").exists()
    assert (tmp_path / "source.db").is_file()

    monkeypatch.setattr(bootstrap, "initialize_archive_database", real_initialize_archive_database)
    bootstrap.initialize_active_archive_root(tmp_path)

    assert (marker_root / ".bootstrap").is_file()
    assert not (marker_root / ".bootstrap.pending").exists()
    assert reconcile_durable_change_train_startup(tmp_path) == ()


def test_fresh_bootstrap_intent_rejects_tampering_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    real_initialize_archive_database = bootstrap.initialize_archive_database

    def fail_embeddings(
        path: Path,
        tier: ArchiveTier,
        *,
        allow_create: bool = True,
        expected_version: int | None = None,
    ) -> None:
        if tier is ArchiveTier.EMBEDDINGS:
            raise RuntimeError("simulated late fresh-bootstrap failure")
        real_initialize_archive_database(path, tier, allow_create=allow_create, expected_version=expected_version)

    monkeypatch.setattr(bootstrap, "initialize_archive_database", fail_embeddings)
    with pytest.raises(RuntimeError, match="simulated late fresh-bootstrap failure"):
        bootstrap.initialize_active_archive_root(tmp_path)

    pending = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap.pending"
    payload = json.loads(pending.read_text(encoding="utf-8"))
    payload["durable_identity_digest"] = "0" * 64
    pending.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="intent durable identity mismatch"):
        bootstrap.initialize_active_archive_root(tmp_path)


def test_pre_marker_current_archive_is_adopted_once(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    marker.unlink()

    initialize_active_archive_root(tmp_path)
    assert marker.is_file()


def test_pre_marker_adoption_refuses_missing_train_directory(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    (marker_root / ".bootstrap").unlink()
    marker_root.rmdir()

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        initialize_active_archive_root(tmp_path)


def test_pre_marker_adoption_requires_all_durable_tiers(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    (tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap").unlink()
    (tmp_path / "user.db").unlink()

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        initialize_active_archive_root(tmp_path)
    assert not (tmp_path / "user.db").exists()


def test_bootstrap_marker_survives_index_generation_replacement(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    (tmp_path / "index.db").unlink()

    initialize_active_archive_root(tmp_path)


def test_fresh_bootstrap_receipt_rejects_archive_identity_mismatch(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["durable_identity_digest"] = "0" * 64
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="durable identity mismatch"):
        reconcile_durable_change_train_startup(tmp_path)


def test_fresh_bootstrap_receipt_rejects_recorded_version_tampering(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    cast(dict[str, int], payload["versions"])["source"] += 1
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="marker digest mismatch"):
        reconcile_durable_change_train_startup(tmp_path)


def test_source_train_identity_survives_late_user_tier_initialization(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    source_path = tmp_path / "source.db"
    initialize_archive_database(source_path, ArchiveTier.SOURCE)
    with sqlite3.connect(source_path) as conn:
        before = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with sqlite3.connect(source_path) as conn:
        after = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)

    assert after.archive_identity_digest == before.archive_identity_digest


def test_future_train_sidecar_hash_and_slot_are_admission_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations_hash"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / "027_future_items.sql"
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, sql_path.name, sql, owner_ref="owner:future-source")
    train = declare_durable_change_train(
        train_id="train:source:v27",
        tier=ArchiveTier.SOURCE,
        current_version=26,
        target_version=27,
        slot=27,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    payload = migration_runner.durable_change_train_to_payload(train)
    cast(dict[str, object], payload["migration"])["sql_sha256"] = "0" * 64
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(payload)
    (source_package / "027.train.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_hash.source",
    )

    with pytest.raises(DurableChangeTrainError, match="SQL SHA-256 mismatch"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))

    cast(dict[str, object], payload["migration"])["sql_sha256"] = claim.sql_sha256
    payload["slot"] = 28
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(payload)
    (source_package / "027.train.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="slot"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))


def test_missing_future_sidecar_is_rejected_at_the_migration_runner_choke_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations_missing"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / "027_future_items.sql"
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, sql_path.name, sql, owner_ref="owner:future-source")
    train = declare_durable_change_train(
        train_id="train:source:v27",
        tier=ArchiveTier.SOURCE,
        current_version=26,
        target_version=27,
        slot=27,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    sidecar = source_package / "027.train.json"
    sidecar.write_text(json.dumps(migration_runner.durable_change_train_to_payload(train)), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_missing.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_missing.source",
    )
    assert migration_runner._load_migrations(ArchiveTier.SOURCE)[0].version == 27
    sidecar.unlink()
    with pytest.raises(MigrationError, match="missing durable migration train sidecar"):
        migration_runner._load_migrations(ArchiveTier.SOURCE)
    policy = durable_change_train_policy_report(ArchiveTier.SOURCE)
    assert policy["ok"] is False
    violations = cast(list[str], policy["violations"])
    assert any("missing durable migration train sidecar" in violation for violation in violations)
    (source_package / "028.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(train)), encoding="utf-8"
    )
    with pytest.raises(DurableChangeTrainError, match="no matching SQL resource"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))


def test_migration_transaction_control_cannot_escape_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(
        tmp_path,
        monkeypatch,
        ArchiveTier.SOURCE,
        sql="-- migration-safety: additive-no-backup\nCREATE TABLE transaction_escape (id INTEGER PRIMARY KEY);\nCOMMIT;\n",
    )

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(MigrationError, match="must not control the existing transaction"):
            migration_runner.migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)
        assert conn.execute("PRAGMA user_version").fetchone() == (_CURRENT_VERSION,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='transaction_escape'").fetchone() is None


def test_canonical_inventory_preserves_trigger_literal_whitespace() -> None:
    def inventory(trigger_literal: str, *, formatted: bool) -> str:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(
                """
                CREATE TABLE items (item_id INTEGER PRIMARY KEY);
                CREATE TABLE audit (payload TEXT NOT NULL);
                """
            )
            if formatted:
                conn.executescript(
                    f"""
                    CREATE TRIGGER audit_items
                    AFTER INSERT ON items
                    BEGIN
                        INSERT INTO audit(payload) VALUES ({trigger_literal});
                    END;
                    """
                )
            else:
                conn.executescript(
                    "CREATE TRIGGER audit_items AFTER INSERT ON items BEGIN "
                    f"INSERT INTO audit(payload) VALUES({trigger_literal}); END;"
                )
            return migration_runner.capture_durable_schema_inventory(conn).sha256

    spaced = inventory("'a  b'", formatted=True)
    same_literal_compact_layout = inventory("'a  b'", formatted=False)
    changed_literal = inventory("'a b'", formatted=False)

    assert spaced == same_literal_compact_layout
    assert spaced != changed_literal


def test_admission_rejects_stale_current_and_target_versions() -> None:
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="stale durable train current"):
        admit_durable_change_train(
            train,
            observed_current_version=0,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:admit",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION,
        )
    with pytest.raises(DurableChangeTrainError, match="stale durable train target"):
        admit_durable_change_train(
            train,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:admit",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION + 1,
        )


def test_source_008_009_collision_names_both_owners_and_blocks_admission() -> None:
    source_migrations = Path(__file__).parents[3] / "polylogue" / "storage" / "sqlite" / "migrations" / "source"
    source_008 = source_migrations / "008_raw_session_capture_mode.sql"
    source_009 = source_migrations / "009_expand_origin_vocabulary.sql"
    first = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        source_008.name,
        source_008.read_text(encoding="utf-8"),
        owner_ref="owner:source-008",
    )
    late_rider = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        "008_expand_origin_vocabulary.sql",
        source_009.read_text(encoding="utf-8"),
        owner_ref="owner:source-009-late-rider",
    )
    report = durable_migration_collision_report((first, late_rider))
    assert report["ok"] is False
    serialized = json.dumps(report)
    assert source_008.name in serialized
    assert "008_expand_origin_vocabulary.sql" in serialized
    assert "owner:source-008" in serialized
    assert "owner:source-009-late-rider" in serialized

    train = declare_durable_change_train(
        train_id="source-v8",
        tier=ArchiveTier.SOURCE,
        current_version=7,
        target_version=8,
        slot=8,
        owner_ref="owner:source-train",
        migration=first,
        riders=(_rider(),),
    )
    parity = DurableFreshDDLParityProof(
        tier=ArchiveTier.SOURCE,
        target_version=8,
        migrated_version=8,
        fresh_version=8,
        migrated_inventory_sha256="a" * 64,
        fresh_inventory_sha256="a" * 64,
        missing_objects=(),
        unexpected_objects=(),
        changed_objects=(),
        evidence_ref="proof:v8-fresh",
        matches=True,
    )
    with pytest.raises(DurableChangeTrainError, match="collision.*rebase/renumber") as exc_info:
        admit_durable_change_train(
            train,
            observed_current_version=7,
            fresh_ddl_parity=parity,
            admission_evidence_ref="proof:v8-admit",
            migration_claims=(first, late_rider),
            canonical_target_version=8,
        )
    assert source_008.name in str(exc_info.value)
    assert "008_expand_origin_vocabulary.sql" in str(exc_info.value)


def test_duplicate_train_ownership_and_late_rider_are_rejected() -> None:
    admitted = _admitted(ArchiveTier.SOURCE)
    duplicate = replace(_declared(ArchiveTier.SOURCE), train_id="train:source:v2:duplicate")
    with pytest.raises(DurableChangeTrainError, match="contention key already owned"):
        admit_durable_change_train(
            duplicate,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:duplicate",
            active_trains=(admitted,),
            migration_claims=(duplicate.migration,),
            canonical_target_version=_TARGET_VERSION,
        )
    with pytest.raises(DurableChangeTrainError, match="late rider.*target v3"):
        add_durable_change_train_rider(admitted, _rider(trust_floor_exception_ref="exception:late"))


def test_schema_only_unproven_and_nonproduction_riders_fail_admission() -> None:
    schema_only = _rider(consumer_count=0, trust_floor_exception_ref="exception:single-consumer-floor")
    with pytest.raises(DurableChangeTrainError, match="schema-only"):
        _admitted(ArchiveTier.SOURCE, rider=schema_only)

    one_consumer = _rider(consumer_count=1)
    with pytest.raises(DurableChangeTrainError, match="fewer than two"):
        _admitted(ArchiveTier.SOURCE, rider=one_consumer)

    test_only = DurableChangeRider(
        rider_id="test-only",
        owner_ref="owner:test-only",
        schema_objects=("table:durable_items",),
        runtime_consumers=(
            DurableRuntimeConsumer("test-a", "tests/unit/test_a.py:test_a", "proof:test-a", ("read",)),
            DurableRuntimeConsumer("test-b", "fixture:test-b", "proof:test-b", ("write",)),
        ),
        behavior_proof_refs=("proof:test-a", "proof:test-b"),
    )
    with pytest.raises(DurableChangeTrainError, match="test-only"):
        _admitted(ArchiveTier.SOURCE, rider=test_only)


def test_fresh_ddl_parity_mismatch_blocks_admission() -> None:
    mismatch = _parity(ArchiveTier.SOURCE, include_durable_items=False)
    assert mismatch.matches is False
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="fresh-DDL parity"):
        admit_durable_change_train(
            train,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=mismatch,
            admission_evidence_ref="proof:mismatch",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION,
        )


def test_backup_authority_timestamp_precedes_its_captured_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorization must not depend on two clock reads landing in one millisecond."""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:source",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
        reserved_at_ms=3,
    )
    timestamps = iter((10, 11))
    monkeypatch.setattr(migration_runner, "_durable_now_ms", lambda: next(timestamps))

    with sqlite3.connect(db_path) as conn:
        authorized = authorize_durable_change_train_backup(
            conn,
            train,
            backup_manifest=None,
            evidence_ref="proof:additive-no-backup",
        )

    assert authorized.backup_authorization is not None
    assert authorized.pre_apply_evidence is not None
    assert authorized.backup_authorization.authorized_at_ms == 10
    assert authorized.pre_apply_evidence.observed_at_ms == 11


def test_missing_backup_authority_stops_a_backup_required_train(tmp_path: Path) -> None:
    backup_sql = "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;\n"
    claim = _claim(ArchiveTier.USER, backup_sql)
    train = _admitted(
        ArchiveTier.USER,
        claim=claim,
        backup_plan_ref="backup-profile:user-overlays",
    )
    db_path = tmp_path / "user.db"
    _create_current_database(db_path)
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:user",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
    )
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(DurableChangeTrainError, match="requires an authenticated backup"):
            authorize_durable_change_train_backup(
                conn,
                train,
                backup_manifest=None,
                evidence_ref="proof:missing-backup",
            )


def test_source_and_user_share_only_the_same_archive_writer_reservation(tmp_path: Path) -> None:
    source = _admitted(ArchiveTier.SOURCE, owner_ref="owner:operator")
    user = _admitted(ArchiveTier.USER, owner_ref="owner:operator")
    source_reserved = reserve_durable_change_train(
        source,
        reservation_id="lease:shared",
        reservation_owner_ref="owner:operator",
        archive_root=tmp_path,
        tier_path=tmp_path / "source.db",
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
    )
    with pytest.raises(DurableChangeTrainError, match="second writer rejected"):
        reserve_durable_change_train(
            user,
            reservation_id="lease:other",
            reservation_owner_ref="owner:operator",
            archive_root=tmp_path,
            tier_path=tmp_path / "user.db",
            daemon_stopped_evidence_ref="proof:stopped",
            single_writer_evidence_ref="proof:other-lease",
            active_trains=(source_reserved,),
        )
    user_reserved = reserve_durable_change_train(
        user,
        reservation_id="lease:shared",
        reservation_owner_ref="owner:operator",
        archive_root=tmp_path,
        tier_path=tmp_path / "user.db",
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
        active_trains=(source_reserved,),
    )
    assert source_reserved.reservation is not None
    assert user_reserved.reservation is not None
    assert user_reserved.reservation.reservation_id == source_reserved.reservation.reservation_id


def test_failed_transaction_exposes_exact_retry_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
INSERT INTO table_that_does_not_exist VALUES (1);
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    claim = _claim(ArchiveTier.SOURCE, failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=claim)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
        assert failed.state is DurableChangeTrainState.FAILED
        assert failed.failure is not None
        assert failed.failure.classification is DurableFailureClassification.ROLLED_BACK_TO_CURRENT
        failure_manifest = tmp_path / "source-failed-train.json"
        write_durable_change_train_manifest(failure_manifest, failed, expected_revision=-1)
        failed = load_durable_change_train_manifest(failure_manifest)
        released_failed = record_durable_writer_release(failed, evidence_ref="proof:failed-writer-release")
        released_manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
        released_manifest.parent.mkdir(parents=True)
        write_durable_change_train_manifest(released_manifest, released_failed, expected_revision=-1)
        with pytest.raises(DurableChangeTrainError, match="unreleased source train"):
            assert_source_continuity_apply_allowed(tmp_path)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == _CURRENT_VERSION
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='durable_items'").fetchone() is None
        recovered = recover_durable_change_train(
            conn,
            failed,
            recovery_evidence_ref="proof:rollback-observed",
            writer_release_evidence_ref="proof:lease-released",
        )
    assert recovered.state is DurableChangeTrainState.ADMITTED
    assert recovered.reservation is None
    assert recovered.backup_authorization is None
    assert recovered.pre_apply_evidence is None


def test_interrupted_commit_recovers_at_applied_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()

        def must_not_reapply(*_args: object, **_kwargs: object) -> None:
            pytest.fail("interrupted target recovery re-entered the migration engine")

        monkeypatch.setattr(migration_runner, "migrate_archive_tier", must_not_reapply)
        recovered = reconcile_interrupted_durable_change_train(
            conn,
            train,
            interruption_evidence_ref="proof:process-died-after-commit",
            writer_release_evidence_ref="proof:lease-expired",
        )
        recovered_manifest = tmp_path / "source-interrupted-recovered.json"
        write_durable_change_train_manifest(recovered_manifest, recovered, expected_revision=-1)
        recovered = load_durable_change_train_manifest(recovered_manifest)
    assert recovered.state is DurableChangeTrainState.APPLIED
    assert recovered.apply_evidence is not None
    assert recovered.apply_evidence.recovered_after_interrupt is True
    assert recovered.reservation is not None and recovered.reservation.active is False


def test_bootstrap_reconciles_and_persists_interrupted_train_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _TARGET_VERSION
    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; "
        "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.RELEASED
    assert recovered.apply_evidence is not None
    assert recovered.apply_evidence.recovered_after_interrupt is True
    assert "proof:startup-recovery:train:source:v2" in recovered.proof_refs


def test_bootstrap_finishes_persisted_applied_train_without_reapplying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _TARGET_VERSION
    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; "
        "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
        train = reconcile_interrupted_durable_change_train(
            conn,
            train,
            interruption_evidence_ref="proof:post-commit-crash",
            writer_release_evidence_ref="proof:lease-expired",
        )
    assert train.reservation is not None
    train = replace(
        train,
        revision=train.revision + 1,
        reservation=replace(train.reservation, active=True, released_at_ms=None, release_evidence_ref=None),
    )
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)
    monkeypatch.setattr(
        migration_runner,
        "migrate_archive_tier",
        lambda *_args, **_kwargs: pytest.fail("startup attempted to reapply a committed train"),
    )

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    assert reconcile_durable_change_trains_on_startup(tmp_path) == (manifest,)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.RELEASED
    assert recovered.proof is not None
    assert recovered.apply_evidence is not None and recovered.apply_evidence.recovered_after_interrupt is True


@pytest.mark.parametrize("tier", (ArchiveTier.SOURCE, ArchiveTier.USER))
@pytest.mark.parametrize("replacement", ("missing", "content", "inode"))
def test_startup_proves_durable_continuity_before_initialization_or_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    replacement: str,
) -> None:
    """A lost or replaced durable file cannot be bootstrapped over an APPLIED train."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    db_path = tmp_path / f"{tier.value}.db"
    _create_current_database(db_path)
    bootstrap.initialize_archive_database(tmp_path / "audit.db", ArchiveTier.AUDIT)
    _install_synthetic_migration(tmp_path, monkeypatch, tier)
    train = _admitted(tier, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)

    if replacement == "missing":
        db_path.unlink()
    elif replacement == "content":
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE base_items SET payload = 'lost' WHERE item_id = 'base-1'")
            conn.commit()
    else:
        replacement_path = tmp_path / "replacement.db"
        replacement_path.write_bytes(db_path.read_bytes())
        os.replace(replacement_path, db_path)

    with pytest.raises(DurableChangeTrainError, match="refusing startup initialization/release"):
        bootstrap.initialize_active_archive_root(tmp_path)

    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.APPLIED
    assert recovered.reservation is not None and recovered.reservation.active is True
    if replacement == "missing":
        assert not db_path.exists()


def test_startup_recovers_persisted_rollback_failure_to_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
INSERT INTO table_that_does_not_exist VALUES (1);
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql))
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    assert reconcile_durable_change_trains_on_startup(tmp_path) == (manifest,)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.ADMITTED
    assert recovered.reservation is None
    assert recovered.failure is None


@pytest.mark.parametrize("replacement", ("content", "inode"))
def test_startup_blocks_persisted_rollback_failure_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
INSERT INTO table_that_does_not_exist VALUES (1);
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql))
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    if replacement == "content":
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE base_items SET payload = 'replaced' WHERE item_id = 'base-1'")
            conn.commit()
    else:
        replacement_path = tmp_path / "replacement.db"
        replacement_path.write_bytes(db_path.read_bytes())
        os.replace(replacement_path, db_path)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    with pytest.raises(DurableChangeTrainError, match="rolled-back recovery durable tier identity/content continuity"):
        reconcile_durable_change_trains_on_startup(tmp_path)
    blocked = load_durable_change_train_manifest(manifest)
    assert blocked.state is DurableChangeTrainState.FAILED
    assert blocked.reservation is not None and blocked.reservation.active is True


def test_startup_keeps_persisted_indeterminate_failure_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        with pytest.raises(DurableChangeTrainRecoveryError) as exc_info:
            reconcile_interrupted_durable_change_train(
                conn,
                train,
                interruption_evidence_ref="proof:unknown-version",
                writer_release_evidence_ref="proof:lease-expired",
            )
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    with pytest.raises(DurableChangeTrainRecoveryError, match="cannot recover automatically"):
        reconcile_durable_change_trains_on_startup(tmp_path)
    blocked = load_durable_change_train_manifest(manifest)
    assert blocked.state is DurableChangeTrainState.FAILED
    assert blocked.reservation is not None and blocked.reservation.active is True


def test_interrupted_unknown_version_requires_authenticated_restore(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        with pytest.raises(
            DurableChangeTrainRecoveryError,
            match="restore the exact authenticated backup",
        ) as exc_info:
            reconcile_interrupted_durable_change_train(
                conn,
                train,
                interruption_evidence_ref="proof:unknown-version",
                writer_release_evidence_ref="proof:lease-expired",
            )
        failed = exc_info.value.failed_train
        assert failed.state is DurableChangeTrainState.FAILED
        assert failed.failure is not None
        assert failed.failure.classification is DurableFailureClassification.INDETERMINATE
        failure_manifest = tmp_path / "source-indeterminate-train.json"
        write_durable_change_train_manifest(failure_manifest, failed, expected_revision=-1)
        failed = load_durable_change_train_manifest(failure_manifest)
        assert failed.reservation is not None and failed.reservation.active is True
        assert failed.failure is not None
        assert "keep the daemon stopped" in failed.failure.required_actions
        with pytest.raises(DurableChangeTrainRecoveryError, match="retain stopped-daemon"):
            record_durable_writer_release(failed, evidence_ref="proof:unsafe-release")


def test_restart_and_every_runtime_consumer_are_required_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "user.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.USER)
    train = _admitted(ArchiveTier.USER)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    train = record_durable_writer_release(train, evidence_ref="proof:lease-release")
    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            actual_parity = prove_durable_fresh_ddl_parity(
                ArchiveTier.USER,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref="proof:actual-fresh",
            )
        incomplete = (DurableRuntimeConsumerResult("consumer-0", "proof:behavior:0", True),)
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=incomplete,
            evidence_ref="proof:incomplete-restart",
        )
    assert restart.converged is False
    with pytest.raises(DurableChangeTrainError, match="runtime proof does not cover"):
        prove_durable_change_train(
            train,
            fresh_ddl_parity=actual_parity,
            runtime_consumers=incomplete,
            restart_convergence=restart,
        )
    with pytest.raises(DurableChangeTrainError, match="only a proven train"):
        release_durable_change_train(train, evidence_ref="proof:premature-release")


def test_manifest_semantics_reject_out_of_order_lifecycle_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    assert train.apply_evidence is not None

    late_post = replace(
        train.apply_evidence.post,
        observed_at_ms=train.apply_evidence.applied_at_ms + 1,
    )
    invalid_apply = replace(
        train,
        apply_evidence=replace(train.apply_evidence, post=late_post),
    )
    with pytest.raises(DurableChangeTrainError, match="apply timestamp predates post-apply"):
        migration_runner.validate_durable_change_train_manifest(invalid_apply)

    release_time = train.apply_evidence.applied_at_ms + 100
    released = record_durable_writer_release(
        train,
        evidence_ref="proof:lease-release",
        released_at_ms=release_time,
    )
    assert released.reservation is not None
    invalid_release = replace(
        released,
        reservation=replace(
            released.reservation,
            released_at_ms=train.apply_evidence.applied_at_ms - 1,
        ),
    )
    with pytest.raises(DurableChangeTrainError, match="writer release timestamp predates"):
        migration_runner.validate_durable_change_train_manifest(invalid_release)

    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            parity = prove_durable_fresh_ddl_parity(
                ArchiveTier.SOURCE,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref="proof:actual-fresh",
            )
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            restarted,
            released,
            runtime_consumers=runtime_results,
            evidence_ref="proof:restart",
        )
    restart_before_release = replace(
        restart,
        observed_at_ms=train.apply_evidence.applied_at_ms + 50,
    )
    with pytest.raises(DurableChangeTrainError, match="restart convergence timestamp predates writer release"):
        prove_durable_change_train(
            released,
            fresh_ddl_parity=parity,
            runtime_consumers=runtime_results,
            restart_convergence=restart_before_release,
            proven_at_ms=release_time + 1,
        )


def test_manifest_checksum_revision_and_unsafe_path_are_enforced(tmp_path: Path) -> None:
    train = _declared(ArchiveTier.SOURCE)
    path = tmp_path / "train.json"
    write_durable_change_train_manifest(path, train, expected_revision=-1)
    with pytest.raises(DurableChangeTrainError, match="revision changed"):
        write_durable_change_train_manifest(path, train, expected_revision=99)
    with pytest.raises(DurableChangeTrainError, match="advance exactly one revision"):
        write_durable_change_train_manifest(path, train, expected_revision=0)
    skipped_revision = replace(train, revision=2)
    with pytest.raises(DurableChangeTrainError, match="advance exactly one revision"):
        write_durable_change_train_manifest(path, skipped_revision, expected_revision=0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["owner_ref"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="checksum mismatch"):
        load_durable_change_train_manifest(path)

    real = tmp_path / "real.json"
    write_durable_change_train_manifest(real, train, expected_revision=-1)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(DurableChangeTrainError, match="not a real single-linked file"):
        load_durable_change_train_manifest(link)

    parent = tmp_path / "real-parent"
    parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(parent, target_is_directory=True)
    with pytest.raises(DurableChangeTrainError, match="parent.*symbolic link"):
        write_durable_change_train_manifest(linked_parent / "train.json", train, expected_revision=-1)

    locked_path = tmp_path / "locked-train.json"
    lock_path = tmp_path / ".locked-train.json.lock"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("not a lock", encoding="utf-8")
    lock_path.symlink_to(lock_target)
    with pytest.raises(DurableChangeTrainError, match="manifest lock safely"):
        write_durable_change_train_manifest(locked_path, train, expected_revision=-1)


def test_rechecks_manifest_semantics_after_a_valid_checksum(tmp_path: Path) -> None:
    train = _admitted(ArchiveTier.USER)
    payload = migration_runner.durable_change_train_to_payload(train)
    parity = payload["fresh_ddl_parity"]
    assert isinstance(parity, dict)
    parity["matches"] = False
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(unsigned)
    path = tmp_path / "forged-train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="fresh-DDL parity is not an exact match"):
        load_durable_change_train_manifest(path)
