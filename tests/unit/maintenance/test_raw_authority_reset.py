"""Real-failure tests for the guarded raw-authority recovery family."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Literal, Never

import pytest

from polylogue.maintenance.raw_authority_recovery import (
    PruneOrphanedIndexRevisionSeedsActuator,
    RawAuthorityRecoveryError,
    RecoveryOperation,
    _canonical_bytes,
    _index_seed_digest,
    _RecoveryArgs,
    _write_recovery_intent,
    apply_raw_authority_recovery,
    inspect_raw_authority_recovery,
    resume_raw_authority_recovery,
    write_recovery_plan,
)
from polylogue.maintenance.raw_authority_reset import (
    prune_orphaned_index_revision_seeds,
    reset_raw_authority_census,
)
from polylogue.operations.mutation_transaction import OperationExecutor
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import write_source_continuity_pending_intent
from polylogue.storage.sqlite.migration_runner import DurableDatabaseEvidence, capture_durable_database_evidence
from polylogue.version import VERSION_INFO


def _seed_ledger(source_db: Path) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO raw_authority_parser_census "
            "(raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms) "
            "VALUES ('r-keep', 'parser-fp', 'complete', '[\"logical\"]', 'kept', 1)"
        )
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO raw_authority_censuses (census_id, sequence_no, scope_json, residual_json, "
            "parser_fingerprint, mode, lifecycle_status, quiescent, inventory_digest, residual_digest, "
            "plan_count, executable_plan_count, residual_plan_count, created_at_ms) "
            "VALUES ('c1',1,'{}','{}','fp','apply','planned',1,?,?,1,1,0,1)",
            (hashlib.sha256(b"inv").hexdigest(), hashlib.sha256(b"res").hexdigest()),
        )
        digest = hashlib.sha256(b"plan-1").hexdigest()
        conn.execute(
            "INSERT INTO raw_authority_plans (plan_id, input_digest, input_raw_ids_json, logical_keys_json, "
            "authority_witness_json, source_preconditions_json, index_preconditions_json, created_at_ms) "
            "VALUES ('plan-1',?,'[\"r1\"]','[]','{}','{}','{}',1)",
            (digest,),
        )
        conn.execute(
            "INSERT INTO raw_authority_blockers (blocker_id, plan_id, census_id, reason, expected_json, "
            "observed_json, created_at_ms) VALUES ('blk-1','plan-1','c1','r','{}','{}',1)"
        )
        conn.execute(
            "INSERT INTO raw_authority_census_plans (census_id, plan_id, ordinal, selected, outcome_status, "
            "reason, next_action, recorded_at_ms) VALUES ('c1','plan-1',0,1,'carried_forward','r','n',1)"
        )
        conn.execute(
            "INSERT INTO raw_authority_census_post_plans (census_id, plan_id, ordinal) VALUES ('c1','plan-1',0)"
        )


def _seed_raw(source_db: Path, raw_id: str) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, "
            "acquired_at_ms, revision_authority) VALUES (?, 'codex-session','/p',0,?,10,1,'byte_proven')",
            (raw_id, bytes.fromhex("01" * 32)),
        )


def _backup_authority(root: Path, monkeypatch: pytest.MonkeyPatch, *, tier: str) -> Path:
    backup = root / f"{tier}-backup" / "manifest.json"
    backup.parent.mkdir()
    backup.write_text("manifest", encoding="utf-8")
    receipt = backup.with_name("verification-receipt.json")
    receipt.write_text("receipt", encoding="utf-8")

    def validate(_path: Path, _tier: object, *, connection: sqlite3.Connection) -> Path:
        assert tuple(connection.execute("SELECT 1").fetchone()) == (1,)
        return receipt

    if tier == "source":
        monkeypatch.setattr("polylogue.maintenance.raw_authority_recovery.validate_migration_backup_manifest", validate)
    else:
        monkeypatch.setattr(
            "polylogue.maintenance.raw_authority_recovery.validate_backup_manifest_covers_derived_tier", validate
        )
    return backup


def test_census_reset_reproduces_poisoned_ledger_and_preserves_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")

    def bypass(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct storage reset bypass was called")

    monkeypatch.setattr("polylogue.storage.raw_authority.reset_raw_authority_census_ledger", bypass)

    before_parser = (
        sqlite3.connect(tmp_path / "source.db").execute("SELECT * FROM raw_authority_parser_census").fetchall()
    )
    dry = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS)
    assert dry.before_counts == {
        "raw_authority_censuses": 1,
        "raw_authority_plans": 1,
        "raw_authority_blockers": 1,
        "raw_authority_census_plans": 1,
        "raw_authority_census_post_plans": 1,
    }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)

    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    report = apply_raw_authority_recovery(plan)
    assert report.status == "applied"
    assert report.postflight == {
        "quick_check": ["ok"],
        "foreign_key_check": [],
        "protected_digest": plan.protected_digest,
    }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for table in (
            "raw_authority_censuses",
            "raw_authority_plans",
            "raw_authority_blockers",
            "raw_authority_census_plans",
            "raw_authority_census_post_plans",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,), table
        assert conn.execute("SELECT revision_authority FROM raw_sessions WHERE raw_id='r-keep'").fetchone() == (
            "byte_proven",
        )
        assert conn.execute("SELECT * FROM raw_authority_parser_census").fetchall() == before_parser

    receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert receipt["operation_id"] == plan.operation_id
    assert receipt["before_counts"] == plan.before_counts
    assert receipt["after_counts"] == dict.fromkeys(plan.before_counts, 0)
    assert (
        receipt["receipt_sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
    )

    repeated = apply_raw_authority_recovery(plan)
    assert repeated.status == "already_satisfied"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_authority_parser_census SET detail = 'changed' WHERE raw_id = 'r-keep'")
    with pytest.raises(RawAuthorityRecoveryError, match="changed"):
        apply_raw_authority_recovery(plan)


def test_census_reset_refuses_a_build_dirtied_after_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An apply cannot bind a plan after its source build changed."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    monkeypatch.setattr(VERSION_INFO, "dirty", False)
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)

    monkeypatch.setattr(VERSION_INFO, "dirty", True)
    with pytest.raises(RawAuthorityRecoveryError, match="clean build"):
        apply_raw_authority_recovery(plan)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_census_reset_refuses_wal_visible_ledger_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed ledger row invalidates a reset plan even when its main file is unchanged."""

    initialize_active_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    _seed_ledger(source_db)
    _seed_raw(source_db, "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    main_database_before = source_db.read_bytes()

    with sqlite3.connect(source_db) as conn:
        conn.execute("UPDATE raw_authority_censuses SET residual_json = '{\"changed\":true}' WHERE census_id = 'c1'")

    assert source_db.read_bytes() == main_database_before
    assert source_db.with_name("source.db-wal").is_file()
    refreshed = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    assert refreshed.ledger_digest != plan.ledger_digest
    assert refreshed.plan_digest != plan.plan_digest
    with pytest.raises(RawAuthorityRecoveryError, match="stale before lease acquisition"):
        apply_raw_authority_recovery(plan)
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT residual_json FROM raw_authority_censuses WHERE census_id = 'c1'").fetchone() == (
            '{"changed":true}',
        )


def test_census_reset_preserves_recovery_evidence_when_final_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed reset must leave restartable evidence before finalization."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    receipt_path = tmp_path / ".maintenance-state" / "raw-authority-recovery" / "nested" / "recovery.json"
    plan = inspect_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.RESET_CENSUS,
        backup_manifest=backup,
        receipt_path=receipt_path,
    )
    plan_file = tmp_path / "external-recovery-plan.json"
    write_recovery_plan(plan, plan_file)

    from polylogue.maintenance import raw_authority_recovery

    original_write = raw_authority_recovery._write_durable_immutable

    def fail_final_receipt(root: Path, path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
        if digest_field == "receipt_sha256":
            raise OSError("injected final receipt write failure")
        return original_write(root, path, payload, digest_field=digest_field)

    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", fail_final_receipt)
    with pytest.raises(RawAuthorityRecoveryError, match="injected final receipt write failure"):
        apply_raw_authority_recovery(plan)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (0,)
    receipt_path = Path(plan.receipt_path)
    intent_path = receipt_path.with_name(f"{receipt_path.name}.intent.json")
    assert intent_path.is_file()
    assert not receipt_path.exists()

    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", original_write)
    operation_id = plan.operation_id
    plan_file.unlink()
    recovered = resume_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.RESET_CENSUS,
        operation_id=operation_id,
        receipt_path=receipt_path,
    )
    assert recovered.status == "already_satisfied"
    assert recovered.receipt_path == receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["operation_id"] == operation_id
    assert receipt["plan_digest"] == recovered.plan.plan_digest


def test_persisted_recovery_plan_ignores_process_scoped_archive_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan written by one CLI process remains authorized in the next one."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    plan_file = tmp_path / "persisted-recovery-plan.json"
    write_recovery_plan(plan, plan_file)

    monkeypatch.setattr("polylogue.storage.archive_identity.os.getpid", lambda: 987654)
    assert apply_raw_authority_recovery(plan_file).status == "applied"


def test_census_reset_persists_source_continuity_intent_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable source-train protocol receives pre-mutation evidence before deletion."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)

    original_write_pending = write_source_continuity_pending_intent
    pending_paths: list[Path] = []

    def record_pending(
        archive_root: Path,
        *,
        mutation_receipt: Path,
        backup_manifest: Path,
        pre_mutation_evidence: DurableDatabaseEvidence,
        operation_id: str,
        evidence_ref: str,
        mutation_kind: Literal["blob_ref_liveness", "raw_authority_recovery"],
    ) -> Path:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)
        assert mutation_kind == "raw_authority_recovery"
        pending = original_write_pending(
            archive_root,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=pre_mutation_evidence,
            operation_id=operation_id,
            evidence_ref=evidence_ref,
            mutation_kind=mutation_kind,
        )
        pending_paths.append(pending)
        assert pending.is_file()
        return pending

    monkeypatch.setattr(
        "polylogue.maintenance.raw_authority_recovery.write_source_continuity_pending_intent", record_pending
    )
    assert apply_raw_authority_recovery(plan).status == "applied"
    assert len(pending_paths) == 1
    assert not pending_paths[0].exists()


def test_census_reset_clears_continuity_intent_when_precommit_backup_revalidation_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset refusal cannot strand a continuity intent without its receipt."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)

    def reject_changed_backup(*_args: object, **_kwargs: object) -> None:
        raise RawAuthorityRecoveryError("backup authority changed before commit")

    monkeypatch.setattr("polylogue.maintenance.raw_authority_recovery._backup_from_plan", reject_changed_backup)
    with pytest.raises(RawAuthorityRecoveryError, match="backup authority changed before commit"):
        apply_raw_authority_recovery(plan)

    pending_root = tmp_path / ".maintenance-state" / "source-continuity-pending"
    assert list(pending_root.glob("*.json")) == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_recovery_protected_digest_streams_rows_without_a_whole_table_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspection hashes protected tables incrementally instead of serializing all rows together."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("CREATE TABLE protected_rows (row_id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
        conn.executemany(
            "INSERT INTO protected_rows (row_id, payload) VALUES (?, ?)",
            [(index, bytes([index]) * 32) for index in range(1, 5)],
        )

    def reject_whole_table_payload(value: object) -> bytes:
        if isinstance(value, dict) and "rows" in value:
            raise AssertionError("protected digest materialized a whole table")
        return _canonical_bytes(value)

    monkeypatch.setattr("polylogue.maintenance.raw_authority_recovery._canonical_bytes", reject_whole_table_payload)
    before = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE protected_rows SET payload = x'ff' WHERE row_id = 2")
    after = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS)

    assert after.protected_digest != before.protected_digest


def test_recovery_receipt_path_must_be_owned_by_the_archive(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")

    with pytest.raises(RawAuthorityRecoveryError, match="archive-owned durable location"):
        inspect_raw_authority_recovery(
            tmp_path,
            RecoveryOperation.RESET_CENSUS,
            receipt_path=tmp_path.parent / "arbitrary-recovery-receipt.json",
        )


def test_recovery_receipt_path_rejects_archive_symlink_escape(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    escaped = tmp_path.parent / "escaped-recovery-receipts"
    escaped.mkdir()
    receipt_dir = tmp_path / ".maintenance-state" / "raw-authority-recovery"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "escape").symlink_to(escaped, target_is_directory=True)

    with pytest.raises(RawAuthorityRecoveryError, match="must not traverse an archive symlink"):
        inspect_raw_authority_recovery(
            tmp_path,
            RecoveryOperation.RESET_CENSUS,
            receipt_path=receipt_dir / "escape" / "receipt.json",
        )


def test_recovery_fsyncs_each_new_durable_receipt_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production receipt writer durably records every created directory entry."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    receipt_path = tmp_path / ".maintenance-state" / "raw-authority-recovery" / "nested" / "recovery.json"
    plan = inspect_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.RESET_CENSUS,
        backup_manifest=backup,
        receipt_path=receipt_path,
    )

    original_fsync = os.fsync
    synced_inodes: set[int] = set()

    def record_fsync(fd: int) -> None:
        synced_inodes.add(os.fstat(fd).st_ino)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    assert apply_raw_authority_recovery(plan).status == "applied"

    expected = {
        (tmp_path / ".maintenance-state").stat().st_ino,
        (tmp_path / ".maintenance-state" / "raw-authority-recovery").stat().st_ino,
        receipt_path.parent.stat().st_ino,
    }
    assert expected <= synced_inodes


def test_recovery_fsyncs_surviving_receipt_parents_after_interrupted_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed writer persists parent entries it did not create itself."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    receipt_parent = tmp_path / ".maintenance-state" / "raw-authority-recovery"
    receipt_parent.mkdir(parents=True)
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)

    original_fsync = os.fsync
    synced_inodes: set[int] = set()

    def record_fsync(fd: int) -> None:
        synced_inodes.add(os.fstat(fd).st_ino)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    assert apply_raw_authority_recovery(plan).status == "applied"
    assert (tmp_path / ".maintenance-state").stat().st_ino in synced_inodes
    assert receipt_parent.stat().st_ino in synced_inodes


def test_recovery_receipt_rejects_fifo_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-regular artifact must not be opened as a receipt stream."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    receipt_path = Path(plan.receipt_path)
    receipt_path.parent.mkdir(parents=True)
    os.mkfifo(receipt_path)
    holder_fd = os.open(receipt_path, os.O_RDWR | os.O_NONBLOCK)
    os.write(holder_fd, b"{}")
    try:
        with pytest.raises(RawAuthorityRecoveryError, match="regular archive-owned file"):
            apply_raw_authority_recovery(plan)
    finally:
        os.close(holder_fd)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_uncommitted_recovery_intent_reauthorizes_through_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intent without committed postflight state is not authorization to mutate."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)

    _write_recovery_intent(plan)

    def require_authorization(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("executor authorization was required")

    monkeypatch.setattr(OperationExecutor, "authorize", require_authorization)
    with pytest.raises(RuntimeError, match="executor authorization was required"):
        apply_raw_authority_recovery(plan)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_census_reset_dry_run_does_not_mutate_and_apply_requires_backup(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    before = (tmp_path / "source.db").read_bytes()
    reset_raw_authority_census(tmp_path, dry_run=True)
    assert (tmp_path / "source.db").read_bytes() == before
    with pytest.raises(RawAuthorityRecoveryError, match="backup authority"):
        reset_raw_authority_census(tmp_path, dry_run=False)
    assert (tmp_path / "source.db").read_bytes() == before


def test_census_reset_refuses_malformed_ledger(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE raw_authority_censuses SET scope_json = 'not-json'")
    with pytest.raises(RawAuthorityRecoveryError, match="malformed"):
        inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS)


def test_census_reset_refuses_running_daemon_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    monkeypatch.setattr("polylogue.maintenance.raw_authority_recovery.running_daemon_pid", lambda _config: 123)

    with pytest.raises(RawAuthorityRecoveryError, match="polylogued is running"):
        apply_raw_authority_recovery(plan)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_census_reset_refuses_changed_source_fingerprint_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_authority_parser_census SET detail = 'changed' WHERE raw_id = 'r-keep'")

    with pytest.raises(RawAuthorityRecoveryError, match="stale"):
        apply_raw_authority_recovery(plan)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def _seed_index_seeds(root: Path) -> Path:
    source_db = root / "source.db"
    index_db = root / "index.db"
    _seed_raw(source_db, "r-present")
    with sqlite3.connect(index_db) as conn:
        for raw_id in ("r-present", "r-gone"):
            conn.execute(
                "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
                "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
                "acquisition_generation, decided_at_ms) VALUES (?,?,?,'sr',?,'byte',1,0,1)",
                (f"k-{raw_id}", f"s-{raw_id}", raw_id, bytes.fromhex("03" * 32)),
            )
            conn.execute(
                "INSERT INTO raw_revision_applications (decision_id, raw_id, session_id, logical_source_key, "
                "source_revision, acquisition_generation, decision, detail, decided_at_ms) "
                "VALUES (?,?,?,?,'sr',0,'selected_baseline','d',1)",
                (f"d-{raw_id}", raw_id, f"s-{raw_id}", f"k-{raw_id}"),
            )
    active_index = root / "active-generation" / "index.db"
    active_index.parent.mkdir()
    shutil.copy2(index_db, active_index)
    (root / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    return active_index


def test_index_prune_reproduces_orphan_failure_and_preserves_present_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    active_index = _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")

    def bypass(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct storage prune bypass was called")

    monkeypatch.setattr("polylogue.storage.raw_authority.prune_orphaned_index_revision_seeds", bypass)
    dry = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS)
    assert dry.before_counts == {"raw_revision_heads": 2, "raw_revision_applications": 2}
    assert sqlite3.connect(active_index).execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (2,)

    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)
    report = apply_raw_authority_recovery(plan)
    assert report.status == "applied"
    assert report.after_counts == {"raw_revision_heads": 1, "raw_revision_applications": 1}
    with sqlite3.connect(active_index) as conn:
        assert {row[0] for row in conn.execute("SELECT accepted_raw_id FROM raw_revision_heads")} == {"r-present"}
        assert {row[0] for row in conn.execute("SELECT raw_id FROM raw_revision_applications")} == {"r-present"}
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert {row[0] for row in conn.execute("SELECT accepted_raw_id FROM raw_revision_heads")} == {
            "r-present",
            "r-gone",
        }
    assert apply_raw_authority_recovery(plan).status == "already_satisfied"


def test_index_seed_digest_avoids_unbounded_sql_parameters(tmp_path: Path) -> None:
    """Candidate exclusion remains usable below SQLite's configured bind limit."""

    initialize_active_archive_root(tmp_path)
    active_index = _seed_index_seeds(tmp_path)
    with sqlite3.connect(active_index) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(tmp_path / "source.db"),))
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 1)
        digest = _index_seed_digest(
            conn,
            excluded_keys={
                "raw_revision_heads": ("k-r-present", "k-r-gone"),
                "raw_revision_applications": ("d-r-present", "d-r-gone"),
            },
        )
    assert len(digest) == 64


def test_index_prune_plan_keeps_retained_seed_evidence_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real recovery plan stores summary proof, never every retained seed identity."""

    initialize_active_archive_root(tmp_path)
    active_index = _seed_index_seeds(tmp_path)
    with sqlite3.connect(active_index) as conn:
        conn.executemany(
            "INSERT INTO raw_revision_applications "
            "(decision_id, raw_id, session_id, logical_source_key, source_revision, acquisition_generation, "
            "decision, detail, decided_at_ms) VALUES (?, 'r-present', ?, ?, 'sr', 1, 'selected_baseline', 'd', 2)",
            [(f"d-retained-{index}", f"s-retained-{index}", f"k-retained-{index}") for index in range(64)],
        )

    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)

    assert plan.post_target_proof is not None
    applications_proof = plan.post_target_proof["raw_revision_applications"]
    assert set(applications_proof) == {
        "excluded_rowids",
        "retained_row_count",
        "retained_rows_sha256",
        "rowid_watermark",
    }
    assert applications_proof["retained_row_count"] == 65
    assert "d-retained-0" not in json.dumps(plan.to_dict(), sort_keys=True)


def test_index_prune_refuses_stale_active_pointer_and_wrong_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)
    other = tmp_path / "other" / "index.db"
    other.parent.mkdir()
    shutil.copy2(tmp_path / "index.db", other)
    (tmp_path / ".index-active-pointer").write_text(str(other), encoding="utf-8")
    with pytest.raises(RawAuthorityRecoveryError, match="stale|changed"):
        apply_raw_authority_recovery(plan)
    with pytest.raises(RawAuthorityRecoveryError, match="does not match"):
        apply_raw_authority_recovery(plan, backup_manifest=tmp_path / "different.json")


def test_index_prune_resume_refuses_an_unbacked_retained_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Receipt finalization refuses a retained head whose raw source no longer exists."""

    initialize_active_archive_root(tmp_path)
    active_index = _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)

    from polylogue.maintenance import raw_authority_recovery

    original_write = raw_authority_recovery._write_durable_immutable

    def fail_final_receipt(root: Path, path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
        if digest_field == "receipt_sha256":
            raise OSError("injected final receipt write failure")
        return original_write(root, path, payload, digest_field=digest_field)

    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", fail_final_receipt)
    with pytest.raises(RawAuthorityRecoveryError, match="injected final receipt write failure"):
        apply_raw_authority_recovery(plan)
    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", original_write)

    with sqlite3.connect(active_index) as conn:
        conn.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = 'r-gone' WHERE logical_source_key = 'k-r-present'"
        )

    with pytest.raises(RawAuthorityRecoveryError, match="unbacked index head"):
        resume_raw_authority_recovery(
            tmp_path,
            RecoveryOperation.PRUNE_INDEX_SEEDS,
            operation_id=plan.operation_id,
        )


def test_index_prune_resume_accepts_source_backed_successor_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt finalization admits later source-backed heads beyond the proof watermark."""

    initialize_active_archive_root(tmp_path)
    monkeypatch.setattr(VERSION_INFO, "dirty", False)
    active_index = _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)

    from polylogue.maintenance import raw_authority_recovery

    original_write = raw_authority_recovery._write_durable_immutable

    def fail_final_receipt(root: Path, path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
        if digest_field == "receipt_sha256":
            raise OSError("injected final receipt write failure")
        return original_write(root, path, payload, digest_field=digest_field)

    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", fail_final_receipt)
    with pytest.raises(RawAuthorityRecoveryError, match="injected final receipt write failure"):
        apply_raw_authority_recovery(plan)
    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", original_write)

    _seed_raw(tmp_path / "source.db", "r-successor")
    with sqlite3.connect(active_index) as conn:
        conn.execute(
            "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) VALUES "
            "('k-successor', 's-successor', 'r-successor', 'sr', ?, 'byte', 1, 0, 2)",
            (bytes.fromhex("03" * 32),),
        )
        conn.execute(
            "INSERT INTO raw_revision_applications "
            "(decision_id, raw_id, session_id, logical_source_key, source_revision, acquisition_generation, "
            "decision, detail, decided_at_ms) VALUES "
            "('d-successor', 'r-successor', 's-successor', 'k-successor', 'sr', 1, 'selected_baseline', 'd', 2)"
        )

    recovered = resume_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.PRUNE_INDEX_SEEDS,
        operation_id=plan.operation_id,
    )
    assert recovered.status == "already_satisfied"
    with sqlite3.connect(active_index) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_revision_heads WHERE logical_source_key = 'k-successor'"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_revision_applications WHERE decision_id = 'd-successor'"
        ).fetchone() == (1,)


def test_index_prune_resume_accepts_source_backed_in_place_successor_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt finalization admits a production-style replacement of a retained head row."""

    initialize_active_archive_root(tmp_path)
    monkeypatch.setattr(VERSION_INFO, "dirty", False)
    active_index = _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)

    from polylogue.maintenance import raw_authority_recovery

    original_write = raw_authority_recovery._write_durable_immutable

    def fail_final_receipt(root: Path, path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
        if digest_field == "receipt_sha256":
            raise OSError("injected final receipt write failure")
        return original_write(root, path, payload, digest_field=digest_field)

    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", fail_final_receipt)
    with pytest.raises(RawAuthorityRecoveryError, match="injected final receipt write failure"):
        apply_raw_authority_recovery(plan)
    monkeypatch.setattr(raw_authority_recovery, "_write_durable_immutable", original_write)

    from polylogue.archive.revision_replay import ApplicationDecision
    from polylogue.storage.sqlite.archive_tiers.revision_application import (
        RevisionApplicationReceipt,
        record_revision_application_sync,
    )

    with sqlite3.connect(active_index) as conn:
        record_revision_application_sync(
            conn,
            RevisionApplicationReceipt(
                raw_id="r-present",
                session_id="s-successor",
                logical_source_key="k-r-present",
                source_revision="sr",
                acquisition_generation=0,
                decision=ApplicationDecision.SELECTED_BASELINE,
                accepted_raw_id="r-present",
                accepted_source_revision="sr",
                accepted_content_hash=bytes.fromhex("04" * 32),
                accepted_frontier_kind="byte",
                accepted_frontier=1,
            ),
            decided_at_ms=2,
        )

    recovered = resume_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.PRUNE_INDEX_SEEDS,
        operation_id=plan.operation_id,
    )
    assert recovered.status == "already_satisfied"
    with sqlite3.connect(active_index) as conn:
        assert conn.execute(
            "SELECT session_id, accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'k-r-present'"
        ).fetchone() == ("s-successor", bytes.fromhex("04" * 32))


def test_census_reset_refuses_a_competing_source_continuity_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset never overlaps another source mutation's recovery evidence."""

    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    backup = _backup_authority(tmp_path, monkeypatch, tier="source")
    monkeypatch.setattr(VERSION_INFO, "dirty", False)
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.RESET_CENSUS, backup_manifest=backup)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        before = capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
    pending = write_source_continuity_pending_intent(
        tmp_path,
        mutation_receipt=tmp_path / "liveness-receipt.jsonl",
        backup_manifest=backup,
        pre_mutation_evidence=before,
        operation_id="other-source-mutation",
        evidence_ref="proof:other-source-mutation",
    )

    with pytest.raises(RawAuthorityRecoveryError, match="continuity recovery is pending"):
        apply_raw_authority_recovery(plan)

    assert pending.exists()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_uncommitted_index_prune_intent_reauthorizes_before_deleting_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intact index-prune intent resumes through executor authorization, not postflight receipt recovery."""

    initialize_active_archive_root(tmp_path)
    active_index = _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)
    _write_recovery_intent(plan)

    original_authorize = OperationExecutor.authorize

    def require_authorization(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("executor authorization was required")

    monkeypatch.setattr(OperationExecutor, "authorize", require_authorization)
    with pytest.raises(RuntimeError, match="executor authorization was required"):
        resume_raw_authority_recovery(
            tmp_path,
            RecoveryOperation.PRUNE_INDEX_SEEDS,
            operation_id=plan.operation_id,
        )
    with sqlite3.connect(active_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone() == (2,)

    monkeypatch.setattr(OperationExecutor, "authorize", original_authorize)
    resumed = resume_raw_authority_recovery(
        tmp_path,
        RecoveryOperation.PRUNE_INDEX_SEEDS,
        operation_id=plan.operation_id,
    )
    assert resumed.status == "applied"


def test_index_recovery_actuator_receipt_retains_authorized_plan_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor receipt remains bound to the plan it authorized."""

    initialize_active_archive_root(tmp_path)
    _seed_index_seeds(tmp_path)
    backup = _backup_authority(tmp_path, monkeypatch, tier="index")
    plan = inspect_raw_authority_recovery(tmp_path, RecoveryOperation.PRUNE_INDEX_SEEDS, backup_manifest=backup)
    args = _RecoveryArgs(
        archive_root=tmp_path,
        operation=RecoveryOperation.PRUNE_INDEX_SEEDS,
        operation_id=plan.operation_id,
        expected_plan_digest=plan.plan_digest,
        backup_manifest=backup,
        receipt_path=Path(plan.receipt_path),
    )
    actuator = PruneOrphanedIndexRevisionSeedsActuator()

    executor = OperationExecutor()
    prepared = executor.prepare(actuator, args)
    assert prepared.target_refs == ("index:prune_orphaned_index_revision_seeds",)
    assert prepared.affected_tiers == ("index",)
    authorization = executor.authorize(
        actuator,
        prepared,
        actor="test:raw-authority",
        role="maintenance",
        capability="archive.raw_authority_recovery",
        confirmation_strength="confirm_flag",
    )
    receipt = executor.execute(actuator, prepared, authorization, args)

    assert authorization.plan_hash == prepared.plan_hash
    assert receipt.plan_hash == prepared.plan_hash
    assert receipt.plan_hash != plan.plan_digest
    domain_plan = receipt.domain_receipt["plan"]
    assert isinstance(domain_plan, dict)
    assert domain_plan["plan_digest"] == plan.plan_digest
    assert receipt.target_refs == ("index:prune_orphaned_index_revision_seeds",)
    assert receipt.affected_count == 2


def test_named_compatibility_facade_requires_index_backup(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_index_seeds(tmp_path)
    with pytest.raises(RawAuthorityRecoveryError, match="backup authority"):
        prune_orphaned_index_revision_seeds(tmp_path, dry_run=False)


def test_storage_compatibility_helpers_refuse_direct_mutation(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")
    _seed_raw(tmp_path / "source.db", "r-keep")
    _seed_index_seeds(tmp_path)

    from polylogue.storage.raw_authority import (
        prune_orphaned_index_revision_seeds as storage_prune_orphaned_index_revision_seeds,
    )
    from polylogue.storage.raw_authority import (
        reset_raw_authority_census_ledger as storage_reset_raw_authority_census_ledger,
    )

    with pytest.raises(RuntimeError, match="direct raw-authority census mutation is disabled"):
        storage_reset_raw_authority_census_ledger(tmp_path, backup_manifest=None, dry_run=False)
    with pytest.raises(RuntimeError, match="direct orphaned-index-seed mutation is disabled"):
        storage_prune_orphaned_index_revision_seeds(tmp_path, dry_run=False)
