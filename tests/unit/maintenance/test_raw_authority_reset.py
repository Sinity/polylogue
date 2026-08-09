"""Real-failure tests for the guarded raw-authority recovery family."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.raw_authority_recovery import (
    RawAuthorityRecoveryError,
    RecoveryOperation,
    apply_raw_authority_recovery,
    inspect_raw_authority_recovery,
)
from polylogue.maintenance.raw_authority_reset import (
    prune_orphaned_index_revision_seeds,
    reset_raw_authority_census,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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
