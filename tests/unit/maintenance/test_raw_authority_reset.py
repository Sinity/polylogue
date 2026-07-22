"""Raw-authority census-ledger reset (convergence recovery)."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.raw_authority_reset import (
    prune_orphaned_index_revision_seeds,
    reset_raw_authority_census,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_ledger(source_db: Path) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")  # seeding only
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


def test_reset_empties_ledger_but_preserves_accepted_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_active_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    _seed_ledger(source_db)
    # Accepted materialization state that MUST survive a census-ledger reset.
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, "
            "acquired_at_ms, revision_authority) VALUES ('r-keep','codex-session','/p',0,?,10,1,'byte_proven')",
            (b"\x01" * 32,),
        )

    dry = reset_raw_authority_census(tmp_path, dry_run=True)
    assert dry.applied is False
    assert (dry.censuses, dry.plans, dry.blockers, dry.census_plans, dry.census_post_plans) == (1, 1, 1, 1, 1)
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0] == 1  # dry: nothing deleted

    validated: list[tuple[Path, object]] = []

    def validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr("polylogue.storage.raw_authority.validate_migration_backup_manifest", validate)
    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = reset_raw_authority_census(tmp_path, backup_manifest=manifest, dry_run=False)
    assert report.applied is True
    assert validated == [(manifest, ArchiveTier.SOURCE)]

    with sqlite3.connect(source_db) as conn:
        for table in (
            "raw_authority_censuses",
            "raw_authority_plans",
            "raw_authority_blockers",
            "raw_authority_census_plans",
            "raw_authority_census_post_plans",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
        # Accepted state preserved.
        row = conn.execute("SELECT revision_authority FROM raw_sessions WHERE raw_id='r-keep'").fetchone()
        assert row == ("byte_proven",)


def test_reset_refuses_to_delete_without_verified_backup_manifest(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _seed_ledger(tmp_path / "source.db")

    with pytest.raises(ValueError, match="verified source backup manifest"):
        reset_raw_authority_census(tmp_path, dry_run=False)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone() == (1,)


def test_prune_orphaned_index_revision_seeds(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    # One present raw in source; the seeds referencing 'r-gone' are orphaned.
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, "
            "acquired_at_ms) VALUES ('r-present','codex-session','/p',0,?,10,1)",
            (b"\x02" * 32,),
        )
    with sqlite3.connect(index_db) as conn:
        for raw_id in ("r-present", "r-gone"):
            conn.execute(
                "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
                "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
                "acquisition_generation, decided_at_ms) VALUES (?,?,?,'sr',?,'byte',1,0,1)",
                (f"k-{raw_id}", f"s-{raw_id}", raw_id, b"\x03" * 32),
            )
            conn.execute(
                "INSERT INTO raw_revision_applications (decision_id, raw_id, session_id, logical_source_key, "
                "source_revision, acquisition_generation, decision, detail, decided_at_ms) "
                "VALUES (?,?,?,?,'sr',0,'selected_baseline','d',1)",
                (f"d-{raw_id}", raw_id, f"s-{raw_id}", f"k-{raw_id}"),
            )

    active_index = tmp_path / "active-generation" / "index.db"
    active_index.parent.mkdir()
    shutil.copy2(index_db, active_index)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    dry = prune_orphaned_index_revision_seeds(tmp_path, dry_run=True)
    assert (dry.revision_heads, dry.revision_applications, dry.applied) == (1, 1, False)

    report = prune_orphaned_index_revision_seeds(tmp_path, dry_run=False)
    assert report.applied is True and report.revision_heads == 1 and report.revision_applications == 1

    with sqlite3.connect(active_index) as conn:
        assert {r[0] for r in conn.execute("SELECT accepted_raw_id FROM raw_revision_heads")} == {"r-present"}
        assert {r[0] for r in conn.execute("SELECT raw_id FROM raw_revision_applications")} == {"r-present"}
    with sqlite3.connect(index_db) as conn:
        assert {r[0] for r in conn.execute("SELECT accepted_raw_id FROM raw_revision_heads")} == {"r-present", "r-gone"}
