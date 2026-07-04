from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.degraded_archive_proof import main, run_degraded_archive_proof


def test_degraded_archive_proof_heals_rebuildable_state(tmp_path: Path) -> None:
    result = run_degraded_archive_proof(tmp_path / "proof")

    assert result.ok is True
    assert result.healing_driver == "daemon_owned_upkeep_primitives"
    assert result.degraded_inputs == (
        "messages_fts_freshness",
        "split_tier_wal",
        "split_tier_sqlite_stat1",
    )
    assert result.daemon_owned_primitives == (
        "repair_stale_fts_rows",
        "maybe_checkpoint_archive_wals",
        "maybe_optimize_archive_tiers",
    )
    assert result.always_running_paths == (
        "daemon_startup_fts_readiness",
        "daemon_convergence_fts_surface_debt",
        "daemon_periodic_wal_checkpoint",
        "daemon_periodic_db_optimize",
        "direct_archive_ingest_post_commit_upkeep",
    )
    assert result.archive_preserved is False
    assert result.demo_verified is True
    assert result.fts_ready_clean is True
    assert result.fts_ready_degraded is False
    assert result.fts_ready_after is True
    assert result.wal_degraded_observed is True
    assert result.fts_repair_success is True
    assert result.optimize_ran >= 1
    assert result.checkpoint_errors == ()
    assert result.optimize_errors == ()
    assert Path(result.artifact_json).exists()
    assert Path(result.artifact_markdown).exists()
    assert not Path(result.archive_root).exists()


def test_degraded_archive_proof_json_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--out-dir", str(tmp_path / "proof"), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["healing_driver"] == "daemon_owned_upkeep_primitives"
    assert payload["degraded_inputs"] == [
        "messages_fts_freshness",
        "split_tier_wal",
        "split_tier_sqlite_stat1",
    ]
    assert payload["daemon_owned_primitives"] == [
        "repair_stale_fts_rows",
        "maybe_checkpoint_archive_wals",
        "maybe_optimize_archive_tiers",
    ]
    assert payload["always_running_paths"] == [
        "daemon_startup_fts_readiness",
        "daemon_convergence_fts_surface_debt",
        "daemon_periodic_wal_checkpoint",
        "daemon_periodic_db_optimize",
        "direct_archive_ingest_post_commit_upkeep",
    ]
    assert payload["archive_preserved"] is False
    assert payload["fts_ready_clean"] is True
    assert payload["fts_ready_degraded"] is False
    assert payload["fts_ready_after"] is True
    assert payload["wal_degraded_observed"] is True
