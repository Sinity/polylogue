from __future__ import annotations

import sqlite3
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from devtools import raw_authority_restart_proof as proof
from devtools.command_catalog import COMMANDS
from polylogue.storage import repair
from polylogue.storage.raw_authority import RawReplayPlanStatus


def test_raw_authority_restart_proof_reaches_conserved_two_census_fixed_point(tmp_path: Path) -> None:
    """Exercise production repair, durable census storage, recovery, and receipt validation."""
    payload = proof.run_raw_authority_restart_proof(tmp_path, keep=True)

    assert payload["schema"] == "polylogue.raw-authority-restart-proof.v1"
    assert cast(str, payload["proof_id"]).startswith("raw-authority-restart-proof:")
    assert payload["production_limits"] == {
        "raw_artifact_limit": None,
        "max_payload_bytes": repair.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
        "parser_census_component_limit": repair.RAW_MATERIALIZATION_CENSUS_COMPONENT_LIMIT,
    }

    cases = cast(list[dict[str, object]], payload["fault_matrix"])
    assert [case["boundary"] for case in cases] == [boundary.value for boundary in proof.FaultBoundary]
    for case in cases:
        assert case["terminal_status_counts"] == {
            RawReplayPlanStatus.DEFERRED.value: 1,
            RawReplayPlanStatus.EXECUTED.value: 2,
            RawReplayPlanStatus.TERMINAL.value: 1,
        }
        topology = cast(dict[str, object], case["topology"])
        assert topology["component_sizes"] == [1, 1, 2, 2]
        assert topology["raw_count"] == 6
        assert topology["membership_row_count"] == 12
        conservation = cast(dict[str, object], case["conservation"])
        assert conservation["initial_plan_count"] == 4
        assert conservation["all_census_rows_conserved"] is True
        assert conservation["each_initial_plan_terminal_once"] is True
        assert conservation["open_blocker_count"] == 0
        assert conservation["planned_census_count"] == 0
        fixed_point = cast(dict[str, object], case["fixed_point"])
        assert fixed_point["second_census_fixed_point"] is True
        assert len(cast(list[str], fixed_point["census_ids"])) == 2

    assert len(cast(list[str], cases[0]["interrupted_census_ids"])) == 1
    assert len(cast(list[str], cases[1]["interrupted_census_ids"])) == 1
    assert len(cast(list[str], cases[2]["interrupted_census_ids"])) == 2
    assert [
        [crash["validated_executed_receipt_count"] for crash in cast(list[dict[str, object]], case["crashes"])]
        for case in cases
    ] == [[1], [2], [1, 1]]


def test_raw_authority_restart_proof_rejects_broken_ledger_conservation(tmp_path: Path) -> None:
    """Removing one recorded outcome must make the source-ledger conservation audit fail."""
    topology = proof._prepare_case(tmp_path / "broken-conservation")
    execution = proof._exercise_fault(topology, proof.FaultBoundary.BEFORE_OUTCOME_COMMIT)

    with sqlite3.connect(topology.archive_root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT cp.census_id, cp.plan_id
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_censuses AS c ON c.census_id = cp.census_id
            WHERE c.sequence_no > ?
              AND c.lifecycle_status IN ('completed', 'interrupted')
              AND cp.selected = 1
              AND cp.outcome_status IN ('executed', 'deferred', 'terminal')
            ORDER BY c.sequence_no, cp.ordinal
            LIMIT 1
            """,
            (topology.preview_sequence_no,),
        ).fetchone()
        assert row is not None
        conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_recorded = 0
            WHERE census_id = ? AND plan_id = ?
            """,
            row,
        )
        conn.commit()

    with pytest.raises(proof.RawAuthorityRestartProofError, match="exactly one recorded outcome per plan"):
        proof._audit_case(topology, proof.FaultBoundary.BEFORE_OUTCOME_COMMIT, execution)


def test_raw_authority_restart_proof_rejects_postcondition_mutation_after_crash(tmp_path: Path) -> None:
    """Mutating an applied membership before restart must drive production recovery fail-closed."""
    topology = proof._prepare_case(tmp_path / "broken-postcondition")
    proof._inject_before_outcome_commit(
        topology,
        expected_role="solo-one",
        label="postcondition mutation crash",
        boundary=proof.FaultBoundary.BEFORE_OUTCOME_COMMIT,
    )

    solo_raw_id = topology.raw_ids_by_role["solo-one"][0]
    with sqlite3.connect(topology.archive_root / "source.db") as conn:
        membership = conn.execute(
            """
            SELECT provider_session_id, logical_source_key
            FROM raw_session_memberships
            WHERE raw_id = ?
            ORDER BY logical_source_key
            LIMIT 1
            """,
            (solo_raw_id,),
        ).fetchone()
        assert membership is not None
        conn.execute(
            """
            DELETE FROM raw_session_memberships
            WHERE raw_id = ? AND provider_session_id = ? AND logical_source_key = ?
            """,
            (solo_raw_id, membership[0], membership[1]),
        )
        conn.commit()

    with pytest.raises(proof.RawAuthorityRestartProofError, match="unresolved durable blocker"):
        proof._resume_and_drain(topology)

    with sqlite3.connect(topology.archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone() == (
            1,
        )
        assert conn.execute(
            """
            SELECT cp.outcome_status
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_censuses AS c ON c.census_id = cp.census_id
            WHERE c.lifecycle_status = 'interrupted'
              AND cp.plan_id = ?
            """,
            (topology.plan_ids_by_role["solo-one"],),
        ).fetchone() == (RawReplayPlanStatus.REJECTED_STALE.value,)


def test_raw_authority_restart_proof_cli_and_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def record_call(workdir: Path, *, keep: bool = False) -> dict[str, object]:
        captured["workdir"] = workdir
        captured["keep"] = keep
        return {"proof_id": "raw-authority-restart-proof:test"}

    monkeypatch.setattr(proof, "run_raw_authority_restart_proof", record_call)
    stdout = StringIO()

    assert proof.main(["--workdir", str(tmp_path), "--keep"], stdout=stdout) == 0
    assert captured == {"workdir": tmp_path, "keep": True}
    assert stdout.getvalue() == "raw-authority-restart-proof:test\n"
    command = COMMANDS["workspace raw-authority-restart-proof"]
    assert command.module == "devtools.raw_authority_restart_proof"
    assert command.examples == (
        "devtools workspace raw-authority-restart-proof --json",
        "devtools workspace raw-authority-restart-proof --workdir .cache/raw-restart-proof --keep --json",
    )
