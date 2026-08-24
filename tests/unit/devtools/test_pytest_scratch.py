"""Regression coverage for owned pytest scratch leases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.pytest_scratch import PytestScratchLease, measure_tree


def test_measure_tree_reports_apparent_allocated_and_file_counts(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    root.joinpath("payload").write_bytes(b"x" * 4_097)
    root.joinpath("nested").mkdir()
    root.joinpath("nested", "second").write_bytes(b"y")

    usage = measure_tree(root)

    assert usage.apparent_bytes == 4_098
    assert usage.allocated_bytes >= usage.apparent_bytes
    assert usage.file_count == 2
    assert usage.directory_count == 2


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "worker_crash"])
def test_terminal_outcomes_reclaim_only_their_own_lease(tmp_path: Path, outcome: str) -> None:
    scratch_root = tmp_path / "scratch"
    evidence = tmp_path / "evidence"
    owner = PytestScratchLease.acquire(
        root=scratch_root, run_id=f"run-{outcome}", lane="parallel", evidence_dir=evidence
    )
    sibling = PytestScratchLease.acquire(
        root=scratch_root, run_id=f"sibling-{outcome}", lane="parallel", evidence_dir=evidence
    )
    owner.basetemp.mkdir(parents=True)
    owner.basetemp.joinpath("diagnostic.txt").write_text("keep bounded evidence", encoding="utf-8")
    owner.basetemp.joinpath("incident-scale.bin").write_bytes(b"x" * (8 * 1024 * 1024 + 1))

    receipt = owner.finalize(outcome)  # type: ignore[arg-type]

    assert receipt["cleanup_complete"] is True
    assert not owner.lease_root.exists()
    assert sibling.lease_root.exists()
    if outcome == "success":
        assert not evidence.joinpath("scratch-failure-artifacts").exists()
    else:
        manifest = json.loads(evidence.joinpath("scratch-failure-artifacts", "manifest.json").read_text())
        assert manifest["copied_files"] == ["diagnostic.txt"]
        assert manifest["skipped"] == [
            {"path": "incident-scale.bin", "reason": "artifact_budget", "size": 8 * 1024 * 1024 + 1}
        ]
    sibling.finalize("success")


def test_new_lease_reclaims_only_dead_owner_markers(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    dead = scratch_root / "runs" / "abandoned" / "parallel"
    dead.mkdir(parents=True)
    dead.joinpath("lease.json").write_text(
        json.dumps({"pid": 999_999_999, "process_start_ticks": "0"}), encoding="utf-8"
    )
    empty = scratch_root / "runs" / "empty-completed-run"
    empty.mkdir(parents=True)
    live = PytestScratchLease.acquire(
        root=scratch_root, run_id="live", lane="parallel", evidence_dir=tmp_path / "evidence"
    )

    assert str(dead) in live.stale_leases_reclaimed
    assert str(empty) in live.stale_leases_reclaimed
    assert not dead.exists()
    assert not empty.exists()
    assert live.lease_root.exists()
    live.finalize("success")


def test_managed_command_rejects_unowned_shared_basetemp(tmp_path: Path) -> None:
    lease = PytestScratchLease.acquire(
        root=tmp_path / "scratch", run_id="run", lane="parallel", evidence_dir=tmp_path / "evidence"
    )

    with pytest.raises(ValueError, match="owns --basetemp"):
        lease.command(["pytest", "--basetemp=/shared"])

    command = lease.command(["pytest", "tests/unit"])
    assert command[-1] == f"--basetemp={lease.basetemp}"
    assert lease.environment({})["POLYLOGUE_PYTEST_SCRATCH_LANE"] == "parallel"
    lease.finalize("success")
