"""Regression coverage for owned pytest scratch leases."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import pytest

from devtools import pytest_scratch
from devtools.pytest_scratch import PytestScratchLease, measure_tree, run_managed_pytest


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


def test_terminal_cleanup_reclaims_mode_hardened_fixture_trees(tmp_path: Path) -> None:
    lease = PytestScratchLease.acquire(
        root=tmp_path / "scratch", run_id="hardened", lane="parallel", evidence_dir=tmp_path / "evidence"
    )
    hardened = lease.basetemp / "fixture" / "nested"
    hardened.mkdir(parents=True)
    hardened.joinpath("payload").write_text("stale", encoding="utf-8")
    hardened.chmod(0o400)

    receipt = lease.finalize("success")

    assert receipt["cleanup_complete"] is True
    assert not lease.lease_root.exists()


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


def test_new_lease_reclaims_markerless_dead_hardened_tree_but_preserves_live_owner(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    dead = scratch_root / "runs" / "20260825T000000Z-all-999999999-deadbeef-s1" / "parallel"
    dead.joinpath("pytest", "nested").mkdir(parents=True)
    dead.joinpath("pytest", "nested", "payload").write_text("stale", encoding="utf-8")
    dead.joinpath("pytest", "nested").chmod(0o400)
    live = scratch_root / "runs" / f"20260825T000000Z-all-{os.getpid()}-cafebabe-s1" / "parallel"
    live.joinpath("pytest").mkdir(parents=True)

    lease = PytestScratchLease.acquire(
        root=scratch_root, run_id="new-run", lane="parallel", evidence_dir=tmp_path / "evidence"
    )

    assert str(dead) in lease.stale_leases_reclaimed
    assert not dead.exists()
    assert live.exists()
    lease.finalize("success")


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


def test_failure_artifacts_do_not_follow_a_destination_symlink(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    evidence = tmp_path / "evidence"
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence.mkdir()
    lease = PytestScratchLease.acquire(root=scratch_root, run_id="run", lane="parallel", evidence_dir=evidence)
    lease.basetemp.mkdir(parents=True)
    lease.basetemp.joinpath("diagnostic.txt").write_text("private", encoding="utf-8")
    evidence.joinpath("scratch-failure-artifacts").symlink_to(outside, target_is_directory=True)

    metrics = lease.finalize("failure")

    assert metrics["failure_artifacts"]["error"] == "destination_not_owned"
    assert not outside.joinpath("diagnostic.txt").exists()


def test_high_water_metrics_use_worker_observations(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    evidence = tmp_path / "evidence"
    events = evidence / "events"
    events.mkdir(parents=True)
    events.joinpath("gw0-1.jsonl").write_text(
        json.dumps(
            {
                "event": "scratch_worker_high_water",
                "high_water": {
                    "apparent_bytes": 999,
                    "allocated_bytes": 8_192,
                    "file_count": 40,
                    "directory_count": 30,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lease = PytestScratchLease.acquire(root=scratch_root, run_id="run", lane="parallel", evidence_dir=evidence)

    metrics = lease.finalize("success")

    assert metrics["terminal_usage"]["apparent_bytes"] < 999
    assert metrics["high_water_scope"] == "observed_test_trees_and_terminal_lease"
    assert metrics["high_water_complete"] is False
    assert metrics["high_water_usage"] == {
        "apparent_bytes": 999,
        "allocated_bytes": 8_192,
        "file_count": 40,
        "directory_count": 30,
    }


def test_managed_pytest_kills_its_process_group_before_propagating_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        pid = 42
        returncode = 130

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                raise KeyboardInterrupt
            return self.returncode

    process = _Process()
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("devtools.pytest_scratch.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "devtools.pytest_scratch.os.killpg",
        lambda pid, signum: killed.append((pid, signum)),
    )

    with pytest.raises(KeyboardInterrupt):
        run_managed_pytest(["pytest"], cwd=Path.cwd(), env={})

    assert killed == [(42, signal.SIGTERM)]


def test_managed_pytest_records_process_group_memory_peaks(tmp_path: Path) -> None:
    metrics_path = tmp_path / "process-memory.json"

    completed = run_managed_pytest(
        [sys.executable, "-c", "import time; time.sleep(0.7)"],
        cwd=tmp_path,
        env={},
        resource_metrics_path=metrics_path,
    )

    assert completed.returncode == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 1
    assert metrics["samples"] >= 1
    assert metrics["aggregate_peak"]["rss_bytes"] > 0
    assert metrics["aggregate_peak"]["pss_bytes"] > 0
    assert metrics["process_peaks"]


def test_workstation_does_not_fallback_to_tmpfs_when_nvme_mount_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pytest_scratch, "DEFAULT_SCRATCH_ROOT", tmp_path / "missing" / "scratch")
    monkeypatch.setattr(pytest_scratch, "running_in_cloud_sandbox", lambda: False)

    with pytest.raises(RuntimeError, match="workstation scratch mount"):
        pytest_scratch.scratch_root_from_environment({})
