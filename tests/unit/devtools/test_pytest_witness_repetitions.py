"""Contract tests for managed pytest witness repetition receipts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from devtools.pytest_witness_repetitions import WITNESSES, run_repetitions


def _write_managed_run(root: Path, *, ordinal: int, nodeid: str) -> None:
    run = root / ".cache" / "verify" / "runs" / f"run-{ordinal}"
    step = run / "steps" / "01-pytest-focused"
    step.mkdir(parents=True)
    archive = root / ".tmpfs" / f"archive-{ordinal}"
    report_path = ".cache/verify/runs/report.json"
    (root / report_path).parent.mkdir(parents=True, exist_ok=True)
    (root / report_path).write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": nodeid,
                        "setup": {"duration": 0.1, "longrepr": "[gw2] linux"},
                        "call": {"duration": 0.2, "longrepr": "[gw2] linux"},
                        "teardown": {"duration": 0.1, "longrepr": "[gw2] linux"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (step / "containment.json").write_text(
        json.dumps({"tmpfs_cleanup_path": str(archive), "controller_group_alive": False}), encoding="utf-8"
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"duration_s": 0.8, "report_path": report_path, "containment_path": str(step / "containment.json")}
                ]
            }
        ),
        encoding="utf-8",
    )


def test_repetitions_retain_every_attempt_with_managed_cleanup_receipts(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str], root: Path, _env: dict[str, str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        _write_managed_run(root, ordinal=len(calls), nodeid=command[4])
        return subprocess.CompletedProcess(command, 0, "", "")

    receipt = run_repetitions(source_root=tmp_path, attempts_per_mode=2, xdist_workers=2, runner=runner)

    assert receipt.ok
    assert len(receipt.attempts) == len(WITNESSES) * 2 * 2
    assert {attempt.mode for attempt in receipt.attempts} == {"isolated", "xdist"}
    assert all(attempt.node_duration_s == 0.4 for attempt in receipt.attempts)
    assert all(attempt.duration_s == 0.8 for attempt in receipt.attempts)
    assert all(attempt.worker_id == "gw2" for attempt in receipt.attempts)
    assert all(attempt.archive_root_cleaned is True for attempt in receipt.attempts)
    assert all(attempt.process_group_cleaned is True for attempt in receipt.attempts)
    assert {command[-1] for command in calls} == {"0", "2"}


def test_repetitions_do_not_retry_or_mask_a_failed_attempt(tmp_path: Path) -> None:
    calls = 0

    def runner(
        command: Sequence[str], root: Path, _env: dict[str, str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        _write_managed_run(root, ordinal=calls, nodeid=command[4])
        return subprocess.CompletedProcess(command, 23 if calls == 1 else 0, "", "named lifecycle failure")

    receipt = run_repetitions(source_root=tmp_path, attempts_per_mode=1, runner=runner)

    assert not receipt.ok
    assert len(receipt.attempts) == len(WITNESSES) * 2
    assert receipt.attempts[0].status == "failed"
    assert receipt.attempts[0].exit_code == 23
    assert receipt.attempts[0].failure == "named lifecycle failure"


def test_repetitions_retain_timeout_without_retry(tmp_path: Path) -> None:
    calls = 0

    def runner(
        command: Sequence[str], _root: Path, _env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0, "", "")

    receipt = run_repetitions(source_root=tmp_path, attempts_per_mode=1, runner=runner)

    assert not receipt.ok
    assert len(receipt.attempts) == len(WITNESSES) * 2
    assert receipt.attempts[0].status == "timed_out"
    assert "15s" in (receipt.attempts[0].failure or "")


def test_repetitions_fail_a_completed_attempt_over_the_evidence_bound(tmp_path: Path) -> None:
    calls = 0

    def runner(
        command: Sequence[str], root: Path, _env: dict[str, str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        _write_managed_run(root, ordinal=calls, nodeid=command[4])
        run = root / ".cache" / "verify" / "runs" / f"run-{calls}" / "run.json"
        payload = json.loads(run.read_text(encoding="utf-8"))
        payload["steps"][0]["duration_s"] = 10.1
        run.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    receipt = run_repetitions(source_root=tmp_path, attempts_per_mode=1, timeout_s=10, runner=runner)

    assert not receipt.ok
    assert receipt.attempts[0].status == "passed"
    assert receipt.attempts[0].duration_s == 10.1
