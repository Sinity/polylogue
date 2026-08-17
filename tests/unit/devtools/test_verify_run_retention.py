"""Contracts for verification-receipt retention.

Six days of ordinary work accumulated 623 run directories and 3.0 GB with no
policy at all. The property under test is that retention is bounded WITHOUT
discarding evidence a merge gate still points at -- over-retaining is the safe
error, a gate whose evidence vanished is not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from devtools.verify_runs import prune_verify_runs


def _make_run(root: Path, run_id: str, mtime: float) -> Path:
    run_dir = root / ".cache" / "verify" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.utime(run_dir, (mtime, mtime))
    return run_dir


def test_keeps_the_most_recent_and_drops_the_rest(tmp_path: Path) -> None:
    for index in range(10):
        _make_run(tmp_path, f"20260101T00000{index}Z-quick-{index}", 1000.0 + index)

    removed = prune_verify_runs(tmp_path, keep=3)

    surviving = sorted(path.name for path in (tmp_path / ".cache/verify/runs").iterdir())
    assert len(surviving) == 3
    assert len(removed) == 7
    assert "20260101T000009Z-quick-9" in surviving, "the newest run must survive"
    assert "20260101T000000Z-quick-0" not in surviving


def test_never_drops_a_run_a_merge_gate_receipt_names(tmp_path: Path) -> None:
    for index in range(6):
        _make_run(tmp_path, f"20260101T00000{index}Z-testmon-{index}", 1000.0 + index)
    gate_dir = tmp_path / ".cache" / "verify" / "merge-gate"
    gate_dir.mkdir(parents=True)
    # Receipts capture command output verbatim, so the reference is textual.
    (gate_dir / "pr-1234.json").write_text(
        json.dumps({"output": "... .cache/verify/runs/20260101T000000Z-testmon-0/steps/01/containment.json ..."}),
        encoding="utf-8",
    )

    prune_verify_runs(tmp_path, keep=2)

    surviving = {path.name for path in (tmp_path / ".cache/verify/runs").iterdir()}
    assert "20260101T000000Z-testmon-0" in surviving, "a run a merge gate points at must never be pruned"
    assert len(surviving) == 3, "the protected run is retained in addition to the keep window"


def test_below_the_threshold_nothing_is_touched(tmp_path: Path) -> None:
    for index in range(3):
        _make_run(tmp_path, f"20260101T00000{index}Z-quick-{index}", 1000.0 + index)

    assert prune_verify_runs(tmp_path, keep=10) == ()
    assert len(list((tmp_path / ".cache/verify/runs").iterdir())) == 3


def test_missing_runs_directory_is_not_an_error(tmp_path: Path) -> None:
    assert prune_verify_runs(tmp_path, keep=5) == ()
