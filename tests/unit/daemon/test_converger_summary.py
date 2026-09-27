"""DaemonConverger.summary() buckets each tracked file exactly once."""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.convergence import DaemonConverger, FileState, StageState


def test_summary_counts_a_converged_after_failure_file_once() -> None:
    """Anti-vacuity (polylogue-fkn5): counting ``error_count > 0`` as failed puts
    the recovered file in both buckets and drives ``in_progress`` negative."""
    converger = DaemonConverger(stages=())
    recovered = FileState(Path("/synthetic/recovered.jsonl"), stages={"parse": StageState.DONE}, error_count=2)
    failing = FileState(Path("/synthetic/failing.jsonl"), stages={"parse": StageState.FAILED})
    pending = FileState(Path("/synthetic/pending.jsonl"), stages={"parse": StageState.PENDING})
    converger._file_states = {state.path: state for state in (recovered, failing, pending)}

    summary = converger.summary()

    assert summary == {"total": 3, "converged": 1, "failed": 1, "in_progress": 1}
    assert summary["converged"] + summary["failed"] + summary["in_progress"] == summary["total"]
