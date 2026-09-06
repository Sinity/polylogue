"""A result must belong to the run that asked for it, and must exist.

Three defects with one shape -- a reader cannot tell whose result it has, or
whether there is one at all:

- ``devtools test`` named ``current-pytest-summary.json`` in its own output and
  never wrote it, so a run that never started and a run that finished and
  failed were indistinguishable;
- a slot client trusted a pueue task id, which is a queue position rather than
  an identity: a reordered queue reassigns ids, and a client then read a
  foreign job's terminal state as its own release;
- the exit status does not survive a pipeline, so `verify | tail` reported
  tail's status and three separate readers called a failing run green.

Anti-vacuity, in order:
- stop writing the summary and ``test_a_summary_exists_even_when_pytest_never_ran``
  goes red;
- drop the ``launch_path`` check from ``_task_result`` and
  ``test_a_reassigned_task_id_is_refused`` goes red -- it returns the other
  job's exit status as this run's;
- delete the verdict line and ``test_the_verdict_survives_a_pipeline`` goes red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.pytest_slot import PytestSlotUnavailableError, _task_result
from devtools.verify_runs import CURRENT_PYTEST_SUMMARY, write_current_pytest_summary


def _status(task_id: str, *, command: str, result: str = "Success") -> str:
    return json.dumps(
        {
            "tasks": {
                task_id: {
                    "command": command,
                    "group": "pytest",
                    "status": {"Done": {"result": result if result != "Success" else "Success"}},
                }
            }
        }
    )


def test_a_task_that_is_still_ours_is_read(tmp_path: Path) -> None:
    launch = tmp_path / "pytest-slot-123.json"
    assert _task_result(_status("7", command=f"agentctl-run {launch}"), "7", launch_path=launch) == 0


def test_a_reassigned_task_id_is_refused(tmp_path: Path) -> None:
    """A reordered queue can hand this id to another worktree's job."""
    launch = tmp_path / "pytest-slot-123.json"
    foreign = "agentctl-run /realm/worktrees/someone-else/.cache/verify/pytest-slot-999.json"
    with pytest.raises(PytestSlotUnavailableError) as caught:
        _task_result(_status("7", command=foreign), "7", launch_path=launch)
    message = str(caught.value)
    assert "no longer this run's task" in message
    assert "must not be read from another job" in message


def test_identity_is_not_checked_when_no_launch_path_is_known(tmp_path: Path) -> None:
    del tmp_path
    assert _task_result(_status("7", command="anything"), "7") == 0


def test_a_summary_exists_even_when_pytest_never_ran(tmp_path: Path) -> None:
    """The distinction a reader needs: never started, versus finished and failed."""
    written = write_current_pytest_summary(
        tmp_path,
        run_id="20260906T000000Z-focused-test-1-abcd",
        tier="focused-test",
        exit_code=125,
        diagnosis="pytest_slot_unavailable",
    )
    assert written == tmp_path / CURRENT_PYTEST_SUMMARY
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 125
    assert payload["diagnosis"] == "pytest_slot_unavailable"
    # The run is named, so a caller can tell this is the result it asked for
    # and not an older one from the same worktree.
    assert payload["run_id"] == "20260906T000000Z-focused-test-1-abcd"
    assert payload["kind"] == "polylogue.pytest-summary"


def test_a_summary_carries_the_pytest_session_when_there_was_one(tmp_path: Path) -> None:
    plugin = tmp_path / "summary.json"
    plugin.write_text(json.dumps({"exitstatus": 1, "selected_count": 4}), encoding="utf-8")
    written = write_current_pytest_summary(
        tmp_path,
        run_id="run-2",
        tier="focused-test",
        exit_code=1,
        diagnosis="pytest_failed",
        statistics={"terminal_count": 4},
        plugin_summary=plugin,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["pytest_session"]["selected_count"] == 4
    assert payload["statistics"]["terminal_count"] == 4


def test_a_summary_replaces_the_previous_run(tmp_path: Path) -> None:
    """A stale read is detectable because the file names its own run."""
    write_current_pytest_summary(tmp_path, run_id="old", tier="focused-test", exit_code=0, diagnosis=None)
    write_current_pytest_summary(tmp_path, run_id="new", tier="focused-test", exit_code=1, diagnosis="pytest_failed")
    payload = json.loads((tmp_path / CURRENT_PYTEST_SUMMARY).read_text(encoding="utf-8"))
    assert payload["run_id"] == "new"


def test_the_verdict_survives_a_pipeline() -> None:
    """The exit status does not survive `| tail`; the last output line does."""
    from devtools.verify import _write_verdict_line

    class _Stream:
        def __init__(self) -> None:
            self.text = ""

        def write(self, value: str) -> int:
            self.text += value
            return len(value)

    failed = _Stream()
    _write_verdict_line({"exit_code": 1, "diagnosis": "gate_failed", "run_id": "r1"}, stream=failed)
    assert "FAILED" in failed.text and "gate_failed" in failed.text and "exit 1" in failed.text

    passed = _Stream()
    _write_verdict_line({"exit_code": 0, "run_id": "r2"}, stream=passed)
    assert "PASSED" in passed.text and "exit 0" in passed.text
