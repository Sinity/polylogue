"""The hosted verify job accepts a run only from its receipt, never from a zero exit.

Anti-vacuity: accepting every successful receipt makes
``test_a_successful_run_without_a_pytest_step_and_no_reason_is_refused`` red;
that receipt is what a skipped, refused or never-started pytest leaves behind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_receipt_check
from devtools.verify_runs import CURRENT_RUN_PATH, VERIFY_RUNS_DIR


def _receipt(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "status": "success",
        "exit_code": 0,
        "steps": [{"name": "gate lint", "status": "success", "exit": 0}],
        "testmon_selection": {"selection_mode": "affected", "selection_reason": None},
    }
    payload.update(overrides)
    return payload


def test_a_successful_pytest_step_is_evidence() -> None:
    steps = [
        {"name": "gate lint", "status": "success", "exit": 0},
        {"name": "pytest (affected)", "status": "success", "exit": 0},
    ]

    assert verify_receipt_check.refusal(_receipt(steps=steps)) is None


def test_a_successful_run_without_a_pytest_step_and_no_reason_is_refused() -> None:
    reason = verify_receipt_check.refusal(_receipt())

    assert reason is not None
    assert "ran no pytest step" in reason


def test_a_recorded_none_selection_is_accepted() -> None:
    selection = {"selection_mode": "none", "selection_reason": "every changed path is documentation"}

    assert verify_receipt_check.refusal(_receipt(testmon_selection=selection)) is None
    assert verify_receipt_check.refusal(_receipt(testmon_selection={"selection_mode": "none"})) is not None


def test_a_failed_run_or_pytest_step_is_refused() -> None:
    assert "ended 'failed'" in str(verify_receipt_check.refusal(_receipt(status="failed", exit_code=125)))
    steps = [{"name": "pytest (affected)", "status": "failed", "exit": 1}]
    assert "did not succeed" in str(verify_receipt_check.refusal(_receipt(steps=steps)))


def test_an_affected_admission_refusal_names_count_reason_and_boundary() -> None:
    reason = verify_receipt_check.refusal(
        _receipt(
            status="failed",
            exit_code=2,
            diagnosis="affected_admission_refused",
            testmon_selection={
                "selection_mode": "affected",
                "admission": {
                    "status": "refused",
                    "selected_count": 1001,
                    "reason": "exceeds cap",
                    "next_boundary": "devtools verify --all at the explicit master/corpus boundary",
                },
            },
        )
    )

    assert reason is not None
    assert "1001" in reason
    assert "exceeds cap" in reason
    assert "verify --all" in reason


def _write_run(root: Path, payload: dict[str, Any]) -> Path:
    directory = root / VERIFY_RUNS_DIR / str(payload["run_id"])
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "run.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt


def _point_current_at(root: Path, run_id: str) -> None:
    current = root / CURRENT_RUN_PATH
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps({"run_id": run_id, "status": "success"}), encoding="utf-8")


def test_main_reads_the_receipt_of_the_named_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The run is bound by id: a newer receipt from another run is not this run's evidence."""
    pytest_step = [{"name": "pytest (affected)", "status": "success", "exit": 0}]
    _write_run(tmp_path, _receipt(run_id="run-mine", steps=pytest_step))
    newer = _write_run(tmp_path, _receipt(run_id="run-other"))
    stamp = newer.stat().st_mtime + 60
    os.utime(newer, (stamp, stamp))

    assert verify_receipt_check.main([str(tmp_path), "--run-id", "run-mine"]) == 0
    assert "run-mine" in capsys.readouterr().err

    assert verify_receipt_check.main([str(tmp_path), "--run-id", "run-other"]) == 1
    assert "ran no pytest step" in capsys.readouterr().err


def test_main_defaults_to_the_run_current_run_json_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_run(tmp_path, _receipt(run_id="run-1", steps=[{"name": "pytest (all)", "status": "success", "exit": 0}]))
    _write_run(tmp_path, _receipt(run_id="run-2"))
    _point_current_at(tmp_path, "run-1")

    assert verify_receipt_check.main([str(tmp_path)]) == 0
    assert "run-1" in capsys.readouterr().err

    _point_current_at(tmp_path, "run-2")

    assert verify_receipt_check.main([str(tmp_path)]) == 1


def test_a_receipt_recording_another_run_id_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = tmp_path / VERIFY_RUNS_DIR / "run-1"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps(_receipt(run_id="run-9", steps=[{"name": "pytest (all)", "status": "success", "exit": 0}])),
        encoding="utf-8",
    )

    assert verify_receipt_check.main([str(tmp_path), "--run-id", "run-1"]) == 1
    assert "records run 'run-9'" in capsys.readouterr().err


def test_main_refuses_without_a_current_run_or_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_receipt_check.main([str(tmp_path)]) == 1
    assert "no current verify run" in capsys.readouterr().err

    assert verify_receipt_check.main([str(tmp_path), "--run-id", "run-missing"]) == 1
    assert "unreadable receipt" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# polylogue-p2mbi: a receipt is evidence for one tree, or for none.
# ---------------------------------------------------------------------------

_CANDIDATE = "a" * 40
_WORKING_BRANCH = "b" * 40
_GREEN = [{"name": "pytest (all)", "status": "success", "exit": 0}]


def _tree_receipt(**overrides: Any) -> dict[str, Any]:
    payload = _receipt(steps=list(_GREEN), git_head=_CANDIDATE, git_dirty=False, final_git_head=_CANDIDATE)
    payload.update(overrides)
    return payload


def test_a_receipt_for_the_candidate_tree_is_accepted() -> None:
    """The control: the refusals below are not refusing everything."""
    assert verify_receipt_check.candidate_tree_refusal(_tree_receipt(), _CANDIDATE) is None


def test_a_receipt_from_another_tree_is_refused_and_names_both_shas() -> None:
    """The scheduled-run defect: green, and about the wrong tree.

    On 2026-09-20 the default checkout sat on branch
    test/dispatcher-intake-measurement while origin/master was 53 commits
    ahead, so every 03:00 corpus run qualified a stale side branch.

    Anti-vacuity: restore ``checkout = "default"`` on ``verify_all`` and a
    scheduled run's receipt carries the working branch's SHA -- exactly the
    payload below -- which this refusal rejects. Drop the comparison and it
    returns None for a receipt that tested a different commit.
    """
    reason = verify_receipt_check.candidate_tree_refusal(
        _tree_receipt(git_head=_WORKING_BRANCH, final_git_head=_WORKING_BRANCH), _CANDIDATE
    )
    assert reason is not None
    assert _WORKING_BRANCH in reason
    assert _CANDIDATE in reason


def test_a_receipt_with_no_recorded_tree_is_evidence_for_nothing() -> None:
    """Criterion 2 and 3 are linked: removing the SHA field must not pass.

    Anti-vacuity: treat a missing ``git_head`` as "nothing to compare" and the
    mismatch refusal becomes opt-out by omission.
    """
    payload = _tree_receipt()
    del payload["git_head"]
    reason = verify_receipt_check.candidate_tree_refusal(payload, _CANDIDATE)
    assert reason is not None
    assert "no tested tree SHA" in reason


def test_a_run_whose_tree_moved_under_it_covers_no_single_tree() -> None:
    """A merge landing mid-run makes the receipt describe two trees.

    Anti-vacuity: compare only ``git_head`` and a run that started on the
    candidate and finished elsewhere is accepted as candidate evidence.
    """
    reason = verify_receipt_check.candidate_tree_refusal(_tree_receipt(final_git_head=_WORKING_BRANCH), _CANDIDATE)
    assert reason is not None
    assert "moved under the run" in reason
    assert _WORKING_BRANCH in reason


def test_a_dirty_tested_tree_is_not_the_named_sha() -> None:
    """Uncommitted or untracked content means what ran was not that commit.

    Anti-vacuity: ignore the dirty flags and a receipt naming the candidate
    SHA is accepted no matter what was actually in the working tree.
    """
    assert verify_receipt_check.candidate_tree_refusal(_tree_receipt(git_dirty=True), _CANDIDATE) is not None
    assert verify_receipt_check.candidate_tree_refusal(_tree_receipt(final_git_dirty=True), _CANDIDATE) is not None


def test_main_refuses_a_wrong_tree_before_it_reports_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is the exit code, not a note beside a pass.

    Anti-vacuity: report the mismatch without failing and this returns 0.
    """
    _write_run(tmp_path, _tree_receipt(run_id="run-1", git_head=_WORKING_BRANCH, final_git_head=_WORKING_BRANCH))
    _point_current_at(tmp_path, "run-1")

    assert verify_receipt_check.main([str(tmp_path), "--candidate", _CANDIDATE]) == 1
    captured = capsys.readouterr().err
    assert _CANDIDATE in captured
    assert _WORKING_BRANCH in captured

    assert verify_receipt_check.main([str(tmp_path), "--candidate", _WORKING_BRANCH]) == 0
    assert _WORKING_BRANCH in capsys.readouterr().err


def test_the_wrong_tree_is_refused_even_when_the_run_is_flawless(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ordering matters: a green run on another tree must not read as evidence.

    Anti-vacuity: run the existing test-evidence refusal first and a perfect
    run on the wrong tree passes both checks.
    """
    _write_run(tmp_path, _tree_receipt(run_id="run-1", git_head=_WORKING_BRANCH, final_git_head=_WORKING_BRANCH))
    _point_current_at(tmp_path, "run-1")

    assert verify_receipt_check.refusal(_tree_receipt(git_head=_WORKING_BRANCH)) is None
    assert verify_receipt_check.main([str(tmp_path), "--candidate", _CANDIDATE]) == 1
    assert "not the candidate" in capsys.readouterr().err
