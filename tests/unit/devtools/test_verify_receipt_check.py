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
