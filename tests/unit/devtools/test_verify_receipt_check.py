"""The hosted verify job accepts a run only from its receipt, never from a zero exit.

Anti-vacuity: accepting every successful receipt makes
``test_a_successful_run_without_a_pytest_step_and_no_reason_is_refused`` red;
that receipt is what a skipped, refused or never-started pytest leaves behind.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_receipt_check
from devtools.verify_runs import VERIFY_RUNS_DIR


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


def test_main_reads_the_newest_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / VERIFY_RUNS_DIR
    older = runs / "run-older"
    newer = runs / "run-newer"
    for directory, payload in (
        (older, _receipt(run_id="run-older", steps=[{"name": "pytest (all)", "status": "success", "exit": 0}])),
        (newer, _receipt(run_id="run-newer")),
    ):
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    stamp = time.time() - 60
    os.utime(older / "run.json", (stamp, stamp))

    assert verify_receipt_check.main([str(tmp_path)]) == 1
    assert "run-newer" in capsys.readouterr().err

    (newer / "run.json").write_text(
        json.dumps(_receipt(run_id="run-newer", steps=[{"name": "pytest (affected)", "status": "success", "exit": 0}])),
        encoding="utf-8",
    )

    assert verify_receipt_check.main([str(tmp_path)]) == 0


def test_main_refuses_without_any_receipt(tmp_path: Path) -> None:
    assert verify_receipt_check.main([str(tmp_path)]) == 1
