"""A queued focused run adjudicates its failures inside the slot it holds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_rerun, pytest_slot
from devtools.pytest_rerun import RERUN_IN_SLOT_ENV, RERUN_IN_SLOT_RESULT, rerun_failed_once


def _failed_report(path: Path, nodeid: str) -> None:
    path.write_text(
        json.dumps({"tests": [{"nodeid": nodeid, "outcome": "failed"}], "summary": {"failed": 1}}),
        encoding="utf-8",
    )


def test_slot_job_reruns_failures_and_records_the_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job writes the rerun record the client adjudicates from.

    Anti-vacuity: drop the ``_rerun_failures_in_slot`` call from the launch
    path's exit-1 branch (or this helper's write) and no record appears, so
    the client queues a second job for its rerun.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_flaky")
    rerun_report = step / "pytest-rerun.json"
    script = (
        "import json, pathlib\n"
        f"pathlib.Path({str(rerun_report)!r}).write_text(json.dumps("
        "{'tests': [{'nodeid': 'tests/test_x.py::test_flaky', 'outcome': 'passed'}]}))\n"
    )
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (["tests/test_x.py::test_flaky"], [sys.executable, "-c", script], rerun_report),
    )
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
        "PATH": "/usr/bin:/bin",
    }

    pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log_path=tmp_path / "slot.log")

    record = json.loads((step / RERUN_IN_SLOT_RESULT).read_text(encoding="utf-8"))
    assert record == {"attempted": ["tests/test_x.py::test_flaky"], "rerun_exit": 0}


def test_client_adjudicates_from_the_slot_record_without_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded in-slot rerun is folded in; the pytest slot is not requested again.

    Anti-vacuity: remove the record branch from ``rerun_failed_once`` and the
    fake slot below is reached, failing the test.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_flaky")
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_flaky", "outcome": "passed"}]}), encoding="utf-8"
    )
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["tests/test_x.py::test_flaky"], "rerun_exit": 0}), encoding="utf-8"
    )

    def must_not_queue(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the client queued a second job for a rerun the slot already ran")

    monkeypatch.setattr(pytest_rerun, "run_pytest", must_not_queue)

    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)

    assert verdict is not None
    assert verdict["still_failed"] == []
    assert verdict["flaky"] == ["tests/test_x.py::test_flaky"]
    assert json.loads(report.read_text(encoding="utf-8"))["exitcode"] == 0


def test_a_rerun_that_fails_again_stays_red(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_real")
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_real", "outcome": "failed"}]}), encoding="utf-8"
    )
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["tests/test_x.py::test_real"], "rerun_exit": 1}), encoding="utf-8"
    )

    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)

    assert verdict is not None
    assert verdict["still_failed"] == ["tests/test_x.py::test_real"]
