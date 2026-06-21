"""Tests for the verification-lab scenario command surface."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_module_imports() -> None:
    assert importlib.import_module("devtools.lab_scenario") is not None


def test_get_tier_0_exercises_returns_structural_data_free_cases() -> None:
    from devtools.lab_scenario import get_tier_0_exercises

    exercises = get_tier_0_exercises()

    assert exercises
    assert all(ex.group == "structural" for ex in exercises)
    assert all(not ex.needs_data for ex in exercises)


def test_list_scenarios_reports_live_paths_without_baseline_counts(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import list_scenarios

    assert list_scenarios(as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    archive = next(entry for entry in payload["scenarios"] if entry["name"] == "archive-smoke")
    visual = next(entry for entry in payload["scenarios"] if entry["name"] == "reader-visual-smoke")
    assert archive == {
        "name": "archive-smoke",
        "kind": "showcase",
        "tier_0_exercise_count": archive["tier_0_exercise_count"],
    }
    assert archive["tier_0_exercise_count"] > 0
    assert visual["command"]


def test_main_prints_direct_stage_summary(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main
    from polylogue.core.outcomes import OutcomeStatus

    result = SimpleNamespace(
        all_passed=True,
        report_dir=None,
        stage_statuses=lambda: {
            "audit": OutcomeStatus.OK,
            "artifact_coverage": OutcomeStatus.SKIP,
            "showcase": OutcomeStatus.OK,
            "invariants": OutcomeStatus.OK,
        },
        failed_stages=lambda: (),
    )

    with patch("devtools.lab_scenario.run_qa_session", return_value=result):
        assert main(["run", "archive-smoke", "--tier", "0"]) == 0

    out = capsys.readouterr().out
    assert "Scenario stages:" in out
    assert "artifact_coverage: skip" in out
    assert "Failed stages: none" in out
