"""Tests for the verification-lab scenario command surface."""

from __future__ import annotations

import importlib
import json
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
    from polylogue.showcase.showcase_runner_models import ShowcaseResult

    with (
        patch("devtools.lab_scenario.ShowcaseRunner") as runner_class,
        patch("devtools.lab_scenario.check_invariants", return_value=[]),
    ):
        runner_class.return_value.run.return_value = ShowcaseResult()
        assert main(["run", "archive-smoke", "--tier", "0"]) == 0

    out = capsys.readouterr().out
    assert "Scenario stages:" in out
    assert "showcase: ok" in out
    assert "invariants: ok" in out
    assert "Failed stages: none" in out


def test_main_json_reports_direct_scenario_payload(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main
    from polylogue.showcase.showcase_runner_models import ShowcaseResult

    with (
        patch("devtools.lab_scenario.ShowcaseRunner") as runner_class,
        patch("devtools.lab_scenario.check_invariants", return_value=[]),
    ):
        runner_class.return_value.run.return_value = ShowcaseResult()
        assert main(["run", "archive-smoke", "--tier", "0", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "scenario": "archive-smoke",
        "stages": {
            "showcase": "ok",
            "invariants": "ok",
        },
        "failed_stages": [],
        "ok": True,
        "report_dir": None,
    }
