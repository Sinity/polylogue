"""Tests for showcase JSON envelope contracts."""

from __future__ import annotations

import json

from polylogue.scenarios import AssertionSpec, polylogue_execution
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_report_payloads import generate_json_report


def _make_result(exercises: list[Exercise] | None = None) -> ShowcaseResult:
    """Build a ShowcaseResult with predictable test data."""
    if exercises is None:
        exercises = [
            Exercise(
                name="test-1",
                group="structural",
                description="Test one",
                execution=polylogue_execution("--help"),
                assertion=AssertionSpec(stdout_contains=("polylogue",)),
            ),
            Exercise(
                name="test-2",
                group="sources",
                description="Test two",
                execution=polylogue_execution("sources"),
            ),
        ]

    result = ShowcaseResult()
    result.results = [
        ExerciseResult(
            exercise=exercises[0],
            passed=True,
            exit_code=0,
            output="Usage: polylogue\n",
            duration_ms=10.0,
        ),
        ExerciseResult(
            exercise=exercises[1],
            passed=False,
            exit_code=1,
            output="error\n",
            error="exit code 1, expected 0",
            duration_ms=20.0,
        ),
    ]
    result.total_duration_ms = 30.0
    return result


class TestJsonEnvelopeValidation:
    """JSON report has correct envelope structure."""

    def test_json_report_is_valid_json(self: object) -> None:
        """generate_json_report produces parseable JSON."""
        result = _make_result()
        report_json = generate_json_report(result)
        data = json.loads(report_json)
        assert isinstance(data, dict)

    def test_json_report_has_required_fields(self: object) -> None:
        """JSON report contains total, passed, failed, skipped, exercises."""
        result = _make_result()
        data = json.loads(generate_json_report(result))

        for field in ("total", "passed", "failed", "skipped", "exercises", "total_duration_ms"):
            assert field in data, f"Missing field: {field}"

    def test_json_report_exercise_entries(self: object) -> None:
        """Each exercise entry has required keys."""
        result = _make_result()
        data = json.loads(generate_json_report(result))

        required_keys = {"name", "group", "description", "passed", "exit_code", "duration_ms"}
        for entry in data["exercises"]:
            assert required_keys <= set(entry.keys()), (
                f"Entry {entry['name']} missing keys: {required_keys - set(entry.keys())}"
            )

    def test_json_report_counts_match(self: object) -> None:
        """Report counts match actual results."""
        result = _make_result()
        data = json.loads(generate_json_report(result))

        assert data["total"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["skipped"] == 0
