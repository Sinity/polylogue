"""Tests for scenario-first showcase generators."""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.cli.click_app import cli as root_cli
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.generators import (
    generate_all_exercises,
    generate_all_scenarios,
    generate_filter_exercises,
    generate_filter_scenarios,
    generate_format_exercises,
    generate_format_scenarios,
    generate_provider_feature_exercises,
    generate_provider_feature_scenarios,
    generate_qa_extra_scenarios,
    generate_schema_exercises,
    generate_schema_scenarios,
)


def _names(items: Sequence[Exercise]) -> list[str]:
    return [item.name for item in items]


def test_generate_filter_exercises_compile_from_scenarios() -> None:
    scenarios = generate_filter_scenarios(root_cli)
    exercises = generate_filter_exercises(root_cli)

    assert _names(exercises) == _names(scenarios)


def test_generate_format_exercises_compile_from_scenarios() -> None:
    scenarios = generate_format_scenarios()
    exercises = generate_format_exercises()

    assert _names(exercises) == _names(scenarios)


def test_generate_schema_exercises_compile_from_scenarios() -> None:
    scenarios = generate_schema_scenarios()
    exercises = generate_schema_exercises()

    assert _names(exercises) == _names(scenarios)


def test_generate_provider_feature_exercises_compile_from_scenarios() -> None:
    scenarios = generate_provider_feature_scenarios()
    exercises = generate_provider_feature_exercises()

    assert _names(exercises) == _names(scenarios)


def test_generate_qa_extra_scenarios_matches_schema_and_format_families() -> None:
    scenarios = generate_qa_extra_scenarios()

    assert _names(scenarios) == _names(generate_schema_scenarios()) + _names(generate_format_scenarios())


def test_generate_all_exercises_compile_from_scenarios() -> None:
    scenarios = generate_all_scenarios(root_cli)
    exercises = generate_all_exercises(root_cli)

    assert _names(exercises) == _names(scenarios)
