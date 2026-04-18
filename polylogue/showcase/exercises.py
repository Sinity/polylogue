"""Typed showcase exercise catalog over the single exercise source model."""

from __future__ import annotations

from polylogue.showcase.catalog_loader import load_exercise_catalog
from polylogue.showcase.exercise_models import AssertionSpec, Exercise
from polylogue.showcase.generators import (
    generate_command_help_scenarios,
    generate_json_contract_scenarios,
    generate_qa_extra_scenarios,
)

_CATALOG = load_exercise_catalog()


def _generated_exercise_scenarios() -> tuple[Exercise, ...]:
    command_help_scenarios = generate_command_help_scenarios()
    json_contract_scenarios = generate_json_contract_scenarios()
    return command_help_scenarios + json_contract_scenarios


_GENERATED_EXERCISE_SCENARIOS = _generated_exercise_scenarios()
_GENERATED_EXERCISE_NAMES = {scenario.name for scenario in _GENERATED_EXERCISE_SCENARIOS}

EXERCISE_SCENARIOS: tuple[Exercise, ...] = (
    tuple(exercise for exercise in _CATALOG.exercises if exercise.name not in _GENERATED_EXERCISE_NAMES)
    + _GENERATED_EXERCISE_SCENARIOS
)
EXERCISES: tuple[Exercise, ...] = EXERCISE_SCENARIOS
QA_EXTRA_SCENARIOS: tuple[Exercise, ...] = generate_qa_extra_scenarios()
QA_EXTRA_EXERCISES: tuple[Exercise, ...] = QA_EXTRA_SCENARIOS

EXERCISE_INDEX: dict[str, Exercise] = {e.name: e for e in EXERCISES}

GROUPS: tuple[str, ...] = _CATALOG.groups


def exercises_by_group() -> dict[str, list[Exercise]]:
    """Return exercises grouped by their group name, in catalog order."""
    result: dict[str, list[Exercise]] = {g: [] for g in GROUPS}
    for ex in EXERCISES:
        result[ex.group].append(ex)
    return result


def vhs_exercises() -> list[Exercise]:
    """Return only exercises where vhs_capture is True."""
    return [e for e in EXERCISES if e.vhs_capture]


def topological_order(exercises: list[Exercise]) -> list[Exercise]:
    """Sort exercises respecting depends_on ordering."""
    index = {e.name: e for e in exercises}
    visited: set[str] = set()
    result: list[Exercise] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        ex = index.get(name)
        if ex is None:
            return
        if ex.depends_on and ex.depends_on in index:
            visit(ex.depends_on)
        visited.add(name)
        result.append(ex)

    for ex in exercises:
        visit(ex.name)
    return result


__all__ = [
    "EXERCISES",
    "EXERCISE_SCENARIOS",
    "EXERCISE_INDEX",
    "GROUPS",
    "QA_EXTRA_EXERCISES",
    "QA_EXTRA_SCENARIOS",
    "Exercise",
    "AssertionSpec",
    "exercises_by_group",
    "topological_order",
    "vhs_exercises",
]
