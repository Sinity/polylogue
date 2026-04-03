"""Typed showcase exercise facade over the serialized catalog."""

from __future__ import annotations

from polylogue.showcase.catalog_loader import load_exercise_catalog
from polylogue.showcase.exercise_models import Exercise, Validation
from polylogue.showcase.generators import generate_root_help_exercises, root_help_exercise_names

_CATALOG = load_exercise_catalog()
_GENERATED_ROOT_HELP_EXERCISES = tuple(generate_root_help_exercises())
_GENERATED_ROOT_HELP_NAMES = root_help_exercise_names()

EXERCISES: tuple[Exercise, ...] = (
    tuple(exercise for exercise in _CATALOG.exercises if exercise.name not in _GENERATED_ROOT_HELP_NAMES)
    + _GENERATED_ROOT_HELP_EXERCISES
)

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
    "EXERCISE_INDEX",
    "GROUPS",
    "Exercise",
    "Validation",
    "exercises_by_group",
    "topological_order",
    "vhs_exercises",
]
