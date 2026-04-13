"""Showcase exercise helpers over the single authored+runnable exercise model."""

from __future__ import annotations

from polylogue.showcase.exercise_models import Exercise


def compile_exercise_scenarios(scenarios: tuple[Exercise, ...]) -> tuple[Exercise, ...]:
    """Lower authored exercise sources into runnable showcase exercises."""
    return scenarios


__all__ = ["Exercise", "compile_exercise_scenarios"]
