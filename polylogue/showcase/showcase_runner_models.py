"""Typed result models for showcase execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polylogue.showcase.exercises import Exercise


@dataclass
class ExerciseResult:
    """Result of running a single exercise."""

    exercise: Exercise
    passed: bool
    exit_code: int
    output: str
    error: str | None = None
    duration_ms: float = 0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class ShowcaseResult:
    """Aggregate result of a full showcase run."""

    results: list[ExerciseResult] = field(default_factory=list)
    total_duration_ms: float = 0
    workspace_dir: Path | None = None
    output_dir: Path | None = None

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed and not result.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed and not result.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.skipped)

    def group_counts(self) -> dict[str, dict[str, int]]:
        """Return pass/fail/skip counts by group."""
        counts: dict[str, dict[str, int]] = {}
        for result in self.results:
            group = result.exercise.group
            if group not in counts:
                counts[group] = {"pass": 0, "fail": 0, "skip": 0}
            if result.skipped:
                counts[group]["skip"] += 1
            elif result.passed:
                counts[group]["pass"] += 1
            else:
                counts[group]["fail"] += 1
        return counts


__all__ = ["ExerciseResult", "ShowcaseResult"]
