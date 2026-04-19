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
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ShowcaseGroupCounts:
    """Pass/fail/skip totals for one exercise group."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "pass": self.passed,
            "fail": self.failed,
            "skip": self.skipped,
        }


@dataclass(frozen=True, slots=True)
class ShowcaseSummary:
    """Aggregate showcase counts and timing."""

    total: int
    passed: int
    failed: int
    skipped: int
    total_duration_ms: float

    def to_payload(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_duration_ms": round(self.total_duration_ms, 1),
        }


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

    def summary(self) -> ShowcaseSummary:
        """Return the typed summary snapshot for this showcase run."""
        return ShowcaseSummary(
            total=len(self.results),
            passed=self.passed,
            failed=self.failed,
            skipped=self.skipped,
            total_duration_ms=round(self.total_duration_ms, 1),
        )

    def group_counts(self) -> dict[str, ShowcaseGroupCounts]:
        """Return pass/fail/skip counts by group."""
        counts: dict[str, ShowcaseGroupCounts] = {}
        for result in self.results:
            group = result.exercise.group
            current = counts.get(group, ShowcaseGroupCounts())
            if result.skipped:
                counts[group] = ShowcaseGroupCounts(
                    passed=current.passed,
                    failed=current.failed,
                    skipped=current.skipped + 1,
                )
            elif result.passed:
                counts[group] = ShowcaseGroupCounts(
                    passed=current.passed + 1,
                    failed=current.failed,
                    skipped=current.skipped,
                )
            else:
                counts[group] = ShowcaseGroupCounts(
                    passed=current.passed,
                    failed=current.failed + 1,
                    skipped=current.skipped,
                )
        return counts


__all__ = [
    "ExerciseResult",
    "ShowcaseGroupCounts",
    "ShowcaseResult",
    "ShowcaseSummary",
]
