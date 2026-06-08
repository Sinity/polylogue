"""Showcase-level invariant checks.

Defines universal invariants that hold across ALL showcase exercises.
These are the showcase equivalent of property tests: conditions that
must be true regardless of which exercise produced the output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from polylogue.core.outcomes import OutcomeStatus
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.runner import ExerciseResult

# Sentinel for skipping an invariant check on a particular exercise
SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class Invariant:
    """A universal invariant that must hold across showcase exercises."""

    name: str
    description: str
    check: Callable[..., str | None]

    def applies_to(self, exercise: Exercise) -> bool:
        return True


@dataclass(slots=True)
class InvariantResult:
    invariant_name: str
    exercise_name: str
    status: OutcomeStatus
    error: str | None = None


def _check_json_valid(result: ExerciseResult) -> str | None:
    args_str = result.exercise.args_text
    if "-f json" not in args_str and "--format json" not in args_str:
        return SKIP
    if result.exercise.output_ext not in (".json", ".jsonl"):
        return SKIP
    if not result.output.strip():
        return SKIP
    if result.exercise.output_ext == ".jsonl":
        for i, line in enumerate(result.output.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return f"Invalid JSON on line {i}: {e}"
        return None
    try:
        json.loads(result.output)
        return None
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


def _check_exit_code(result: ExerciseResult) -> str | None:
    expected = result.exercise.assertion.exit_code
    if expected is None:
        return SKIP
    if result.exit_code != expected:
        return f"exit code {result.exit_code}, expected {expected}"
    return None


def _check_nonempty_output(result: ExerciseResult) -> str | None:
    if result.exercise.writes:
        return SKIP
    args_str = result.exercise.args_text
    if "--count" in args_str:
        return SKIP
    if "--help" in args_str or "--version" in args_str:
        return SKIP
    if not result.output.strip():
        return "Empty output for read command"
    return None


SHOWCASE_INVARIANTS: list[Invariant] = [
    Invariant("json_valid", "All -f json output parses as valid JSON", _check_json_valid),
    Invariant("exit_code", "Exit code matches validation spec", _check_exit_code),
    Invariant("nonempty_output", "Non-count read commands produce non-empty output", _check_nonempty_output),
]


def check_invariants(results: list[ExerciseResult]) -> list[InvariantResult]:
    invariant_results: list[InvariantResult] = []
    for result in results:
        if result.skipped:
            continue
        for invariant in SHOWCASE_INVARIANTS:
            if not invariant.applies_to(result.exercise):
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.SKIP))
                continue
            try:
                error = invariant.check(result)
            except Exception as e:
                error = f"invariant check crashed: {e}"
            if error == SKIP:
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.SKIP))
            elif error is None:
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.OK))
            else:
                invariant_results.append(
                    InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.ERROR, error=error)
                )
    return invariant_results


def format_invariant_summary(results: list[InvariantResult]) -> str:
    passed = sum(1 for r in results if r.status is OutcomeStatus.OK)
    failed = sum(1 for r in results if r.status is OutcomeStatus.ERROR)
    skipped = sum(1 for r in results if r.status is OutcomeStatus.SKIP)
    lines = [f"Invariant Checks: {passed} pass, {failed} fail, {skipped} skip"]
    failures = [r for r in results if r.status is OutcomeStatus.ERROR]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for f in failures:
            lines.append(f"  {f.invariant_name} @ {f.exercise_name}: {f.error}")
    return "\n".join(lines)


__all__ = [
    "Invariant",
    "InvariantResult",
    "SHOWCASE_INVARIANTS",
    "check_invariants",
    "format_invariant_summary",
]
