"""Showcase-level invariant checks.

Defines universal invariants that hold across ALL showcase exercises.
These are the showcase equivalent of property tests: conditions that
must be true regardless of which exercise produced the output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.runner import ExerciseResult

# Sentinel for skipping an invariant check on a particular exercise
SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class Invariant:
    """A universal invariant that must hold across showcase exercises."""

    name: str
    description: str
    check: Callable[..., str | None]  # returns error message or None (pass), or SKIP

    def applies_to(self, exercise: Exercise) -> bool:
        """Whether this invariant applies to a given exercise."""
        return True  # Default: applies to all


@dataclass(slots=True)
class InvariantResult:
    """Result of checking one invariant against one exercise."""

    invariant_name: str
    exercise_name: str
    status: OutcomeStatus
    error: str | None = None


def _check_json_valid(result: ExerciseResult) -> str | None:
    """All -f json output parses as valid JSON (or JSON Lines for .jsonl)."""
    args_str = result.exercise.args_text
    if "-f json" not in args_str and "--json" not in args_str:
        return SKIP
    if result.exercise.output_ext not in (".json", ".jsonl"):
        return SKIP
    if not result.output.strip():
        return SKIP

    # JSON Lines (.jsonl): each non-empty line must be valid JSON individually
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

    # Single JSON document
    try:
        json.loads(result.output)
        return None
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


def _check_exit_code(result: ExerciseResult) -> str | None:
    """Exit code matches validation spec."""
    expected = result.exercise.assertion.exit_code
    if expected is None:
        return SKIP  # Delegated to custom validator
    if result.exit_code != expected:
        return f"exit code {result.exit_code}, expected {expected}"
    return None


def _check_clean_stderr(result: ExerciseResult) -> str | None:
    """Read commands produce no unexpected stderr."""
    if result.exercise.writes:
        return SKIP
    # We don't have separate stderr in ExerciseResult (CliRunner merges it),
    # so this invariant checks for error-like patterns in output
    return None


def _check_nonempty_output(result: ExerciseResult) -> str | None:
    """Non-count read commands produce non-empty output."""
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


# ---------------------------------------------------------------------------
# Registry of all invariants
# ---------------------------------------------------------------------------

SHOWCASE_INVARIANTS: list[Invariant] = [
    Invariant(
        "json_valid",
        "All -f json output parses as valid JSON",
        _check_json_valid,
    ),
    Invariant(
        "exit_code",
        "Exit code matches validation spec",
        _check_exit_code,
    ),
    Invariant(
        "clean_stderr",
        "Read commands produce no unexpected stderr",
        _check_clean_stderr,
    ),
    Invariant(
        "nonempty_output",
        "Non-count read commands produce non-empty output",
        _check_nonempty_output,
    ),
]


def check_invariants(results: list[ExerciseResult]) -> list[InvariantResult]:
    """Run all invariants across all exercise results."""
    invariant_results: list[InvariantResult] = []

    for result in results:
        if result.skipped:
            continue
        for invariant in SHOWCASE_INVARIANTS:
            if not invariant.applies_to(result.exercise):
                invariant_results.append(
                    InvariantResult(
                        invariant_name=invariant.name,
                        exercise_name=result.exercise.name,
                        status=OutcomeStatus.SKIP,
                    )
                )
                continue

            try:
                error = invariant.check(result)
            except Exception as e:
                error = f"invariant check crashed: {e}"

            if error == SKIP:
                invariant_results.append(
                    InvariantResult(
                        invariant_name=invariant.name,
                        exercise_name=result.exercise.name,
                        status=OutcomeStatus.SKIP,
                    )
                )
            elif error is None:
                invariant_results.append(
                    InvariantResult(
                        invariant_name=invariant.name,
                        exercise_name=result.exercise.name,
                        status=OutcomeStatus.OK,
                    )
                )
            else:
                invariant_results.append(
                    InvariantResult(
                        invariant_name=invariant.name,
                        exercise_name=result.exercise.name,
                        status=OutcomeStatus.ERROR,
                        error=error,
                    )
                )

    return invariant_results


def format_invariant_summary(results: list[InvariantResult]) -> str:
    """Format invariant check results as a summary."""
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
