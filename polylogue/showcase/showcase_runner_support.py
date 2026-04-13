"""Execution helpers for showcase runs."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from polylogue.showcase.exercises import EXERCISES, Exercise, topological_order
from polylogue.showcase.showcase_runner_models import ExerciseResult
from polylogue.showcase.workspace import (
    build_synthetic_corpus_specs,
    create_verification_workspace,
    generate_synthetic_fixtures,
    run_pipeline_for_fixture_workspace,
    seed_workspace_from_specs,
)


def select_exercises(
    *,
    live: bool,
    tier_filter: int | None,
    extra_exercises: list[Exercise],
) -> list[Exercise]:
    """Filter exercises based on mode, env, and tier."""
    selected: list[Exercise] = []
    for exercise in list(EXERCISES) + list(extra_exercises):
        if exercise.env == "live" and not live:
            continue
        if exercise.env == "seeded" and live:
            continue
        if live and exercise.writes:
            continue
        if tier_filter is not None and exercise.tier != tier_filter:
            continue
        selected.append(exercise)
    return topological_order(selected)


def seed_workspace(workspace_dir: Path, *, synthetic_count: int) -> dict[str, str]:
    """Populate an isolated verification workspace and ingest its fixtures."""
    workspace = create_verification_workspace(workspace_dir)
    seed_workspace_from_specs(
        workspace,
        corpus_specs=build_synthetic_corpus_specs(
            count=synthetic_count,
            style="showcase",
        ),
    )
    return dict(workspace.env_vars)


def seed_workspace_with(
    workspace_dir: Path,
    *,
    synthetic_count: int,
    generate_fixtures: Callable[[Path], None],
) -> dict[str, str]:
    """Populate a workspace using an injected fixture-generation callback."""
    workspace = create_verification_workspace(workspace_dir)
    generate_fixtures(workspace.fixture_dir)
    run_pipeline_for_fixture_workspace(workspace)
    return dict(workspace.env_vars)


def generate_showcase_fixtures(fixture_dir: Path, *, count: int) -> None:
    """Generate schema-driven synthetic fixtures for all providers."""
    generate_synthetic_fixtures(fixture_dir, count=count, style="showcase")


def run_exercise(
    exercise: Exercise,
    *,
    env_vars: dict[str, str],
    invoke_showcase_cli_fn: Callable[..., object],
) -> ExerciseResult:
    """Run one showcase exercise and capture its result."""
    started = time.monotonic()
    env = dict(env_vars)
    env["POLYLOGUE_FORCE_PLAIN"] = "1"
    args = ["--plain", *exercise.args]

    try:
        cli_result = invoke_showcase_cli_fn(args, env=env, timeout=exercise.timeout_s)
        output = cli_result.output
        exit_code = cli_result.exit_code
    except subprocess.TimeoutExpired:
        duration = (time.monotonic() - started) * 1000
        return ExerciseResult(
            exercise=exercise,
            passed=False,
            exit_code=-1,
            output="",
            error=f"timed out after {exercise.timeout_s:.0f}s",
            duration_ms=duration,
        )
    except Exception as exc:
        duration = (time.monotonic() - started) * 1000
        return ExerciseResult(
            exercise=exercise,
            passed=False,
            exit_code=-1,
            output="",
            error=f"invoke crashed: {exc}",
            duration_ms=duration,
        )

    duration = (time.monotonic() - started) * 1000
    error = validate_exercise_output(exercise, output, exit_code)
    return ExerciseResult(
        exercise=exercise,
        passed=error is None,
        exit_code=exit_code,
        output=output,
        error=error,
        duration_ms=duration,
    )


def validate_exercise_output(exercise: Exercise, output: str, exit_code: int) -> str | None:
    """Validate exercise output against its validation spec."""
    validation = exercise.validation

    if validation.exit_code is not None and exit_code != validation.exit_code:
        return f"exit code {exit_code}, expected {validation.exit_code}"

    for needle in validation.stdout_contains:
        if needle not in output:
            return f"output missing {needle!r}"

    for needle in validation.stdout_not_contains:
        if needle in output:
            return f"output unexpectedly contains {needle!r}"

    if validation.stdout_is_valid_json:
        try:
            json.loads(output)
        except json.JSONDecodeError as exc:
            return f"invalid JSON: {exc}"

    if validation.stdout_min_lines is not None:
        line_count = len(output.strip().splitlines())
        if line_count < validation.stdout_min_lines:
            return f"only {line_count} lines, expected >= {validation.stdout_min_lines}"

    if validation.custom:
        return validation.custom(output, exit_code)

    return None


__all__ = [
    "generate_showcase_fixtures",
    "run_exercise",
    "seed_workspace",
    "seed_workspace_with",
    "select_exercises",
    "validate_exercise_output",
]
