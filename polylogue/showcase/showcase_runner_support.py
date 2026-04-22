"""Execution helpers for showcase runs."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.authored_payloads import canonical_payload_json
from polylogue.scenarios import CorpusRequest, CorpusSpec, ExecutionSpec
from polylogue.showcase.corpus_requests import showcase_corpus_request
from polylogue.showcase.exercises import EXERCISES, Exercise, topological_order
from polylogue.showcase.showcase_runner_models import ExerciseResult
from polylogue.showcase.workspace import (
    build_synthetic_corpus_scenarios,
    create_verification_workspace,
    generate_synthetic_fixtures,
    run_pipeline_for_fixture_workspace,
    seed_workspace_from_scenarios,
    seed_workspace_from_specs,
)

if TYPE_CHECKING:
    from polylogue.showcase.cli_boundary import ShowcaseCliResult


class _ShowcaseCliInvoker(Protocol):
    def __call__(
        self,
        execution: ExecutionSpec,
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 60.0,
    ) -> ShowcaseCliResult: ...


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


def seed_workspace(
    workspace_dir: Path,
    *,
    corpus_request: CorpusRequest | None = None,
) -> dict[str, str]:
    """Populate an isolated verification workspace and ingest its fixtures."""
    workspace = create_verification_workspace(workspace_dir)
    request = corpus_request or showcase_corpus_request()
    seed_workspace_from_scenarios(
        workspace,
        corpus_scenarios=build_synthetic_corpus_scenarios(
            request=request,
        ),
    )
    return dict(workspace.env_vars)


def seed_workspace_with(
    workspace_dir: Path,
    *,
    corpus_request: CorpusRequest | None = None,
    exercises: tuple[Exercise, ...] = (),
    generate_fixtures: Callable[[Path, CorpusRequest], None],
) -> dict[str, str]:
    """Populate a workspace using an injected fixture-generation callback."""
    workspace = create_verification_workspace(workspace_dir)
    request = corpus_request or showcase_corpus_request()
    corpus_specs = _merge_exercise_corpus_specs(exercises)
    if corpus_specs:
        seed_workspace_from_specs(workspace, corpus_specs=corpus_specs)
        return dict(workspace.env_vars)
    generate_fixtures(workspace.fixture_dir, request)
    run_pipeline_for_fixture_workspace(workspace)
    return dict(workspace.env_vars)


def generate_showcase_fixtures(
    fixture_dir: Path,
    *,
    request: CorpusRequest | None = None,
) -> None:
    """Generate schema-driven synthetic fixtures for all providers."""
    generate_synthetic_fixtures(fixture_dir, request=request or showcase_corpus_request())


def _merge_exercise_corpus_specs(exercises: tuple[Exercise, ...]) -> tuple[CorpusSpec, ...]:
    """Return ordered unique corpus specs referenced by the selected exercises."""
    seen: set[str] = set()
    merged: list[CorpusSpec] = []
    for exercise in exercises:
        for corpus_spec in exercise.corpus_specs:
            key = canonical_payload_json(corpus_spec.to_payload())
            if key in seen:
                continue
            seen.add(key)
            merged.append(corpus_spec)
    return tuple(merged)


def run_exercise(
    exercise: Exercise,
    *,
    env_vars: dict[str, str],
    invoke_showcase_cli_fn: _ShowcaseCliInvoker,
) -> ExerciseResult:
    """Run one showcase exercise and capture its result."""
    started = time.monotonic()
    env = dict(env_vars)
    env["POLYLOGUE_FORCE_PLAIN"] = "1"

    try:
        cli_result = invoke_showcase_cli_fn(exercise.execution, env=env, timeout=exercise.timeout_s)
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
    """Validate exercise output against its assertion spec."""
    return exercise.assertion.validate_process(output, exit_code)


__all__ = [
    "_merge_exercise_corpus_specs",
    "generate_showcase_fixtures",
    "run_exercise",
    "seed_workspace",
    "seed_workspace_with",
    "select_exercises",
    "validate_exercise_output",
]
