"""Showcase runner: prepare verification workspace, execute exercises, collect results."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from polylogue.scenarios import CorpusRequest
from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.corpus_requests import showcase_corpus_request
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.showcase_runner_models import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_runner_support import (
    generate_showcase_fixtures,
    run_exercise,
    seed_workspace_with,
    select_exercises,
    validate_exercise_output,
)


class ShowcaseRunner:
    """Execute showcase exercises against seeded or live data."""

    def __init__(
        self,
        *,
        live: bool = False,
        output_dir: Path | None = None,
        fail_fast: bool = False,
        verbose: bool = False,
        corpus_request: CorpusRequest | None = None,
        tier_filter: int | None = None,
        extra_exercises: list[Exercise] | None = None,
        workspace_env: dict[str, str] | None = None,
    ) -> None:
        self.live = live
        self.output_dir = output_dir
        self.fail_fast = fail_fast
        self.verbose = verbose
        self.corpus_request = corpus_request or showcase_corpus_request()
        self.tier_filter = tier_filter
        self.extra_exercises = extra_exercises or []
        self._env_vars: dict[str, str] = {}
        self._workspace_dir: Path | None = None
        self._shared_state: dict[str, Any] = {}
        self._workspace_env = workspace_env

    def run(self) -> ShowcaseResult:
        """Execute all applicable exercises and return results."""
        import tempfile

        started = time.monotonic()
        result = ShowcaseResult()

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="polylogue-showcase-"))
        result.output_dir = self.output_dir

        exercises_dir = self.output_dir / "exercises"
        exercises_dir.mkdir(exist_ok=True)
        selected_exercises = self._select_exercises()

        if not self.live:
            if self._workspace_env:
                self._env_vars = dict(self._workspace_env)
            else:
                self._workspace_dir = self.output_dir / "workspace"
                self._seed_workspace(self._workspace_dir, exercises=selected_exercises)
                result.workspace_dir = self._workspace_dir

        completed: set[str] = set()
        for exercise in selected_exercises:
            if exercise.depends_on and exercise.depends_on not in completed:
                result.results.append(
                    ExerciseResult(
                        exercise=exercise,
                        passed=False,
                        exit_code=-1,
                        output="",
                        skipped=True,
                        skip_reason=f"dependency {exercise.depends_on!r} not completed",
                    )
                )
                continue

            exercise_result = self._run_exercise(exercise)
            result.results.append(exercise_result)

            if exercise_result.passed and not exercise_result.skipped:
                completed.add(exercise.name)

            safe_name = f"{exercise.group}--{exercise.name}{exercise.output_ext}"
            (exercises_dir / safe_name).write_text(exercise_result.output or "")

            if self.fail_fast and not exercise_result.passed and not exercise_result.skipped:
                break

        result.total_duration_ms = (time.monotonic() - started) * 1000
        return result

    def _select_exercises(self) -> list[Exercise]:
        return select_exercises(
            live=self.live,
            tier_filter=self.tier_filter,
            extra_exercises=self.extra_exercises,
        )

    def _seed_workspace(self, workspace_dir: Path, *, exercises: list[Exercise]) -> None:
        self._env_vars = seed_workspace_with(
            workspace_dir,
            corpus_request=self.corpus_request,
            exercises=tuple(exercises),
            generate_fixtures=lambda fixture_dir, request: self._generate_synthetic_fixtures(
                fixture_dir, request=request
            ),
        )

    def _generate_synthetic_fixtures(self, fixture_dir: Path, *, request: CorpusRequest) -> None:
        generate_showcase_fixtures(fixture_dir, request=request)

    def _run_exercise(self, exercise: Exercise) -> ExerciseResult:
        return run_exercise(
            exercise,
            env_vars=self._env_vars,
            invoke_showcase_cli_fn=invoke_showcase_cli,
        )

    def _validate(self, exercise: Exercise, output: str, exit_code: int) -> str | None:
        return validate_exercise_output(exercise, output, exit_code)


__all__ = ["ExerciseResult", "ShowcaseResult", "ShowcaseRunner"]
