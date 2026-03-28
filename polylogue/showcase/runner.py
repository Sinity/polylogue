"""Showcase runner: prepare verification workspace, execute exercises, collect results."""

from __future__ import annotations

<<<<<<< HEAD
import json
||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
import json
import subprocess
=======
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)
import time
<<<<<<< HEAD
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
from dataclasses import dataclass, field
=======
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)
from pathlib import Path
from typing import Any

<<<<<<< HEAD
from click.testing import CliRunner

from polylogue.showcase.exercises import (
    EXERCISES,
    Exercise,
    topological_order,
||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.exercises import (
    EXERCISES,
    Exercise,
    topological_order,
=======
from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.showcase_runner_models import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_runner_support import (
    generate_showcase_fixtures,
    run_exercise,
    seed_workspace_with,
    select_exercises,
    validate_exercise_output,
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)
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
        synthetic_count: int = 3,
        tier_filter: int | None = None,
        extra_exercises: list[Exercise] | None = None,
        workspace_env: dict[str, str] | None = None,
    ) -> None:
        self.live = live
        self.output_dir = output_dir
        self.fail_fast = fail_fast
        self.verbose = verbose
        self.synthetic_count = synthetic_count
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

        if not self.live:
            if self._workspace_env:
                self._env_vars = dict(self._workspace_env)
            else:
                self._workspace_dir = self.output_dir / "workspace"
                self._seed_workspace(self._workspace_dir)
                result.workspace_dir = self._workspace_dir

        completed: set[str] = set()
        for exercise in self._select_exercises():
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

    def _seed_workspace(self, workspace_dir: Path) -> None:
        self._env_vars = seed_workspace_with(
            workspace_dir,
            synthetic_count=self.synthetic_count,
            generate_fixtures=lambda fixture_dir: self._generate_synthetic_fixtures(
                fixture_dir,
                count=self.synthetic_count,
            ),
        )

    def _generate_synthetic_fixtures(self, fixture_dir: Path, *, count: int) -> None:
        generate_showcase_fixtures(fixture_dir, count=count)

    def _run_exercise(self, exercise: Exercise) -> ExerciseResult:
<<<<<<< HEAD
        """Run a single exercise and validate the result."""
        from polylogue.cli.click_app import cli

        t0 = time.monotonic()

        # Build env vars for CliRunner
        env = dict(self._env_vars) if self._env_vars else {}
        env["POLYLOGUE_FORCE_PLAIN"] = "1"

        # Always pass --plain to ensure deterministic output
        args = ["--plain"] + list(exercise.args)

        runner = CliRunner()

        def _invoke() -> tuple[str, int]:
            res = runner.invoke(cli, args, env=env, catch_exceptions=True)
            return res.output or "", res.exit_code

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_invoke)
                output, exit_code = future.result(timeout=exercise.timeout_s)
        except FuturesTimeoutError:
            duration = (time.monotonic() - t0) * 1000
            return ExerciseResult(
                exercise=exercise,
                passed=False,
                exit_code=-1,
                output="",
                error=f"timed out after {exercise.timeout_s:.0f}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - t0) * 1000
            return ExerciseResult(
                exercise=exercise,
                passed=False,
                exit_code=-1,
                output="",
                error=f"invoke crashed: {e}",
                duration_ms=duration,
            )

        duration = (time.monotonic() - t0) * 1000

        # Validate
        error = self._validate(exercise, output, exit_code)

        return ExerciseResult(
            exercise=exercise,
            passed=error is None,
            exit_code=exit_code,
            output=output,
            error=error,
            duration_ms=duration,
||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
        """Run a single exercise and validate the result."""
        t0 = time.monotonic()

        env = dict(self._env_vars) if self._env_vars else {}
        env["POLYLOGUE_FORCE_PLAIN"] = "1"
        args = ["--plain"] + list(exercise.args)

        try:
            cli_result = invoke_showcase_cli(args, env=env, timeout=exercise.timeout_s)
            output = cli_result.output
            exit_code = cli_result.exit_code
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - t0) * 1000
            return ExerciseResult(
                exercise=exercise,
                passed=False,
                exit_code=-1,
                output="",
                error=f"timed out after {exercise.timeout_s:.0f}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - t0) * 1000
            return ExerciseResult(
                exercise=exercise,
                passed=False,
                exit_code=-1,
                output="",
                error=f"invoke crashed: {e}",
                duration_ms=duration,
            )

        duration = (time.monotonic() - t0) * 1000

        # Validate
        error = self._validate(exercise, output, exit_code)

        return ExerciseResult(
            exercise=exercise,
            passed=error is None,
            exit_code=exit_code,
            output=output,
            error=error,
            duration_ms=duration,
=======
        return run_exercise(
            exercise,
            env_vars=self._env_vars,
            invoke_showcase_cli_fn=invoke_showcase_cli,
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)
        )

    def _validate(self, exercise: Exercise, output: str, exit_code: int) -> str | None:
        return validate_exercise_output(exercise, output, exit_code)


<<<<<<< HEAD
        for needle in v.stdout_contains:
            if needle not in output:
                return f"output missing {needle!r}"

        for needle in v.stdout_not_contains:
            if needle in output:
                return f"output unexpectedly contains {needle!r}"

        if v.stdout_is_valid_json:
            try:
                json.loads(output)
            except json.JSONDecodeError as e:
                return f"invalid JSON: {e}"

        if v.stdout_min_lines is not None:
            line_count = len(output.strip().splitlines())
            if line_count < v.stdout_min_lines:
                return f"only {line_count} lines, expected >= {v.stdout_min_lines}"

        if v.custom:
            return v.custom(output, exit_code)

        return None
||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
        for needle in v.stdout_contains:
            if needle not in output:
                return f"output missing {needle!r}"

        for needle in v.stdout_not_contains:
            if needle in output:
                return f"output unexpectedly contains {needle!r}"

        if v.stdout_is_valid_json:
            try:
                json.loads(output)
            except json.JSONDecodeError as exc:
                return f"invalid JSON: {exc}"

        if v.stdout_min_lines is not None:
            line_count = len(output.strip().splitlines())
            if line_count < v.stdout_min_lines:
                return f"only {line_count} lines, expected >= {v.stdout_min_lines}"

        if v.custom:
            return v.custom(output, exit_code)

        return None
=======
__all__ = ["ExerciseResult", "ShowcaseResult", "ShowcaseRunner"]
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)
