"""Showcase runner: prepare verification workspace, execute exercises, collect results."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.exercises import (
    EXERCISES,
    Exercise,
    topological_order,
)
from polylogue.showcase.workspace import (
    create_verification_workspace,
    generate_synthetic_fixtures,
    run_pipeline_for_fixture_workspace,
)


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
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    def group_counts(self) -> dict[str, dict[str, int]]:
        """Return pass/fail/skip counts by group."""
        counts: dict[str, dict[str, int]] = {}
        for r in self.results:
            g = r.exercise.group
            if g not in counts:
                counts[g] = {"pass": 0, "fail": 0, "skip": 0}
            if r.skipped:
                counts[g]["skip"] += 1
            elif r.passed:
                counts[g]["pass"] += 1
            else:
                counts[g]["fail"] += 1
        return counts


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

        t0 = time.monotonic()
        result = ShowcaseResult()

        # Set up output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="polylogue-showcase-"))
        result.output_dir = self.output_dir

        exercises_dir = self.output_dir / "exercises"
        exercises_dir.mkdir(exist_ok=True)

        # Seed workspace if not live mode
        if not self.live:
            if self._workspace_env:
                # Workspace already prepared externally (by qa_runner)
                self._env_vars = dict(self._workspace_env)
            else:
                self._workspace_dir = self.output_dir / "workspace"
                self._seed_workspace(self._workspace_dir)
                result.workspace_dir = self._workspace_dir

        # Build exercise list
        exercises = self._select_exercises()
        exercises = topological_order(exercises)

        # Track completed exercises for dependency resolution
        completed: set[str] = set()

        # Execute
        for exercise in exercises:
            # Check dependencies
            if exercise.depends_on and exercise.depends_on not in completed:
                er = ExerciseResult(
                    exercise=exercise,
                    passed=False,
                    exit_code=-1,
                    output="",
                    skipped=True,
                    skip_reason=f"dependency {exercise.depends_on!r} not completed",
                )
                result.results.append(er)
                continue

            er = self._run_exercise(exercise)
            result.results.append(er)

            if er.passed and not er.skipped:
                completed.add(exercise.name)

            # Save output
            safe_name = f"{exercise.group}--{exercise.name}{exercise.output_ext}"
            (exercises_dir / safe_name).write_text(er.output or "")

            if self.fail_fast and not er.passed and not er.skipped:
                break

        result.total_duration_ms = (time.monotonic() - t0) * 1000
        return result

    def _select_exercises(self) -> list[Exercise]:
        """Filter exercises based on mode, env, and tier."""
        # Combine static catalog with any dynamically generated exercises
        all_exercises = list(EXERCISES) + list(self.extra_exercises)

        selected: list[Exercise] = []
        for ex in all_exercises:
            # Env-based filtering
            if ex.env == "live" and not self.live:
                continue  # live-only exercises require --live mode
            if ex.env == "seeded" and self.live:
                continue  # seeded-only exercises don't run against live data

            # Skip writes in live mode (protect real data)
            if self.live and ex.writes:
                continue

            if self.tier_filter is not None and ex.tier != self.tier_filter:
                continue
            selected.append(ex)
        return selected

    def _seed_workspace(self, workspace_dir: Path) -> None:
        """Populate an isolated verification workspace and ingest its fixtures."""
        workspace = create_verification_workspace(workspace_dir)
        self._generate_synthetic_fixtures(workspace.fixture_dir, count=self.synthetic_count)
        run_pipeline_for_fixture_workspace(workspace)
        self._env_vars = dict(workspace.env_vars)

    def _generate_synthetic_fixtures(self, fixture_dir: Path, *, count: int) -> None:
        """Generate schema-driven synthetic fixtures for all providers."""
        generate_synthetic_fixtures(fixture_dir, count=count, style="showcase")

    def _run_exercise(self, exercise: Exercise) -> ExerciseResult:
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
        )

    def _validate(self, exercise: Exercise, output: str, exit_code: int) -> str | None:
        """Validate exercise output against its Validation spec. Returns error or None."""
        v = exercise.validation

        if v.exit_code is not None and exit_code != v.exit_code:
            return f"exit code {exit_code}, expected {v.exit_code}"

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
