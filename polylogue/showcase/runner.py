"""Showcase runner: seed workspace, execute exercises, collect results."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from polylogue.showcase.exercises import (
    EXERCISES,
    Exercise,
    Validation,
    exercises_by_group,
    topological_order,
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
    ) -> None:
        self.live = live
        self.output_dir = output_dir
        self.fail_fast = fail_fast
        self.verbose = verbose
        self._env_vars: dict[str, str] = {}
        self._workspace_dir: Path | None = None
        self._shared_state: dict[str, Any] = {}

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
        """Filter exercises based on mode."""
        selected: list[Exercise] = []
        for ex in EXERCISES:
            if self.live:
                # Live mode: skip writes and exercises that need seeded data
                if ex.writes:
                    continue
            selected.append(ex)
        return selected

    def _seed_workspace(self, workspace_dir: Path) -> None:
        """Copy static fixtures, run pipeline, configure env vars."""
        data_home = workspace_dir / "data"
        state_home = workspace_dir / "state"
        archive_root = workspace_dir / "archive"
        render_root = archive_root / "render"
        fixture_dir = workspace_dir / "fixtures"
        fake_home = workspace_dir / "home"

        for d in [data_home, state_home, archive_root, render_root, fake_home]:
            d.mkdir(parents=True, exist_ok=True)

        # Copy static fixtures from package resources
        self._copy_fixtures(fixture_dir)

        # Also copy fixtures into the inbox location so get_sources() finds them
        # inbox_root() = {XDG_DATA_HOME}/polylogue/inbox
        inbox_dir = data_home / "polylogue" / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        for provider_dir in fixture_dir.iterdir():
            if provider_dir.is_dir():
                dest = inbox_dir / provider_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in provider_dir.iterdir():
                    if f.is_file():
                        (dest / f.name).write_bytes(f.read_bytes())

        # Set environment for fully isolated workspace.
        # HOME is overridden so that auto-discovered paths like
        # ~/.claude/projects and ~/.codex/sessions don't resolve
        # to real user data.
        self._env_vars = {
            "HOME": str(fake_home),
            "XDG_DATA_HOME": str(data_home),
            "XDG_STATE_HOME": str(state_home),
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "POLYLOGUE_RENDER_ROOT": str(render_root),
            "POLYLOGUE_FORCE_PLAIN": "1",
        }

        # Apply env vars to current process for pipeline
        old_env: dict[str, str | None] = {}
        for key, value in self._env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Reset singletons so they pick up new env vars
            from polylogue import services
            services.reset()
            services._backend = None
            services._repository = None

            # Build config with fixture sources
            from polylogue.config import Config
            from polylogue.paths import Source

            sources: list[Source] = []
            for provider_dir in sorted(fixture_dir.iterdir()):
                if provider_dir.is_dir():
                    sources.append(Source(name=provider_dir.name, path=provider_dir))

            config = Config(
                archive_root=archive_root,
                render_root=render_root,
                sources=sources,
            )

            # Run the async pipeline
            from polylogue.pipeline.runner import run_sources
            asyncio.run(run_sources(
                config=config,
                stage="all",
                plan=None,
                ui=None,
                source_names=None,
            ))
        finally:
            # Restore original env
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def _copy_fixtures(self, fixture_dir: Path) -> None:
        """Copy static fixtures from package resources to workspace."""
        fixture_dir.mkdir(parents=True, exist_ok=True)

        pkg_fixtures = importlib_resources.files("polylogue.showcase") / "fixtures"
        for provider_entry in pkg_fixtures.iterdir():
            if not provider_entry.is_dir():
                continue
            dest_dir = fixture_dir / provider_entry.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_entry in provider_entry.iterdir():
                if file_entry.is_file():
                    dest_file = dest_dir / file_entry.name
                    dest_file.write_bytes(file_entry.read_bytes())

    def _run_exercise(self, exercise: Exercise) -> ExerciseResult:
        """Run a single exercise and validate the result."""
        from polylogue import services
        from polylogue.cli.click_app import cli

        t0 = time.monotonic()

        # Reset singletons between exercises to ensure clean state
        services.reset()
        services._backend = None
        services._repository = None

        # Build env vars for CliRunner
        env = dict(self._env_vars) if self._env_vars else {}
        env["POLYLOGUE_FORCE_PLAIN"] = "1"

        # Always pass --plain to ensure deterministic output
        args = ["--plain"] + list(exercise.args)

        runner = CliRunner()
        try:
            result = runner.invoke(cli, args, env=env, catch_exceptions=True)
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
        output = result.output or ""

        # Validate
        error = self._validate(exercise, output, result.exit_code)

        return ExerciseResult(
            exercise=exercise,
            passed=error is None,
            exit_code=result.exit_code,
            output=output,
            error=error,
            duration_ms=duration,
        )

    def _validate(self, exercise: Exercise, output: str, exit_code: int) -> str | None:
        """Validate exercise output against its Validation spec. Returns error or None."""
        v = exercise.validation

        if exit_code != v.exit_code:
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
            except json.JSONDecodeError as e:
                return f"invalid JSON: {e}"

        if v.stdout_min_lines is not None:
            line_count = len(output.strip().splitlines())
            if line_count < v.stdout_min_lines:
                return f"only {line_count} lines, expected >= {v.stdout_min_lines}"

        if v.custom:
            return v.custom(output, exit_code)

        return None
