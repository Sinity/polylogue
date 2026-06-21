"""Verification-lab showcase scenario runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from devtools import repo_root as _get_root
from polylogue.core.outcomes import OutcomeStatus
from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.exercises import EXERCISES, Exercise
from polylogue.showcase.invariants import InvariantResult, check_invariants
from polylogue.showcase.runner import ShowcaseRunner
from polylogue.showcase.showcase_runner_models import ShowcaseResult
from polylogue.showcase.showcase_runner_support import run_exercise

_SCENARIO_NAMES = ("archive-smoke", "reader-visual-smoke")
TIER_0_GROUPS = frozenset({"structural"})
_ENV_DEPENDENT: frozenset[str] = frozenset({"version"})


class _ScenarioResult(Protocol):
    report_dir: Path | None

    def stage_statuses(self) -> dict[str, OutcomeStatus]: ...

    def failed_stages(self) -> tuple[str, ...]: ...


class ArchiveSmokeResult:
    """Direct result wrapper for the archive-smoke lab scenario."""

    def __init__(
        self,
        *,
        showcase_result: ShowcaseResult,
        invariant_results: list[InvariantResult],
        report_dir: Path | None,
    ) -> None:
        self.showcase_result = showcase_result
        self.invariant_results = invariant_results
        self.report_dir = report_dir

    @property
    def all_passed(self) -> bool:
        return not self.failed_stages()

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        invariant_status = (
            OutcomeStatus.ERROR
            if any(result.status is OutcomeStatus.ERROR for result in self.invariant_results)
            else OutcomeStatus.OK
        )
        return {
            "showcase": OutcomeStatus.OK if self.showcase_result.failed == 0 else OutcomeStatus.ERROR,
            "invariants": invariant_status,
        }

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(name for name, status in self.stage_statuses().items() if status is OutcomeStatus.ERROR)


def get_tier_0_exercises() -> list[Exercise]:
    """Return tier-0 exercises with stable committed baseline output."""
    return [ex for ex in EXERCISES if ex.group in TIER_0_GROUPS and not ex.needs_data and ex.name not in _ENV_DEPENDENT]


def run_tier_0() -> dict[str, str]:
    """Run tier-0 showcase exercises and return output by exercise name."""
    from polylogue.showcase.exercises import topological_order

    exercises = topological_order(get_tier_0_exercises())
    results: dict[str, str] = {}
    failures: list[str] = []
    total = len(exercises)
    for index, exercise in enumerate(exercises, start=1):
        print(f"  [{index:03d}/{total:03d}] {exercise.name}", flush=True)
        result = run_exercise(
            exercise,
            env_vars={},
            invoke_showcase_cli_fn=invoke_showcase_cli,
        )
        results[exercise.name] = result.output or ""
        if not result.passed:
            failures.append(f"{exercise.name}: exit {result.exit_code}: {result.error or 'failed'}")
    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"tier-0 showcase exercises failed:\n  - {joined}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verification-lab showcase scenarios.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a named showcase scenario set.")
    run_parser.add_argument("scenario", choices=_SCENARIO_NAMES, help="Scenario set to run.")
    run_parser.add_argument(
        "--live", action="store_true", help="Run against the active archive instead of a seeded workspace."
    )
    run_parser.add_argument("--tier", type=int, default=None, help="Only run exercises at this tier.")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Directory for scenario artifacts.")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable QA session payload.")
    run_parser.add_argument("--verbose", action="store_true", help="Print exercise outputs.")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first exercise failure.")

    list_parser = subparsers.add_parser("list", help="List available showcase scenarios.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def list_scenarios(*, as_json: bool) -> int:
    """List available showcase scenarios with their tier-0 exercise inventory."""
    tier_0 = get_tier_0_exercises()
    scenarios: list[dict[str, object]] = [
        {
            "name": "archive-smoke",
            "kind": "showcase",
            "tier_0_exercise_count": len(tier_0),
        },
        {
            "name": "reader-visual-smoke",
            "kind": "reader-visual",
            "command": f"{sys.executable} -m pytest -q tests/visual",
        },
    ]
    payload = {"scenarios": scenarios}
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    for entry in scenarios:
        name = str(entry["name"])
        if name == "reader-visual-smoke":
            print(f"{name:<20s}  command: {entry['command']}")
            continue
        print(f"{name:<20s}  tier-0 exercises: {entry['tier_0_exercise_count']}")
    return 0


def run_reader_visual_smoke(*, report_dir: Path | None, as_json: bool) -> int:
    """Run the daemon reader visual/DOM smoke lane."""
    command = [sys.executable, "-m", "pytest", "-q", "tests/visual"]
    result = subprocess.run(
        command,
        cwd=_get_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    payload: dict[str, object] = {
        "scenario": "reader-visual-smoke",
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "reader-visual-smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("Running reader visual DOM smoke...")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.returncode


def _format_scenario_summary(result: _ScenarioResult) -> str:
    """Format the scenario runner's direct stage result without QA reports."""
    stage_statuses = result.stage_statuses()
    failed_stages = result.failed_stages()
    lines = ["Scenario stages:"]
    for name, status in stage_statuses.items():
        lines.append(f"  {name}: {status.value}")
    if failed_stages:
        lines.append(f"Failed stages: {', '.join(failed_stages)}")
    else:
        lines.append("Failed stages: none")
    if result.report_dir is not None:
        lines.append(f"Artifacts: {result.report_dir}")
    return "\n".join(lines)


def run_archive_smoke(
    *,
    live: bool,
    tier: int | None,
    report_dir: Path | None,
    verbose: bool,
    fail_fast: bool,
) -> ArchiveSmokeResult:
    """Run the archive-smoke lab scenario without QA session wrapping."""
    runner = ShowcaseRunner(
        live=live,
        output_dir=report_dir,
        fail_fast=fail_fast,
        verbose=verbose,
        tier_filter=tier,
    )
    showcase_result = runner.run()
    invariant_results = check_invariants(showcase_result.results)
    return ArchiveSmokeResult(
        showcase_result=showcase_result,
        invariant_results=invariant_results,
        report_dir=showcase_result.output_dir,
    )


def _scenario_payload(result: _ScenarioResult) -> dict[str, object]:
    """Return the direct lab-scenario payload without QA report wrapping."""
    stage_statuses = result.stage_statuses()
    failed_stages = result.failed_stages()
    return {
        "scenario": "archive-smoke",
        "stages": {name: status.value for name, status in stage_statuses.items()},
        "failed_stages": list(failed_stages),
        "ok": not failed_stages,
        "report_dir": str(result.report_dir) if result.report_dir is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "list":
        return list_scenarios(as_json=bool(args.json))
    if args.action != "run":
        parser.error(f"unknown action: {args.action}")
    if args.scenario == "reader-visual-smoke":
        return run_reader_visual_smoke(report_dir=args.report_dir, as_json=bool(args.json))
    result = run_archive_smoke(
        live=bool(args.live),
        tier=args.tier,
        report_dir=args.report_dir,
        verbose=bool(args.verbose),
        fail_fast=bool(args.fail_fast),
    )
    if args.json:
        print(json.dumps(_scenario_payload(result), indent=2))
    else:
        print(_format_scenario_summary(result))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
