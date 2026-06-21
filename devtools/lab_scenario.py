"""Verification-lab smoke scenario runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from devtools import repo_root as _get_root
from polylogue.core.outcomes import OutcomeStatus
from polylogue.scenarios import AssertionSpec, ExecutionSpec, polylogue_execution
from polylogue.showcase.cli_boundary import invoke_showcase_cli

_SCENARIO_NAMES = ("archive-smoke", "reader-visual-smoke")
_ARCHIVE_SMOKE_TIER = 0


class _ScenarioResult(Protocol):
    report_dir: Path | None

    def stage_statuses(self) -> dict[str, OutcomeStatus]: ...

    def failed_stages(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ArchiveSmokeCheck:
    name: str
    execution: ExecutionSpec
    assertion: AssertionSpec
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class ArchiveSmokeCheckResult:
    check: ArchiveSmokeCheck
    passed: bool
    exit_code: int
    output: str
    duration_ms: float
    error: str | None = None


_ARCHIVE_SMOKE_CHECKS: tuple[ArchiveSmokeCheck, ...] = (
    ArchiveSmokeCheck(
        name="help-main",
        execution=polylogue_execution("--help"),
        assertion=AssertionSpec(stdout_contains=("polylogue",)),
    ),
    ArchiveSmokeCheck(
        name="help-mark-candidates",
        execution=polylogue_execution("mark", "candidates", "--help"),
        assertion=AssertionSpec(stdout_contains=("candidates",)),
    ),
    ArchiveSmokeCheck(
        name="completions-bash",
        execution=polylogue_execution("config", "completions", "--shell", "bash"),
        assertion=AssertionSpec(stdout_contains=("complete",)),
    ),
)


class ArchiveSmokeResult:
    """Direct result wrapper for the archive-smoke lab scenario."""

    def __init__(
        self,
        *,
        check_results: list[ArchiveSmokeCheckResult],
        report_dir: Path | None,
        unsupported_reason: str | None = None,
    ) -> None:
        self.check_results = check_results
        self.report_dir = report_dir
        self.unsupported_reason = unsupported_reason

    @property
    def all_passed(self) -> bool:
        return not self.failed_stages()

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        status = (
            OutcomeStatus.ERROR
            if self.unsupported_reason is not None or any(not result.passed for result in self.check_results)
            else OutcomeStatus.OK
        )
        return {
            "cli": status,
        }

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(name for name, status in self.stage_statuses().items() if status is OutcomeStatus.ERROR)


def get_archive_smoke_checks() -> tuple[ArchiveSmokeCheck, ...]:
    """Return direct CLI checks for the archive-smoke lab scenario."""
    return _ARCHIVE_SMOKE_CHECKS


def run_tier_0() -> dict[str, str]:
    """Run direct archive-smoke checks and return output by check name."""
    checks = get_archive_smoke_checks()
    results: dict[str, str] = {}
    failures: list[str] = []
    total = len(checks)
    for index, check in enumerate(checks, start=1):
        print(f"  [{index:03d}/{total:03d}] {check.name}", flush=True)
        result = _run_archive_smoke_check(check)
        results[check.name] = result.output or ""
        if not result.passed:
            failures.append(f"{check.name}: exit {result.exit_code}: {result.error or 'failed'}")
    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"archive-smoke checks failed:\n  - {joined}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verification-lab smoke scenarios.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a named smoke scenario set.")
    run_parser.add_argument("scenario", choices=_SCENARIO_NAMES, help="Scenario set to run.")
    run_parser.add_argument(
        "--live", action="store_true", help="Run against the active archive instead of a seeded workspace."
    )
    run_parser.add_argument("--tier", type=int, default=None, help="Only run smoke checks at this tier.")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Directory for scenario artifacts.")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable scenario payload.")
    run_parser.add_argument("--verbose", action="store_true", help="Print smoke-check outputs.")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first smoke-check failure.")

    list_parser = subparsers.add_parser("list", help="List available smoke scenarios.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def list_scenarios(*, as_json: bool) -> int:
    """List available scenario checks."""
    archive_checks = get_archive_smoke_checks()
    scenarios: list[dict[str, object]] = [
        {
            "name": "archive-smoke",
            "kind": "cli-smoke",
            "tier_0_check_count": len(archive_checks),
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
        print(f"{name:<20s}  tier-0 checks: {entry['tier_0_check_count']}")
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
    """Format the scenario runner's direct stage result without report wrapping."""
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
    as_json: bool = False,
) -> ArchiveSmokeResult:
    """Run direct archive-smoke CLI checks without showcase catalog wrapping."""
    if tier not in (None, _ARCHIVE_SMOKE_TIER):
        return ArchiveSmokeResult(
            check_results=[],
            report_dir=report_dir,
            unsupported_reason=f"archive-smoke only supports tier {_ARCHIVE_SMOKE_TIER} direct CLI checks",
        )
    check_results = _run_archive_smoke_checks(
        verbose=verbose,
        fail_fast=fail_fast,
        progress_stream=sys.stderr if as_json else sys.stdout,
    )
    _write_archive_smoke_report(report_dir, check_results=check_results, live=live, tier=tier)
    return ArchiveSmokeResult(
        check_results=check_results,
        report_dir=report_dir,
    )


def _scenario_payload(result: _ScenarioResult) -> dict[str, object]:
    """Return the direct lab-scenario payload without report wrapping."""
    stage_statuses = result.stage_statuses()
    failed_stages = result.failed_stages()
    return {
        "scenario": "archive-smoke",
        "stages": {name: status.value for name, status in stage_statuses.items()},
        "failed_stages": list(failed_stages),
        "ok": not failed_stages,
        "report_dir": str(result.report_dir) if result.report_dir is not None else None,
    }


def _run_archive_smoke_check(check: ArchiveSmokeCheck) -> ArchiveSmokeCheckResult:
    started = time.monotonic()
    try:
        cli_result = invoke_showcase_cli(check.execution, env={"POLYLOGUE_FORCE_PLAIN": "1"}, timeout=check.timeout_s)
    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - started) * 1000
        return ArchiveSmokeCheckResult(
            check=check,
            passed=False,
            exit_code=-1,
            output="",
            duration_ms=duration_ms,
            error=f"timed out after {check.timeout_s:.0f}s",
        )
    except Exception as exc:
        duration_ms = (time.monotonic() - started) * 1000
        return ArchiveSmokeCheckResult(
            check=check,
            passed=False,
            exit_code=-1,
            output="",
            duration_ms=duration_ms,
            error=f"invoke crashed: {exc}",
        )
    output = cli_result.output
    error = check.assertion.validate_process(output, cli_result.exit_code)
    duration_ms = (time.monotonic() - started) * 1000
    return ArchiveSmokeCheckResult(
        check=check,
        passed=error is None,
        exit_code=cli_result.exit_code,
        output=output,
        duration_ms=duration_ms,
        error=error,
    )


def _run_archive_smoke_checks(
    *,
    verbose: bool,
    fail_fast: bool,
    progress_stream: TextIO,
) -> list[ArchiveSmokeCheckResult]:
    results: list[ArchiveSmokeCheckResult] = []
    checks = get_archive_smoke_checks()
    for index, check in enumerate(checks, start=1):
        print(f"  [{index:03d}/{len(checks):03d}] {check.name}", flush=True, file=progress_stream)
        result = _run_archive_smoke_check(check)
        results.append(result)
        if verbose and result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n", file=progress_stream)
        if fail_fast and not result.passed:
            break
    return results


def _write_archive_smoke_report(
    report_dir: Path | None,
    *,
    check_results: list[ArchiveSmokeCheckResult],
    live: bool,
    tier: int | None,
) -> None:
    if report_dir is None:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": "archive-smoke",
        "live": live,
        "tier": tier,
        "checks": [
            {
                "name": result.check.name,
                "exit_code": result.exit_code,
                "passed": result.passed,
                "duration_ms": round(result.duration_ms, 1),
                "error": result.error,
            }
            for result in check_results
        ],
    }
    (report_dir / "archive-smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        as_json=bool(args.json),
    )
    if args.json:
        print(json.dumps(_scenario_payload(result), indent=2))
    else:
        print(_format_scenario_summary(result))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
