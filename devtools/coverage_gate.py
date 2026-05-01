"""Coverage gate shared by CI and local release-readiness checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib

CoverageThreshold = int | float


def read_coverage_threshold(pyproject_path: Path) -> CoverageThreshold:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    threshold: object = data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"{pyproject_path} does not define tool.coverage.report.fail_under")
    return threshold


def _format_threshold(threshold: CoverageThreshold) -> str:
    if isinstance(threshold, int) or threshold.is_integer():
        return str(int(threshold))
    return str(threshold)


def build_coverage_command(
    *,
    pyproject_path: Path,
    ignore_integration: bool,
    term_missing: bool,
    extra_args: Sequence[str] = (),
) -> list[str]:
    threshold = read_coverage_threshold(pyproject_path)
    command = [
        "pytest",
        "--cov=polylogue",
        "--cov-report=xml",
    ]
    if term_missing:
        command.append("--cov-report=term-missing:skip-covered")
    command.extend(["--cov-fail-under", _format_threshold(threshold), "-q"])
    if ignore_integration:
        command.append("--ignore=tests/integration")
    command.extend(extra_args)
    return command


def _strip_arg_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest coverage using tool.coverage.report.fail_under from pyproject.toml.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to the pyproject.toml file containing the coverage threshold.",
    )
    parser.add_argument(
        "--ignore-integration",
        action="store_true",
        help="Skip tests/integration for local ratchet measurements.",
    )
    parser.add_argument(
        "--term-missing",
        action="store_true",
        help="Also render missing-line coverage in the terminal.",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Extra arguments passed to pytest after `--`.")
    args = parser.parse_args(argv)

    threshold = read_coverage_threshold(args.pyproject)
    command = build_coverage_command(
        pyproject_path=args.pyproject,
        ignore_integration=bool(args.ignore_integration),
        term_missing=bool(args.term_missing),
        extra_args=_strip_arg_separator(list(args.pytest_args)),
    )
    sys.stderr.write(f"coverage-gate: enforcing fail_under={_format_threshold(threshold)} from {args.pyproject}\n")
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
