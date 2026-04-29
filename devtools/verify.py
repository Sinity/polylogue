"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Steps:
  1. ruff format --check (fast, <2s)
  2. ruff check (fast, <2s)
  3. mypy (polylogue + tests + devtools) (warm ~1s, cold ~25s)
  4. devtools render-all --check (fast, <5s)
  5. pytest --ignore=tests/integration (slow but essential, ~3min)

Use --quick to run only steps 1-4 (suitable for pre-commit).
Use --lab to add verification-lab scenario checks through lab commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run(label: str, cmd: list[str], *, cwd: str | None = None) -> int:
    t0 = time.monotonic()
    sys.stderr.write(f"  {label} ... ")
    sys.stderr.flush()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        sys.stderr.write(f"ok ({elapsed:.1f}s)\n")
    else:
        sys.stderr.write(f"FAILED ({elapsed:.1f}s)\n")
        if result.stdout.strip():
            sys.stderr.write(result.stdout + "\n")
        if result.stderr.strip():
            sys.stderr.write(result.stderr + "\n")
    return result.returncode


def build_verify_steps(*, quick: bool, lab: bool) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", ["mypy"]),
        ("render-all", ["devtools", "render-all", "--check"]),
        ("verify-topology", ["devtools", "verify-topology"]),
        ("verify-file-budgets", ["devtools", "verify-file-budgets"]),
        ("verify-test-ownership", ["devtools", "verify-test-ownership"]),
        ("verify-migrations", ["devtools", "verify-migrations"]),
        ("verify-cross-cuts", ["devtools", "verify-cross-cuts"]),
        ("verify-suppressions", ["devtools", "verify-suppressions"]),
        ("verify-manifests", ["devtools", "verify-manifests"]),
    ]

    if not quick:
        steps.append(
            ("pytest", ["pytest", "-q", "--tb=short", "--ignore=tests/integration"]),
        )

    if lab:
        steps.append(("lab scenario", ["devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"]))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--lab",
        action="store_true",
        help="Delegate additional domain proof checks through verification-lab commands.",
    )
    args = parser.parse_args(argv)

    sys.stderr.write("verify: running local verification baseline\n")

    exit_code = 0
    steps = build_verify_steps(quick=bool(args.quick), lab=bool(args.lab))

    for label, cmd in steps:
        rc = _run(label, cmd)
        if rc != 0:
            exit_code = rc

    if exit_code != 0:
        sys.stderr.write("\nverify: FAILED — fix before pushing\n")
    else:
        sys.stderr.write("\nverify: all checks passed\n")

    return exit_code
