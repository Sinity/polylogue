"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Steps:
  1. ruff format --check (fast, <2s)
  2. ruff check (fast, <2s)
  3. mypy polylogue/ (warm ~1s, cold ~25s)
  4. devtools render-all --check (fast, <5s)
  5. pytest --ignore=tests/integration (slow but essential, ~3min)

Use --quick to run only steps 1-4 (suitable for pre-commit).
"""

from __future__ import annotations

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


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    quick = "--quick" in args

    sys.stderr.write("verify: running local verification baseline\n")

    exit_code = 0

    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", ["mypy", "polylogue/"]),
        ("render-all", ["devtools", "render-all", "--check"]),
    ]

    if not quick:
        steps.append(
            ("pytest", ["pytest", "-q", "--tb=short", "--ignore=tests/integration"]),
        )

    for label, cmd in steps:
        rc = _run(label, cmd)
        if rc != 0:
            exit_code = rc

    if exit_code != 0:
        sys.stderr.write("\nverify: FAILED — fix before pushing\n")
    else:
        sys.stderr.write("\nverify: all checks passed\n")

    return exit_code
