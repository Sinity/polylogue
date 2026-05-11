"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy + proof-pack (~3s warm).
  --quick    Pre-push tier: all non-pytest gates (~15s warm).
  (default)  Full baseline: all gates + pytest (~3 min).
  --lab      Full baseline + verification-lab scenario checks.

Individual steps:
  1. ruff format --check (fast, <2s)
  2. ruff check (fast, <2s)
  3. mypy (polylogue + tests + devtools) (warm ~1s, cold ~25s)
  4. devtools render-all --check (fast, <5s)
  5. devtools proof-pack --check (fast proof-policy gate)
  6..15. verify-* structural gates (fast, <1s each)
  16. pytest --ignore=tests/integration (slow but essential, ~3min)

Use --skip-slow to exclude @pytest.mark.slow tests from the pytest step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _mypy_cmd() -> list[str]:
    """Return the mypy command, preferring dmypy for warm-cache speed.

    Probes whether the dmypy daemon is already running (fast status check).
    If alive, subsequent ``dmypy run`` calls check only changed files
    (~0.5s vs ~13s). If the daemon isn't running, fall back to ``mypy``
    to avoid the cold-start penalty on every invocation.
    """
    try:
        result = subprocess.run(
            ["dmypy", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return ["dmypy", "run", "--", "--no-error-summary"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["mypy"]


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


def build_verify_steps(*, quick: bool, lab: bool, skip_slow: bool, commit: bool = False) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]

    # Structural verify-* steps and proof-pack — skip in --commit tier
    # (which is designed for fast pre-commit confidence).
    if not commit:
        steps.extend(
            [
                ("render-all", ["devtools", "render-all", "--check"]),
                ("verify-topology", ["devtools", "verify-topology"]),
                ("verify-layering", ["devtools", "verify-layering"]),
                ("verify-file-budgets", ["devtools", "verify-file-budgets"]),
                ("verify-test-ownership", ["devtools", "verify-test-ownership"]),
                ("verify-schema-roundtrip", ["devtools", "verify-schema-roundtrip", "--all"]),
                ("verify-cross-cuts", ["devtools", "verify-cross-cuts"]),
                ("verify-suppressions", ["devtools", "verify-suppressions"]),
                ("verify-manifests", ["devtools", "verify-manifests"]),
                ("verify-witness-lifecycle", ["devtools", "verify-witness-lifecycle"]),
                ("verify-lane-assertions", ["devtools", "verify-lane-assertions"]),
            ]
        )

    # proof-pack runs in all tiers (it's fast and load-bearing)
    steps.append(("proof-pack check", ["devtools", "proof-pack", "--check"]))

    # pytest is skipped in --quick and --commit tiers.
    if not quick and not commit:
        pytest_cmd = ["pytest", "-q", "--tb=short", "--ignore=tests/integration"]
        if skip_slow:
            pytest_cmd.extend(["-m", "not slow"])
        steps.append(("pytest", pytest_cmd))

    if lab:
        steps.append(("lab scenario", ["devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"]))
        steps.append(("verify-slos", ["devtools", "verify-slos"]))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--commit", action="store_true", help="Pre-commit tier: format + lint + mypy + proof-pack only."
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Exclude @pytest.mark.slow tests from the pytest step.",
    )
    parser.add_argument(
        "--lab",
        action="store_true",
        help="Delegate additional domain proof checks through verification-lab commands.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    sys.stderr.write("verify: running local verification baseline\n")

    exit_code = 0
    steps = build_verify_steps(
        quick=bool(args.quick),
        commit=bool(args.commit),
        lab=bool(args.lab),
        skip_slow=bool(args.skip_slow),
    )

    for label, cmd in steps:
        rc = _run(label, cmd)
        if rc != 0:
            exit_code = rc

    if exit_code != 0:
        sys.stderr.write("\nverify: FAILED — fix before pushing\n")
    else:
        sys.stderr.write("\nverify: all checks passed\n")
        _stamp_head()

    return exit_code


def _stamp_head() -> None:
    """Record HEAD SHA so the pre-push hook can skip already-verified commits."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return
    stamp_dir = __import__("pathlib").Path(".cache")
    stamp_dir.mkdir(parents=True, exist_ok=True)
    (stamp_dir / "last-verify-head").write_text(result.stdout.strip() + "\n")
