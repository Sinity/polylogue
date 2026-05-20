"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy + verification-impact check (~3s warm).
  --quick    Pre-push tier: all non-pytest gates (~15s warm).
  (default)  Baseline with pytest-testmon affected tests.
  --seed-testmon
             Full non-integration pytest run that seeds/updates .testmondata.
  --all/--full
             Explicit full non-integration pytest diagnostic.
  --lab      Default testmon baseline plus verification-lab scenario and SLO checks.

Output formats:
  --json     Machine-readable JSON to stdout (human progress to stderr).
  (default)  Human-readable text when stdout is a TTY; auto-JSON otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── mypy daemon probe ──────────────────────────────────────────────


def _mypy_cmd() -> list[str]:
    """Return the mypy command, preferring dmypy for warm-cache speed."""
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


def _devtools_cmd(*args: str) -> list[str]:
    """Run repository devtools from the current checkout, not a stale PATH wrapper."""
    return [sys.executable, "-m", "devtools", *args]


# ── resource preflight ─────────────────────────────────────────────


def _check_available_memory() -> tuple[int, int] | None:
    """Return (available_kb, total_kb) from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
    except OSError:
        return None
    avail = total = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1])
        elif line.startswith("MemTotal:"):
            total = int(line.split()[1])
    if avail is not None and total is not None:
        return avail, total
    return None


_MEM_WARN_GB = 2


def _warn_low_memory() -> None:
    mem = _check_available_memory()
    if mem is None:
        return
    avail_gb = mem[0] / (1024 * 1024)
    if avail_gb < _MEM_WARN_GB:
        sys.stderr.write(f"verify: low memory ({avail_gb:.1f} GB free) — pytest may be slow or OOM\n")


# ── completion notification ────────────────────────────────────────


def _notify(summary: str) -> None:
    """Send desktop notification if notify-send is available."""
    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "polylogue verify", summary],
            capture_output=True,
            timeout=5,
        )


def _format_completion_notification(
    *,
    exit_code: int,
    total_duration: float,
    step_results: list[dict[str, Any]],
) -> str:
    """Build the desktop notification summary for a completed verify run."""
    if exit_code == 0:
        msg = f"PASS ({total_duration:.0f}s)"
        pytest_step = next((s for s in step_results if str(s["name"]).startswith("pytest")), None)
        if pytest_step is not None and "count" in pytest_step:
            msg += f", {pytest_step['count']} tests"
        return msg
    failed = [s["name"] for s in step_results if s["exit"] != 0]
    return f"FAIL ({total_duration:.0f}s) — {', '.join(failed)}"


# ── history (JSONL) ────────────────────────────────────────────────


HISTORY_PATH = Path(".cache/verify-history.jsonl")
TESTMON_DATA = Path(".testmondata")
TESTMON_SEED_STAMP = Path(".cache/testmon/seed.json")
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
TESTMON_GLOBAL_INVALIDATORS = (
    ".python-version",
    "conftest.py",
    "noxfile.py",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "uv.lock",
    "tests/conftest.py",
    "tests/infra/",
)


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    entries.append(json.loads(line))
    return entries


def _save_history(entry: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _print_history(file: Path | None = None) -> None:
    """Print last 10 verify runs as a compact table."""
    entries = _load_history()
    if not entries:
        print("verify: no history yet")
        return
    print(f"{'time':<20} {'tier':<8} {'head':<10} {'dur':>7} {'exit':>4}  steps")
    print("-" * 75)
    for entry in entries[-10:]:
        ts = entry["timestamp"][5:19]  # MM-DD HH:MM
        tier = entry["tier"][:8]
        head = entry["git_head"][:8]
        dur = f"{entry['total_duration_s']:.0f}s"
        ec = entry["exit_code"]
        steps = ", ".join(f"{s['name']}({s['duration_s']:.0f}s{' FAIL' if s['exit'] else ''})" for s in entry["steps"])
        print(f"{ts:<20} {tier:<8} {head:<10} {dur:>7} {ec:>4}  {steps}")


# ── step runner ─────────────────────────────────────────────────────


_PYTEST_COUNT_RE = re.compile(
    r"\b(?P<count>\d+)\s+"
    r"(?P<status>passed|failed|error|errors|skipped|xfailed|xpassed|rerun|reruns)\b"
)


def _parse_pytest_test_count(output: str) -> int | None:
    """Return the total executed-test count from pytest's terminal summary.

    Used only as a fallback when the structured JSON report is missing or
    unreadable. The primary path is `_read_pytest_report()`.
    """
    if "no tests ran" in output:
        return 0
    counts = [int(match.group("count")) for match in _PYTEST_COUNT_RE.finditer(output)]
    if not counts:
        return None
    return sum(counts)


def _read_pytest_report(path: Path = PYTEST_REPORT_PATH) -> dict[str, Any] | None:
    """Load the structured pytest-json-report artifact, or None if absent/bad."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _pytest_metadata_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project a pytest-json-report dict into verify-step metadata."""
    summary = report.get("summary")
    metadata: dict[str, Any] = {"report_path": str(PYTEST_REPORT_PATH)}
    if isinstance(summary, dict):
        # `total` is collected count; sum the executed outcomes for the
        # number we previously scraped from the terminal.
        outcome_keys = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        executed = sum(int(summary.get(k, 0) or 0) for k in outcome_keys)
        metadata["count"] = executed
        for key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "total"):
            value = summary.get(key)
            if isinstance(value, int):
                metadata[key] = value
    duration = report.get("duration")
    if isinstance(duration, (int, float)):
        metadata["pytest_duration_s"] = round(float(duration), 2)
    return metadata


def _clear_pytest_report() -> None:
    """Remove a stale report before a pytest step runs."""
    with contextlib.suppress(FileNotFoundError):
        PYTEST_REPORT_PATH.unlink()


def _run(label: str, cmd: list[str], *, cwd: str | None = None) -> tuple[int, float, dict[str, Any]]:
    t0 = time.monotonic()
    sys.stderr.write(f"  {label} ... ")
    sys.stderr.flush()
    if label.startswith("pytest"):
        _clear_pytest_report()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    metadata: dict[str, Any] = {}
    if label.startswith("pytest"):
        report = _read_pytest_report()
        if report is not None:
            metadata.update(_pytest_metadata_from_report(report))
        else:
            # Fallback: terminal scraping when the structured report is
            # missing (pytest crashed before writing it, or the plugin is
            # disabled in some lab profile).
            fallback = _parse_pytest_test_count(result.stdout + "\n" + result.stderr)
            if fallback is not None:
                metadata["count"] = fallback
            metadata["report_path"] = None
    if result.returncode == 0:
        sys.stderr.write(f"ok ({elapsed:.1f}s)\n")
    else:
        sys.stderr.write(f"FAILED ({elapsed:.1f}s)\n")
        if result.stdout.strip():
            sys.stderr.write(result.stdout + "\n")
        if result.stderr.strip():
            sys.stderr.write(result.stderr + "\n")
    return result.returncode, elapsed, metadata


def _stop_after_failed_step(label: str) -> bool:
    return label.startswith("pytest") or label in {"lab scenario", "verify-slos"}


# ── step builder ────────────────────────────────────────────────────


def build_verify_steps(
    *,
    quick: bool,
    lab: bool,
    skip_slow: bool,
    commit: bool = False,
    seed_testmon: bool = False,
    full_pytest: bool = False,
    testmon_global: bool = False,
) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]

    if not commit:
        steps.extend(
            [
                ("render-all", _devtools_cmd("render-all", "--check")),
                ("verify-topology", _devtools_cmd("verify-topology")),
                ("verify-layering", _devtools_cmd("verify-layering")),
                ("verify-file-budgets", _devtools_cmd("verify-file-budgets")),
                (
                    "verify-provider-meta-policy",
                    _devtools_cmd("verify-provider-meta-policy"),
                ),
                ("verify-test-ownership", _devtools_cmd("verify-test-ownership")),
                ("verify-closure-matrix", _devtools_cmd("verify-closure-matrix")),
                ("verify-schema-roundtrip", _devtools_cmd("verify-schema-roundtrip", "--all")),
                (
                    "verify-suppressions",
                    _devtools_cmd("verify-suppressions", "--enforce-kinds", "type_ignore,noqa,no_cover"),
                ),
                ("verify-manifests", _devtools_cmd("verify-manifests")),
                ("verify-ci-workflows", _devtools_cmd("verify-ci-workflows")),
                ("verify-doc-commands", _devtools_cmd("verify-doc-commands")),
                ("verify-witness-lifecycle", _devtools_cmd("verify-witness-lifecycle")),
                # Recommended (not required): runs in soft mode so missing
                # witnesses on merged fix PRs are reported without blocking
                # ``devtools verify``. Nightly jobs invoke the command
                # without ``--soft`` for the hard gate. See issue #1227.
                (
                    "verify-witness-coverage",
                    _devtools_cmd("verify-witness-coverage", "--soft"),
                ),
                ("verify-lane-assertions", _devtools_cmd("verify-lane-assertions")),
                ("verify-test-infra-currency", _devtools_cmd("verify-test-infra-currency")),
                ("verify-test-clock-hygiene", _devtools_cmd("verify-test-clock-hygiene")),
            ]
        )

    steps.append(("verification-impact check", _devtools_cmd("verification-impact", "--full", "--check")))

    if not quick and not commit:
        _report_dir = Path(".cache/test-reports")
        _report_dir.mkdir(parents=True, exist_ok=True)
        PYTEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        # Scale-tier policy (issue #1183): default verify includes
        # ``scale_small`` but excludes ``scale_medium`` / ``scale_large``.
        # ``--lab`` lets the medium tier in; the large tier is reserved
        # for nightly CI and explicit ``devtools benchmark-campaign``
        # invocations.
        scale_marker_expr = "not scale_large" if lab else "not scale_medium and not scale_large"
        pytest_cmd = [
            "pytest",
            "-q",
            "--tb=short",
            "--ignore=tests/integration",
            "--durations=10",
            f"--junitxml={_report_dir}/verify-latest.xml",
            "--json-report",
            "--json-report-omit=collectors,log,streams,warnings",
            f"--json-report-file={PYTEST_REPORT_PATH}",
        ]
        if skip_slow:
            pytest_cmd.extend(["-m", f"not slow and {scale_marker_expr}"])
        else:
            pytest_cmd.extend(["-m", scale_marker_expr])
        if seed_testmon:
            pytest_cmd.extend(["--testmon", "--testmon-noselect", *_pytest_worker_args(default="16")])
            steps.append(("pytest seed-testmon", pytest_cmd))
        elif full_pytest:
            pytest_cmd.extend(_pytest_worker_args(default="16"))
            steps.append(("pytest full", pytest_cmd))
        elif testmon_global:
            pytest_cmd.extend(["--testmon", "--testmon-noselect", *_pytest_worker_args(default="16")])
            steps.append(("pytest testmon-global", pytest_cmd))
        else:
            pytest_cmd.extend(["--testmon", "-n", "0"])
            if skip_slow:
                pytest_cmd.append("--testmon-forceselect")
            steps.append(("pytest testmon", pytest_cmd))

    if lab:
        steps.append(("lab scenario", _devtools_cmd("lab-scenario", "run", "archive-smoke", "--tier", "0")))
        steps.append(("verify-slos", _devtools_cmd("verify-slos", "--include-lab")))
        steps.append(("verify-schema-upgrade-lane", _devtools_cmd("verify-schema-upgrade-lane")))
    return steps


# ── comparison against last run ────────────────────────────────────


def _compare_against_last(step_results: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable regression flags."""
    entries = _load_history()
    if len(entries) < 1:
        return []
    last = entries[-1]
    last_steps = {s["name"]: s["duration_s"] for s in last.get("steps", [])}
    flags: list[str] = []
    for s in step_results:
        prev = last_steps.get(s["name"])
        if prev is not None and prev > 0:
            delta = s["duration_s"] - prev
            pct = (delta / prev) * 100
            if pct > 50 and delta > 5.0:
                flags.append(
                    f"  {s['name']}: {s['duration_s']:.1f}s "
                    f"(+{delta:.0f}s, +{pct:.0f}% vs last — unexpected regression?)"
                )
    return flags


# ── structured output ───────────────────────────────────────────────


def _print_json(result: dict[str, Any]) -> None:
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


# ── stamp ───────────────────────────────────────────────────────────


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _stamp_head() -> None:
    head = _git_head()
    if head is None:
        return
    stamp_dir = Path(".cache")
    stamp_dir.mkdir(parents=True, exist_ok=True)
    (stamp_dir / "last-verify-head").write_text(head + "\n")


def _file_fingerprint(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return "unreadable"
    return h.hexdigest()


def _pytest_worker_args(*, default: str) -> list[str]:
    workers = os.environ.get("POLYLOGUE_PYTEST_WORKERS", default).strip() or default
    return ["-n", workers]


def _read_testmon_seed_stamp() -> dict[str, Any] | None:
    try:
        raw: object = json.loads(TESTMON_SEED_STAMP.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _git_changed_paths(base: str | None) -> set[str] | None:
    changed: set[str] = set()
    commands: list[list[str]] = []
    if base:
        commands.append(["git", "diff", "--name-only", f"{base}..HEAD"])
    commands.append(["git", "diff", "--name-only", "HEAD"])
    commands.append(["git", "ls-files", "--others", "--exclude-standard"])

    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        changed.update(line for line in result.stdout.splitlines() if line.strip())
    return changed


def _is_testmon_global_invalidator(path: str) -> bool:
    if not path:
        return False
    if path.endswith((".toml", ".ini", ".cfg")) and (
        path.startswith("tests/") or path in {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"}
    ):
        return True
    return any(path == invalidator or path.startswith(invalidator) for invalidator in TESTMON_GLOBAL_INVALIDATORS)


def _testmon_requires_global_collection() -> bool:
    seed = _read_testmon_seed_stamp()
    base = seed.get("git_head") if seed else None
    changed = _git_changed_paths(base if isinstance(base, str) else None)
    if changed is None:
        return True
    return any(_is_testmon_global_invalidator(path) for path in changed)


def _testmon_preflight(*, seed_testmon: bool, full_pytest: bool, quick: bool, commit: bool) -> str | None:
    if quick or commit or seed_testmon or full_pytest:
        return None
    if TESTMON_DATA.exists() and TESTMON_SEED_STAMP.exists():
        return None
    return (
        "verify: pytest-testmon is not seeded; run `devtools verify --seed-testmon` "
        "to create .testmondata and .cache/testmon/seed.json before using the default affected-test path.\n"
    )


def _write_testmon_seed_stamp(result: dict[str, Any]) -> None:
    TESTMON_SEED_STAMP.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": result.get("git_head"),
        "testmon_data": _file_fingerprint(TESTMON_DATA),
        "steps": result.get("steps", []),
    }
    TESTMON_SEED_STAMP.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# ── main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--seed-testmon",
        action="store_true",
        help="Run full non-integration pytest with --testmon-noselect to seed/update .testmondata.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Force the full non-integration pytest diagnostic instead of testmon."
    )
    parser.add_argument(
        "--full", action="store_true", help="Alias for --all: run full non-integration pytest diagnostic."
    )
    parser.add_argument(
        "--commit", action="store_true", help="Pre-commit tier: format + lint + mypy + verification-impact check only."
    )
    parser.add_argument(
        "--skip-slow", action="store_true", help="Exclude @pytest.mark.slow tests from the pytest step."
    )
    parser.add_argument(
        "--lab",
        action="store_true",
        help=(
            "Run the default pytest-testmon baseline plus verification-lab "
            "scenario and verify-slos checks; does not imply --all."
        ),
    )
    parser.add_argument("--history", action="store_true", help="Print last 10 verify runs and exit.")
    parser.add_argument("--json", action="store_true", default=None, help="Write structured JSON to stdout.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.history:
        _print_history()
        return 0

    # Auto-detect JSON when stdout is not a TTY (agent/pipe context).
    use_json = args.json if args.json is not None else not sys.stdout.isatty()

    tier = "full"
    if args.commit:
        tier = "commit"
    elif args.quick:
        tier = "quick"
    elif args.seed_testmon:
        tier = "seed-testmon"
    elif args.all or args.full:
        tier = "full"
    elif args.lab:
        tier = "lab"
    else:
        tier = "testmon"

    full_pytest = bool(args.all or args.full)
    preflight_error = _testmon_preflight(
        seed_testmon=bool(args.seed_testmon),
        full_pytest=full_pytest,
        quick=bool(args.quick),
        commit=bool(args.commit),
    )
    if preflight_error is not None:
        sys.stderr.write(preflight_error)
        return 2

    testmon_global = (
        not (args.quick or args.commit or args.seed_testmon or full_pytest) and _testmon_requires_global_collection()
    )

    head = _git_head()
    t0 = time.monotonic()

    if not use_json:
        sys.stderr.write("verify: running local verification baseline\n")

    # Resource preflight before heavy steps.
    if not args.quick and not args.commit:
        _warn_low_memory()

    exit_code = 0
    steps = build_verify_steps(
        quick=bool(args.quick),
        commit=bool(args.commit),
        lab=bool(args.lab),
        skip_slow=bool(args.skip_slow),
        seed_testmon=bool(args.seed_testmon),
        full_pytest=full_pytest,
        testmon_global=testmon_global,
    )

    step_results: list[dict[str, Any]] = []

    for label, cmd in steps:
        if label.startswith("pytest"):
            _warn_low_memory()  # check again right before the heavy step
        rc, elapsed, metadata = _run(label, cmd)
        step_result: dict[str, Any] = {"name": label, "duration_s": round(elapsed, 2), "exit": rc}
        step_result.update(metadata)
        step_results.append(step_result)
        if rc != 0:
            exit_code = rc
            if _stop_after_failed_step(label):
                break

    total_duration = round(time.monotonic() - t0, 2)

    # Build history entry.
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "tier": tier,
        "steps": step_results,
        "total_duration_s": total_duration,
        "exit_code": exit_code,
    }

    if use_json:
        _print_json(history_entry)
    else:
        if exit_code == 0:
            # Compare against last run, flag regressions.
            flags = _compare_against_last(step_results)
            sys.stderr.write(f"\nverify: all checks passed ({total_duration:.1f}s total)")
            if flags:
                sys.stderr.write(" — " + "; ".join(flags) if len(flags) == 1 else "")
                sys.stderr.write("\n")
                for flag in flags:
                    sys.stderr.write(flag + "\n")
            else:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(f"\nverify: FAILED ({total_duration:.1f}s) — fix before pushing\n")

    # Persist history and stamp.
    _save_history(history_entry)
    if exit_code == 0:
        _stamp_head()
        if args.seed_testmon or testmon_global:
            _write_testmon_seed_stamp(history_entry)

    # Notify on long-running verify.
    if total_duration > 30:
        _notify(
            _format_completion_notification(
                exit_code=exit_code,
                total_duration=total_duration,
                step_results=step_results,
            )
        )

    return exit_code
