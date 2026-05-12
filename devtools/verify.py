"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy + proof-pack (~3s warm).
  --quick    Pre-push tier: all non-pytest gates (~15s warm).
  (default)  Full baseline: all gates + pytest (~3 min).
  --lab      Full baseline + verification-lab scenario checks.

Output formats:
  --json     Machine-readable JSON to stdout (human progress to stderr).
  (default)  Human-readable text when stdout is a TTY; auto-JSON otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import json
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


# ── history (JSONL) ────────────────────────────────────────────────


HISTORY_PATH = Path(".cache/verify-history.jsonl")


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


def _run(label: str, cmd: list[str], *, cwd: str | None = None) -> tuple[int, float]:
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
    return result.returncode, elapsed


# ── step builder ────────────────────────────────────────────────────


def build_verify_steps(*, quick: bool, lab: bool, skip_slow: bool, commit: bool = False) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]

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

    steps.append(("proof-pack check", ["devtools", "proof-pack", "--check"]))

    if not quick and not commit:
        pytest_cmd = ["pytest", "-q", "--tb=short", "--ignore=tests/integration"]
        if skip_slow:
            pytest_cmd.extend(["-m", "not slow"])
        steps.append(("pytest", pytest_cmd))

    if lab:
        steps.append(("lab scenario", ["devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"]))
        steps.append(("verify-slos", ["devtools", "verify-slos"]))
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


# ── worktree-fingerprint result cache ──────────────────────────────

RESULT_CACHE = Path(".cache/last-verify-result.json")


def _worktree_fingerprint() -> str:
    """Return a hash of HEAD + dirty file paths so unchanged worktrees
    skip redundant verification entirely."""
    head = _git_head() or ""
    diff = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
    )
    diff_files = diff.stdout.strip()
    unstaged = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    untracked = unstaged.stdout.strip()
    payload = f"{head}\n{diff_files}\n{untracked}"
    return _hash_text(payload)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _read_cached_result(fp: str) -> dict[str, Any] | None:
    """Return the cached result if the worktree fingerprint matches, else None."""
    if not RESULT_CACHE.exists():
        return None
    try:
        cached = json.loads(RESULT_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("fingerprint") == fp:
        return cached.get("result")
    return None


def _write_cached_result(fp: str, result: dict[str, Any]) -> None:
    RESULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE.write_text(json.dumps({"fingerprint": fp, "result": result}, ensure_ascii=False) + "\n")


# ── main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--commit", action="store_true", help="Pre-commit tier: format + lint + mypy + proof-pack only."
    )
    parser.add_argument(
        "--skip-slow", action="store_true", help="Exclude @pytest.mark.slow tests from the pytest step."
    )
    parser.add_argument(
        "--lab", action="store_true", help="Delegate additional domain proof checks through verification-lab."
    )
    parser.add_argument("--history", action="store_true", help="Print last 10 verify runs and exit.")
    parser.add_argument("--json", action="store_true", default=None, help="Write structured JSON to stdout.")
    parser.add_argument("--force", action="store_true", help="Bypass worktree-fingerprint cache and re-verify.")
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
    elif args.lab:
        tier = "lab"

    # Worktree-fingerprint cache: if nothing changed since last verify,
    # replay the cached result instantly instead of re-running tests.
    if not args.commit and not args.quick and not args.lab and not args.force:
        fp = _worktree_fingerprint()
        cached = _read_cached_result(fp)
        if cached is not None:
            ec = cached.get("exit_code", 0)
            dur = cached.get("total_duration_s", 0)
            if use_json:
                _print_json(cached)
            elif ec == 0:
                sys.stderr.write(f"verify: HEAD already verified ({dur:.0f}s cached); nothing to do.\n")
            else:
                failed = [s["name"] for s in cached.get("steps", []) if s["exit"] != 0]
                sys.stderr.write(f"verify: HEAD unchanged — cached failures: {', '.join(failed)}\n")
            return ec

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
    )

    step_results: list[dict[str, Any]] = []

    for label, cmd in steps:
        if label == "pytest":
            _warn_low_memory()  # check again right before the heavy step
        rc, elapsed = _run(label, cmd)
        step_results.append({"name": label, "duration_s": round(elapsed, 2), "exit": rc})
        if rc != 0:
            exit_code = rc

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

    # Persist history, stamp, and worktree-fingerprint cache.
    _save_history(history_entry)
    if exit_code == 0:
        _stamp_head()
    # Cache the result keyed by worktree fingerprint — next invocation
    # at the same tree state replays this instantly.
    if not args.commit and not args.quick and not args.lab:
        fp = _worktree_fingerprint()
        _write_cached_result(fp, history_entry)

    # Notify on long-running verify.
    if total_duration > 30:
        if exit_code == 0:
            test_count = next((s for s in step_results if s["name"] == "pytest"), None)
            msg = f"PASS ({total_duration:.0f}s)"
            if test_count:
                msg += f", {test_count.get('count', '?')} tests"
            _notify(msg)
        else:
            failed = [s["name"] for s in step_results if s["exit"] != 0]
            _notify(f"FAIL ({total_duration:.0f}s) — {', '.join(failed)}")

    return exit_code
