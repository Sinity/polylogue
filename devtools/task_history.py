"""Agent-visible task execution history.

Maintains an append-only JSONL log of task executions under
``.agent/task-history/tasks.jsonl`` for use by agents and operators.

Subcommands:

- ``log`` — Append a structured task record.
- ``recent`` — Show the N most recent tasks.
- ``stats`` — Aggregate summary over all recorded tasks.
- ``replay`` — Re-run a previously logged ``devtools`` invocation.
- ``budget`` — Enforce per-class p95 latency budgets.
- ``prune`` — Bound the on-disk JSONL log size.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root
from devtools.verify_runs import CURRENT_RUN_PATH
from polylogue.core.json import JSONDocument

TaskRecord = JSONDocument


# ---------------------------------------------------------------------------
# Storage helpers (path-injectable for tests)
# ---------------------------------------------------------------------------


def task_history_file_path() -> Path:
    """Return the active task-history JSONL path.

    Honors ``POLYLOGUE_TASK_HISTORY_FILE`` (used by tests and one-off overrides);
    otherwise defaults to ``<repo_root>/.agent/task-history/tasks.jsonl``.
    """
    override = os.environ.get("POLYLOGUE_TASK_HISTORY_FILE")
    if override:
        return Path(override)
    return _get_root() / ".agent" / "task-history" / "tasks.jsonl"


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _read_tasks(path: Path | None = None) -> list[TaskRecord]:
    target = path or task_history_file_path()
    _ensure_file(target)
    tasks: list[TaskRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip malformed lines
    return tasks


def _append_task(task: TaskRecord, path: Path | None = None) -> None:
    target = path or task_history_file_path()
    _ensure_file(target)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(task, sort_keys=True) + "\n")


def _latest_verify_run_metadata(command: str) -> dict[str, Any]:
    if command not in {"verify", "test"}:
        return {}
    path = _get_root() / CURRENT_RUN_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    latest_pytest = next(
        (
            step
            for step in reversed(payload.get("steps", []))
            if isinstance(step, dict) and str(step.get("name", "")).startswith("pytest")
        ),
        {},
    )
    metadata: dict[str, Any] = {
        "verify_run_id": payload.get("run_id"),
        "verify_artifact_dir": payload.get("artifact_dir"),
        "verify_status": payload.get("status"),
        "verify_diagnosis": payload.get("diagnosis"),
    }
    if isinstance(latest_pytest, dict):
        for key in (
            "diagnosis",
            "selected_count",
            "deselected_count",
            "count",
            "peak_tree_rss_mb",
            "peak_tree_pss_mb",
            "peak_process_count",
            "resource_sample_count",
        ):
            if key in latest_pytest:
                metadata[f"pytest_{key}"] = latest_pytest[key]
    return {key: value for key, value in metadata.items() if value is not None}


# ---------------------------------------------------------------------------
# Command-class taxonomy
# ---------------------------------------------------------------------------


_CLASS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("verify", "verify"),
    ("render", "render"),
    ("release build-package", "render"),
    ("lab", "lab"),
    ("witness", "witness"),
    ("lab lanes", "verify"),
    ("bench mutation", "campaign"),
    ("bench campaign", "campaign"),
    ("bench synthetic", "campaign"),
    ("schema", "verify"),
    ("evidence", "query"),
    ("lab graph", "query"),
    ("lab probe pipeline", "query"),
    ("bench memory", "query"),
    ("lab probe capture-regression", "query"),
    ("lab projections", "query"),
    ("verify coverage", "verify"),
    ("workspace tasks", "query"),
    ("workspace failure-context", "query"),
    ("workspace worktree-gc", "query"),
    ("status", "query"),
)


def classify_command(command_name: str) -> str:
    """Classify a devtools command name into a coarse class bucket.

    Classes: ``verify``, ``render``, ``lab``, ``witness``, ``campaign``,
    ``query``, ``other``.
    """
    if not command_name:
        return "other"
    for prefix, klass in _CLASS_PREFIXES:
        if command_name == prefix or command_name.startswith(f"{prefix} "):
            return klass
        head = command_name.split()[0]
        if head == prefix or head.startswith(f"{prefix}-") or head.startswith(f"{prefix}_"):
            return klass
    return "other"


# ---------------------------------------------------------------------------
# Auto-log entrypoint (called from click_dispatch.main)
# ---------------------------------------------------------------------------


def record_invocation(
    *,
    command: str,
    args: list[str],
    duration_ms: float,
    exit_code: int,
    cwd: str | None = None,
    path: Path | None = None,
) -> None:
    """Append a single invocation record; never raise.

    Intended for the ``devtools`` harness wrapper. Failures are swallowed
    so a broken task history path does not prevent commands from returning.
    """
    try:
        task: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "args": list(args),
            "duration_ms": round(float(duration_ms), 3),
            "exit_code": int(exit_code),
            "cwd": cwd if cwd is not None else os.getcwd(),
            "class": classify_command(command),
        }
        task.update(_latest_verify_run_metadata(command))
        _append_task(task, path=path)
    except Exception:
        # Auto-log must never crash the wrapped command.
        return


def auto_log_disabled() -> bool:
    """Return True when auto-logging is suppressed via env."""
    return os.environ.get("POLYLOGUE_TASK_HISTORY_DISABLE", "") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Subcommand: log
# ---------------------------------------------------------------------------


def _cmd_log(args: argparse.Namespace) -> int:
    task: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": args.command or "",
    }
    if args.duration_ms is not None:
        task["duration_ms"] = args.duration_ms
    if args.exit_code is not None:
        task["exit_code"] = args.exit_code
    if args.tags:
        task["tags"] = args.tags
    if args.cwd is not None:
        task["cwd"] = str(args.cwd)
    if args.note is not None:
        task["note"] = args.note
    if args.command:
        task["class"] = classify_command(args.command)

    _append_task(task)
    if args.json:
        print(json.dumps(task, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: recent
# ---------------------------------------------------------------------------


def _cmd_recent(args: argparse.Namespace) -> int:
    tasks = _read_tasks()
    recent = tasks[-args.count :] if args.count else tasks
    if args.json:
        print(json.dumps(recent, indent=2, sort_keys=True))
        return 0
    if not recent:
        print("no tasks recorded")
        return 0
    for idx, task in enumerate(reversed(recent), start=1):
        ts: str = task.get("timestamp", "?")  # type: ignore[assignment]
        cmd: str = task.get("command", "?")  # type: ignore[assignment]
        code = task.get("exit_code")
        dur = task.get("duration_ms")
        parts: list[str] = [f"#{idx}", ts, cmd]
        if code is not None:
            parts.append(f"exit={code}")
        if dur is not None:
            parts.append(f"{dur}ms")
        if task.get("verify_run_id"):
            parts.append(f"run={task['verify_run_id']}")
        if task.get("verify_diagnosis") or task.get("pytest_diagnosis"):
            parts.append(f"diagnosis={task.get('verify_diagnosis') or task.get('pytest_diagnosis')}")
        if task.get("pytest_peak_tree_rss_mb") is not None:
            parts.append(f"rss_peak={task['pytest_peak_tree_rss_mb']}MiB")
        print("  ".join(parts))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Return the linear-interpolated percentile of a numeric list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = k - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def _class_distributions(tasks: list[TaskRecord]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for task in tasks:
        dur = task.get("duration_ms")
        if dur is None:
            continue
        klass = task.get("class") or classify_command(str(task.get("command", "")))
        buckets.setdefault(str(klass), []).append(float(dur))  # type: ignore[arg-type]
    return {
        klass: {
            "count": float(len(durations)),
            "median_ms": _percentile(durations, 50),
            "p95_ms": _percentile(durations, 95),
            "max_ms": float(max(durations)),
            "sum_ms": float(sum(durations)),
        }
        for klass, durations in buckets.items()
    }


def _phase_duration(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    duration = value.get("duration")
    return float(duration) if isinstance(duration, (int, float)) else 0.0


def _latest_pytest_slow_tests(limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    report_path = _get_root() / ".cache" / "verify" / "last-pytest.json"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tests = payload.get("tests")
    if not isinstance(tests, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        nodeid = item.get("nodeid")
        if not isinstance(nodeid, str):
            continue
        setup_s = _phase_duration(item.get("setup"))
        call_s = _phase_duration(item.get("call"))
        teardown_s = _phase_duration(item.get("teardown"))
        total_s = setup_s + call_s + teardown_s
        if total_s <= 0:
            continue
        rows.append(
            {
                "nodeid": nodeid,
                "total_s": round(total_s, 4),
                "setup_s": round(setup_s, 4),
                "call_s": round(call_s, 4),
                "teardown_s": round(teardown_s, 4),
                "outcome": item.get("outcome"),
            }
        )
    rows.sort(key=lambda row: float(row["total_s"]), reverse=True)
    return rows[:limit]


def _cmd_stats(args: argparse.Namespace) -> int:
    tasks = _read_tasks()
    if not tasks:
        if args.json:
            print(json.dumps({"total": 0, "by_command": {}, "by_exit_code": {}}, indent=2))
        else:
            print("no tasks recorded")
        return 0

    total = len(tasks)
    by_command: dict[str, int] = {}
    by_exit_code: dict[str, int] = {}
    total_duration_ms: float = 0.0
    duration_count = 0

    for task in tasks:
        cmd: str = task.get("command", "(unknown)")  # type: ignore[assignment]
        by_command[cmd] = by_command.get(cmd, 0) + 1
        code = task.get("exit_code")
        code_key: str = str(code) if code is not None else "(none)"
        by_exit_code[code_key] = by_exit_code.get(code_key, 0) + 1
        dur = task.get("duration_ms")
        if dur is not None:
            total_duration_ms += float(dur)  # type: ignore[arg-type]
            duration_count += 1

    stats: dict[str, Any] = {
        "total": total,
        "by_command": by_command,
        "by_exit_code": by_exit_code,
    }
    if duration_count > 0:
        stats["total_duration_ms"] = total_duration_ms
        stats["avg_duration_ms"] = total_duration_ms / duration_count

    if args.by_class:
        stats["by_class"] = _class_distributions(tasks)

    peaks: list[float] = []
    for task in tasks:
        value = task.get("pytest_peak_tree_rss_mb")
        if isinstance(value, (int, float)):
            peaks.append(float(value))
    if args.resources and peaks:
        stats["resources"] = {
            "count": len(peaks),
            "peak_rss_mb_max": max(peaks),
            "peak_rss_mb_p95": _percentile(peaks, 95),
        }
    slow_tests = _latest_pytest_slow_tests(args.slow_tests)
    if slow_tests:
        stats["slow_tests"] = slow_tests

    slowest: list[TaskRecord] = []
    if args.slowest and args.slowest > 0:
        timed = [t for t in tasks if t.get("duration_ms") is not None]
        timed.sort(key=lambda t: float(t.get("duration_ms", 0)), reverse=True)  # type: ignore[arg-type]
        slowest = timed[: args.slowest]
        stats["slowest"] = slowest

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0

    print(f"total tasks: {total}")
    print(f"by command ({len(by_command)}):")
    for cmd, count in sorted(by_command.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}x  {cmd}")
    print(f"by exit code ({len(by_exit_code)}):")
    for code_key, count in sorted(by_exit_code.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}x  {code_key}")
    if duration_count > 0:
        print(f"total duration: {total_duration_ms:.0f}ms")
        print(f"avg duration:   {stats['avg_duration_ms']:.0f}ms")
    if args.by_class:
        print(f"by class ({len(stats['by_class'])}):")
        for klass, dist in sorted(stats["by_class"].items()):
            print(
                f"  {klass:<10} n={int(dist['count']):>4}  "
                f"median={dist['median_ms']:.0f}ms  p95={dist['p95_ms']:.0f}ms  "
                f"max={dist['max_ms']:.0f}ms"
            )
    if args.resources and "resources" in stats:
        res = stats["resources"]
        print(
            f"resources: n={res['count']} "
            f"peak_rss_max={res['peak_rss_mb_max']:.1f}MiB "
            f"peak_rss_p95={res['peak_rss_mb_p95']:.1f}MiB"
        )
    if slow_tests:
        print(f"slow tests from latest pytest report ({len(slow_tests)}):")
        for test in slow_tests:
            print(
                f"  {test['total_s']:.2f}s "
                f"setup={test['setup_s']:.2f}s "
                f"call={test['call_s']:.2f}s "
                f"teardown={test['teardown_s']:.2f}s "
                f"{test['nodeid']}"
            )
    if slowest:
        print(f"slowest {len(slowest)}:")
        for task in slowest:
            cmd_str: str = task.get("command", "?")  # type: ignore[assignment]
            dur = task.get("duration_ms")
            ts: str = task.get("timestamp", "?")  # type: ignore[assignment]
            print(f"  {dur}ms  {cmd_str}  {ts}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: replay
# ---------------------------------------------------------------------------


def _cmd_replay(args: argparse.Namespace) -> int:
    tasks = _read_tasks()
    if not tasks:
        print("no tasks recorded", file=sys.stderr)
        return 1
    # ``index`` is 1-based from the most recent task (replay 1 = last task).
    index = max(1, args.index)
    if index > len(tasks):
        print(
            f"replay index {index} exceeds {len(tasks)} recorded tasks",
            file=sys.stderr,
        )
        return 1
    task = tasks[-index]

    command_name: str = str(task.get("command", "")).strip()
    raw_args = task.get("args", [])
    invocation_args: list[str] = [str(item) for item in raw_args] if isinstance(raw_args, list) else []

    if not command_name:
        print("task has no recorded command name", file=sys.stderr)
        return 1

    argv = [command_name, *invocation_args]
    if args.dry_run or args.json:
        payload = {
            "index": index,
            "timestamp": task.get("timestamp"),
            "command": command_name,
            "args": invocation_args,
            "argv": argv,
            "cwd": task.get("cwd"),
            "previous_exit_code": task.get("exit_code"),
            "previous_duration_ms": task.get("duration_ms"),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        if args.dry_run:
            return 0

    # Re-run via the devtools entrypoint as a subprocess so the inner run
    # gets its own auto-log record without recursing inside this process.
    env = os.environ.copy()
    cwd = task.get("cwd") if isinstance(task.get("cwd"), str) else None
    if cwd is not None and not Path(cwd).exists():
        cwd = None
    cmd = [sys.executable, "-m", "devtools", *argv]
    try:
        completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    except FileNotFoundError as exc:  # python missing — surface clearly
        print(f"replay failed: {exc}", file=sys.stderr)
        return 1
    return int(completed.returncode)


# ---------------------------------------------------------------------------
# Subcommand: budget
# ---------------------------------------------------------------------------


def _cmd_budget(args: argparse.Namespace) -> int:
    tasks = _read_tasks()
    distributions = _class_distributions(tasks)
    target_class = args.class_name
    dist = distributions.get(target_class)
    if dist is None or dist["count"] == 0:
        payload = {
            "class": target_class,
            "count": 0,
            "max_ms": args.max_ms,
            "status": "no-data",
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"no data for class {target_class}")
        return 0 if args.allow_empty else 1

    p95 = dist["p95_ms"]
    within = p95 <= float(args.max_ms)
    payload = {
        "class": target_class,
        "count": int(dist["count"]),
        "median_ms": dist["median_ms"],
        "p95_ms": p95,
        "max_ms_observed": dist["max_ms"],
        "budget_ms": float(args.max_ms),
        "within_budget": within,
        "status": "ok" if within else "over-budget",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        symbol = "OK" if within else "OVER"
        print(
            f"[{symbol}] class={target_class} n={int(dist['count'])} "
            f"p95={p95:.0f}ms budget={args.max_ms:.0f}ms "
            f"median={dist['median_ms']:.0f}ms max={dist['max_ms']:.0f}ms"
        )
    return 0 if within else 2


# ---------------------------------------------------------------------------
# Subcommand: prune
# ---------------------------------------------------------------------------


def _cmd_prune(args: argparse.Namespace) -> int:
    tasks = _read_tasks()
    if args.keep < 0:
        print("--keep must be >= 0", file=sys.stderr)
        return 2
    before = len(tasks)
    if args.keep >= before:
        payload = {"before": before, "after": before, "removed": 0, "keep": args.keep}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"nothing to prune (have {before}, keep {args.keep})")
        return 0
    keep_tasks = tasks[-args.keep :] if args.keep > 0 else []
    path = task_history_file_path()
    _ensure_file(path)
    # Atomic-ish rewrite: write to a sibling temp file then replace.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for task in keep_tasks:
            fh.write(json.dumps(task, sort_keys=True) + "\n")
    tmp.replace(path)
    removed = before - len(keep_tasks)
    payload = {"before": before, "after": len(keep_tasks), "removed": removed, "keep": args.keep}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"pruned {removed} records (kept {len(keep_tasks)} of {before})")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devtools workspace tasks", description="Task execution history.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    log_parser = subparsers.add_parser("log", help="Append a task record.")
    log_parser.add_argument("--command", "-c", default=None, help="Command that was run.")
    log_parser.add_argument("--duration-ms", type=float, default=None, help="Duration in milliseconds.")
    log_parser.add_argument("--exit-code", type=int, default=None, help="Exit code.")
    log_parser.add_argument("--tags", action="append", default=None, help="Tags for the task.")
    log_parser.add_argument("--cwd", default=None, help="Working directory.")
    log_parser.add_argument("--note", default=None, help="Free-text note.")
    log_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    recent_parser = subparsers.add_parser("recent", help="Show recent tasks.")
    recent_parser.add_argument("--count", "-n", type=int, default=10, help="Number of recent tasks (default: 10).")
    recent_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    stats_parser = subparsers.add_parser("stats", help="Task statistics.")
    stats_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    stats_parser.add_argument(
        "--by-class",
        action="store_true",
        help="Add per-class duration distributions (median/p95/max).",
    )
    stats_parser.add_argument(
        "--slowest",
        type=int,
        default=0,
        metavar="N",
        help="Include the N slowest invocations by duration_ms.",
    )
    stats_parser.add_argument(
        "--resources",
        action="store_true",
        help="Add pytest resource distributions when verify/test records carry them.",
    )
    stats_parser.add_argument(
        "--slow-tests",
        type=int,
        default=0,
        metavar="N",
        help="Include the N slowest tests from .cache/verify/last-pytest.json.",
    )

    replay_parser = subparsers.add_parser("replay", help="Re-run the Nth most recent task (default 1).")
    replay_parser.add_argument(
        "index", type=int, nargs="?", default=1, help="1-based offset from most recent (default 1)."
    )
    replay_parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved invocation without executing."
    )
    replay_parser.add_argument(
        "--json", action="store_true", help="Emit the resolved invocation as JSON before running."
    )

    budget_parser = subparsers.add_parser("budget", help="Enforce a p95 latency budget for a command class.")
    budget_parser.add_argument(
        "--class",
        dest="class_name",
        required=True,
        help="Command class to evaluate (verify, render, lab, witness, campaign, query, other).",
    )
    budget_parser.add_argument("--max-ms", type=float, required=True, help="Budget ceiling in milliseconds (p95).")
    budget_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Treat missing data for the class as a pass rather than a failure.",
    )
    budget_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    prune_parser = subparsers.add_parser("prune", help="Bound the JSONL log size.")
    prune_parser.add_argument("--keep", type=int, required=True, help="Number of most recent records to retain.")
    prune_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    parsed = parser.parse_args(argv)

    if parsed.subcommand == "log":
        return _cmd_log(parsed)
    elif parsed.subcommand == "recent":
        return _cmd_recent(parsed)
    elif parsed.subcommand == "stats":
        return _cmd_stats(parsed)
    elif parsed.subcommand == "replay":
        return _cmd_replay(parsed)
    elif parsed.subcommand == "budget":
        return _cmd_budget(parsed)
    elif parsed.subcommand == "prune":
        return _cmd_prune(parsed)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
