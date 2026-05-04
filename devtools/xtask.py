"""Agent-visible task execution history (xtask).

Maintains an append-only JSONL log of task executions under
``.agent/xtask/tasks.jsonl`` for use by agents and operators.

Subcommands:

- ``log`` — Append a structured task record.
- ``recent`` — Show the N most recent tasks.
- ``stats`` — Aggregate summary over all recorded tasks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDocument

XTaskDict = JSONDocument

_XTASK_DIR = Path(__file__).resolve().parent.parent / ".agent" / "xtask"
_XTASK_FILE = _XTASK_DIR / "tasks.jsonl"


def _ensure_file() -> None:
    _XTASK_DIR.mkdir(parents=True, exist_ok=True)
    if not _XTASK_FILE.exists():
        _XTASK_FILE.write_text("", encoding="utf-8")


def _read_tasks() -> list[XTaskDict]:
    _ensure_file()
    tasks: list[XTaskDict] = []
    for line in _XTASK_FILE.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip malformed lines
    return tasks


def _append_task(task: XTaskDict) -> None:
    _ensure_file()
    with _XTASK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(task, sort_keys=True) + "\n")


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
    for task in reversed(recent):
        ts: str = task.get("timestamp", "?")  # type: ignore[assignment]
        cmd: str = task.get("command", "?")  # type: ignore[assignment]
        code = task.get("exit_code")
        dur = task.get("duration_ms")
        parts: list[str] = [ts, cmd]
        if code is not None:
            parts.append(f"exit={code}")
        if dur is not None:
            parts.append(f"{dur}ms")
        print("  ".join(parts))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


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
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devtools xtask", description="Task execution history.")
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

    parsed = parser.parse_args(argv)

    if parsed.subcommand == "log":
        return _cmd_log(parsed)
    elif parsed.subcommand == "recent":
        return _cmd_recent(parsed)
    elif parsed.subcommand == "stats":
        return _cmd_stats(parsed)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
