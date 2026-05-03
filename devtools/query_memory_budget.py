"""Run a command under an explicit RSS budget and emit a machine-readable summary.

Usage:
    devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import TypedDict


class MemoryBudgetResult(TypedDict):
    command: list[str]
    exit_code: int
    max_rss_mb: int
    peak_parent_rss_mb: float
    peak_rss_mb: float
    within_budget: bool


def _read_vm_rss_kb(pid: int) -> int:
    """Read VmRSS from /proc for a running process."""
    status_path = Path("/proc") / str(pid) / "status"
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except FileNotFoundError:
        return 0
    return 0


def _read_child_pids(pid: int) -> list[int]:
    """Read direct child process IDs from procfs."""
    children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        child_text = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not child_text:
        return []
    return [int(value) for value in child_text.split()]


def _process_tree_pids(root_pid: int) -> list[int]:
    """Return live process IDs under ``root_pid``, including the root."""
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(_read_child_pids(pid))
    return list(seen)


def _read_process_tree_rss_kb(pid: int) -> int:
    """Read aggregate current RSS for a process tree."""
    return sum(_read_vm_rss_kb(tree_pid) for tree_pid in _process_tree_pids(pid))


def run_memory_budget(command: list[str], *, max_rss_mb: int, poll_interval_s: float = 0.05) -> MemoryBudgetResult:
    """Execute a command and track peak process-tree RSS."""
    proc = subprocess.Popen(command)
    peak_parent_rss_kb = 0
    peak_rss_kb = 0

    while proc.poll() is None:
        peak_parent_rss_kb = max(peak_parent_rss_kb, _read_vm_rss_kb(proc.pid))
        peak_rss_kb = max(peak_rss_kb, _read_process_tree_rss_kb(proc.pid))
        time.sleep(poll_interval_s)

    peak_parent_rss_kb = max(peak_parent_rss_kb, _read_vm_rss_kb(proc.pid))
    peak_rss_kb = max(peak_rss_kb, _read_process_tree_rss_kb(proc.pid))
    exit_code = int(proc.returncode or 0)
    peak_parent_rss_mb = round(peak_parent_rss_kb / 1024, 1)
    peak_rss_mb = round(peak_rss_kb / 1024, 1)

    return {
        "command": command,
        "exit_code": exit_code,
        "max_rss_mb": int(max_rss_mb),
        "peak_parent_rss_mb": peak_parent_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "within_budget": bool(exit_code == 0 and peak_rss_mb <= max_rss_mb),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rss-mb",
        type=int,
        required=True,
        help="Maximum allowed peak RSS in MiB",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute after '--'",
    )
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after '--'")

    result = run_memory_budget(command, max_rss_mb=args.max_rss_mb)
    print(json.dumps(result, indent=2, sort_keys=True))

    exit_code = result["exit_code"]
    within_budget = result["within_budget"]
    if exit_code != 0:
        return exit_code
    return 0 if within_budget else 3


if __name__ == "__main__":
    raise SystemExit(main())
