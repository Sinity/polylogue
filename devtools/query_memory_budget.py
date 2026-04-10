"""Run a command under an explicit RSS budget and emit a machine-readable summary.

Usage:
    devtools query-memory-budget --max-rss-mb 1536 -- python -m polylogue --plain --stats
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


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


def run_memory_budget(command: list[str], *, max_rss_mb: int, poll_interval_s: float = 0.05) -> dict[str, object]:
    """Execute a command and track peak RSS."""
    proc = subprocess.Popen(command)
    peak_rss_kb = 0

    while proc.poll() is None:
        peak_rss_kb = max(peak_rss_kb, _read_vm_rss_kb(proc.pid))
        time.sleep(poll_interval_s)

    peak_rss_kb = max(peak_rss_kb, _read_vm_rss_kb(proc.pid))
    exit_code = int(proc.returncode or 0)
    peak_rss_mb = round(peak_rss_kb / 1024, 1)

    return {
        "command": command,
        "exit_code": exit_code,
        "max_rss_mb": int(max_rss_mb),
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

    if result["exit_code"] != 0:
        return int(result["exit_code"])
    return 0 if bool(result["within_budget"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
