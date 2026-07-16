"""Run a command under an explicit RSS budget and emit a machine-readable summary.

Usage:
    devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

from polylogue.core.json import JSONDocument
from polylogue.scenarios import (
    BudgetMeasure,
    BudgetSemantics,
    MeasurementScope,
    WorkloadBudget,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)


class MemoryBudgetResult(TypedDict):
    command: list[str]
    exit_code: int
    max_rss_mb: int
    peak_parent_rss_mb: float
    peak_rss_mb: float
    within_budget: bool
    duration_ms: float
    workload_receipt: JSONDocument


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


def _command_input_id(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"command:sha256:{hashlib.sha256(encoded).hexdigest()}"


def run_memory_budget(command: list[str], *, max_rss_mb: int, poll_interval_s: float = 0.05) -> MemoryBudgetResult:
    """Execute a command and track peak process-tree RSS."""
    started = time.perf_counter()
    proc = subprocess.Popen(command)
    peak_parent_rss_kb = 0
    peak_rss_kb = 0

    while proc.poll() is None:
        peak_parent_rss_kb = max(peak_parent_rss_kb, _read_vm_rss_kb(proc.pid))
        peak_rss_kb = max(peak_rss_kb, _read_process_tree_rss_kb(proc.pid))
        time.sleep(poll_interval_s)

    peak_parent_rss_kb = max(peak_parent_rss_kb, _read_vm_rss_kb(proc.pid))
    peak_rss_kb = max(peak_rss_kb, _read_process_tree_rss_kb(proc.pid))
    peak_rss_kb = max(peak_rss_kb, peak_parent_rss_kb)
    exit_code = int(proc.returncode or 0)
    peak_parent_rss_mb = round(peak_parent_rss_kb / 1024, 1)
    peak_rss_mb = round(peak_rss_kb / 1024, 1)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    workload_spec = WorkloadEnvelopeSpec(
        workload_id="devtools:query-memory-budget",
        family_id="query-memory",
        version=1,
        inputs=(WorkloadInputRef(input_id=_command_input_id(command)),),
        phases=("execute",),
        measurement_scope=MeasurementScope.PROCESS_TREE,
        budgets=(
            WorkloadBudget(
                measure=BudgetMeasure.PEAK_RSS_BYTES,
                maximum=max_rss_mb * 1024 * 1024,
                semantics=BudgetSemantics.REGRESSION_GATE,
            ),
        ),
    )
    receipt = WorkloadReceipt.from_observations(
        spec=workload_spec,
        status=WorkloadRunStatus.SUCCEEDED if exit_code == 0 else WorkloadRunStatus.FAILED,
        build_id=None,
        runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(
                name="execute",
                wall_ms=duration_ms,
                peak_rss_bytes=peak_rss_kb * 1024,
                unavailable=(
                    "cpu_ms",
                    "current_rss_bytes",
                    "peak_pss_bytes",
                    "current_pss_bytes",
                    "anon_bytes",
                    "file_cache_bytes",
                    "swap_bytes",
                    "temp_storage_bytes",
                    "storage_bytes",
                    "read_io_bytes",
                    "write_io_bytes",
                    "response_bytes",
                    "cancellation_latency_ms",
                    "progress_completed",
                    "progress_total",
                    "queue_depth",
                    "backpressure_ms",
                    "cleanup_reclaimed_bytes",
                ),
            ),
        ),
        notes=("Legacy process-tree sampler adapter; unmeasured dimensions are explicit.",),
    )

    return {
        "command": command,
        "exit_code": exit_code,
        "max_rss_mb": int(max_rss_mb),
        "peak_parent_rss_mb": peak_parent_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "within_budget": bool(exit_code == 0 and peak_rss_mb <= max_rss_mb),
        "duration_ms": duration_ms,
        "workload_receipt": receipt.to_payload(),
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
