"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy (~3s warm).
  --quick    Pre-push tier: all non-pytest gates (~15s warm).
  (default)  Baseline with pytest-testmon affected tests.
  --seed-testmon
             Full non-integration pytest run that seeds/updates .cache/testmon/testmondata.
  --all/--full
             Explicit full non-integration pytest diagnostic.
  --lab      Default testmon baseline plus lab smoke and SLO checks.

Output formats:
  --json     Machine-readable JSON to stdout (human progress to stderr).
  (default)  Human-readable text when stdout is a TTY; auto-JSON otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.pytest_progress_plugin import merge_worker_collection_payloads
from devtools.pytest_supervisor import (
    SupervisorLaunch,
    build_supervisor_launch,
    descendant_process_identities,
    enable_child_subreaper,
    read_receipt,
    reap_exited_children,
    signal_descendant_identities,
    signal_owned_process_group,
    signal_process_identity,
    update_receipt,
    write_termination_request,
)
from devtools.testmon_bootstrap import maybe_bootstrap_testmon_seed
from devtools.testmon_state import (
    SUCCESSFUL_NODE_OUTCOMES,
    TERMINAL_NODE_OUTCOMES,
    BindingMode,
    GraphStatus,
    SeedAttemptOutcome,
    SeedShardStatus,
    TerminalAuthorization,
    TestmonBinding,
    TestmonSeedStamp,
    VerificationScope,
    inspect_testmon_database,
    refresh_stamp,
    seed_shard_ledger_is_terminal,
    seed_shard_plan,
    stamp_from_attempt,
    testmon_runtime_identity,
    validate_seed_shard_ledger,
    validate_stamp,
)
from devtools.verify_runs import (
    CURRENT_CONTAINMENT_PATH,
    CURRENT_EVENTS_DIR,
    CURRENT_POSTMORTEM_PATH,
    CURRENT_RESOURCES_PATH,
    CURRENT_STATISTICS_PATH,
    PYTEST_CANONICAL_REPORT_NAME,
    PYTEST_EXPLICIT_BASETEMP_ENV,
    VERIFY_HISTORY_PATH,
    CheckoutMutationMonitor,
    PytestResourceError,
    PytestStepArtifacts,
    ResourceSampler,
    VerifyRun,
    adaptive_pytest_worker_count,
    append_verify_history,
    apply_managed_pytest_runtime_policy,
    classify_pytest_result,
    cleanup_managed_pytest_basetemp,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    finalize_checkout_mutation_monitors,
    finish_checkout_mutation_monitor,
    force_managed_pytest_scratch,
    latest_event_from_paths,
    normalize_pytest_basetemp_env,
    pytest_basetemp_path,
    pytest_command_worker_request,
    pytest_tmpfs_budget_exceeded,
    pytest_tmpfs_budget_kb,
    start_checkout_mutation_monitor,
    utc_now,
    worktree_fingerprint,
    xdist_uninterruptible_stall_reason,
)
from polylogue.scenarios.workload import (
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

ROOT = Path(__file__).resolve().parents[1]


def _anchor_verification_paths() -> None:
    """Use the checkout root for relative verification state when invoked inside it."""
    current = Path.cwd().resolve()
    try:
        current.relative_to(ROOT.resolve())
    except ValueError:
        return
    os.chdir(ROOT)


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
    return [sys.executable, "-m", "devtools", *(part for arg in args for part in arg.split())]


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


HISTORY_PATH = VERIFY_HISTORY_PATH
TESTMON_DATA = Path(".cache/testmon/testmondata")
TESTMON_SEED_STAMP = Path(".cache/testmon/seed.json")
TESTMON_SEED_ATTEMPT = Path(".cache/testmon/seed-attempt.json")
TESTMON_AFFECTED_STAMP = Path(".cache/testmon/affected.json")
TESTMON_SEED_PROTOCOL_VERSION = 7
# Keep resumable checkpoints coarse enough that controller startup and
# per-shard testmon initialization do not dominate the seed.  The seed still
# records every node outcome, so a failed shard remains retryable at node
# resolution; this size yields six shards for the current correctness corpus.
TESTMON_SEED_SHARD_SIZE = 4096
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
PYTEST_JUNIT_REPORT_DIR = Path(".cache/test-reports")
PYTEST_JUNIT_REPORT_PATH = PYTEST_JUNIT_REPORT_DIR / "verify-latest.xml"
PYTEST_PROGRESS_PATH = PYTEST_REPORT_DIR / "current-pytest-progress.json"
PYTEST_EVENTS_PATH = PYTEST_REPORT_DIR / "current-pytest-events.jsonl"
PYTEST_EVENTS_DIR = CURRENT_EVENTS_DIR
PYTEST_SELECTION_PATH = PYTEST_REPORT_DIR / "current-pytest-selection.json"
PYTEST_SUMMARY_PATH = PYTEST_REPORT_DIR / "current-pytest-summary.json"
PYTEST_OUTPUT_PATH = PYTEST_REPORT_DIR / "current-pytest-output.log"
PYTEST_CONTAINMENT_PATH = CURRENT_CONTAINMENT_PATH
PYTEST_HEARTBEAT_ENV = "POLYLOGUE_VERIFY_HEARTBEAT_S"
PYTEST_TIMEOUT_ENV = "POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S"
PYTEST_STALL_TIMEOUT_ENV = "POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S"
PYTEST_TERM_GRACE_ENV = "POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S"
PYTEST_RESOURCE_INTERVAL_ENV = "POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S"
DEFAULT_PYTEST_HEARTBEAT_S = 30.0
DEFAULT_PYTEST_TIMEOUT_S = 45 * 60.0
DEFAULT_PYTEST_STALL_TIMEOUT_S = 10 * 60.0
DEFAULT_PYTEST_TERM_GRACE_S = 5.0
DEFAULT_PYTEST_RESOURCE_INTERVAL_S = 2.0


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
    append_verify_history(entry, path=HISTORY_PATH)


def _print_history(file: Path | None = None) -> None:
    """Print last 10 verify runs as a compact table."""
    entries = _load_history()
    if not entries:
        print("verify: no history yet")
        return
    print(f"{'time':<20} {'tier':<8} {'head':<10} {'dur':>7} {'exit':>4}  steps")
    print("-" * 75)
    for entry in entries[-10:]:
        timestamp = str(entry.get("timestamp") or entry.get("finished_at") or entry.get("started_at") or "unknown")
        ts = timestamp[5:19] if timestamp != "unknown" else timestamp
        tier = str(entry.get("tier") or "unknown")[:8]
        head = str(entry.get("git_head") or "unknown")[:8]
        duration = entry.get("total_duration_s", entry.get("duration_s", 0.0))
        try:
            dur = f"{float(duration or 0.0):.0f}s"
        except (TypeError, ValueError):
            dur = "0s"
        raw_exit = entry.get("exit_code", 1)
        try:
            ec = int(raw_exit if raw_exit is not None else 1)
        except (TypeError, ValueError):
            ec = 1
        rendered_steps: list[str] = []
        for step in entry.get("steps", []):
            if not isinstance(step, dict):
                continue
            try:
                step_duration = float(step.get("duration_s") or 0.0)
            except (TypeError, ValueError):
                step_duration = 0.0
            raw_step_exit = step.get("exit", 1)
            try:
                step_exit = int(raw_step_exit if raw_step_exit is not None else 1)
            except (TypeError, ValueError):
                step_exit = 1
            rendered_steps.append(f"{step.get('name', 'unknown')}({step_duration:.0f}s{' FAIL' if step_exit else ''})")
        steps = ", ".join(rendered_steps)
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


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_latest_pytest_event(
    path: Path = PYTEST_EVENTS_PATH,
    *,
    events_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return the latest valid pytest event from the live JSONL ledger."""
    if events_dir is not None:
        return latest_event_from_paths(events_dir, path)
    if path == PYTEST_EVENTS_PATH:
        return latest_event_from_paths(PYTEST_EVENTS_DIR, path)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            lines = handle.read().splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            return event
    return None


def _pytest_metadata_from_report(report: dict[str, Any], *, report_path: Path) -> dict[str, Any]:
    """Project a pytest-json-report dict into verify-step metadata."""
    summary = report.get("summary")
    metadata: dict[str, Any] = {"report_path": str(report_path)}
    if isinstance(summary, dict):
        # Prefer pytest-json-report's explicit total when present. Older or
        # reduced reports still get a stable executed count from outcome keys.
        outcome_keys = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        total = summary.get("total")
        executed = int(total) if isinstance(total, int) else sum(int(summary.get(k, 0) or 0) for k in outcome_keys)
        metadata["count"] = executed
        for key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "total"):
            value = summary.get(key)
            if isinstance(value, int):
                metadata[key] = value
    duration = report.get("duration")
    if isinstance(duration, (int, float)):
        metadata["pytest_duration_s"] = round(float(duration), 2)
    return metadata


def _pytest_command_metadata(cmd: list[str]) -> dict[str, Any]:
    """Return verify metadata that explains the pytest worker policy."""
    metadata: dict[str, Any] = {}
    metadata["pytest_workers"] = pytest_command_worker_request(cmd) or "unset"
    if "--testmon" in cmd:
        metadata["pytest_selection"] = "testmon-noselect" if "--testmon-noselect" in cmd else "testmon"
    else:
        metadata["pytest_selection"] = "full"
    return metadata


def _nonnegative_int_delta(
    first: Mapping[str, object] | None,
    last: Mapping[str, object] | None,
    key: str,
) -> int | None:
    if first is None or last is None:
        return None
    first_value = first.get(key)
    last_value = last.get(key)
    if not isinstance(first_value, int) or not isinstance(last_value, int):
        return None
    return max(0, last_value - first_value)


def _pytest_workload_receipt(
    *,
    label: str,
    cmd: list[str],
    elapsed_s: float,
    returncode: int,
    termination_reason: str | None,
    resource_summary: Mapping[str, object],
    last_resource_sample: Mapping[str, object] | None,
    tmpfs_budget_mb: float | None,
    basetemp_cleanup: Path | None,
    concurrency: int,
) -> dict[str, Any]:
    """Adapt managed-pytest accounting to the shared workload receipt."""
    input_digest = hashlib.sha256(
        json.dumps(cmd, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    budgets: list[WorkloadBudget] = []
    if (timeout_s := _pytest_timeout_s()) > 0:
        budgets.append(
            WorkloadBudget(
                BudgetMeasure.WALL_MS,
                timeout_s * 1000,
                BudgetSemantics.CONTAINMENT,
                phase="execute",
            )
        )
    if tmpfs_budget_mb is not None:
        budgets.append(
            WorkloadBudget(
                BudgetMeasure.TEMP_STORAGE_BYTES,
                tmpfs_budget_mb * 1024 * 1024,
                BudgetSemantics.CONTAINMENT,
                phase="execute",
            )
        )
    spec = WorkloadEnvelopeSpec(
        workload_id=f"verify:{label}",
        family_id="verify-pytest",
        version=1,
        inputs=(WorkloadInputRef(input_id=f"pytest-selection:sha256:{input_digest}"),),
        phases=("execute", "quiescent"),
        measurement_scope=MeasurementScope.PROCESS_TREE,
        concurrency=concurrency,
        admission="memory-headroom",
        budgets=tuple(budgets),
    )
    peak_rss_kb = resource_summary.get("peak_tree_rss_kb")
    peak_pss_kb = resource_summary.get("peak_tree_pss_kb")
    peak_anon_pss_kb = resource_summary.get("peak_tree_anon_pss_kb")
    peak_file_pss_kb = resource_summary.get("peak_tree_file_pss_kb")
    peak_swap_pss_kb = resource_summary.get("peak_tree_swap_pss_kb")
    read_bytes = resource_summary.get("tree_read_bytes_delta")
    write_bytes = resource_summary.get("tree_write_bytes_delta")
    peak_basetemp_kb = resource_summary.get("peak_basetemp_allocated_kb")
    logical_basetemp_kb = resource_summary.get("peak_basetemp_size_kb")
    final_rss_kb = last_resource_sample.get("tree_rss_kb") if last_resource_sample is not None else None
    final_pss_kb = last_resource_sample.get("tree_pss_kb") if last_resource_sample is not None else None
    total_cpu_s = last_resource_sample.get("tree_cpu_s") if last_resource_sample is not None else None
    execute_unavailable = [
        "current_rss_bytes",
        "current_pss_bytes",
        "storage_bytes",
        "response_bytes",
        "cancellation_latency_ms",
        "progress_completed",
        "progress_total",
        "queue_depth",
        "backpressure_ms",
        "cleanup_reclaimed_bytes",
    ]
    if not isinstance(peak_rss_kb, int):
        execute_unavailable.append("peak_rss_bytes")
    if not isinstance(peak_pss_kb, int):
        execute_unavailable.append("peak_pss_bytes")
    if not isinstance(peak_basetemp_kb, int):
        execute_unavailable.append("temp_storage_bytes")
    if not isinstance(peak_anon_pss_kb, int):
        execute_unavailable.append("anon_bytes")
    if not isinstance(peak_file_pss_kb, int):
        execute_unavailable.append("file_cache_bytes")
    if not isinstance(peak_swap_pss_kb, int):
        execute_unavailable.append("swap_bytes")
    if not isinstance(read_bytes, int):
        execute_unavailable.append("read_io_bytes")
    if not isinstance(write_bytes, int):
        execute_unavailable.append("write_io_bytes")
    if not isinstance(total_cpu_s, int | float):
        execute_unavailable.append("cpu_ms")
    quiescent_unavailable = [
        "wall_ms",
        "cpu_ms",
        "peak_rss_bytes",
        "peak_pss_bytes",
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
    ]
    if not isinstance(final_rss_kb, int):
        quiescent_unavailable.append("current_rss_bytes")
    if not isinstance(final_pss_kb, int):
        quiescent_unavailable.append("current_pss_bytes")
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=(
            WorkloadRunStatus.CANCELLED
            if termination_reason is not None
            else WorkloadRunStatus.SUCCEEDED
            if returncode == 0
            else WorkloadRunStatus.FAILED
        ),
        build_id=f"git:{head}" if (head := _git_head()) is not None else None,
        runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(
                name="execute",
                wall_ms=elapsed_s * 1000,
                cpu_ms=float(total_cpu_s) * 1000 if isinstance(total_cpu_s, int | float) else None,
                peak_rss_bytes=peak_rss_kb * 1024 if isinstance(peak_rss_kb, int) else None,
                peak_pss_bytes=peak_pss_kb * 1024 if isinstance(peak_pss_kb, int) else None,
                anon_bytes=peak_anon_pss_kb * 1024 if isinstance(peak_anon_pss_kb, int) else None,
                file_cache_bytes=peak_file_pss_kb * 1024 if isinstance(peak_file_pss_kb, int) else None,
                swap_bytes=peak_swap_pss_kb * 1024 if isinstance(peak_swap_pss_kb, int) else None,
                temp_storage_bytes=peak_basetemp_kb * 1024 if isinstance(peak_basetemp_kb, int) else None,
                read_io_bytes=read_bytes if isinstance(read_bytes, int) else None,
                write_io_bytes=write_bytes if isinstance(write_bytes, int) else None,
                unavailable=tuple(execute_unavailable),
            ),
            WorkloadPhaseObservation(
                name="quiescent",
                current_rss_bytes=final_rss_kb * 1024 if isinstance(final_rss_kb, int) else None,
                current_pss_bytes=final_pss_kb * 1024 if isinstance(final_pss_kb, int) else None,
                cleanup_complete=True if basetemp_cleanup is not None else None,
                quiescent=final_rss_kb == 0,
                unavailable=tuple(quiescent_unavailable),
            ),
        ),
        cancellation_requested=termination_reason is not None,
        cleanup_complete=True if basetemp_cleanup is not None else None,
        notes=(
            "Managed pytest process-tree sampler adapter.",
            (
                f"Logical basetemp peak retained as diagnostic evidence: {logical_basetemp_kb * 1024} bytes."
                if isinstance(logical_basetemp_kb, int)
                else "Logical basetemp peak unavailable."
            ),
        ),
    )
    return dict(receipt.to_payload())


def _pytest_artifact_paths(cmd: Sequence[str]) -> tuple[Path, ...]:
    json_paths: list[Path] = []
    junit_paths: list[Path] = []
    for arg in cmd:
        if arg.startswith("--json-report-file="):
            json_paths.append(Path(arg.split("=", 1)[1]))
        elif arg.startswith("--junitxml="):
            junit_paths.append(Path(arg.split("=", 1)[1]))
    paths = [*(json_paths or [PYTEST_REPORT_PATH]), *(junit_paths or [PYTEST_JUNIT_REPORT_PATH])]
    return tuple(dict.fromkeys(paths))


def _pytest_json_report_path(cmd: Sequence[str]) -> Path:
    for arg in reversed(cmd):
        if arg.startswith("--json-report-file="):
            return Path(arg.split("=", 1)[1])
    return PYTEST_REPORT_PATH


def _clear_pytest_report(cmd: Sequence[str] = ()) -> None:
    """Remove a stale report before a pytest step runs."""
    for path in (
        *_pytest_artifact_paths(cmd),
        PYTEST_PROGRESS_PATH,
        PYTEST_EVENTS_PATH,
        PYTEST_EVENTS_DIR,
        PYTEST_SELECTION_PATH,
        PYTEST_SUMMARY_PATH,
        PYTEST_OUTPUT_PATH,
        CURRENT_RESOURCES_PATH,
        CURRENT_POSTMORTEM_PATH,
        CURRENT_CONTAINMENT_PATH,
        CURRENT_STATISTICS_PATH,
    ):
        with contextlib.suppress(FileNotFoundError):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _write_pytest_output(stdout: str, stderr: str) -> None:
    """Persist captured pytest output for killed runs and post-mortem review."""
    PYTEST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PYTEST_OUTPUT_PATH.write_text(stdout + ("\n" if stdout and stderr else "") + stderr, encoding="utf-8")


def _persist_pytest_output(stdout: str, stderr: str, *, artifacts: PytestStepArtifacts | None) -> None:
    """Persist drained pytest output on both ordinary and exceptional exits."""
    with contextlib.suppress(OSError):
        _write_pytest_output(stdout, stderr)
    if artifacts is not None:
        for path, content in (
            (artifacts.stdout_path, stdout),
            (artifacts.stderr_path, stderr),
            (artifacts.output_path, stdout + stderr),
        ):
            with contextlib.suppress(OSError):
                path.write_text(content, encoding="utf-8")


def _write_pytest_progress(
    *,
    event: str,
    cmd: list[str],
    started_at: float,
    pid: int | None = None,
    returncode: int | None = None,
    elapsed_s: float | None = None,
    idle_s: float | None = None,
    output_bytes: Mapping[str, int] | None = None,
    status: Mapping[str, str | int | None] | None = None,
    cpu_pct: float | None = None,
    termination_reason: str | None = None,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    resources: Mapping[str, Any] | None = None,
    containment: Mapping[str, Any] | None = None,
    events_path: Path = PYTEST_EVENTS_PATH,
    events_dir: Path | None = None,
) -> None:
    """Write a live pytest progress artifact for long verify runs."""
    if elapsed_s is None:
        elapsed_s = time.monotonic() - started_at
    payload: dict[str, Any] = {
        "event": event,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s, 2),
        "command": cmd,
    }
    if pid is not None:
        payload["pid"] = pid
    if returncode is not None:
        payload["returncode"] = returncode
    if idle_s is not None:
        payload["idle_s"] = round(idle_s, 2)
    if output_bytes is not None:
        payload["output_bytes"] = dict(output_bytes)
    if status is not None:
        payload["process_state"] = status.get("state")
        payload["rss_kb"] = status.get("rss_kb")
    if cpu_pct is not None:
        payload["cpu_pct"] = round(cpu_pct, 2)
    if termination_reason is not None:
        payload["termination_reason"] = termination_reason
    if run_id is not None:
        payload["run_id"] = run_id
    if artifact_dir is not None:
        payload["artifact_dir"] = artifact_dir
    if resources is not None:
        payload["resources"] = dict(resources)
    if containment is not None:
        payload["containment"] = dict(containment)
    latest_event = _read_latest_pytest_event(events_path, events_dir=events_dir)
    if latest_event is not None:
        payload["latest_test_event"] = {
            key: latest_event[key]
            for key in ("event", "nodeid", "when", "outcome", "duration_s", "updated_at")
            if key in latest_event
        }
        if latest_event.get("event") == "test_started" and isinstance(latest_event.get("nodeid"), str):
            payload["current_test_nodeid"] = latest_event["nodeid"]
    targets = [PYTEST_PROGRESS_PATH]
    if artifact_dir is not None:
        targets.insert(0, Path(artifact_dir) / "progress.json")
    for target in dict.fromkeys(targets):
        with contextlib.suppress(OSError):
            _atomic_write_json(target, payload)


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        fields = raw.rsplit(") ", 1)[1].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (float(fields[11]) + float(fields[12])) / float(ticks)
    except (IndexError, OSError, ValueError):
        return None


def _process_status(pid: int) -> dict[str, str | int | None]:
    status: dict[str, str | int | None] = {"state": None, "rss_kb": None}
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return status
    for line in lines:
        if line.startswith("State:"):
            status["state"] = line.split(":", 1)[1].strip()
        elif line.startswith("VmRSS:"):
            with contextlib.suppress(ValueError, IndexError):
                status["rss_kb"] = int(line.split()[1])
    return status


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(value, 0.0)


def _pytest_heartbeat_interval() -> float:
    return _float_env(PYTEST_HEARTBEAT_ENV, DEFAULT_PYTEST_HEARTBEAT_S)


def _pytest_timeout_s() -> float:
    return _float_env(PYTEST_TIMEOUT_ENV, DEFAULT_PYTEST_TIMEOUT_S)


def _pytest_stall_timeout_s() -> float:
    return _float_env(PYTEST_STALL_TIMEOUT_ENV, DEFAULT_PYTEST_STALL_TIMEOUT_S)


def _pytest_term_grace_s() -> float:
    return _float_env(PYTEST_TERM_GRACE_ENV, DEFAULT_PYTEST_TERM_GRACE_S)


def _pytest_resource_interval_s() -> float:
    return _float_env(PYTEST_RESOURCE_INTERVAL_ENV, DEFAULT_PYTEST_RESOURCE_INTERVAL_S)


def _containment_summary(launch: SupervisorLaunch, receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": launch.mode,
        "unit": launch.unit,
        "receipt_path": str(launch.receipt_path),
        "runtime_cap_s": launch.runtime_cap_s,
    }
    if receipt is not None:
        for key in (
            "supervisor_pid",
            "controller_pid",
            "controller_pgid",
            "controller_sid",
            "cgroup_path",
            "cgroup_owned",
            "signals_sent",
            "escalated_to_sigkill",
            "controller_group_alive",
        ):
            if key in receipt:
                payload[key] = receipt[key]
    return payload


def _request_supervisor_termination(
    process: subprocess.Popen[bytes],
    launch: SupervisorLaunch,
    *,
    reason: str,
) -> None:
    write_termination_request(launch.request_path, reason=reason)
    receipt = read_receipt(launch.receipt_path)
    supervisor_pid = receipt.get("supervisor_pid") if receipt is not None else None
    supervisor_start = receipt.get("supervisor_start_ticks") if receipt is not None else None
    signalled = False
    if isinstance(supervisor_pid, int) and isinstance(supervisor_start, int):
        signalled = signal_process_identity(supervisor_pid, supervisor_start, signal.SIGTERM)
    if not signalled and process.poll() is None:
        process.send_signal(signal.SIGTERM)


class PytestContainmentError(RuntimeError):
    """Raised when an interrupted pytest supervisor cannot be confirmed stopped."""


def _await_interrupted_pytest_containment(
    process: subprocess.Popen[bytes],
    launch: SupervisorLaunch,
    *,
    term_grace_s: float,
    preserved_runner_descendants: Sequence[tuple[int, int]],
) -> None:
    """Wait for an interrupted pytest supervisor before its caller cleans up."""
    if process.poll() is None:
        _request_supervisor_termination(process, launch, reason="pytest runner interrupted")
    try:
        process.wait(timeout=max(1.0, term_grace_s + 1.0))
    except subprocess.TimeoutExpired:
        _force_kill_owned_run(
            process,
            launch,
            preserved_runner_descendants=preserved_runner_descendants,
        )
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise PytestContainmentError(
                "pytest containment did not quiesce after forced termination; leaving its basetemp intact"
            ) from exc
    reap_exited_children()
    receipt = read_receipt(launch.receipt_path)
    remaining_descendants = tuple(
        identity
        for identity in descendant_process_identities(os.getpid())
        if identity not in preserved_runner_descendants
    )
    if (
        receipt is None
        or receipt.get("status") not in {"finished", "terminated"}
        or receipt.get("controller_group_alive") is not False
        or remaining_descendants
    ):
        raise PytestContainmentError(
            "pytest containment did not quiesce its owned process tree; leaving its basetemp intact"
        )


def _supervised_tmpfs_cleanup_path(*, root: Path, run_id: str, env: dict[str, str]) -> Path | None:
    """Return only a supervisor-owned tmpfs path eligible for cleanup."""
    if env.get(PYTEST_EXPLICIT_BASETEMP_ENV) or pytest_tmpfs_budget_kb(env) is None:
        return None
    return pytest_basetemp_path(root=root, run_id=run_id, env=env)


def _force_kill_owned_run(
    process: subprocess.Popen[bytes],
    launch: SupervisorLaunch,
    *,
    preserved_runner_descendants: Sequence[tuple[int, int]] = (),
) -> None:
    """Escalate only through identities recorded for this pytest run."""
    receipt = read_receipt(launch.receipt_path)
    controller_pgid = receipt.get("controller_pgid") if receipt is not None else None
    controller_sid = receipt.get("controller_sid") if receipt is not None else None
    controller_start = receipt.get("controller_start_ticks") if receipt is not None else None
    supervisor_pid = receipt.get("supervisor_pid") if receipt is not None else None
    supervisor_start = receipt.get("supervisor_start_ticks") if receipt is not None else None
    if launch.unit is not None and shutil.which("systemctl"):
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            result = subprocess.run(
                ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", launch.unit],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                return
    if isinstance(controller_pgid, int) and isinstance(controller_sid, int):
        signal_owned_process_group(
            pgid=controller_pgid,
            sid=controller_sid,
            leader_start_ticks=controller_start if isinstance(controller_start, int) else None,
            sig=signal.SIGKILL,
        )
    if isinstance(supervisor_pid, int) and isinstance(supervisor_start, int):
        signal_process_identity(supervisor_pid, supervisor_start, signal.SIGKILL)
    signal_descendant_identities(
        os.getpid(),
        signal.SIGKILL,
        preserved_roots=preserved_runner_descendants,
    )
    if process.poll() is None:
        process.kill()


def _wait_for_supervisor_start(
    process: subprocess.Popen[bytes],
    launch: SupervisorLaunch,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        receipt = read_receipt(launch.receipt_path)
        if receipt is not None:
            return receipt
        if process.poll() is not None:
            return None
        time.sleep(0.01)
    return read_receipt(launch.receipt_path)


def _startup_wait_s(*, t0: float, timeout_s: float) -> float:
    if timeout_s <= 0:
        return 5.0
    return max(0.0, min(5.0, t0 + timeout_s - time.monotonic()))


def _finish_supervisor_startup_failure(
    cmd: list[str],
    *,
    launch: SupervisorLaunch,
    process: subprocess.Popen[bytes],
    t0: float,
    timeout_s: float,
    term_grace_s: float,
    reason: str,
    exit_code: int,
    stdout: bytes,
    stderr: bytes,
    runner_subreaper_enabled: bool,
    run: VerifyRun | None,
    artifacts: PytestStepArtifacts | None,
) -> subprocess.CompletedProcess[str]:
    receipt = update_receipt(
        launch.receipt_path,
        {
            "schema_version": 1,
            "status": "terminated",
            "finished_at": utc_now(),
            "duration_s": round(time.monotonic() - t0, 4),
            "controller_command": list(cmd),
            "mode": launch.mode,
            "unit": launch.unit,
            "cgroup_owned": launch.mode == "systemd-scope",
            "timeout_s": timeout_s,
            "term_grace_s": term_grace_s,
            "runtime_cap_s": launch.runtime_cap_s,
            "exit_code": exit_code,
            "supervisor_exit_code": process.poll(),
            "termination_reason": reason,
            "signals_sent": ["SIGKILL"],
            "escalated_to_sigkill": True,
            "runner_forced_cleanup": True,
            "runner_forced_at": utc_now(),
            "runner_pid": os.getpid(),
            "runner_subreaper_enabled": runner_subreaper_enabled,
            "startup_failure": True,
        },
    )
    rendered_stdout = stdout.decode(errors="replace")
    rendered_stderr = stderr.decode(errors="replace")
    rendered_stderr = f"{rendered_stderr}\nverify: {reason}; pytest supervisor startup was contained\n"
    output_bytes = {"stdout": len(stdout), "stderr": len(rendered_stderr.encode())}
    _write_pytest_progress(
        event="terminated",
        cmd=cmd,
        started_at=t0,
        pid=process.pid,
        returncode=exit_code,
        output_bytes=output_bytes,
        termination_reason=reason,
        run_id=run.run_id if run is not None else None,
        artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
        containment=_containment_summary(launch, receipt),
    )
    _write_pytest_output(rendered_stdout, rendered_stderr)
    if artifacts is not None:
        artifacts.stdout_path.write_text(rendered_stdout, encoding="utf-8")
        artifacts.stderr_path.write_text(rendered_stderr, encoding="utf-8")
        artifacts.output_path.write_text(rendered_stdout + rendered_stderr, encoding="utf-8")
    reap_exited_children()
    return subprocess.CompletedProcess(cmd, exit_code, rendered_stdout, rendered_stderr)


def _run_pytest_with_heartbeat(
    cmd: list[str],
    *,
    cwd: str | None,
    env: dict[str, str],
    t0: float,
    run: VerifyRun | None = None,
    artifacts: PytestStepArtifacts | None = None,
) -> subprocess.CompletedProcess[str]:
    heartbeat_s = _pytest_heartbeat_interval()
    timeout_s = _pytest_timeout_s()
    stall_timeout_s = _pytest_stall_timeout_s()
    term_grace_s = _pytest_term_grace_s()
    resource_interval_s = _pytest_resource_interval_s()
    tmpfs_budget_kb = pytest_tmpfs_budget_kb(env)
    events_path = Path(env.get("POLYLOGUE_PYTEST_EVENTS_PATH", str(PYTEST_EVENTS_PATH)))
    events_dir = Path(env.get("POLYLOGUE_PYTEST_EVENTS_DIR", str(PYTEST_EVENTS_DIR)))
    runner_subreaper_enabled = enable_child_subreaper()
    preserved_runner_descendants = tuple(descendant_process_identities(os.getpid()))
    receipt_path = (
        artifacts.containment_path
        if artifacts is not None
        else Path(env.get("POLYLOGUE_PYTEST_CONTAINMENT_PATH", str(Path.cwd() / PYTEST_CONTAINMENT_PATH)))
    )
    pytest_run_id = run.run_id if run is not None else env.get("POLYLOGUE_PYTEST_RUN_ID", str(os.getpid()))
    tmpfs_cleanup_path = _supervised_tmpfs_cleanup_path(
        root=Path(cwd) if cwd is not None else Path.cwd(),
        run_id=pytest_run_id,
        env=env,
    )
    launch = build_supervisor_launch(
        cmd,
        owner_pid=os.getpid(),
        timeout_s=timeout_s,
        term_grace_s=term_grace_s,
        receipt_path=receipt_path,
        run_id=pytest_run_id,
        env=env,
        cleanup_path=tmpfs_cleanup_path,
    )
    sys.stderr.write(f"\n    command: {shlex.join(cmd)}\n")
    sys.stderr.write(
        f"    containment: mode={launch.mode}"
        f"{f', unit={launch.unit}' if launch.unit is not None else ''}, receipt={launch.receipt_path}\n"
    )
    sys.stderr.flush()
    if not runner_subreaper_enabled:
        reason = "pytest runner could not become a Linux child subreaper"
        preflight_receipt = update_receipt(
            launch.receipt_path,
            {
                "schema_version": 1,
                "status": "terminated",
                "started_at": utc_now(),
                "finished_at": utc_now(),
                "duration_s": round(time.monotonic() - t0, 4),
                "runner_pid": os.getpid(),
                "runner_subreaper_enabled": False,
                "controller_command": list(cmd),
                "mode": launch.mode,
                "unit": launch.unit,
                "cgroup_owned": False,
                "timeout_s": timeout_s,
                "term_grace_s": term_grace_s,
                "runtime_cap_s": launch.runtime_cap_s,
                "exit_code": 125,
                "termination_reason": reason,
                "signals_sent": [],
                "escalated_to_sigkill": False,
                "controller_group_alive": False,
                "startup_failure": True,
            },
        )
        rendered_stderr = f"verify: {reason}; refusing an unowned pytest launch\n"
        _write_pytest_progress(
            event="terminated",
            cmd=cmd,
            started_at=t0,
            returncode=125,
            termination_reason=reason,
            run_id=run.run_id if run is not None else None,
            artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
            containment=_containment_summary(launch, preflight_receipt),
        )
        _write_pytest_output("", rendered_stderr)
        if artifacts is not None:
            artifacts.stdout_path.write_text("", encoding="utf-8")
            artifacts.stderr_path.write_text(rendered_stderr, encoding="utf-8")
            artifacts.output_path.write_text(rendered_stderr, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 125, "", rendered_stderr)

    def _spawn_supervisor(argv: Sequence[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def _stop_startup_attempt(
        startup_process: subprocess.Popen[bytes],
        startup_launch: SupervisorLaunch,
    ) -> tuple[bytes, bytes]:
        _force_kill_owned_run(
            startup_process,
            startup_launch,
            preserved_runner_descendants=preserved_runner_descendants,
        )
        try:
            result = startup_process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            startup_process.kill()
            result = startup_process.communicate(timeout=1)
        reap_exited_children()
        return result

    process = _spawn_supervisor(launch.argv)
    assert process.stdout is not None
    assert process.stderr is not None
    startup_receipt = _wait_for_supervisor_start(process, launch, timeout_s=_startup_wait_s(t0=t0, timeout_s=timeout_s))
    startup_stdout = b""
    startup_stderr = b""
    if startup_receipt is None:
        startup_stdout, startup_stderr = _stop_startup_attempt(process, launch)
        if timeout_s > 0 and time.monotonic() - t0 >= timeout_s:
            return _finish_supervisor_startup_failure(
                cmd,
                launch=launch,
                process=process,
                t0=t0,
                timeout_s=timeout_s,
                term_grace_s=term_grace_s,
                reason=f"pytest runtime exceeded {timeout_s:g}s",
                exit_code=124,
                stdout=startup_stdout,
                stderr=startup_stderr,
                runner_subreaper_enabled=runner_subreaper_enabled,
                run=run,
                artifacts=artifacts,
            )
        if launch.fallback_argv is None:
            detail = startup_stderr.decode(errors="replace").strip()
            reason = "pytest supervisor failed before publishing ownership"
            if detail:
                reason = f"{reason}: {detail}"
            return _finish_supervisor_startup_failure(
                cmd,
                launch=launch,
                process=process,
                t0=t0,
                timeout_s=timeout_s,
                term_grace_s=term_grace_s,
                reason=reason,
                exit_code=125,
                stdout=startup_stdout,
                stderr=startup_stderr,
                runner_subreaper_enabled=runner_subreaper_enabled,
                run=run,
                artifacts=artifacts,
            )
        fallback_message = (
            b"pytest supervisor: systemd scope launch failed; retrying with the Linux process-group boundary\n"
        )
        startup_stderr += fallback_message
        sys.stderr.write((startup_stdout + startup_stderr).decode(errors="replace"))
        sys.stderr.flush()
        launch = SupervisorLaunch(
            launch.fallback_argv,
            launch.receipt_path,
            launch.request_path,
            "process-group",
            None,
            None,
        )
        process = _spawn_supervisor(launch.argv)
        assert process.stdout is not None
        assert process.stderr is not None
        startup_receipt = _wait_for_supervisor_start(
            process,
            launch,
            timeout_s=_startup_wait_s(t0=t0, timeout_s=timeout_s),
        )
        if startup_receipt is None:
            fallback_stdout, fallback_stderr = _stop_startup_attempt(process, launch)
            startup_stdout += fallback_stdout
            startup_stderr += fallback_stderr
            timed_out = timeout_s > 0 and time.monotonic() - t0 >= timeout_s
            reason = (
                f"pytest runtime exceeded {timeout_s:g}s"
                if timed_out
                else "pytest process-group supervisor failed before publishing ownership"
            )
            return _finish_supervisor_startup_failure(
                cmd,
                launch=launch,
                process=process,
                t0=t0,
                timeout_s=timeout_s,
                term_grace_s=term_grace_s,
                reason=reason,
                exit_code=124 if timed_out else 125,
                stdout=startup_stdout,
                stderr=startup_stderr,
                runner_subreaper_enabled=runner_subreaper_enabled,
                run=run,
                artifacts=artifacts,
            )
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    assert stdout_pipe is not None
    assert stderr_pipe is not None
    sampler = (
        ResourceSampler(
            root_pid=process.pid,
            run_id=run.run_id if run is not None else env.get("POLYLOGUE_PYTEST_RUN_ID", str(process.pid)),
            root=Path(cwd) if cwd is not None else Path.cwd(),
            env=env,
            output_path=artifacts.resources_path if artifacts is not None else Path.cwd() / CURRENT_RESOURCES_PATH,
        )
        if artifacts is not None or resource_interval_s > 0
        else None
    )
    if sampler is not None:
        sampler.sample(event="started")
    _write_pytest_progress(
        event="started",
        cmd=cmd,
        started_at=t0,
        pid=process.pid,
        output_bytes={"stdout": 0, "stderr": 0},
        run_id=run.run_id if run is not None else None,
        artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
        containment=_containment_summary(launch, startup_receipt),
        events_path=events_path,
        events_dir=events_dir,
    )
    selector = selectors.DefaultSelector()
    selector.register(stdout_pipe, selectors.EVENT_READ, "stdout")
    selector.register(stderr_pipe, selectors.EVENT_READ, "stderr")
    output: dict[str, list[bytes]] = {
        "stdout": [startup_stdout] if startup_stdout else [],
        "stderr": [startup_stderr] if startup_stderr else [],
    }
    output_bytes = {"stdout": len(startup_stdout), "stderr": len(startup_stderr)}
    last_cpu = _process_cpu_seconds(process.pid)
    last_sample = time.monotonic()
    last_output = last_sample
    last_resource_sample = last_sample
    termination_reason: str | None = None
    termination_requested_at: float | None = None
    forced_cleanup = False
    supervisor_finished_at: float | None = None
    post_exit_forced_at: float | None = None
    forced_returncode: int | None = None
    # Test-event progress, not raw output bytes: an xdist master can keep
    # emitting output (its own heartbeat chatter) while every worker is
    # wedged (e.g. a D-state deadlock), so output-silence alone never fires
    # the stall detector (polylogue-27rb). last_progress_marker tracks the
    # latest test event's own updated_at timestamp across all workers
    # (devtools/pytest_progress_plugin.py); last_progress_at is the local
    # monotonic time that marker was last seen to change.
    initial_event = _read_latest_pytest_event(events_path, events_dir=events_dir)
    last_progress_marker: str | None = initial_event.get("updated_at") if initial_event is not None else None
    last_progress_at = last_sample
    seen_any_progress_event = initial_event is not None
    xdist_uninterruptible_since: float | None = None

    def _refresh_progress_marker(at: float, latest: dict[str, Any] | None = None) -> None:
        nonlocal last_progress_marker, last_progress_at, seen_any_progress_event
        if latest is None:
            latest = _read_latest_pytest_event(events_path, events_dir=events_dir)
        if latest is None:
            return
        marker = latest.get("updated_at")
        seen_any_progress_event = True
        if marker != last_progress_marker:
            last_progress_marker = marker
            last_progress_at = at

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t0
            idle = now - last_output
            progress_idle = now - last_progress_at
            receipt = read_receipt(launch.receipt_path)
            if receipt is not None and receipt.get("status") in {"finished", "terminated"} and selector.get_map():
                if supervisor_finished_at is None:
                    supervisor_finished_at = now
                elif now - supervisor_finished_at >= max(0.2, term_grace_s) and post_exit_forced_at is None:
                    receipt_reason = receipt.get("termination_reason")
                    termination_reason = (
                        termination_reason
                        or (str(receipt_reason) if isinstance(receipt_reason, str) else None)
                        or "pytest supervisor exited while owned output pipes remained open"
                    )
                    _force_kill_owned_run(
                        process,
                        launch,
                        preserved_runner_descendants=preserved_runner_descendants,
                    )
                    forced_cleanup = True
                    forced_returncode = 125
                    post_exit_forced_at = now
                    receipt = update_receipt(
                        launch.receipt_path,
                        {
                            "status": "terminated",
                            "supervisor_exit_code": receipt.get("exit_code"),
                            "exit_code": forced_returncode,
                            "termination_reason": termination_reason,
                            "runner_forced_cleanup": True,
                            "runner_forced_at": datetime.now(timezone.utc).isoformat(),
                            "runner_pid": os.getpid(),
                            "runner_subreaper_enabled": runner_subreaper_enabled,
                        },
                    )
                elif post_exit_forced_at is not None and now - post_exit_forced_at >= 1.0:
                    for selector_key in list(selector.get_map().values()):
                        with contextlib.suppress(KeyError):
                            selector.unregister(selector_key.fileobj)
                        if selector_key.fileobj is stdout_pipe:
                            stdout_pipe.close()
                        elif selector_key.fileobj is stderr_pipe:
                            stderr_pipe.close()
                        else:
                            with contextlib.suppress(OSError):
                                os.close(selector_key.fd)
                    if process.poll() is None:
                        process.kill()
            if termination_reason is None and timeout_s > 0 and elapsed >= timeout_s:
                termination_reason = f"pytest runtime exceeded {timeout_s:g}s"
            elif termination_reason is None and stall_timeout_s > 0 and idle >= stall_timeout_s:
                termination_reason = f"pytest produced no output for {stall_timeout_s:g}s"
            elif (
                termination_reason is None
                and stall_timeout_s > 0
                and seen_any_progress_event
                and progress_idle >= stall_timeout_s
                and xdist_uninterruptible_since is None
            ):
                termination_reason = (
                    f"pytest reported no test progress for {stall_timeout_s:g}s "
                    f"(output kept flowing; last progress marker: {last_progress_marker})"
                )
            if termination_reason is not None and termination_requested_at is None:
                _request_supervisor_termination(process, launch, reason=termination_reason)
                termination_requested_at = now
            if (
                termination_requested_at is not None
                and now - termination_requested_at >= term_grace_s + 1.0
                and not forced_cleanup
            ):
                _force_kill_owned_run(
                    process,
                    launch,
                    preserved_runner_descendants=preserved_runner_descendants,
                )
                forced_cleanup = True
                forced_returncode = 124
                receipt = update_receipt(
                    launch.receipt_path,
                    {
                        "status": "terminated",
                        "supervisor_exit_code": process.poll(),
                        "exit_code": forced_returncode,
                        "termination_reason": termination_reason,
                        "runner_forced_cleanup": True,
                        "runner_forced_at": datetime.now(timezone.utc).isoformat(),
                        "runner_pid": os.getpid(),
                        "runner_subreaper_enabled": runner_subreaper_enabled,
                    },
                )

            deadlines: list[float] = []
            if heartbeat_s > 0:
                deadlines.append(heartbeat_s)
            if timeout_s > 0 and termination_reason is None:
                deadlines.append(max(timeout_s - elapsed, 0.0))
            if stall_timeout_s > 0 and termination_reason is None:
                deadlines.append(max(stall_timeout_s - idle, 0.0))
            if stall_timeout_s > 0 and seen_any_progress_event and termination_reason is None:
                deadlines.append(max(stall_timeout_s - progress_idle, 0.0))
            if termination_requested_at is not None and not forced_cleanup:
                deadlines.append(max(term_grace_s + 1.0 - (now - termination_requested_at), 0.0))
            if supervisor_finished_at is not None and post_exit_forced_at is None:
                deadlines.append(max(max(0.2, term_grace_s) - (now - supervisor_finished_at), 0.0))
            if post_exit_forced_at is not None:
                deadlines.append(max(1.0 - (now - post_exit_forced_at), 0.0))
            selector_timeout = min(deadlines) if deadlines else None
            events = selector.select(timeout=selector_timeout)
            if events:
                for selector_key, _mask in events:
                    chunk = os.read(selector_key.fd, 65536)
                    if chunk:
                        stream_name = str(selector_key.data)
                        output[stream_name].append(chunk)
                        output_bytes[stream_name] += len(chunk)
                        sys.stderr.write(chunk.decode(errors="replace"))
                        sys.stderr.flush()
                        last_output = time.monotonic()
                        _refresh_progress_marker(last_output)
                        receipt = read_receipt(launch.receipt_path)
                        _write_pytest_progress(
                            event="output",
                            cmd=cmd,
                            started_at=t0,
                            pid=process.pid,
                            idle_s=last_output - last_progress_at,
                            output_bytes=output_bytes,
                            status=_process_status(process.pid),
                            run_id=run.run_id if run is not None else None,
                            artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
                            containment=_containment_summary(launch, receipt),
                            events_path=events_path,
                            events_dir=events_dir,
                        )
                    else:
                        selector.unregister(selector_key.fileobj)
            else:
                status = _process_status(process.pid)
                cpu_now = _process_cpu_seconds(process.pid)
                cpu_pct = None
                sample_now = time.monotonic()
                if cpu_now is not None and last_cpu is not None and sample_now > last_sample:
                    cpu_pct = ((cpu_now - last_cpu) / (sample_now - last_sample)) * 100.0
                last_cpu = cpu_now
                last_sample = sample_now
                rss = status["rss_kb"]
                rss_text = f", rss={int(rss) // 1024} MiB" if isinstance(rss, int) else ""
                cpu_text = f", cpu={cpu_pct:.0f}%" if cpu_pct is not None else ""
                state_text = f", state={status['state']}" if status["state"] is not None else ""
                latest_event = _read_latest_pytest_event(events_path, events_dir=events_dir)
                _refresh_progress_marker(sample_now, latest_event)
                if latest_event is not None:
                    event = latest_event.get("event")
                    nodeid = latest_event.get("nodeid")
                    node_text = (
                        f", latest={event}:{nodeid}" if isinstance(event, str) and isinstance(nodeid, str) else ""
                    )
                else:
                    node_text = ""
                progress_idle_text = (
                    f", progress_idle={sample_now - last_progress_at:.0f}s" if seen_any_progress_event else ""
                )
                receipt = read_receipt(launch.receipt_path)
                controller_pid = receipt.get("controller_pid") if receipt is not None else None
                controller_text = f", controller={controller_pid}" if isinstance(controller_pid, int) else ""
                sys.stderr.write(
                    f"    still running: supervisor={process.pid}{controller_text}, elapsed={sample_now - t0:.0f}s, "
                    f"idle={sample_now - last_output:.0f}s{progress_idle_text}{state_text}{cpu_text}{rss_text}{node_text}\n"
                )
                sys.stderr.flush()
                _write_pytest_progress(
                    event="heartbeat",
                    cmd=cmd,
                    started_at=t0,
                    pid=process.pid,
                    elapsed_s=sample_now - t0,
                    idle_s=sample_now - last_progress_at,
                    output_bytes=output_bytes,
                    status=status,
                    cpu_pct=cpu_pct,
                    run_id=run.run_id if run is not None else None,
                    artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
                    containment=_containment_summary(launch, receipt),
                    events_path=events_path,
                    events_dir=events_dir,
                )
            sample_now = time.monotonic()
            if (
                sampler is not None
                and resource_interval_s > 0
                and sample_now - last_resource_sample >= resource_interval_s
            ):
                resource_sample = sampler.sample(event="sample")
                if (
                    termination_reason is None
                    and tmpfs_budget_kb is not None
                    and pytest_tmpfs_budget_exceeded(resource_sample, budget_kb=tmpfs_budget_kb)
                ):
                    basetemp_allocated_kb = int(resource_sample["basetemp_allocated_kb"])
                    termination_reason = (
                        f"pytest tmpfs budget exceeded: {basetemp_allocated_kb / 1024:.1f} MiB allocated "
                        f"> {tmpfs_budget_kb / 1024:.0f} MiB"
                    )
                if resource_sample.get("all_xdist_workers_uninterruptible") is True:
                    if xdist_uninterruptible_since is None:
                        xdist_uninterruptible_since = sample_now
                    elif termination_reason is None:
                        termination_reason = xdist_uninterruptible_stall_reason(
                            resource_sample,
                            started_at=xdist_uninterruptible_since,
                            now=sample_now,
                            timeout_s=stall_timeout_s,
                        )
                else:
                    xdist_uninterruptible_since = None
                last_resource_sample = sample_now
            if process.poll() is not None and not selector.get_map():
                break
    except BaseException:
        try:
            _await_interrupted_pytest_containment(
                process,
                launch,
                term_grace_s=term_grace_s,
                preserved_runner_descendants=preserved_runner_descendants,
            )
        finally:
            _persist_pytest_output(
                b"".join(output["stdout"]).decode(errors="replace"),
                b"".join(output["stderr"]).decode(errors="replace"),
                artifacts=artifacts,
            )
        raise
    finally:
        selector.close()
    reap_exited_children()

    for stream in (stdout_pipe, stderr_pipe):
        if stream.closed:
            continue
        with contextlib.suppress(OSError):
            remaining = stream.read()
        if remaining:
            stream_name = "stdout" if stream is stdout_pipe else "stderr"
            output[stream_name].append(remaining)
            output_bytes[stream_name] += len(remaining)
    stdout = b"".join(output["stdout"]).decode(errors="replace")
    stderr = b"".join(output["stderr"]).decode(errors="replace")
    receipt = read_receipt(launch.receipt_path)
    if termination_reason is None and receipt is not None and receipt.get("status") == "terminated":
        receipt_reason = receipt.get("termination_reason")
        if isinstance(receipt_reason, str):
            termination_reason = receipt_reason
    receipt_exit = receipt.get("exit_code") if receipt is not None else None
    returncode = (
        forced_returncode
        if forced_returncode is not None
        else (int(receipt_exit) if isinstance(receipt_exit, int) else (process.returncode or 0))
    )
    containment = _containment_summary(launch, receipt)
    resource_summary: dict[str, Any] = {}
    if sampler is not None:
        sampler.sample(event="finished" if termination_reason is None else "terminated")
        resource_summary = sampler.summary()
    if termination_reason is not None:
        controller_pgid = containment.get("controller_pgid", process.pid)
        stderr = (
            f"{stderr}\nverify: {termination_reason}; terminated owned pytest process group "
            f"{controller_pgid} ({launch.mode})\n"
        )
        _write_pytest_progress(
            event="terminated",
            cmd=cmd,
            started_at=t0,
            pid=process.pid,
            returncode=returncode,
            output_bytes=output_bytes,
            termination_reason=termination_reason,
            run_id=run.run_id if run is not None else None,
            artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
            resources=resource_summary,
            containment=containment,
            events_path=events_path,
            events_dir=events_dir,
        )
    else:
        _write_pytest_progress(
            event="finished",
            cmd=cmd,
            started_at=t0,
            pid=process.pid,
            returncode=returncode,
            output_bytes=output_bytes,
            run_id=run.run_id if run is not None else None,
            artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
            resources=resource_summary,
            containment=containment,
            events_path=events_path,
            events_dir=events_dir,
        )
    _persist_pytest_output(stdout, stderr, artifacts=artifacts)
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _run(
    label: str,
    cmd: list[str],
    *,
    cwd: str | None = None,
    run: VerifyRun | None = None,
) -> tuple[int, float, dict[str, Any]]:
    t0 = time.monotonic()
    sys.stderr.write(f"  {label} ... ")
    sys.stderr.flush()
    is_pytest = label.startswith("pytest")
    # ``bench slo`` starts pytest-benchmark itself, so it needs the same
    # bounded temp policy and run marker as a direct pytest step.
    has_managed_pytest_child = label == "bench slo"
    if is_pytest and run is not None:
        isolated_report = run.run_dir / f"pytest-report-{uuid.uuid4().hex}.json"
        cmd = [f"--json-report-file={isolated_report}" if arg.startswith("--json-report-file=") else arg for arg in cmd]
    if is_pytest:
        _clear_pytest_report(cmd)
    artifacts = run.start_step(label=label, cmd=cmd) if run is not None else None
    env = _subprocess_env()
    explicit_basetemp = _pytest_command_basetemp(cmd, cwd=cwd, env=env)
    if explicit_basetemp is not None:
        env[PYTEST_EXPLICIT_BASETEMP_ENV] = str(explicit_basetemp)
    pytest_tmpfs = False
    pytest_tmpfs_budget_mb: float | None = None
    runtime_policy = None
    pytest_concurrency = 0
    basetemp_cleanup: Path | None = None
    if is_pytest or has_managed_pytest_child:
        try:
            if has_managed_pytest_child:
                # The outer benchmark process is not supervised as pytest, so
                # its nested pytest cannot safely consume a bounded tmpfs run:
                # nobody samples or terminates it at the tmpfs cap. Preserve a
                # custom disk root, but replace inherited /dev/shm placement
                # with the managed scratch candidate before admission.
                env = force_managed_pytest_scratch(env)
            if is_pytest:
                pytest_concurrency = _pytest_command_concurrency(cmd, env=env)
            env, runtime_policy = apply_managed_pytest_runtime_policy(
                env,
                worker_count=pytest_concurrency,
                full_suite=_pytest_uses_full_suite_basetemp(label),
            )
        except PytestResourceError as exc:
            elapsed = time.monotonic() - t0
            sys.stderr.write(f"FAILED ({elapsed:.1f}s)\nverify: {exc}\n")
            refusal_metadata: dict[str, Any] = {
                "diagnosis": "pytest_resource_preflight_failed",
                "error": str(exc),
                "termination_reason": "pytest resource preflight refused basetemp admission",
                "verification_scope": "narrow-terminal",
                "release_baseline_allowed": False,
            }
            if run is not None and artifacts is not None:
                finalized_step = run.finish_step(
                    step_id=artifacts.step_id,
                    result={"duration_s": round(elapsed, 2), "exit": 125, **refusal_metadata},
                )
                if isinstance(finalized_step, dict):
                    for key in ("statistics", "statistics_path"):
                        if key in finalized_step:
                            refusal_metadata[key] = finalized_step[key]
            return 125, elapsed, refusal_metadata
        pytest_tmpfs = env.get("POLYLOGUE_PYTEST_TMPFS") == "1"
        budget_kb = pytest_tmpfs_budget_kb(env)
        pytest_tmpfs_budget_mb = budget_kb / 1024 if budget_kb is not None else None
        if label.startswith("pytest seed-testmon"):
            # A complete corpus is currently ~16K nodes. Preserve the whole
            # selection in the attempt receipt so interrupted seeds can prove
            # eventual coverage instead of relying on a 500-node sample.
            env["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] = "50000"
        if run is not None and artifacts is not None:
            env = env_for_pytest_step(env, run=run, artifacts=artifacts)
    interrupted = False
    pytest_containment_quiescent = True
    containment_error: str | None = None
    if is_pytest:
        try:
            try:
                result = _run_pytest_with_heartbeat(cmd, cwd=cwd, env=env, t0=t0, run=run, artifacts=artifacts)
            except PytestContainmentError as exc:
                pytest_containment_quiescent = False
                containment_error = str(exc)
                result = subprocess.CompletedProcess(args=cmd, returncode=125, stdout="", stderr=str(exc))
            except KeyboardInterrupt:
                interrupted = True
                result = subprocess.CompletedProcess(args=cmd, returncode=130, stdout="", stderr="")
        finally:
            if pytest_containment_quiescent:
                basetemp_cleanup = cleanup_managed_pytest_basetemp(
                    root=ROOT,
                    run_id=env.get("POLYLOGUE_PYTEST_RUN_ID", ""),
                    env=env,
                )
    else:
        try:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        except KeyboardInterrupt:
            interrupted = True
            result = subprocess.CompletedProcess(args=cmd, returncode=130, stdout="", stderr="")
    elapsed = time.monotonic() - t0
    metadata: dict[str, Any] = {}
    if artifacts is not None:
        metadata["run_id"] = run.run_id if run is not None else None
        metadata["artifact_dir"] = str(artifacts.step_dir.relative_to(Path.cwd()))
    if is_pytest:
        if containment_error is not None:
            metadata["diagnosis"] = "pytest_containment_unproven"
            metadata["termination_reason"] = f"pytest containment did not quiesce: {containment_error}"
        metadata.update(_pytest_command_metadata(cmd))
        metadata["heartbeat_s"] = _pytest_heartbeat_interval()
        metadata["timeout_s"] = _pytest_timeout_s()
        metadata["stall_timeout_s"] = _pytest_stall_timeout_s()
        metadata["term_grace_s"] = _pytest_term_grace_s()
        metadata["resource_interval_s"] = _pytest_resource_interval_s()
        metadata["pytest_tmpfs"] = pytest_tmpfs
        metadata["pytest_tmpfs_budget_mb"] = pytest_tmpfs_budget_mb
        metadata["pytest_runtime_policy"] = runtime_policy.to_dict() if runtime_policy is not None else None
        metadata["progress_path"] = str(PYTEST_PROGRESS_PATH)
        metadata["events_path"] = str(PYTEST_EVENTS_PATH)
        metadata["events_dir"] = str(PYTEST_EVENTS_DIR)
        metadata["selection_path"] = str(PYTEST_SELECTION_PATH)
        metadata["summary_path"] = str(PYTEST_SUMMARY_PATH)
        metadata["output_path"] = str(PYTEST_OUTPUT_PATH)
        metadata["resources_path"] = str(CURRENT_RESOURCES_PATH)
        metadata["postmortem_path"] = str(CURRENT_POSTMORTEM_PATH)
        metadata["containment_path"] = str(PYTEST_CONTAINMENT_PATH)
        metadata["basetemp_cleanup"] = str(basetemp_cleanup) if basetemp_cleanup is not None else None
        junit_paths = [
            str(path) for path in _pytest_artifact_paths(cmd) if path.suffix == ".xml" or path.name.endswith(".xml")
        ]
        if junit_paths:
            metadata["junitxml_path"] = junit_paths[-1]
        report_path = _pytest_json_report_path(cmd)
        report = _read_pytest_report(report_path)
        if report is not None:
            metadata.update(_pytest_metadata_from_report(report, report_path=report_path))
            metadata["report_status"] = "present"
            if artifacts is not None:
                durable_report_path = artifacts.step_dir / PYTEST_CANONICAL_REPORT_NAME
                try:
                    shutil.copyfile(report_path, durable_report_path)
                except OSError:
                    metadata["report_path"] = None
                else:
                    metadata["report_path"] = str(
                        durable_report_path.relative_to(run.root if run is not None else ROOT)
                    )
                    if report_path != durable_report_path:
                        with contextlib.suppress(OSError):
                            report_path.unlink()
        else:
            # Fallback: terminal scraping when the structured report is
            # missing (pytest crashed before writing it, or the plugin is
            # disabled in some lab profile).
            fallback = _parse_pytest_test_count(result.stdout + "\n" + result.stderr)
            if fallback is not None:
                metadata["count"] = fallback
            metadata["report_path"] = None
            metadata["report_status"] = "missing"
        progress_path = (
            artifacts.progress_path
            if artifacts is not None and artifacts.progress_path.exists()
            else PYTEST_PROGRESS_PATH
        )
        progress = _read_json_artifact(progress_path)
        if progress is not None:
            event = progress.get("event")
            if isinstance(event, str):
                metadata["progress_event"] = event
            termination_reason = progress.get("termination_reason")
            if isinstance(termination_reason, str):
                metadata["termination_reason"] = termination_reason
        selection_path = artifacts.selection_path if artifacts is not None else PYTEST_SELECTION_PATH
        if interrupted or containment_error is not None:
            _recover_worker_collection_facts(
                events_dir=(
                    artifacts.events_dir
                    if artifacts is not None
                    else Path(env.get("POLYLOGUE_PYTEST_EVENTS_DIR", str(PYTEST_EVENTS_DIR)))
                ),
                selection_path=selection_path,
            )
        selection = _read_json_artifact(selection_path)
        if selection is not None:
            selected_count = selection.get("selected_count")
            deselected_count = selection.get("deselected_count")
            if isinstance(selected_count, int):
                metadata["selected_count"] = selected_count
            if isinstance(deselected_count, int):
                metadata["deselected_count"] = deselected_count
            collection_duration_s = selection.get("collection_duration_s")
            if isinstance(collection_duration_s, (int, float)):
                metadata["collection_duration_s"] = collection_duration_s
        summary_path = artifacts.summary_path if artifacts is not None else PYTEST_SUMMARY_PATH
        summary = _read_json_artifact(summary_path)
        if summary is not None:
            slowest_reports = summary.get("slowest_reports")
            if isinstance(slowest_reports, list):
                metadata["slowest_report_count"] = len(slowest_reports)
        containment_path = artifacts.containment_path if artifacts is not None else PYTEST_CONTAINMENT_PATH
        containment = _read_json_artifact(containment_path)
        if containment is not None:
            for source_key, metadata_key in (
                ("mode", "containment_mode"),
                ("unit", "containment_unit"),
                ("cgroup_path", "containment_cgroup_path"),
                ("controller_pid", "pytest_controller_pid"),
                ("controller_pgid", "pytest_controller_pgid"),
                ("signals_sent", "containment_signals_sent"),
                ("escalated_to_sigkill", "containment_escalated_to_sigkill"),
            ):
                if source_key in containment:
                    metadata[metadata_key] = containment[source_key]
        resource_summary: dict[str, Any] = {}
        last_resource_row: dict[str, Any] | None = None
        if artifacts is not None and artifacts.resources_path.exists():
            sample_count = 0
            first_resource_row: dict[str, Any] | None = None
            peak_rss = 0
            peak_pss: int | None = None
            peak_anon_pss: int | None = None
            peak_file_pss: int | None = None
            peak_swap_pss: int | None = None
            peak_process_count = 0
            peak_basetemp_size_kb: int | None = None
            peak_basetemp_allocated_kb: int | None = None
            with artifacts.resources_path.open(encoding="utf-8") as resource_handle:
                for line in resource_handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    if first_resource_row is None:
                        first_resource_row = row
                    last_resource_row = row
                    sample_count += 1
                    peak_rss = max(peak_rss, int(row.get("tree_rss_kb") or 0))
                    if row.get("tree_pss_kb") is not None:
                        peak_pss = max(peak_pss or 0, int(row["tree_pss_kb"]))
                    if row.get("tree_anon_pss_kb") is not None:
                        peak_anon_pss = max(peak_anon_pss or 0, int(row["tree_anon_pss_kb"]))
                    if row.get("tree_file_pss_kb") is not None:
                        peak_file_pss = max(peak_file_pss or 0, int(row["tree_file_pss_kb"]))
                    if row.get("tree_swap_pss_kb") is not None:
                        peak_swap_pss = max(peak_swap_pss or 0, int(row["tree_swap_pss_kb"]))
                    peak_process_count = max(peak_process_count, int(row.get("process_count") or 0))
                    if row.get("basetemp_size_kb") is not None:
                        peak_basetemp_size_kb = max(
                            peak_basetemp_size_kb or 0,
                            int(row["basetemp_size_kb"]),
                        )
                    if row.get("basetemp_allocated_kb") is not None:
                        peak_basetemp_allocated_kb = max(
                            peak_basetemp_allocated_kb or 0,
                            int(row["basetemp_allocated_kb"]),
                        )
            if sample_count:
                resource_summary = {
                    "resource_sample_count": sample_count,
                    "peak_tree_rss_kb": peak_rss,
                    "peak_tree_rss_mb": round(peak_rss / 1024, 1),
                    "peak_tree_pss_kb": peak_pss,
                    "peak_tree_pss_mb": round(peak_pss / 1024, 1) if peak_pss is not None else None,
                    "peak_tree_anon_pss_kb": peak_anon_pss,
                    "peak_tree_file_pss_kb": peak_file_pss,
                    "peak_tree_swap_pss_kb": peak_swap_pss,
                    "start_tree_swap_pss_kb": (
                        first_resource_row.get("tree_swap_pss_kb") if first_resource_row is not None else None
                    ),
                    "final_tree_swap_pss_kb": (
                        last_resource_row.get("tree_swap_pss_kb") if last_resource_row is not None else None
                    ),
                    "tree_swap_pss_delta_kb": _nonnegative_int_delta(
                        first_resource_row, last_resource_row, "tree_swap_pss_kb"
                    ),
                    "tree_read_bytes": last_resource_row.get("tree_read_bytes") if last_resource_row else None,
                    "tree_write_bytes": last_resource_row.get("tree_write_bytes") if last_resource_row else None,
                    "tree_cancelled_write_bytes": (
                        last_resource_row.get("tree_cancelled_write_bytes") if last_resource_row else None
                    ),
                    "tree_read_bytes_delta": _nonnegative_int_delta(
                        first_resource_row, last_resource_row, "tree_read_bytes"
                    ),
                    "tree_write_bytes_delta": _nonnegative_int_delta(
                        first_resource_row, last_resource_row, "tree_write_bytes"
                    ),
                    "tree_cancelled_write_bytes_delta": _nonnegative_int_delta(
                        first_resource_row, last_resource_row, "tree_cancelled_write_bytes"
                    ),
                    "peak_process_count": peak_process_count,
                    "peak_basetemp_size_kb": peak_basetemp_size_kb,
                    "peak_basetemp_size_mb": (
                        round(peak_basetemp_size_kb / 1024, 1) if peak_basetemp_size_kb is not None else None
                    ),
                    "peak_basetemp_allocated_kb": peak_basetemp_allocated_kb,
                    "peak_basetemp_allocated_mb": (
                        round(peak_basetemp_allocated_kb / 1024, 1) if peak_basetemp_allocated_kb is not None else None
                    ),
                }
                metadata.update(resource_summary)
        diagnosis = classify_pytest_result(
            returncode=result.returncode,
            termination_reason=metadata.get("termination_reason")
            if isinstance(metadata.get("termination_reason"), str)
            else None,
            report_present=metadata.get("report_status") == "present",
            summary=summary if isinstance(summary, dict) else None,
            progress_event=metadata.get("progress_event") if isinstance(metadata.get("progress_event"), str) else None,
        )
        if containment_error is not None:
            diagnosis = "pytest_containment_unproven"
        if interrupted:
            diagnosis = "pytest_interrupted"
            metadata["termination_reason"] = "operator_interrupt"
        metadata["diagnosis"] = diagnosis
        termination_reason = (
            metadata.get("termination_reason") if isinstance(metadata.get("termination_reason"), str) else None
        )
        workload_receipt = _pytest_workload_receipt(
            label=label,
            cmd=cmd,
            elapsed_s=elapsed,
            returncode=result.returncode,
            termination_reason=termination_reason,
            resource_summary=resource_summary,
            last_resource_sample=last_resource_row,
            tmpfs_budget_mb=pytest_tmpfs_budget_mb,
            basetemp_cleanup=basetemp_cleanup,
            concurrency=max(1, pytest_concurrency),
        )
        metadata["workload_receipt"] = workload_receipt
        if artifacts is not None:
            postmortem = {
                "updated_at": utc_now(),
                "diagnosis": diagnosis,
                "returncode": result.returncode,
                "report_status": metadata.get("report_status"),
                "progress_event": metadata.get("progress_event"),
                "summary_exitstatus": summary.get("exitstatus") if isinstance(summary, dict) else None,
                "containment_mode": metadata.get("containment_mode"),
                "containment_unit": metadata.get("containment_unit"),
                "containment_cgroup_path": metadata.get("containment_cgroup_path"),
                "pytest_controller_pid": metadata.get("pytest_controller_pid"),
                "pytest_controller_pgid": metadata.get("pytest_controller_pgid"),
                "containment_signals_sent": metadata.get("containment_signals_sent"),
                "containment_escalated_to_sigkill": metadata.get("containment_escalated_to_sigkill"),
                "workload_receipt": workload_receipt,
                **resource_summary,
            }
            artifacts.postmortem_path.write_text(json.dumps(postmortem, indent=2, ensure_ascii=False) + "\n")
    elif interrupted:
        metadata.update({"diagnosis": "verification_interrupted", "termination_reason": "operator_interrupt"})
    if result.returncode == 0:
        sys.stderr.write(f"ok ({elapsed:.1f}s)\n")
    else:
        sys.stderr.write(f"FAILED ({elapsed:.1f}s)\n")
        if result.stdout.strip():
            sys.stderr.write(result.stdout + "\n")
        if result.stderr.strip():
            sys.stderr.write(result.stderr + "\n")
    if run is not None and artifacts is not None:
        finalized_step = run.finish_step(
            step_id=artifacts.step_id, result={"duration_s": round(elapsed, 2), "exit": result.returncode, **metadata}
        )
        if isinstance(finalized_step, dict):
            for key in ("statistics", "statistics_path"):
                if key in finalized_step:
                    metadata[key] = finalized_step[key]
    if is_pytest and artifacts is not None:
        copy_current_pytest_artifacts(
            Path.cwd(),
            artifacts,
            legacy_paths={
                "progress_path": PYTEST_PROGRESS_PATH,
                "events_merged_path": PYTEST_EVENTS_PATH,
                "selection_path": PYTEST_SELECTION_PATH,
                "summary_path": PYTEST_SUMMARY_PATH,
                "output_path": PYTEST_OUTPUT_PATH,
            },
        )
    return result.returncode, elapsed, metadata


def _pytest_command_basetemp(
    cmd: Sequence[str], *, cwd: str | None, env: Mapping[str, str] | None = None
) -> Path | None:
    """Return the effective explicit pytest basetemp, if the command has one."""
    raw_path: str | None = None
    addopts: list[str] = []
    if env is not None:
        with contextlib.suppress(ValueError):
            addopts = shlex.split(env.get("PYTEST_ADDOPTS", ""))
    arguments = [*addopts, *cmd]
    for index, argument in enumerate(arguments):
        if argument.startswith("--basetemp="):
            raw_path = argument.partition("=")[2]
        elif argument == "--basetemp" and index + 1 < len(arguments):
            raw_path = arguments[index + 1]
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(cwd) if cwd is not None else Path.cwd()) / path


def _subprocess_env() -> dict[str, str]:
    env = normalize_pytest_basetemp_env(os.environ)
    # Tests and verification helpers may inspect Git, but observational reads
    # must not refresh the index and invalidate the exact-head mutation watch.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["POLYLOGUE_ROOT"] = str(ROOT)
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not inherited_pythonpath else f"{ROOT}{os.pathsep}{inherited_pythonpath}"
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pycache")
    TESTMON_DATA.parent.mkdir(parents=True, exist_ok=True)
    env["TESTMON_DATAFILE"] = str(TESTMON_DATA)
    env["POLYLOGUE_PYTEST_EVENTS_DIR"] = str(ROOT / PYTEST_EVENTS_DIR)
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(ROOT / PYTEST_EVENTS_PATH)
    env["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(ROOT / PYTEST_SELECTION_PATH)
    env["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(ROOT / PYTEST_SUMMARY_PATH)
    return env


def _stop_after_failed_step(label: str) -> bool:
    return label.startswith("pytest") or label in {"lab smoke", "bench slo"}


def _seed_shard_failure_requires_stop(step: Mapping[str, Any], *, shard_complete: bool) -> bool:
    """Stop shard admission after harness failure while retaining red-test evidence.

    A normal pytest exit 1 with a structured ``pytest_failed`` diagnosis is
    useful seed evidence: later shards can still populate the resumable
    dependency graph. Timeouts, resource refusals, worker/internal errors,
    usage errors, and unclassified failures mean the harness is no longer
    healthy enough to admit another expensive shard.
    """
    exit_code = step.get("exit")
    if exit_code == 0:
        return False
    return not (exit_code == 1 and step.get("diagnosis") == "pytest_failed" and shard_complete)


# ── step builder ────────────────────────────────────────────────────


def build_verify_steps(
    *,
    quick: bool,
    lab: bool,
    skip_slow: bool,
    commit: bool = False,
    seed_testmon: bool = False,
    resume_testmon_seed: bool = False,
    full_pytest: bool = False,
    broad_testmon: bool = False,
) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]

    if not commit:
        steps.extend(
            [
                ("render all", _devtools_cmd("render all", "--check")),
                ("verify layering", _devtools_cmd("verify layering")),
                ("lab graph strict", _devtools_cmd("lab graph", "--strict")),
                ("verify closure-matrix", _devtools_cmd("verify closure-matrix")),
                ("lab schema roundtrip", _devtools_cmd("lab schema roundtrip", "--all")),
                ("verify manifests", _devtools_cmd("verify manifests")),
                ("verify ci-workflows", _devtools_cmd("verify ci-workflows")),
                ("verify catalog-bypasses", _devtools_cmd("verify catalog-bypasses")),
                ("verify doc-commands", _devtools_cmd("verify doc-commands")),
                ("verify docs-coverage", _devtools_cmd("verify docs-coverage")),
                ("verify test-infra-currency", _devtools_cmd("verify test-infra-currency")),
                ("verify pytest-timeout-overrides", _devtools_cmd("verify pytest-timeout-overrides")),
                ("verify degrade-loudly", _devtools_cmd("verify degrade-loudly")),
                # Static, archive-independent, sub-second: an index bump that
                # lands without its lifecycle.py delta declaration silently
                # downgrades every existing generation to a full raw replay
                # (polylogue-9rw0). Gated here, not behind --lab, because the
                # failure surfaces as an unqueryable live archive rather than
                # as a test failure.
                ("lab policy schema-versioning", _devtools_cmd("lab policy schema-versioning")),
                # Static, archive-independent, sub-second: catches the gap the
                # schema-versioning gate above cannot see -- a parser/classifier
                # changing what it accepts for identical input bytes with no
                # version bump at all (polylogue-gucv; PR #3428 is the
                # concrete case that shipped green against the version-keyed
                # gate above).
                ("lab policy classifier-fingerprints", _devtools_cmd("lab policy classifier-fingerprints")),
                # ~15-30s, fully deterministic (wall-clock timing is masked
                # before comparison, see devtools/verify_demo_tour_freshness.py).
                # Unlike backlog-hygiene/bead-graph below, this check's failure
                # count does not scale with total backlog/bead-corpus size --
                # it is a fixed-cost diff against one committed fixture that
                # only drifts when demo/insight code actually changes shape
                # (polylogue-ze5i: moved out of --lab after the committed
                # fixture was regenerated to match a `healed_tiers` field the
                # demo receipts code had already grown).
                ("lab policy demo-tour-freshness", _devtools_cmd("lab policy demo-tour-freshness")),
                # Static, archive-independent, sub-second: forbids the exact
                # byte-mutation-before-hashing pattern that produced
                # polylogue-u19l's Codex append-header bug (a synthesized
                # literal spliced onto captured bytes before they reached the
                # content hasher, permanently defeating live-source
                # byte-identity verification for ~59GB of raw rows).
                ("lab policy raw-payload-hash-purity", _devtools_cmd("lab policy raw-payload-hash-purity")),
                # Static, archive-independent, sub-second: forbids a NEW
                # occurrence of polylogue-hith/qkuq's already-fixed
                # attachment-id bug shape (comparison identity minted from
                # positional/index data, unstable across export vintages
                # that reorder entries) -- polylogue-gysk3 found the same
                # hazard still live for provider_message_id.
                ("lab policy position-derived-identity", _devtools_cmd("lab policy position-derived-identity")),
                # Static, archive-independent, sub-second: forbids a NEW
                # unreachable (frontier state, dispatched actuator) pairing in
                # polylogue/storage/raw_reconciler.py -- polylogue-w32w found
                # UNRESOLVED_PROVENANCE paired with the dispatched
                # REFINE_QUARANTINE actuator, an actuator no path could ever
                # select, and 4,174 blockers accumulated behind it for weeks
                # before anyone noticed. The runtime constructor guard
                # (RawAuthorityFrontierItem.__post_init__) only fires when
                # something actually constructs the bad combination; this
                # lint re-checks every literal pairing at review time.
                (
                    "lab policy raw-authority-frontier-executability",
                    _devtools_cmd("lab policy raw-authority-frontier-executability"),
                ),
                # Static, archive-independent, sub-second: forbids a NEW
                # top-level def named table_exists/column_exists/index_exists
                # (or a _-prefixed/_sync/_async variant) outside
                # polylogue/storage/introspection.py -- the ~25-copy
                # duplication polylogue-48h consolidated into that module.
                (
                    "lab policy table-exists-duplication",
                    _devtools_cmd("lab policy table-exists-duplication"),
                ),
                # Publication gate. Committed provider schema packages are
                # public artifacts; this blocks local provenance
                # (bundle_scopes/representative_paths) and scans for secrets.
                # It exits non-zero on blockers and had never been wired to
                # anything, while 76 blockers sat in the committed tree.
                (
                    "schema promotion audit",
                    [
                        sys.executable,
                        "-m",
                        "polylogue.schemas.promotion_audit",
                        "polylogue/schemas",
                        "--output",
                        str(PYTEST_REPORT_DIR / "schema-promotion-audit.json"),
                    ],
                ),
                # Static, archive-independent: the committed incident ledger
                # must agree with the structured Beads dependency graph and
                # every receipt/reference must resolve before quick verify is
                # allowed to report green.
                (
                    "incident coverage ledger",
                    [
                        sys.executable,
                        "-m",
                        "devtools.incident_coverage_ledger",
                        "--beads-export",
                        str(ROOT / ".beads" / "issues.jsonl"),
                    ],
                ),
            ]
        )

    if not quick and not commit:
        _report_dir = PYTEST_JUNIT_REPORT_DIR
        _report_dir.mkdir(parents=True, exist_ok=True)
        PYTEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        # Scale-tier policy (issue #1183): default verify includes
        # ``scale_small`` but excludes ``scale_medium`` / ``scale_large``.
        # ``--lab`` lets the medium tier in; the large tier is reserved
        # for nightly CI and explicit ``devtools bench campaign``
        # invocations.
        scale_marker_expr = "not scale_large" if lab else "not scale_medium and not scale_large"
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--ignore=tests/integration",
            # Benchmark files are an explicit campaign surface.  A number of
            # them are correctness-shaped and lack the benchmark marker, so a
            # marker expression alone cannot keep performance probes out of
            # the correctness/testmon corpus.
            "--ignore=tests/benchmarks",
            "--durations=10",
            f"--junitxml={_report_dir}/verify-latest.xml",
            "--json-report",
            "--json-report-omit=collectors,log,streams,warnings",
            f"--json-report-file={PYTEST_REPORT_PATH}",
            "-p",
            "devtools.pytest_progress_plugin",
        ]
        # Benchmark cases are an explicit campaign surface, not part of the
        # correctness/testmon seed.  Keeping them out here is important: a
        # benchmark marker is not necessarily paired with ``slow`` or a scale
        # marker, and a serial shard would otherwise spend minutes executing a
        # performance probe before it can checkpoint any correctness nodes.
        base_marker = f"not benchmark and {scale_marker_expr}"
        if skip_slow:
            base_marker = f"not slow and {base_marker}"
        if seed_testmon:
            # Collection produces the exact corpus contract before any testmon
            # write. Shards below are generated from this ledger and run one
            # at a time, so pytest-testmon has exactly one SQLite writer.
            pytest_cmd.extend(["-m", base_marker, "--collect-only", "-n", "0"])
            label = "pytest seed-testmon collect (resume)" if resume_testmon_seed else "pytest seed-testmon collect"
            steps.append((label, pytest_cmd))
        elif full_pytest:
            # #1775: the full diagnostic runs as two lanes. The bulk lane keeps
            # xdist parallelism but deselects wall-clock-bound tests; the
            # isolated lane reruns those (``load_sensitive``/``tui`` — timing
            # budgets, loopback-socket timeouts, TUI render timing) single-
            # process with a stable order, so worker contention can no longer
            # flake them. Both lanes are correctness blockers; the split only
            # removes the scheduling jitter that made ``--all`` an unreliable
            # completion gate.
            bulk_cmd = [
                *pytest_cmd,
                "-m",
                f"({base_marker}) and not load_sensitive and not tui",
                *_pytest_worker_args(),
            ]
            steps.append((BROAD_PYTEST_STEP_LABELS["full_parallel"], bulk_cmd))

            def _isolated_report_arg(arg: str) -> str:
                # Keep the bulk lane's canonical report artifacts intact for
                # _compare_against_last; the isolated lane writes its own files.
                if arg.startswith("--junitxml="):
                    return f"--junitxml={_report_dir}/verify-latest-isolated.xml"
                if arg.startswith("--json-report-file="):
                    return f"--json-report-file={PYTEST_REPORT_DIR / 'last-pytest-isolated.json'}"
                return arg

            isolated_cmd = [_isolated_report_arg(arg) for arg in pytest_cmd]
            isolated_cmd.extend(["-m", f"({base_marker}) and (load_sensitive or tui)", "-p", "no:randomly", "-n", "0"])
            steps.append((BROAD_PYTEST_STEP_LABELS["load_sensitive"], isolated_cmd))
        else:
            pytest_cmd.extend(["-m", base_marker, "--testmon", *_pytest_worker_args()])
            pytest_cmd.append("--testmon-forceselect")
            label = BROAD_PYTEST_STEP_LABELS["testmon_broad"] if broad_testmon else "pytest testmon"
            steps.append((label, pytest_cmd))

    if lab:
        steps.append(("lab smoke", _devtools_cmd("lab smoke", "run", "archive-smoke", "--tier", "0")))
        steps.append(("bench slo", _devtools_cmd("bench slo", "--include-lab")))
        steps.append(("lab policy timestamp-doctrine", _devtools_cmd("lab policy timestamp-doctrine")))
        steps.append(("lab policy insight-honesty", _devtools_cmd("lab policy insight-honesty")))
        steps.append(("lab policy demo-packet-registry", _devtools_cmd("lab policy demo-packet-registry")))
        steps.append(("lab policy docs-drift", _devtools_cmd("lab policy docs-drift")))
        steps.append(
            ("lab policy campaign-archive-boundaries", _devtools_cmd("lab policy campaign-archive-boundaries"))
        )
        steps.append(("lab policy acceptance-contracts", _devtools_cmd("lab policy acceptance-contracts")))
        steps.append(
            (
                "lab policy acceptance-contract-reconcile",
                _devtools_cmd("lab policy acceptance-contract-reconcile"),
            )
        )
        steps.append(
            (
                "lab policy acceptance-contract-apply",
                _devtools_cmd("lab policy acceptance-contract-apply"),
            )
        )
        # backlog-hygiene and bead-graph are corpus-wide backlog-debt scans
        # (findings scale with the total count of open Beads issues, not
        # with this change's diff) -- they stay --lab-only/scheduled rather
        # than default- or CI-gated. Gating either on a merge would block
        # every PR in the repo until the entire pre-existing backlog is
        # cleaned up (485 backlog-hygiene findings / 225 missing-AC beads
        # measured 2026-08-02), which is periodic hygiene debt, not a
        # per-change regression signal. Wired into the CircleCI nightly
        # schedule instead (polylogue-ze5i) so continuous failure is at
        # least visible, and the backlog itself is tracked by follow-up
        # beads rather than left to rot silently.
        steps.append(("lab policy backlog-hygiene", _devtools_cmd("lab policy backlog-hygiene")))
        steps.append(("lab policy bead-graph", _devtools_cmd("lab policy bead-graph")))
    return steps


# ── comparison against last run ────────────────────────────────────


def _compare_against_last(step_results: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable regression flags."""
    entries = _load_history()
    if len(entries) < 1:
        return []
    flags: list[str] = []
    for s in step_results:
        name = s.get("name")
        if not isinstance(name, str):
            continue
        prev = next(
            (
                prior.get("duration_s")
                for entry in reversed(entries)
                if entry.get("tier") != "focused-test"
                for prior in entry.get("steps", [])
                if isinstance(prior, dict)
                and prior.get("name") == name
                and isinstance(prior.get("duration_s"), (int, float))
            ),
            None,
        )
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
    """Resolve HEAD through the bounded authority-sensitive Git probe."""
    return _git_commit("HEAD")


def _git_committed_tree() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _git_commit(ref: str) -> str | None:
    """Resolve a mutable Git ref once for an authority-sensitive run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=ROOT,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or result.stderr.strip():
        return None
    commit = result.stdout.strip()
    return commit or None


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


def _pytest_worker_args(*, maximum: int | None = None) -> list[str]:
    """Return the managed worker count, optionally capped for a bounded lane."""
    workers = adaptive_pytest_worker_count(os.environ)
    if maximum is not None:
        workers = min(workers, maximum)
    return ["--dist=loadgroup", "-n", str(workers)]


BROAD_PYTEST_STEP_LABELS = {
    "seed": "pytest seed-testmon",
    "seed_resume": "pytest seed-testmon (resume)",
    "full_parallel": "pytest full (parallel)",
    "load_sensitive": "pytest load-sensitive (isolated)",
    "testmon_broad": "pytest testmon (broad)",
}


def _pytest_command_concurrency(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> int:
    """Return a fail-closed reservation for the final pytest command.

    ``-n auto`` can launch one worker per logical CPU.  Reserve that maximum
    instead of guessing one worker; an unrecognised xdist value is treated the
    same way so malformed or future values cannot weaken admission.
    """
    request = pytest_command_worker_request(cmd)
    if request is None:
        return 0
    if request == "auto":
        auto_workers = (env if env is not None else os.environ).get("PYTEST_XDIST_AUTO_NUM_WORKERS", "").strip()
        if auto_workers:
            try:
                configured = int(auto_workers)
            except ValueError:
                configured = 0
            if configured > 0:
                return configured
    try:
        return max(0, int(request))
    except ValueError:
        return max(1, os.cpu_count() or 1)


def _pytest_uses_full_suite_basetemp(label: str) -> bool:
    """Whether this pytest step can materialize the measured full-suite tree."""
    return label in BROAD_PYTEST_STEP_LABELS.values() or label.startswith("pytest seed-testmon shard ")


_BROAD_TESTMON_CHANGED_PATHS = {
    "pyproject.toml",
    "tests/conftest.py",
}


def _changed_paths(base_commit: str, head_commit: str) -> set[str]:
    """Return changes between immutable start-time Git authorities."""
    changed: set[str] = set()
    commands = (
        ["git", "diff", "--no-renames", "--name-only", "-z", head_commit, "--"],
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{base_commit}...{head_commit}", "--"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--"],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=5,
                cwd=ROOT,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PytestResourceError("testmon changed-path authority is unavailable") from exc
        if result.returncode != 0 or result.stderr.strip():
            raise PytestResourceError("testmon changed-path authority is unavailable")
        changed.update(os.fsdecode(raw_path) for raw_path in result.stdout.split(b"\0") if raw_path)
    return changed


def _default_testmon_is_broad_change(base_commit: str, head_commit: str) -> bool:
    """Return true when affected-test selection should be treated as broad."""
    return bool(_changed_paths(base_commit, head_commit) & _BROAD_TESTMON_CHANGED_PATHS)


def _changed_executable_paths(base_commit: str, head_commit: str) -> tuple[str, ...]:
    """Return changed paths whose behavior should select at least one test."""
    roots = ("polylogue/", "devtools/", "tests/", "packaging/")
    exact = {"pyproject.toml", "uv.lock"}
    return tuple(
        sorted(path for path in _changed_paths(base_commit, head_commit) if path in exact or path.startswith(roots))
    )


def _testmon_coverage_identity(executable_paths: Sequence[str]) -> dict[str, Any]:
    """Identify the exact worktree contents covered by an affected/full run."""
    return {
        "worktree_fingerprint": worktree_fingerprint(),
        "executable_paths": list(executable_paths),
    }


def _matching_testmon_coverage(executable_paths: Sequence[str]) -> str | None:
    """Return the receipt kind proving that zero new selection is legitimate."""
    identity = _testmon_coverage_identity(executable_paths)
    affected = _read_json_artifact(TESTMON_AFFECTED_STAMP)
    selected_count = affected.get("selected_count") if isinstance(affected, dict) else None
    if (
        isinstance(affected, dict)
        and affected.get("protocol_version") == 1
        and affected.get("status") == "complete"
        and isinstance(affected.get("timestamp"), str)
        and bool(affected.get("timestamp"))
        and isinstance(affected.get("run_id"), str)
        and bool(affected.get("run_id"))
        and isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and selected_count > 0
        and affected.get("identity") == identity
    ):
        return "successful_affected_run"
    return None


def _record_testmon_affected_coverage(*, executable_paths: Sequence[str], selected_count: int, run_id: str) -> None:
    """Persist proof that testmon exercised dependencies for these contents."""
    _atomic_write_json(
        TESTMON_AFFECTED_STAMP,
        {
            "protocol_version": 1,
            "status": "complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "selected_count": selected_count,
            "identity": _testmon_coverage_identity(executable_paths),
        },
    )


def _testmon_preflight(*, seed_testmon: bool, full_pytest: bool, quick: bool, commit: bool) -> str | None:
    if quick or commit or seed_testmon or full_pytest:
        return None
    seed_message = (
        "verify: pytest-testmon is not seeded; run `devtools verify --seed-testmon` "
        "to create .cache/testmon/testmondata and .cache/testmon/seed.json "
        "before using the default affected-test path.\n"
    )
    if not TESTMON_DATA.exists():
        return seed_message
    if not TESTMON_SEED_STAMP.exists():
        attempt = _read_testmon_seed_attempt()
        if (
            attempt is not None
            and stamp_from_attempt(
                attempt,
                TESTMON_DATA,
                checkout_root=ROOT,
                protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
                published_marker=False,
            )
            is not None
        ):
            sys.stderr.write(
                "verify: using a validated complete pytest-testmon graph from a red seed attempt; "
                "the release baseline remains red.\n"
            )
            return None
        return seed_message
    stamp = validate_stamp(
        TESTMON_SEED_STAMP,
        TESTMON_DATA,
        checkout_root=ROOT,
        protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
    )
    if stamp is None:
        return (
            "verify: pytest-testmon seed state is unreadable, stale, malformed, or not graph-complete; run "
            "`devtools verify --seed-testmon` to rebuild the dependency baseline.\n"
        )
    return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _testmon_seed_identity(
    *,
    git_head: str | None,
    git_tree: str | None = None,
    skip_slow: bool,
    lab: bool,
    terminal_authorization: str | None = None,
) -> dict[str, Any]:
    runtime_identity = testmon_runtime_identity(ROOT)
    if runtime_identity is None:
        raise RuntimeError("could not identify the active dependency environment and pytest harness")
    dependency_environment, pytest_harness = runtime_identity
    return {
        "git_head": git_head,
        "git_tree": git_tree,
        "worktree_fingerprint": worktree_fingerprint(),
        "python": sys.version,
        "skip_slow": skip_slow,
        "lab": lab,
        "terminal_authorization": terminal_authorization,
        "dependency_environment": dependency_environment,
        "pytest_harness": pytest_harness,
    }


def _read_testmon_seed_attempt() -> dict[str, Any] | None:
    payload = _read_json_artifact(TESTMON_SEED_ATTEMPT)
    return payload if isinstance(payload, dict) else None


def _recover_worker_collection_facts(*, events_dir: Path, selection_path: Path) -> bool:
    """Publish xdist worker collection facts when its controller never finishes.

    The progress plugin normally merges these facts during controller
    ``pytest_sessionfinish``. Interrupted containment bypasses that hook, so
    the runner recovers the same canonical worker fact before it terminalizes
    the durable step record.
    """
    merged = merge_worker_collection_payloads(events_dir)
    if merged is None:
        return False
    selection = dict(merged)
    selection.update(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": "runner",
            "pid": os.getpid(),
            "recovered_after_interruption": True,
        }
    )
    try:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = selection_path.with_name(f"{selection_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(selection_path)
    except OSError:
        return False
    return True


def _flatten_seed_outcomes(attempt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten outcomes from every interrupted attempt, newest result winning."""
    if attempt is None:
        return []
    flattened: dict[str, dict[str, Any]] = {}
    for field in ("prior_node_outcomes", "node_outcomes"):
        raw = attempt.get(field)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("nodeid"), str) and item["nodeid"]:
                flattened[item["nodeid"]] = dict(item)
    return [flattened[nodeid] for nodeid in sorted(flattened)]


def _testmon_release_baseline_permission() -> bool | None:
    """Return release permission for current testmon state, or ``None`` when not applicable."""
    if TESTMON_SEED_STAMP.exists():
        stamp = validate_stamp(
            TESTMON_SEED_STAMP,
            TESTMON_DATA,
            checkout_root=ROOT,
            protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
        )
        return stamp.release_baseline_allowed if stamp is not None else False
    attempt = _read_testmon_seed_attempt()
    if attempt is None:
        return False
    stamp = stamp_from_attempt(
        attempt,
        TESTMON_DATA,
        checkout_root=ROOT,
        protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
        published_marker=False,
    )
    return stamp.release_baseline_allowed if stamp is not None else False


def _safe_testmon_artifact_dir(raw: object, *, require_run_root: bool = False) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    checkout_root = Path.cwd().resolve()
    if require_run_root and path.is_absolute():
        return None
    resolved = (path if path.is_absolute() else checkout_root / path).resolve()
    try:
        resolved.relative_to(checkout_root)
        if require_run_root:
            resolved.relative_to((checkout_root / ".cache" / "verify" / "runs").resolve())
    except ValueError:
        return None
    return resolved


def _testmon_seed_expected_nodeids(attempt: Mapping[str, Any]) -> list[str]:
    """Recover the seed ledger, including after an abrupt outer-run exit."""
    expected = attempt.get("expected_nodeids")
    if isinstance(expected, list) and expected:
        if (
            any(not isinstance(nodeid, str) or not nodeid for nodeid in expected)
            or len(set(expected)) != len(expected)
            or not isinstance(attempt.get("expected_count"), int)
            or isinstance(attempt.get("expected_count"), bool)
            or attempt.get("expected_count") != len(expected)
            or not isinstance(attempt.get("expected_digest"), str)
            or attempt.get("expected_digest") != hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
        ):
            return []
        return list(expected)

    artifact_dir = _safe_testmon_artifact_dir(attempt.get("artifact_dir"), require_run_root=True)
    if artifact_dir is None:
        return []
    for selection_path in sorted(artifact_dir.glob("steps/*/selection.json")):
        selection = _read_json_artifact(selection_path)
        if not isinstance(selection, dict):
            continue
        omitted = selection.get("selected_nodeids_omitted")
        selected_count = selection.get("selected_count")
        if (
            not isinstance(omitted, int)
            or isinstance(omitted, bool)
            or omitted != 0
            or not isinstance(selected_count, int)
            or isinstance(selected_count, bool)
        ):
            continue
        selected = selection.get("selected_nodeids")
        if (
            isinstance(selected, list)
            and selected
            and all(isinstance(nodeid, str) and nodeid for nodeid in selected)
            and len(set(selected)) == len(selected)
            and selected_count == len(selected)
        ):
            return list(selected)
    return []


def _testmon_seed_resume_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return inputs that change which corpus a seed promises to cover."""
    return {
        key: identity.get(key)
        for key in (
            "git_tree",
            "worktree_fingerprint",
            "python",
            "skip_slow",
            "lab",
            "terminal_authorization",
            "dependency_environment",
            "pytest_harness",
        )
    }


def _testmon_seed_can_resume(identity: Mapping[str, Any]) -> bool:
    attempt = _read_testmon_seed_attempt()
    if attempt is None or not TESTMON_DATA.exists():
        return False
    prior_identity = attempt.get("identity")
    contract = _testmon_seed_resume_contract(identity)
    return (
        attempt.get("protocol_version") == TESTMON_SEED_PROTOCOL_VERSION
        and attempt.get("status") in {"running", "incomplete"}
        and isinstance(prior_identity, dict)
        and isinstance(contract["git_tree"], str)
        and bool(contract["git_tree"])
        and _testmon_seed_resume_contract(prior_identity) == contract
        and bool(_testmon_seed_expected_nodeids(attempt))
    )


def _prepare_testmon_seed_attempt(
    *,
    identity: Mapping[str, Any],
    run: VerifyRun,
    resume: bool,
) -> dict[str, Any]:
    prior = _read_testmon_seed_attempt() if resume else None
    expected = sorted(_testmon_seed_expected_nodeids(prior)) if prior is not None else []
    prior_outcomes = _flatten_seed_outcomes(prior)
    payload = {
        "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
        "status": "running",
        "identity": dict(identity),
        "resume": resume,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest() if expected else None,
        "prior_node_outcomes": prior_outcomes,
        "shards": list(prior.get("shards", [])) if prior is not None and isinstance(prior.get("shards"), list) else [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "artifact_dir": str(run.relative_run_dir),
        "testmon_data_before": _file_fingerprint(TESTMON_DATA),
        "binding": TestmonBinding(BindingMode.EXACT, str(ROOT.resolve())).as_dict(),
    }
    TESTMON_SEED_STAMP.unlink(missing_ok=True)
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    return payload


def _seed_selection_nodeids(selection: Mapping[str, Any]) -> list[str] | None:
    """Accept only a complete, untruncated collection ledger."""
    nodeids = selection.get("selected_nodeids")
    selected_count = selection.get("selected_count")
    omitted = selection.get("selected_nodeids_omitted")
    if (
        not isinstance(nodeids, list)
        or not nodeids
        or any(not isinstance(nodeid, str) or not nodeid for nodeid in nodeids)
        or len(set(nodeids)) != len(nodeids)
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count != len(nodeids)
        or not isinstance(omitted, int)
        or isinstance(omitted, bool)
        or omitted != 0
    ):
        return None
    return sorted(nodeids)


def _prepare_testmon_seed_shards(
    prepared: Mapping[str, Any],
    *,
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Persist the full planned corpus before the first testmon DB mutation."""
    expected = sorted(_testmon_seed_expected_nodeids(prepared)) if prepared.get("resume") else []
    if not expected:
        expected = _seed_selection_nodeids(selection or {}) or []
    prior_shards = validate_seed_shard_ledger(prepared.get("shards"), expected_nodeids=expected)
    shards = (
        prior_shards
        if prior_shards is not None
        else (
            seed_shard_plan(
                expected,
                shard_size=TESTMON_SEED_SHARD_SIZE,
                serial_nodeids=[
                    nodeid
                    for nodeid, markers in (selection or {}).get("selected_node_markers", {}).items()
                    if "load_sensitive" in markers or "tui" in markers
                ],
            )
            if expected
            else []
        )
    )
    payload = {
        **dict(prepared),
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(expected).encode()).hexdigest() if expected else None,
        "selection": dict(selection or {}),
        "shard_size": TESTMON_SEED_SHARD_SIZE,
        "shards": shards,
    }
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    return payload


def _seed_shard_command(
    collection_command: Sequence[str],
    shard: Mapping[str, Any],
    *,
    nodeids_file: Path,
) -> list[str]:
    """Build a bounded-argv, dynamically balanced pytest-testmon invocation.

    A full shard's node IDs can exceed the host's ``execve`` argument budget
    once ``systemd-run`` and the managed environment are included.  Pytest's
    response-file syntax keeps the authoritative node list in the run
    artifact while making the child command size independent of shard size.
    """
    nodeids = shard.get("nodeids")
    if not isinstance(nodeids, list) or not nodeids:
        raise ValueError("testmon seed shard is missing nodeids")
    nodeids_file.parent.mkdir(parents=True, exist_ok=True)
    nodeids_file.write_text("\n".join(nodeids) + "\n", encoding="utf-8")
    command: list[str] = []
    skip_next = False
    for argument in collection_command:
        if skip_next:
            skip_next = False
            continue
        if argument == "--collect-only":
            continue
        if argument in {"-n", "--numprocesses"}:
            skip_next = True
            continue
        if argument.startswith("--numprocesses=") or (argument.startswith("-n") and len(argument) > 2):
            continue
        command.append(argument)
    # Collection is deliberately serial, but execution is not. pytest-testmon
    # has an xdist-aware controller database; retaining the managed worker pool
    # here avoids turning a 20k-node seed into hours of serial fixture setup.
    if shard.get("execution_mode") == "serial":
        command.extend(["-n", "0", "--testmon", "--testmon-noselect", f"@{nodeids_file}"])
    else:
        command.extend(
            [
                *_pytest_worker_args(maximum=10),
                "--testmon",
                "--testmon-noselect",
                f"@{nodeids_file}",
            ]
        )
    return command


def _canonical_seed_nodeid(nodeid: str, expected_nodeids: Sequence[str]) -> str:
    """Map xdist's ``nodeid@group`` reports back to the collected node ID."""
    if nodeid in expected_nodeids:
        return nodeid
    candidates = [expected for expected in expected_nodeids if nodeid.startswith(expected + "@")]
    return max(candidates, key=len, default=nodeid)


def _seed_shard_outcomes(shards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the shard ledger in canonical node order for legacy readers."""
    outcomes: dict[str, dict[str, Any]] = {}
    for shard in shards:
        raw_outcomes = shard.get("node_outcomes")
        if not isinstance(raw_outcomes, list):
            continue
        for item in raw_outcomes:
            if isinstance(item, Mapping) and isinstance(item.get("nodeid"), str):
                outcomes[str(item["nodeid"])] = dict(item)
    return [outcomes[nodeid] for nodeid in sorted(outcomes)]


def _checkpoint_testmon_seed_shard(
    *,
    prepared: Mapping[str, Any],
    shard_index: int,
    step: Mapping[str, Any],
) -> dict[str, Any]:
    """Record one shard's result atomically before another shard may start."""
    expected = sorted(_testmon_seed_expected_nodeids(prepared))
    shards = validate_seed_shard_ledger(prepared.get("shards"), expected_nodeids=expected)
    if shards is None or shard_index < 1 or shard_index > len(shards):
        raise ValueError("testmon seed shard ledger is malformed")
    shard = dict(shards[shard_index - 1])
    nodeids = shard["nodeids"]
    artifact_dir = _safe_testmon_artifact_dir(step.get("artifact_dir"))
    selection = _read_json_artifact(artifact_dir / "selection.json") if artifact_dir is not None else None
    selected_raw = _seed_selection_nodeids(selection) if isinstance(selection, Mapping) else None
    selected = (
        sorted(_canonical_seed_nodeid(nodeid, nodeids) for nodeid in selected_raw) if selected_raw is not None else None
    )
    database = _testmon_database_state(nodeids)
    prior = {
        str(item["nodeid"]): item
        for item in shard.get("node_outcomes", [])
        if isinstance(item, Mapping) and isinstance(item.get("nodeid"), str)
    }
    outcomes = _seed_node_outcomes_from_events(
        artifact_dir / "events.jsonl" if artifact_dir is not None else Path(".missing-testmon-events"),
        expected_nodeids=nodeids,
        database=database,
        pytest_step=step,
        prior_node_outcomes=prior,
        use_database_fallback=False,
    )
    terminal = all(item.get("outcome") in TERMINAL_NODE_OUTCOMES for item in outcomes)
    selection_matches = selected == nodeids
    shard.update(
        {
            "status": SeedShardStatus.COMPLETE.value
            if selection_matches and terminal
            else SeedShardStatus.INCOMPLETE.value,
            "started_at": shard.get("started_at") or datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": step.get("exit"),
            "artifact_dir": step.get("artifact_dir"),
            "selection": dict(selection) if isinstance(selection, Mapping) else None,
            "database": database,
            "node_outcomes": outcomes,
            "pytest_step": dict(step),
        }
    )
    shards[shard_index - 1] = shard
    payload = {
        **dict(prepared),
        "status": "running",
        "shards": shards,
        "node_outcomes": _seed_shard_outcomes(shards),
        "testmon_data": _file_fingerprint(TESTMON_DATA),
    }
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    return payload


def _testmon_seed_terminal_authorized(prepared: Mapping[str, Any]) -> bool:
    identity = prepared.get("identity")
    return (
        isinstance(identity, Mapping)
        and identity.get("skip_slow") is True
        and identity.get("terminal_authorization") == TerminalAuthorization.NARROW_TERMINAL.value
    )


def _testmon_database_state(expected_nodeids: Sequence[str]) -> dict[str, Any]:
    graph = inspect_testmon_database(TESTMON_DATA, expected_nodeids)
    expected = set(expected_nodeids)
    failed = list(graph.failed_nodeids)
    return {
        "recorded_count": graph.recorded_count,
        "failed_count": len(failed),
        "dependency_edge_count": graph.dependency_edge_count,
        "missing_nodeids": list(graph.missing_nodeids),
        "failed_nodeids": failed,
        "node_outcomes": {
            nodeid: ("failed" if nodeid in failed else "passed" if nodeid not in graph.missing_nodeids else "missing")
            for nodeid in sorted(expected)
        },
        "error": graph.error,
        "graph_status": graph.status.value,
        "orphan_execution_edges": graph.orphan_execution_edges,
        "orphan_fingerprint_edges": graph.orphan_fingerprint_edges,
    }


def _seed_node_outcomes_from_events(
    path: Path,
    *,
    expected_nodeids: Sequence[str],
    database: Mapping[str, Any],
    pytest_step: Mapping[str, Any] | None,
    use_database_fallback: bool = True,
    prior_node_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Classify every promised seed node into one explicit terminal state."""
    reports: dict[str, list[dict[str, Any]]] = {}
    started: set[str] = set()
    finished: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    event = json.loads(line)
                    nodeid = event.get("nodeid")
                    if not isinstance(nodeid, str) or not nodeid:
                        continue
                    nodeid = _canonical_seed_nodeid(nodeid, expected_nodeids)
                    if event.get("event") == "test_started":
                        started.add(nodeid)
                    elif event.get("event") == "test_finished":
                        finished.add(nodeid)
                    elif event.get("event") == "test_report":
                        reports.setdefault(nodeid, []).append(event)
    except OSError:
        pass

    database_outcomes = database.get("node_outcomes")
    recorded = database_outcomes if isinstance(database_outcomes, dict) else {}
    diagnosis = str((pytest_step or {}).get("diagnosis") or "").lower()
    results: list[dict[str, Any]] = []
    for nodeid in expected_nodeids:
        node_reports = reports.get(nodeid, [])
        failed_reports = [report for report in node_reports if report.get("outcome") == "failed"]
        call_reports = [report for report in node_reports if report.get("when") == "call"]
        longrepr = "\n".join(str(report.get("longrepr") or "") for report in failed_reports).lower()
        outcome: str
        reason: str
        if "timeout" in longrepr:
            outcome, reason = "timeout", "pytest-timeout report"
        elif any(report.get("when") in {"setup", "teardown"} for report in failed_reports):
            outcome, reason = "error", "fixture setup/teardown failed"
        elif any(report.get("outcome") == "xfailed" for report in node_reports):
            outcome, reason = "xfailed", "pytest expected failure"
        elif any(report.get("outcome") == "xpassed" for report in node_reports):
            outcome, reason = "xpassed", "pytest unexpected pass"
        elif any(report.get("outcome") == "failed" for report in call_reports):
            outcome, reason = "failed", "test call failed"
        elif any(report.get("outcome") == "passed" for report in call_reports):
            outcome, reason = "passed", "test call passed"
        elif any(report.get("outcome") == "skipped" for report in call_reports):
            outcome, reason = "skipped", "test call skipped"
        elif any(report.get("outcome") == "skipped" for report in node_reports):
            outcome, reason = "skipped", "test setup or teardown skipped"
        elif nodeid in finished and any(
            report.get("when") == "teardown" and report.get("outcome") == "passed" for report in node_reports
        ):
            # Teardown describes fixture cleanup, not the test body.  It may
            # corroborate a terminal testmon row, but it cannot replace a
            # missing call report: a failed call can still end with a passing
            # teardown, and an unrecorded call must remain resumable.
            if recorded.get(nodeid) == "passed":
                outcome, reason = "passed", "passing teardown corroborated by testmon success"
            elif recorded.get(nodeid) == "failed":
                outcome, reason = "failed", "passing teardown contradicted by testmon failure"
            else:
                outcome, reason = "missing", "passing teardown without call report or testmon result"
        elif nodeid in started and nodeid not in finished and "timeout" in diagnosis:
            outcome, reason = "timeout", "supervisor timed out while node was active"
        elif nodeid in started and nodeid not in finished and "worker" in diagnosis:
            outcome, reason = "worker_crash", "worker exited while node was active"
        elif (
            nodeid in started
            and nodeid not in finished
            and any(marker in diagnosis for marker in ("interrupt", "signal", "terminated"))
        ):
            outcome, reason = "interrupted", "run ended while node was active"
        elif use_database_fallback and recorded.get(nodeid) == "passed":
            outcome, reason = "passed", "testmon database recorded success"
        elif use_database_fallback and recorded.get(nodeid) == "failed":
            outcome, reason = "failed", "testmon database recorded failure"
        elif prior_node_outcomes is not None and nodeid in prior_node_outcomes:
            prior = prior_node_outcomes[nodeid]
            prior_outcome = prior.get("outcome")
            if prior_outcome in TERMINAL_NODE_OUTCOMES:
                outcome, reason = str(prior_outcome), "terminal outcome carried from the prior seed attempt"
            else:
                outcome, reason = "missing", "prior seed attempt has no terminal outcome"
        else:
            outcome, reason = "missing", "no terminal report or testmon execution row"
        results.append(
            {
                "nodeid": nodeid,
                "outcome": outcome,
                "reason": reason,
                "started": nodeid in started,
                "finished": nodeid in finished,
                "phases": [
                    {
                        "when": report.get("when"),
                        "outcome": report.get("outcome"),
                        "duration_s": report.get("duration_s"),
                    }
                    for report in node_reports
                ],
            }
        )
    return results


def _seed_attempt_outcome(
    *,
    release_eligible: bool,
    terminal_graph: bool,
    exit_code: int,
    pytest_step: Mapping[str, Any] | None,
) -> SeedAttemptOutcome:
    """Classify the terminal seed result without hiding a bounded resource stop."""
    if release_eligible:
        return SeedAttemptOutcome.GREEN_RELEASE_BASELINE
    if terminal_graph:
        return SeedAttemptOutcome.RED_BASELINE if exit_code != 0 else SeedAttemptOutcome.SELECTION_ONLY
    diagnosis = str((pytest_step or {}).get("diagnosis") or "").casefold()
    termination_reason = str((pytest_step or {}).get("termination_reason") or "").casefold()
    if diagnosis in {"pytest_timeout", "pytest_stall_timeout", "pytest_resource_preflight_failed"} or any(
        marker in termination_reason
        for marker in ("runtime exceeded", "tmpfs budget exceeded", "resource budget", "resource limit")
    ):
        return SeedAttemptOutcome.RESOURCE_TIMEOUT
    return SeedAttemptOutcome.INCOMPLETE


def _finalize_testmon_seed_attempt(
    *,
    prepared: Mapping[str, Any],
    step_results: Sequence[Mapping[str, Any]],
    exit_code: int,
) -> dict[str, Any]:
    pytest_step = next(
        (step for step in step_results if str(step.get("name", "")).startswith("pytest seed-testmon")), None
    )
    selection: dict[str, Any] = {}
    events_path: Path | None = None
    if pytest_step is not None:
        artifact_dir = _safe_testmon_artifact_dir(pytest_step.get("artifact_dir"))
        if artifact_dir is not None:
            selection_payload = _read_json_artifact(artifact_dir / "selection.json")
            if isinstance(selection_payload, dict):
                selection = selection_payload
            events_path = artifact_dir / "events.jsonl"

    raw_omitted = selection.get("selected_nodeids_omitted")
    raw_selected_count = selection.get("selected_count")
    selected_nodeids = selection.get("selected_nodeids")
    selection_valid = (
        isinstance(raw_omitted, int)
        and not isinstance(raw_omitted, bool)
        and raw_omitted >= 0
        and isinstance(raw_selected_count, int)
        and not isinstance(raw_selected_count, bool)
        and isinstance(selected_nodeids, list)
        and all(isinstance(nodeid, str) and nodeid for nodeid in selected_nodeids)
        and len(set(selected_nodeids)) == len(selected_nodeids)
        and raw_selected_count == len(selected_nodeids)
    )
    prepared_expected = prepared.get("expected_nodeids")
    expected_raw = prepared_expected if isinstance(prepared_expected, list) and prepared_expected else selected_nodeids
    expected = list(expected_raw) if isinstance(expected_raw, list) else []
    shards = validate_seed_shard_ledger(prepared.get("shards"), expected_nodeids=expected)
    sharded = shards is not None
    if sharded:
        assert shards is not None
        selection_valid = seed_shard_ledger_is_terminal(shards)
        omitted = 0
        database = _testmon_database_state(expected)
        outcome_by_node = {item["nodeid"]: item for item in _seed_shard_outcomes(shards)}
        node_outcomes = [
            outcome_by_node.get(nodeid, {"nodeid": nodeid, "outcome": "missing", "reason": "shard not completed"})
            for nodeid in expected
        ]
        shard_steps: list[Mapping[str, Any]] = []
        for shard in shards:
            raw_step = shard.get("pytest_step")
            if isinstance(raw_step, Mapping):
                shard_steps.append(raw_step)
        if shard_steps:
            pytest_step = dict(shard_steps[-1])
            timed_out = next(
                (
                    step
                    for step in shard_steps
                    if str(step.get("diagnosis")) in {"pytest_timeout", "pytest_stall_timeout"}
                ),
                None,
            )
            if timed_out is not None:
                pytest_step = dict(timed_out)
        selection = {
            "selected_count": len(expected),
            "selected_nodeids_omitted": 0,
            "shard_count": len(shards),
            "completed_shard_count": sum(shard.get("status") == SeedShardStatus.COMPLETE.value for shard in shards),
        }
    else:
        omitted = raw_omitted if selection_valid and isinstance(raw_omitted, int) else 1
        database = _testmon_database_state(expected)
        node_outcomes = _seed_node_outcomes_from_events(
            events_path or Path(".missing-testmon-events"),
            expected_nodeids=expected,
            database=database,
            pytest_step=pytest_step,
            use_database_fallback=False,
            prior_node_outcomes={
                str(item["nodeid"]): item
                for item in prepared.get("prior_node_outcomes", [])
                if isinstance(item, Mapping) and isinstance(item.get("nodeid"), str)
            },
        )
    unsuccessful_nodeids = [
        str(item["nodeid"]) for item in node_outcomes if item.get("outcome") not in SUCCESSFUL_NODE_OUTCOMES
    ]
    green_complete = (
        exit_code == 0
        and bool(expected)
        and selection_valid
        and omitted == 0
        and database["error"] is None
        and database["graph_status"] == "complete"
        and not database["missing_nodeids"]
        and not database["failed_nodeids"]
        and database["orphan_execution_edges"] == 0
        and database["orphan_fingerprint_edges"] == 0
        and not unsuccessful_nodeids
    )
    identity = prepared.get("identity")
    narrow_terminal = isinstance(identity, Mapping) and identity.get("skip_slow") is True
    terminal_authorized = _testmon_seed_terminal_authorized(prepared)
    release_eligible = green_complete and (not narrow_terminal or terminal_authorized)
    terminal_graph = (
        database["error"] is None
        and database["graph_status"] == GraphStatus.COMPLETE.value
        and not database["missing_nodeids"]
        and database["orphan_execution_edges"] == 0
        and database["orphan_fingerprint_edges"] == 0
        and all(item.get("outcome") in TERMINAL_NODE_OUTCOMES for item in node_outcomes)
    )
    outcome = _seed_attempt_outcome(
        release_eligible=release_eligible,
        terminal_graph=terminal_graph,
        exit_code=exit_code,
        pytest_step=pytest_step,
    )
    shard_ledger = (
        shards if sharded else seed_shard_plan(expected, shard_size=max(1, len(expected))) if expected else []
    )
    if not sharded and shard_ledger:
        shard_ledger[0].update(
            {
                "status": (
                    SeedShardStatus.COMPLETE.value
                    if all(item.get("outcome") in TERMINAL_NODE_OUTCOMES for item in node_outcomes)
                    else SeedShardStatus.INCOMPLETE.value
                ),
                "node_outcomes": node_outcomes,
            }
        )
    seed_scope = (
        VerificationScope.NARROW_TERMINAL.value if narrow_terminal else VerificationScope.RELEASE_BASELINE.value
    )
    attempt_candidate = {
        **dict(prepared),
        "status": "complete" if release_eligible else "reusable" if terminal_graph else "incomplete",
        "outcome": outcome.value,
        "exit_code": exit_code,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest() if expected else None,
        "selection": {
            **selection,
            # A resumed run inherits the complete collection ledger from its
            # original selection. The current pytest step may select only a
            # subset while it repairs missing graph edges.
            "selected_count": len(expected)
            if prepared.get("resume") and selection_valid
            else selection.get("selected_count"),
            "selected_nodeids_omitted": 0 if prepared.get("resume") and selection_valid else omitted,
        },
        "shards": shard_ledger,
        "node_outcomes": node_outcomes,
        "identity": prepared.get("identity"),
        "run_id": prepared.get("run_id"),
        "artifact_dir": prepared.get("artifact_dir"),
        "testmon_data": _file_fingerprint(TESTMON_DATA),
        "verification_scope": seed_scope,
        "terminal_authorization": (TerminalAuthorization.NARROW_TERMINAL.value if terminal_authorized else None),
        "release_baseline_allowed": release_eligible,
    }
    reusable_stamp = stamp_from_attempt(
        attempt_candidate,
        TESTMON_DATA,
        checkout_root=Path.cwd(),
        protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
    )
    reusable = reusable_stamp is not None
    release_permission = bool(
        reusable
        and reusable_stamp is not None
        and reusable_stamp.release_baseline_allowed
        and (not narrow_terminal or terminal_authorized)
    )
    attempt_status = "complete" if green_complete and release_permission else "reusable" if reusable else "incomplete"
    payload = {
        **dict(prepared),
        "status": attempt_status,
        "outcome": outcome.value,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest() if expected else None,
        "selection": {
            key: (
                len(expected)
                if key == "selected_count" and prepared.get("resume") and selection_valid
                else 0
                if key == "selected_nodeids_omitted" and prepared.get("resume") and selection_valid
                else selection.get(key)
            )
            for key in (
                "selected_count",
                "deselected_count",
                "selected_nodeids_omitted",
                "deselected_nodeids_omitted",
                "collection_duration_s",
            )
        },
        "shards": shard_ledger,
        "database": database,
        "node_outcomes": node_outcomes,
        "node_outcome_counts": dict(
            sorted(
                {
                    outcome: sum(1 for item in node_outcomes if item.get("outcome") == outcome)
                    for outcome in {str(item.get("outcome")) for item in node_outcomes}
                }.items()
            )
        ),
        "unsuccessful_nodeids": unsuccessful_nodeids,
        "testmon_data": _file_fingerprint(TESTMON_DATA),
        "pytest_step": dict(pytest_step) if pytest_step is not None else None,
        "binding": TestmonBinding(BindingMode.EXACT, str(ROOT.resolve())).as_dict(),
        "verification_scope": seed_scope,
        "terminal_authorization": (TerminalAuthorization.NARROW_TERMINAL.value if terminal_authorized else None),
    }
    payload["release_baseline_allowed"] = release_permission
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    if release_permission and reusable_stamp is not None:
        _atomic_write_json(TESTMON_SEED_STAMP, reusable_stamp.as_dict())
    else:
        TESTMON_SEED_STAMP.unlink(missing_ok=True)
    return payload


def _refresh_testmon_selection_attempt(
    *,
    step: Mapping[str, Any],
    run: VerifyRun,
    exit_code: int,
) -> None:
    """Refresh a reusable red graph after every completed affected run."""
    attempt = _read_testmon_seed_attempt()
    if attempt is None or attempt.get("release_baseline_allowed") is True:
        return
    expected = _testmon_seed_expected_nodeids(attempt)
    if not expected:
        return
    database = _testmon_database_state(expected)
    artifact_dir = _safe_testmon_artifact_dir(step.get("artifact_dir"))
    events_path = artifact_dir / "events.jsonl" if artifact_dir is not None else Path(".missing-testmon-events")
    prior = {
        str(item["nodeid"]): item
        for item in attempt.get("node_outcomes", [])
        if isinstance(item, Mapping) and isinstance(item.get("nodeid"), str)
    }
    node_outcomes = _seed_node_outcomes_from_events(
        events_path,
        expected_nodeids=expected,
        database=database,
        pytest_step=step,
        use_database_fallback=False,
        prior_node_outcomes=prior,
    )
    graph_complete = (
        database.get("graph_status") == GraphStatus.COMPLETE.value
        and not database.get("missing_nodeids")
        and database.get("error") is None
        and database.get("orphan_execution_edges") == 0
        and database.get("orphan_fingerprint_edges") == 0
    )
    terminal = all(item.get("outcome") in TERMINAL_NODE_OUTCOMES for item in node_outcomes)
    prior_selection = attempt.get("selection")
    payload = {
        **attempt,
        "status": "reusable" if graph_complete and terminal else "incomplete",
        "outcome": (
            SeedAttemptOutcome.RED_BASELINE.value
            if graph_complete and terminal
            else SeedAttemptOutcome.INCOMPLETE.value
        ),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "selection": {
            **(dict(prior_selection) if isinstance(prior_selection, Mapping) else {}),
            "selected_count": len(expected),
            "selected_nodeids_omitted": 0,
        },
        "database": database,
        "node_outcomes": node_outcomes,
        "node_outcome_counts": dict(
            sorted(
                {
                    outcome: sum(1 for item in node_outcomes if item.get("outcome") == outcome)
                    for outcome in {str(item.get("outcome")) for item in node_outcomes}
                }.items()
            )
        ),
        "unsuccessful_nodeids": [
            str(item["nodeid"]) for item in node_outcomes if item.get("outcome") not in SUCCESSFUL_NODE_OUTCOMES
        ],
        "testmon_data": _file_fingerprint(TESTMON_DATA),
        "run_id": run.run_id,
        "artifact_dir": str(run.relative_run_dir),
        "pytest_step": dict(step),
        "release_baseline_allowed": False,
        "verification_scope": VerificationScope.AFFECTED.value,
    }
    raw_binding = attempt.get("binding")
    if not isinstance(raw_binding, Mapping):
        payload["binding"] = TestmonBinding(BindingMode.EXACT, str(ROOT.resolve())).as_dict()
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)


def _discard_testmon_dependency_authority() -> None:
    """Remove a dependency graph learned while checkout authority was unstable."""
    for path in (
        TESTMON_SEED_STAMP,
        TESTMON_SEED_ATTEMPT,
        TESTMON_DATA,
        Path(f"{TESTMON_DATA}-wal"),
        Path(f"{TESTMON_DATA}-shm"),
        Path(f"{TESTMON_DATA}-journal"),
    ):
        path.unlink(missing_ok=True)


# ── main ────────────────────────────────────────────────────────────


@finalize_checkout_mutation_monitors
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--seed-testmon",
        action="store_true",
        help="Run full non-integration pytest with --testmon-noselect to seed/update .cache/testmon/testmondata.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Force the full non-integration pytest diagnostic instead of testmon."
    )
    parser.add_argument(
        "--full", action="store_true", help="Alias for --all: run full non-integration pytest diagnostic."
    )
    parser.add_argument("--commit", action="store_true", help="Pre-commit tier: format + lint + mypy only.")
    parser.add_argument(
        "--skip-slow", action="store_true", help="Exclude @pytest.mark.slow tests from the pytest step."
    )
    parser.add_argument(
        "--terminal-authorization",
        choices=[TerminalAuthorization.NARROW_TERMINAL.value],
        help="Typed authorization for a narrow terminal verification that skips slow tests.",
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

    _anchor_verification_paths()
    bootstrap_message = maybe_bootstrap_testmon_seed(
        ROOT,
        protocol_version=TESTMON_SEED_PROTOCOL_VERSION,
    )
    if bootstrap_message is not None:
        sys.stderr.write(bootstrap_message + "\n")
    try:
        fingerprint = assert_polylogue_matches_checkout(ROOT, context="devtools verify")
    except CheckoutImportMismatchError as exc:
        sys.stderr.write(f"verify: {exc}\n")
        return 125
    polylogue_import_path = fingerprint.polylogue_import_path
    environment_fingerprint = fingerprint.as_dict()
    sys.stderr.write(f"verify: polylogue package → {polylogue_import_path}\n")

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

    head = _git_head()
    full_pytest = bool(args.all or args.full)
    affected_testmon = not (args.quick or args.commit or args.seed_testmon or full_pytest)
    testmon_base_commit = _git_commit("origin/master") if affected_testmon else None
    testmon_head_commit = head if affected_testmon else None
    if affected_testmon and (testmon_base_commit is None or testmon_head_commit is None):
        sys.stderr.write("verify: cannot resolve immutable Git refs for affected-test authority.\n")
        return 125
    if args.terminal_authorization is not None and not ((full_pytest or args.seed_testmon) and args.skip_slow):
        parser.error("--terminal-authorization requires --all, --full, or --seed-testmon with --skip-slow")
    preflight_error = _testmon_preflight(
        seed_testmon=bool(args.seed_testmon),
        full_pytest=full_pytest,
        quick=bool(args.quick),
        commit=bool(args.commit),
    )
    if preflight_error is not None:
        sys.stderr.write(preflight_error)
        return 2

    t0 = time.monotonic()
    mutation_monitor = CheckoutMutationMonitor(ROOT)
    start_checkout_mutation_monitor(mutation_monitor)
    checkout_fingerprint = worktree_fingerprint(ROOT)
    verify_run = VerifyRun(
        tier=tier,
        argv=list(sys.argv[1:] if argv is None else argv),
        git_head=head,
        polylogue_import_path=str(polylogue_import_path),
        environment_fingerprint=environment_fingerprint,
        worktree_fingerprint=checkout_fingerprint,
    )
    seed_identity: dict[str, Any] | None = None
    resume_testmon_seed = False
    prepared_seed_attempt: dict[str, Any] | None = None
    if args.seed_testmon:
        try:
            seed_identity = _testmon_seed_identity(
                git_head=head,
                git_tree=_git_committed_tree(),
                skip_slow=bool(args.skip_slow),
                lab=bool(args.lab),
                terminal_authorization=args.terminal_authorization,
            )
        except RuntimeError as exc:
            finish_checkout_mutation_monitor(mutation_monitor)
            sys.stderr.write(f"verify: {exc}\n")
            early_payload = verify_run.finish(
                exit_code=125,
                duration_s=time.monotonic() - t0,
                diagnosis="testmon_environment_identity_unavailable",
            )
            _save_history(early_payload)
            return 125
        resume_testmon_seed = _testmon_seed_can_resume(seed_identity)
        prepared_seed_attempt = _prepare_testmon_seed_attempt(
            identity=seed_identity,
            run=verify_run,
            resume=resume_testmon_seed,
        )
        if resume_testmon_seed:
            sys.stderr.write("verify: resuming the matching incomplete pytest-testmon seed\n")

    if not use_json:
        sys.stderr.write("verify: running local verification baseline\n")

    # Resource preflight before heavy steps.
    if not args.quick and not args.commit:
        _warn_low_memory()

    exit_code = 0
    try:
        steps = build_verify_steps(
            quick=bool(args.quick),
            commit=bool(args.commit),
            lab=bool(args.lab),
            skip_slow=bool(args.skip_slow),
            seed_testmon=bool(args.seed_testmon),
            resume_testmon_seed=resume_testmon_seed,
            full_pytest=full_pytest,
            broad_testmon=(
                _default_testmon_is_broad_change(testmon_base_commit, testmon_head_commit)
                if testmon_base_commit is not None and testmon_head_commit is not None
                else False
            ),
        )
    except PytestResourceError as exc:
        finish_checkout_mutation_monitor(mutation_monitor)
        sys.stderr.write(f"verify: {exc}\n")
        early_payload = verify_run.finish(
            exit_code=125,
            duration_s=time.monotonic() - t0,
            diagnosis="pytest_resource_preflight_failed",
        )
        _save_history(early_payload)
        return 125

    step_results: list[dict[str, Any]] = []
    pending_testmon_stamp: TestmonSeedStamp | None = None
    pending_affected_coverage: tuple[tuple[str, ...], int] | None = None
    pending_selection_refresh: tuple[dict[str, Any], int] | None = None
    testmon_graph_touched = False
    changed_path_authority_failed = False
    for label, cmd in steps:
        if label.startswith("pytest"):
            _warn_low_memory()  # check again right before the heavy step
        rc, elapsed, metadata = _run(label, cmd, run=verify_run)
        if label in {"pytest testmon", "pytest testmon (broad)"} or label.startswith("pytest seed-testmon"):
            testmon_graph_touched = True
        if rc == 0 and label in {"pytest testmon", "pytest testmon (broad)"}:
            raw_stamp = _read_json_artifact(TESTMON_SEED_STAMP)
            try:
                current_stamp = (
                    TestmonSeedStamp.from_mapping(raw_stamp, protocol_version=TESTMON_SEED_PROTOCOL_VERSION)
                    if isinstance(raw_stamp, Mapping)
                    else None
                )
            except ValueError:
                current_stamp = None
            if current_stamp is not None:
                refreshed_stamp = refresh_stamp(current_stamp, TESTMON_DATA)
                if refreshed_stamp is not None:
                    pending_testmon_stamp = refreshed_stamp
            assert testmon_base_commit is not None
            assert testmon_head_commit is not None
            try:
                executable_paths = _changed_executable_paths(testmon_base_commit, testmon_head_commit)
            except PytestResourceError as exc:
                changed_path_authority_failed = True
                executable_paths = ()
                rc = 125
                metadata["diagnosis"] = "testmon_changed_path_authority_unavailable"
                metadata["error"] = str(exc)
                pending_testmon_stamp = None
                sys.stderr.write(
                    "verify: changed-path authority became unavailable after pytest; "
                    "discarding the affected dependency graph.\n"
                )
            selected_count = metadata.get("selected_count")
            if selected_count == 0 and executable_paths:
                coverage = _matching_testmon_coverage(executable_paths)
                if coverage is None:
                    rc = 5
                    metadata["diagnosis"] = "zero_testmon_selection_for_executable_change"
                    metadata["zero_selection_changed_paths"] = list(executable_paths)
                    sys.stderr.write(
                        "verify: pytest-testmon selected zero tests for executable changes and no "
                        "matching successful coverage receipt exists; refresh the seed or repair "
                        "dependency capture: " + ", ".join(executable_paths) + "\n"
                    )
                else:
                    metadata["zero_selection_coverage"] = coverage
            elif isinstance(selected_count, int) and selected_count > 0:
                pending_affected_coverage = (tuple(executable_paths), selected_count)
        step_result: dict[str, Any] = {"name": label, "duration_s": round(elapsed, 2), "exit": rc}
        step_result.update(metadata)
        step_results.append(step_result)
        if args.seed_testmon and label.startswith("pytest seed-testmon collect"):
            if rc != 0:
                exit_code = rc
                break
            artifact_dir = _safe_testmon_artifact_dir(metadata.get("artifact_dir"))
            selection = _read_json_artifact(artifact_dir / "selection.json") if artifact_dir is not None else None
            assert prepared_seed_attempt is not None
            prepared_seed_attempt = _prepare_testmon_seed_shards(
                prepared_seed_attempt,
                selection=selection if isinstance(selection, Mapping) else None,
            )
            expected = _testmon_seed_expected_nodeids(prepared_seed_attempt)
            shards = validate_seed_shard_ledger(prepared_seed_attempt.get("shards"), expected_nodeids=expected)
            if shards is None:
                exit_code = 5
                step_result["exit"] = 5
                step_result["diagnosis"] = "testmon_seed_collection_incomplete"
                sys.stderr.write("verify: pytest-testmon collection did not produce a complete shard plan.\n")
                break
            for shard in shards:
                if shard.get("status") == SeedShardStatus.COMPLETE.value:
                    continue
                shard_index = int(shard["index"])
                shard_label = f"pytest seed-testmon shard {shard_index}/{len(shards)}"
                shard_args_path = verify_run.run_dir / "seed-shards" / f"{shard_index:04d}.args"
                try:
                    shard_cmd = _seed_shard_command(cmd, shard, nodeids_file=shard_args_path)
                except (OSError, PytestResourceError) as exc:
                    resource_failure_result = {
                        "name": shard_label,
                        "duration_s": 0.0,
                        "exit": 125,
                        "diagnosis": (
                            "pytest_resource_refusal"
                            if isinstance(exc, PytestResourceError)
                            else "testmon_seed_args_file_write_failed"
                        ),
                        "error": str(exc),
                        "shard_index": shard_index,
                        "shard_count": len(shards),
                        "shard_nodeid_count": len(shard["nodeids"]),
                    }
                    step_results.append(resource_failure_result)
                    prepared_seed_attempt = _checkpoint_testmon_seed_shard(
                        prepared=prepared_seed_attempt,
                        shard_index=shard_index,
                        step=resource_failure_result,
                    )
                    exit_code = 125
                    break
                _warn_low_memory()
                shard_rc, shard_elapsed, shard_metadata = _run(shard_label, shard_cmd, run=verify_run)
                shard_result: dict[str, Any] = {
                    "name": shard_label,
                    "duration_s": round(shard_elapsed, 2),
                    "exit": shard_rc,
                    "shard_index": shard_index,
                    "shard_count": len(shards),
                    "shard_nodeid_count": len(shard["nodeids"]),
                }
                shard_result.update(shard_metadata)
                step_results.append(shard_result)
                prepared_seed_attempt = _checkpoint_testmon_seed_shard(
                    prepared=prepared_seed_attempt,
                    shard_index=shard_index,
                    step=shard_result,
                )
                checkpointed_shards = prepared_seed_attempt.get("shards")
                if (
                    not isinstance(checkpointed_shards, list)
                    or shard_index > len(checkpointed_shards)
                    or not isinstance(checkpointed_shards[shard_index - 1], Mapping)
                ):
                    raise RuntimeError("testmon seed shard checkpoint is malformed")
                shard_complete = checkpointed_shards[shard_index - 1].get("status") == SeedShardStatus.COMPLETE.value
                if shard_rc != 0:
                    stop_seed = _seed_shard_failure_requires_stop(
                        shard_result,
                        shard_complete=shard_complete,
                    )
                    if exit_code == 0 or stop_seed:
                        # A later infrastructure failure is the terminal
                        # condition even when an earlier shard recorded
                        # ordinary red-test evidence.
                        exit_code = shard_rc
                    if stop_seed:
                        break
            continue
        if label in {"pytest testmon", "pytest testmon (broad)"} and not args.seed_testmon and not full_pytest:
            pending_selection_refresh = (step_result, rc)
        if rc != 0:
            exit_code = rc
            if rc == 130 or _stop_after_failed_step(label):
                break

    final_head = _git_head()
    final_checkout_fingerprint = worktree_fingerprint(ROOT)
    mutation_observation = finish_checkout_mutation_monitor(mutation_monitor)
    checkout_stable = True
    if (
        changed_path_authority_failed
        or head is None
        or final_head is None
        or "unavailable" in {checkout_fingerprint, final_checkout_fingerprint}
        or mutation_observation.unavailable
    ):
        checkout_stable = False
        step_results.append(
            {
                "name": "checkout stability",
                "duration_s": 0.0,
                "exit": 125,
                "diagnosis": (
                    "testmon_changed_path_authority_unavailable"
                    if changed_path_authority_failed
                    else "checkout_fingerprint_unavailable"
                ),
                "initial_git_head": head,
                "final_git_head": final_head,
                "initial_worktree_fingerprint": checkout_fingerprint,
                "final_worktree_fingerprint": final_checkout_fingerprint,
            }
        )
        if exit_code == 0:
            exit_code = 125
        sys.stderr.write("verify: checkout fingerprint unavailable; evidence is not exact-head.\n")
    elif final_head != head or mutation_observation.changed or final_checkout_fingerprint != checkout_fingerprint:
        checkout_stable = False
        step_results.append(
            {
                "name": "checkout stability",
                "duration_s": 0.0,
                "exit": 125,
                "diagnosis": "checkout_changed_during_verification",
                "initial_git_head": head,
                "final_git_head": final_head,
                "initial_worktree_fingerprint": checkout_fingerprint,
                "final_worktree_fingerprint": final_checkout_fingerprint,
                "transient_checkout_mutation": mutation_observation.changed,
                "checkout_mutation_path": mutation_observation.observed_path,
            }
        )
        if exit_code == 0:
            exit_code = 125
        sys.stderr.write("verify: checkout contents changed during verification; evidence is not exact-head.\n")

    seed_receipt: dict[str, Any] | None = None
    if checkout_stable:
        if pending_testmon_stamp is not None:
            _atomic_write_json(TESTMON_SEED_STAMP, pending_testmon_stamp.as_dict())
        if pending_affected_coverage is not None:
            executable_paths, selected_count = pending_affected_coverage
            _record_testmon_affected_coverage(
                executable_paths=executable_paths,
                selected_count=selected_count,
                run_id=verify_run.run_id,
            )
        if pending_selection_refresh is not None:
            step_result, selection_exit_code = pending_selection_refresh
            _refresh_testmon_selection_attempt(
                step=step_result,
                run=verify_run,
                exit_code=selection_exit_code,
            )
        if prepared_seed_attempt is not None:
            seed_receipt = _finalize_testmon_seed_attempt(
                prepared=prepared_seed_attempt,
                step_results=step_results,
                exit_code=exit_code,
            )
            if exit_code == 0 and seed_receipt["status"] != "complete":
                exit_code = 5
                sys.stderr.write(
                    "verify: pytest passed but the testmon dependency baseline is incomplete; "
                    f"inspect {TESTMON_SEED_ATTEMPT}.\n"
                )
    elif testmon_graph_touched:
        _discard_testmon_dependency_authority()
        if prepared_seed_attempt is not None:
            seed_receipt = {
                "status": "discarded",
                "outcome": SeedAttemptOutcome.INCOMPLETE.value,
                "resume": False,
                "expected_count": len(_testmon_seed_expected_nodeids(prepared_seed_attempt)),
                "release_baseline_allowed": False,
            }

    total_duration = round(time.monotonic() - t0, 2)

    # Build history entry.
    history_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "final_git_head": final_head,
        "tier": tier,
        "run_id": verify_run.run_id,
        "checkout_root": str(ROOT.resolve()),
        "worktree_fingerprint": checkout_fingerprint,
        "final_worktree_fingerprint": final_checkout_fingerprint,
        "artifact_dir": str(verify_run.relative_run_dir),
        "steps": step_results,
        "total_duration_s": total_duration,
        "exit_code": exit_code,
    }
    checkout_stability_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in reversed(step_results)
            if step.get("name") == "checkout stability" and "diagnosis" in step
        ),
        None,
    )
    fallback_pytest_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in reversed(step_results)
            if str(step.get("name", "")).startswith("pytest") and "diagnosis" in step
        ),
        None,
    )
    pytest_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in reversed(step_results)
            if str(step.get("name", "")).startswith("pytest") and step.get("exit") == exit_code and "diagnosis" in step
        ),
        fallback_pytest_diagnosis,
    )
    run_diagnosis = checkout_stability_diagnosis or pytest_diagnosis
    if run_diagnosis is not None:
        history_entry["diagnosis"] = run_diagnosis
    if seed_receipt is not None:
        history_entry["testmon_seed"] = {
            "status": seed_receipt["status"],
            "outcome": seed_receipt["outcome"],
            "resume": seed_receipt["resume"],
            "expected_count": seed_receipt["expected_count"],
            "attempt_path": str(TESTMON_SEED_ATTEMPT),
            "stamp_path": str(TESTMON_SEED_STAMP) if seed_receipt["release_baseline_allowed"] else None,
            "release_baseline_allowed": seed_receipt["release_baseline_allowed"],
        }

    if args.quick or args.commit:
        verification_scope = VerificationScope.NON_TEST
        # Non-test verification is intentionally not release authority, but it
        # is still a typed verification receipt.  ``None`` made merge-gate
        # treat an explicit quick receipt as malformed instead of as a valid
        # non-release gate.
        release_baseline_allowed: bool | None = False
    elif full_pytest or args.seed_testmon:
        narrow_terminal = bool(args.skip_slow)
        authorized_narrow_terminal = args.terminal_authorization == TerminalAuthorization.NARROW_TERMINAL.value
        verification_scope = (
            VerificationScope.NARROW_TERMINAL if narrow_terminal else VerificationScope.RELEASE_BASELINE
        )
        if full_pytest:
            release_baseline_allowed = exit_code == 0 and (not narrow_terminal or authorized_narrow_terminal)
        else:
            release_baseline_allowed = _testmon_release_baseline_permission() and (
                not narrow_terminal or authorized_narrow_terminal
            )
    else:
        verification_scope = VerificationScope.AFFECTED
        release_baseline_allowed = _testmon_release_baseline_permission()
    history_entry["verification_scope"] = verification_scope.value
    history_entry["release_baseline_allowed"] = release_baseline_allowed
    history_entry["terminal_authorization"] = args.terminal_authorization
    if release_baseline_allowed is False and tier in {"testmon", "lab", "seed-testmon"}:
        sys.stderr.write(
            "verify: affected-test selection is usable, but the current testmon state does not grant "
            "release-baseline permission.\n"
        )

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
    verify_run.finish(
        exit_code=exit_code,
        duration_s=total_duration,
        diagnosis=run_diagnosis,
        verification_scope=verification_scope.value,
        release_baseline_allowed=release_baseline_allowed,
        terminal_authorization=args.terminal_authorization,
        final_worktree_fingerprint=final_checkout_fingerprint,
        checkout_mutation_path=mutation_observation.observed_path,
    )
    if exit_code == 0:
        _stamp_head()

    # Notify only on failure. Passing runs stay silent — the terminal
    # already shows the green summary and a desktop popup per run is
    # spammy when verify is invoked on every push.
    if exit_code != 0:
        _notify(
            _format_completion_notification(
                exit_code=exit_code,
                total_duration=total_duration,
                step_results=step_results,
            )
        )

    return exit_code
