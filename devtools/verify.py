"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy (~3s warm).
  --quick    Pre-push tier: all non-pytest gates (~15s warm).
  (default)  Baseline with pytest-testmon affected tests.
  --all/--full
             Complete pytest correctness corpus in the current native
             testmon environment (performance benchmarks excluded).
  --lab      Default testmon baseline plus lab smoke and SLO checks.

Output formats:
  --json     Machine-readable JSON to stdout (human progress to stderr).
  (default)  Human-readable text when stdout is a TTY; auto-JSON otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
from devtools.testmon_bootstrap import (
    NativeTestmonDeadlineError,
    NativeTestmonPreparation,
    NativeTestmonRepairError,
    NativeTestmonState,
    classify_native_testmon_changes,
    inspect_native_testmon_environment,
    prepare_native_testmon_environment,
    remove_invalid_native_testmon_state,
    validate_native_testmon_state_ownership,
)
from devtools.verification_contracts import VerificationScope
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
    CheckoutMutationObservation,
    PytestResourceError,
    PytestStepArtifacts,
    ResourceSampler,
    VerifyRun,
    adaptive_pytest_worker_count,
    aggregate_native_testmon_run,
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
    pytest_step_run_id,
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
_PYTEST_CLEAR_CONFIGURED_ADDOPTS = "--override-ini=addopts="
_PYTEST_MANAGED_PLUGIN_NAMES = (
    "anyio",
    "asyncio",
    "hypothesispytest",
    "benchmark",
    "pytest_cov",
    "pytest_jsonreport",
    "randomly",
    "syrupy",
    "timeout",
    "xdist",
    "pytest-testmon",
)
_PYTEST_MANAGED_PLUGIN_ARGS = tuple(argument for name in _PYTEST_MANAGED_PLUGIN_NAMES for argument in ("-p", name))
_PYTEST_CLOSED_WORLD_COLLECTION_ARGS = (
    _PYTEST_CLEAR_CONFIGURED_ADDOPTS,
    "--override-ini=python_files=test_*.py *_test.py fuzz_*.py",
    "--override-ini=python_classes=Test",
    "--override-ini=python_functions=test",
    "--override-ini=norecursedirs=",
    "tests",
)
NATIVE_TESTMON_LIFECYCLE_LOCK_TIMEOUT_S = 60.0


def _normalize_managed_pytest_environment(
    env: dict[str, str],
    *,
    disable_plugin_autoload: bool = True,
) -> None:
    """Remove ambient pytest options and extensions from a managed child."""
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    if disable_plugin_autoload:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    else:
        env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)


def _python_optimization_level() -> int:
    """Return the active interpreter optimization level."""
    return int(sys.flags.optimize)


@dataclass(slots=True)
class _OwnedNativeTestmonState:
    descriptor: int
    data_path: Path

    def close(self) -> None:
        os.close(self.descriptor)


def _open_owned_native_testmon_state(repo_root: Path) -> _OwnedNativeTestmonState:
    """Bind managed SQLite access to one no-follow checkout directory."""
    validate_native_testmon_state_ownership(repo_root)
    raw_data = TESTMON_DATA if TESTMON_DATA.is_absolute() else repo_root.resolve() / TESTMON_DATA
    parent = raw_data.parent
    parent.mkdir(parents=True, exist_ok=True)
    validate_native_testmon_state_ownership(repo_root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(parent, flags)
        opened = os.fstat(descriptor)
        current = parent.lstat()
    except OSError as exc:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        raise NativeTestmonRepairError(f"cannot bind owned testmon directory {parent}: {exc}") from exc
    assert descriptor is not None
    if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        os.close(descriptor)
        raise NativeTestmonRepairError(f"owned testmon directory changed while binding: {parent}")
    bound = Path(f"/proc/{os.getpid()}/fd/{descriptor}") / raw_data.name
    return _OwnedNativeTestmonState(descriptor=descriptor, data_path=bound)


@contextlib.contextmanager
def _native_testmon_lifecycle_lock(
    repo_root: Path,
    *,
    timeout_s: float = NATIVE_TESTMON_LIFECYCLE_LOCK_TIMEOUT_S,
) -> Iterator[None]:
    """Serialize one checkout's native testmon preparation, lanes, and inspection."""
    cache = repo_root.resolve() / ".cache"
    try:
        mode = cache.lstat().st_mode
    except FileNotFoundError:
        cache.mkdir(exist_ok=True)
        mode = cache.lstat().st_mode
    except OSError as exc:
        raise NativeTestmonRepairError(f"cannot inspect native testmon lock parent {cache}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise NativeTestmonRepairError(f"native testmon lock parent is not an owned directory: {cache}")
    lock_path = cache / "native-testmon-lifecycle.lock"
    directory_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            cache,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_directory = os.fstat(directory_descriptor)
        current_directory = cache.lstat()
        if not stat.S_ISDIR(opened_directory.st_mode) or (opened_directory.st_dev, opened_directory.st_ino) != (
            current_directory.st_dev,
            current_directory.st_ino,
        ):
            raise NativeTestmonRepairError(f"native testmon lock parent changed while binding: {cache}")
        lock_descriptor = os.open(
            lock_path.name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        opened_lock = os.fstat(lock_descriptor)
        current_lock = os.stat(lock_path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened_lock.st_mode)
            or not stat.S_ISREG(current_lock.st_mode)
            or opened_lock.st_nlink != 1
            or current_lock.st_nlink != 1
            or (opened_lock.st_dev, opened_lock.st_ino) != (current_lock.st_dev, current_lock.st_ino)
        ):
            raise NativeTestmonRepairError(
                f"native testmon lifecycle lock is not an owned single-link regular file: {lock_path}"
            )
    except OSError as exc:
        if lock_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(lock_descriptor)
            lock_descriptor = None
        if directory_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(directory_descriptor)
            directory_descriptor = None
        raise NativeTestmonRepairError(f"cannot bind native testmon lifecycle lock {lock_path}: {exc}") from exc
    try:
        assert lock_descriptor is not None
        with os.fdopen(lock_descriptor, "r+", encoding="utf-8") as handle:
            lock_descriptor = None
            deadline = time.monotonic() + max(0.0, timeout_s)
            announced_wait = False
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PytestResourceError(
                            f"timed out waiting for native testmon lifecycle lock after {timeout_s:.1f}s"
                        ) from exc
                    if not announced_wait:
                        handle.seek(0)
                        holder = handle.read().strip() or "another verify invocation"
                        sys.stderr.write(f"verify: waiting for native testmon lifecycle lock ({holder})\n")
                        sys.stderr.flush()
                        announced_wait = True
                    time.sleep(min(0.05, remaining))
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}")
            handle.flush()
            try:
                yield
            finally:
                handle.seek(0)
                handle.truncate()
    finally:
        if lock_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(lock_descriptor)
        if directory_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(directory_descriptor)


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
            cwd=ROOT,
            env=_subprocess_env(),
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
    timeout_s: float,
) -> dict[str, Any]:
    """Adapt managed-pytest accounting to the shared workload receipt."""
    input_digest = hashlib.sha256(
        json.dumps(cmd, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    budgets: list[WorkloadBudget] = []
    if timeout_s > 0:
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
    timeout_override_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    heartbeat_s = _pytest_heartbeat_interval()
    timeout_s = _pytest_timeout_s() if timeout_override_s is None else max(0.0, timeout_override_s)
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
    # Prefer the value pytest itself will read: the basetemp directory is named
    # from it, and it is lane-scoped so the parallel and serial lanes of one
    # verify run do not share a tree.
    pytest_run_id = env.get("POLYLOGUE_PYTEST_RUN_ID") or (run.run_id if run is not None else str(os.getpid()))
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
            run_id=env.get("POLYLOGUE_PYTEST_RUN_ID") or (run.run_id if run is not None else str(process.pid)),
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


def _recover_worker_collection_facts(*, events_dir: Path, selection_path: Path) -> bool:
    """Recover xdist collection evidence if interruption skips sessionfinish."""
    merged = merge_worker_collection_payloads(events_dir)
    if merged is None:
        return False
    selection = {
        **merged,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": "runner",
        "pid": os.getpid(),
        "recovered_after_interruption": True,
    }
    try:
        _atomic_write_json(selection_path, selection)
    except OSError:
        return False
    return True


def _run(
    label: str,
    cmd: list[str],
    *,
    cwd: str | None = None,
    run: VerifyRun | None = None,
    timeout_s: float | None = None,
) -> tuple[int, float, dict[str, Any]]:
    if not label.startswith("pytest native"):
        return _run_step(label, cmd, cwd=cwd, run=run, timeout_s=timeout_s)
    state = _ACTIVE_VERIFY_RUN.owned_native_testmon_state if _ACTIVE_VERIFY_RUN is not None else None
    temporary_state = state is None
    if state is None:
        state = _open_owned_native_testmon_state(ROOT)
    try:
        return _run_step(
            label,
            cmd,
            cwd=cwd,
            run=run,
            timeout_s=timeout_s,
            native_testmon_data=state.data_path,
        )
    finally:
        if temporary_state:
            state.close()


def _run_step(
    label: str,
    cmd: list[str],
    *,
    cwd: str | None = None,
    run: VerifyRun | None = None,
    timeout_s: float | None = None,
    native_testmon_data: Path | None = None,
) -> tuple[int, float, dict[str, Any]]:
    t0 = time.monotonic()
    sys.stderr.write(f"  {label} ... ")
    sys.stderr.flush()
    is_pytest = label.startswith("pytest")
    managed_native_lane = label.startswith("pytest native")
    closed_world_command = _native_pytest_command_is_closed_world(label, cmd)
    # ``bench slo`` starts pytest-benchmark itself, so it needs the same
    # bounded temp policy and run marker as a direct pytest step.
    has_managed_pytest_child = label == "bench slo"
    owns_pytest_environment = managed_native_lane or has_managed_pytest_child
    if is_pytest and run is not None:
        isolated_report = run.run_dir / f"pytest-report-{uuid.uuid4().hex}.json"
        cmd = [f"--json-report-file={isolated_report}" if arg.startswith("--json-report-file=") else arg for arg in cmd]
    if is_pytest:
        _clear_pytest_report(cmd)
    artifacts = run.start_step(label=label, cmd=cmd) if run is not None else None
    env = _subprocess_env(native_testmon_data=native_testmon_data)
    if is_pytest and run is not None and artifacts is not None:
        # Stamp this step's pytest identity before basetemp admission, not just
        # before launch. The identity names the basetemp directory, so a
        # preflight run against an inherited one admits (or refuses) a tree this
        # step will never use. It refuses whenever the ambient environment
        # belongs to another managed run — a nested `verify.main()` inside a
        # test inherits the enclosing run's id, resolves to a basetemp that run
        # already claimed, and exits 125 on a claim it should never have
        # contended for.
        env["POLYLOGUE_PYTEST_RUN_ID"] = pytest_step_run_id(run.run_id, artifacts.step_id)
    external_addopts_neutralized = False
    external_plugins_neutralized = False
    if owns_pytest_environment:
        _normalize_managed_pytest_environment(env, disable_plugin_autoload=managed_native_lane)
    if managed_native_lane and _pytest_uses_full_suite_basetemp(label):
        env["HYPOTHESIS_PROFILE"] = "default"
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
        if label.startswith("pytest native"):
            # The invocation aggregate compares the exact two-lane collection
            # with the native environment corpus before granting release
            # authority. Keep the complete node set in these bounded artifacts.
            env["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] = "50000"
        if run is not None and artifacts is not None:
            env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        if owns_pytest_environment:
            _normalize_managed_pytest_environment(env, disable_plugin_autoload=managed_native_lane)
        if managed_native_lane:
            external_addopts_neutralized = _PYTEST_CLEAR_CONFIGURED_ADDOPTS in cmd
            external_plugins_neutralized = (
                "PYTEST_PLUGINS" not in env and env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
            )
    closed_world_collection = closed_world_command and external_addopts_neutralized and external_plugins_neutralized
    interrupted = False
    pytest_containment_quiescent = True
    containment_error: str | None = None
    if is_pytest:
        try:
            try:
                result = _run_pytest_with_heartbeat(
                    cmd,
                    cwd=cwd,
                    env=env,
                    t0=t0,
                    run=run,
                    artifacts=artifacts,
                    timeout_override_s=timeout_s,
                )
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
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            captured_stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else ""
            captured_stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else ""
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=captured_stdout,
                stderr=captured_stderr + "\nverify: invocation deadline exhausted\n",
            )
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
        metadata["external_addopts_neutralized"] = external_addopts_neutralized
        metadata["external_plugins_neutralized"] = external_plugins_neutralized
        metadata["closed_world_collection"] = closed_world_collection
        metadata["heartbeat_s"] = _pytest_heartbeat_interval()
        metadata["timeout_s"] = _pytest_timeout_s() if timeout_s is None else timeout_s
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
        if (
            label.startswith("pytest native")
            and result.returncode == 5
            and metadata.get("selected_count") == 0
            and metadata.get("report_status") == "present"
            and isinstance(summary, Mapping)
            and summary.get("exitstatus") == 5
        ):
            # Either semantic partition may legitimately be empty. Collection
            # still synchronized the complete native corpus, so an empty lane
            # is successful evidence rather than a pytest usage failure.
            result.returncode = 0
            metadata["empty_semantic_lane"] = True
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
            timeout_s=_pytest_timeout_s() if timeout_s is None else timeout_s,
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


def _subprocess_env(*, native_testmon_data: Path | None = None) -> dict[str, str]:
    env = normalize_pytest_basetemp_env(os.environ)
    # Tests and verification helpers may inspect Git, but observational reads
    # must not refresh the index and invalidate the exact-head mutation watch.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["POLYLOGUE_ROOT"] = str(ROOT)
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    # Managed verification owns Python startup. A checkout path on PYTHONPATH
    # lets sitecustomize alter pytest controls before the managed command runs.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONOPTIMIZE", None)
    # Bytecode writing must stay enabled: PYTHONPYCACHEPREFIX below sends it to
    # .cache/pycache, and without it pytest re-runs assertion rewriting over
    # every test module in every worker on every run.
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONUSERBASE", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pycache")
    env["TESTMON_DATAFILE"] = str(native_testmon_data or TESTMON_DATA)
    env["POLYLOGUE_PYTEST_EVENTS_DIR"] = str(ROOT / PYTEST_EVENTS_DIR)
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(ROOT / PYTEST_EVENTS_PATH)
    env["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(ROOT / PYTEST_SELECTION_PATH)
    env["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(ROOT / PYTEST_SUMMARY_PATH)
    return env


def _stop_after_failed_step(label: str) -> bool:
    return label in {"lab smoke", "bench slo"}


def _native_lane_failure_requires_stop(step: Mapping[str, Any]) -> bool:
    """Continue the serial lane only after an ordinary test failure."""
    return not (step.get("exit") == 1 and step.get("diagnosis") == "pytest_failed")


# ── step builder ────────────────────────────────────────────────────


def _native_pytest_steps(
    *,
    testmon_mode: str,
    testmon_environment: str,
    parallel_worker_args: Sequence[str],
) -> list[tuple[str, list[str]]]:
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--ignore=tests/benchmarks",
        "--durations=10",
        f"--junitxml={PYTEST_JUNIT_REPORT_DIR}/verify-latest.xml",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={PYTEST_REPORT_PATH}",
        "-p",
        "devtools.pytest_progress_plugin",
    ]
    pytest_cmd.extend(_PYTEST_MANAGED_PLUGIN_ARGS)
    pytest_cmd.extend(_PYTEST_CLOSED_WORLD_COLLECTION_ARGS)
    native_args = ["--testmon", f"--testmon-env={testmon_environment}"]
    if testmon_mode == "affected":
        native_args.append("--testmon-forceselect")
    else:
        native_args.append("--testmon-noselect")

    parallel_cmd = [
        *pytest_cmd,
        "-m",
        "not load_sensitive",
        *native_args,
        *parallel_worker_args,
    ]

    def _serial_report_arg(arg: str) -> str:
        if arg.startswith("--junitxml="):
            return f"--junitxml={PYTEST_JUNIT_REPORT_DIR}/verify-latest-serial.xml"
        if arg.startswith("--json-report-file="):
            return f"--json-report-file={PYTEST_REPORT_DIR / 'last-pytest-serial.json'}"
        return arg

    serial_cmd = [_serial_report_arg(arg) for arg in pytest_cmd]
    serial_cmd.extend(
        [
            "-m",
            "load_sensitive",
            *native_args,
            "-p",
            "no:randomly",
            "-n",
            "0",
        ]
    )
    return [
        (f"pytest native parallel ({testmon_mode})", parallel_cmd),
        (f"pytest native serial ({testmon_mode})", serial_cmd),
    ]


def _native_pytest_command_is_closed_world(label: str, cmd: Sequence[str]) -> bool:
    """Accept only a command produced by the managed native-lane builder."""
    match = re.fullmatch(r"pytest native (parallel|serial) \((affected|bootstrap|full)\)", label)
    if match is None:
        return False
    environment_args = [arg for arg in cmd if arg.startswith("--testmon-env=")]
    worker_request = pytest_command_worker_request(cmd)
    if len(environment_args) != 1 or worker_request is None or not worker_request.isdigit():
        return False
    expected_steps = _native_pytest_steps(
        testmon_mode=match.group(2),
        testmon_environment=environment_args[0].removeprefix("--testmon-env="),
        parallel_worker_args=("--dist=loadgroup", "-n", worker_request),
    )
    expected = dict(expected_steps).get(label)
    return expected is not None and list(cmd) == expected


def build_verify_steps(
    *,
    quick: bool,
    lab: bool,
    commit: bool = False,
    testmon_mode: str = "affected",
    testmon_environment: str = "",
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
                ("verify ci-commands", _devtools_cmd("verify ci-commands")),
                ("verify doc-commands", _devtools_cmd("verify doc-commands")),
                ("lab schema roundtrip", _devtools_cmd("lab schema roundtrip", "--all")),
                # Static, archive-independent, sub-second: an index bump that
                # lands without its lifecycle.py delta declaration silently
                # downgrades every existing generation to a full raw replay
                # (polylogue-9rw0). Gated here, not behind --lab, because the
                # failure surfaces as an unqueryable live archive rather than
                # as a test failure.
                ("lab policy schema-versioning", _devtools_cmd("lab policy schema-versioning")),
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
            ]
        )

    if not quick and not commit:
        _report_dir = PYTEST_JUNIT_REPORT_DIR
        _report_dir.mkdir(parents=True, exist_ok=True)
        PYTEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        if testmon_mode not in {"affected", "bootstrap", "full"}:
            raise ValueError(f"unknown native testmon mode: {testmon_mode}")
        if not testmon_environment:
            raise ValueError("native testmon environment is required for pytest verification")
        # Every native command owns its collection, option, and plugin surface;
        # the benchmark root remains the one explicit non-correctness corpus.
        steps.extend(
            _native_pytest_steps(
                testmon_mode=testmon_mode,
                testmon_environment=testmon_environment,
                parallel_worker_args=_pytest_worker_args(),
            )
        )

    if lab:
        steps.append(("lab smoke", _devtools_cmd("lab smoke", "run", "archive-smoke", "--tier", "0")))
        steps.append(("bench slo", _devtools_cmd("bench slo", "--include-lab")))
        steps.append(("lab policy timestamp-doctrine", _devtools_cmd("lab policy timestamp-doctrine")))
        steps.append(("lab policy insight-honesty", _devtools_cmd("lab policy insight-honesty")))
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pytest_worker_args(*, maximum: int | None = None) -> list[str]:
    """Return the managed worker count, optionally capped for a bounded lane."""
    workers = adaptive_pytest_worker_count(os.environ)
    if maximum is not None:
        workers = min(workers, maximum)
    return ["--dist=loadgroup", "-n", str(workers)]


def _pytest_command_concurrency(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> int:
    """Return a fail-closed reservation for the final pytest command."""
    request = pytest_command_worker_request(cmd)
    if request is None:
        return 0
    if request == "auto":
        auto_workers = (env if env is not None else os.environ).get("PYTEST_XDIST_AUTO_NUM_WORKERS", "").strip()
        if auto_workers:
            with contextlib.suppress(ValueError):
                configured = int(auto_workers)
                if configured > 0:
                    return configured
    try:
        return max(0, int(request))
    except ValueError:
        return max(1, os.cpu_count() or 1)


def _pytest_uses_full_suite_basetemp(label: str) -> bool:
    """Whether this semantic lane may materialize the complete corpus tree."""
    return label.startswith("pytest native") and ("(bootstrap)" in label or "(full)" in label)


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


def _changed_test_relevant_paths(base_commit: str, head_commit: str) -> tuple[str, ...]:
    roots = ("polylogue/", "devtools/", "tests/", "packaging/")
    exact = {"pyproject.toml", "uv.lock", "pytest.ini", "tox.ini", "setup.cfg"}
    return tuple(
        sorted(path for path in _changed_paths(base_commit, head_commit) if path in exact or path.startswith(roots))
    )


@dataclass(slots=True)
class _ActiveVerifyRun:
    run: VerifyRun
    started_at: float
    verification_scope: VerificationScope
    head: str | None
    mutation_monitor: CheckoutMutationMonitor | None = None
    initial_worktree_fingerprint: str | None = None
    owned_native_testmon_state: _OwnedNativeTestmonState | None = None


_ACTIVE_VERIFY_RUN: _ActiveVerifyRun | None = None


def _start_active_checkout_mutation_monitor(monitor: CheckoutMutationMonitor) -> None:
    start_checkout_mutation_monitor(monitor)
    if _ACTIVE_VERIFY_RUN is not None:
        _ACTIVE_VERIFY_RUN.mutation_monitor = monitor


def _finish_active_checkout_mutation_monitor(monitor: CheckoutMutationMonitor) -> CheckoutMutationObservation:
    try:
        return finish_checkout_mutation_monitor(monitor)
    finally:
        if _ACTIVE_VERIFY_RUN is not None and _ACTIVE_VERIFY_RUN.mutation_monitor is monitor:
            _ACTIVE_VERIFY_RUN.mutation_monitor = None


def _close_active_native_testmon_state() -> None:
    if _ACTIVE_VERIFY_RUN is None or _ACTIVE_VERIFY_RUN.owned_native_testmon_state is None:
        return
    state = _ACTIVE_VERIFY_RUN.owned_native_testmon_state
    _ACTIVE_VERIFY_RUN.owned_native_testmon_state = None
    state.close()


def _planned_verification_scope(
    args: argparse.Namespace,
    *,
    testmon_mode: str | None,
) -> VerificationScope:
    if args.quick or args.commit:
        return VerificationScope.NON_TEST
    if testmon_mode in {"bootstrap", "full"}:
        return VerificationScope.RELEASE_BASELINE
    return VerificationScope.AFFECTED


def _pytest_profile() -> str:
    return "correctness=complete"


def _native_pytest_environment(*, force_release_profile: bool) -> dict[str, str | None]:
    environment = {
        # Hypothesis uses its default profile when the variable is absent.
        # Record that effective value in the testmon environment identity so a
        # bootstrap graph is reusable by the following affected invocation.
        "HYPOTHESIS_PROFILE": os.environ.get("HYPOTHESIS_PROFILE") or "default",
        "POLYLOGUE_CI": os.environ.get("POLYLOGUE_CI"),
    }
    if force_release_profile:
        environment["HYPOTHESIS_PROFILE"] = "default"
    return environment


def _native_environment_after_run(
    preparation: NativeTestmonPreparation,
    *,
    required_executable_paths: Sequence[str],
) -> NativeTestmonState:
    state = _ACTIVE_VERIFY_RUN.owned_native_testmon_state if _ACTIVE_VERIFY_RUN is not None else None
    temporary_state = state is None
    if state is None:
        state = _open_owned_native_testmon_state(ROOT)
    try:
        return inspect_native_testmon_environment(
            state.data_path,
            environment_name=preparation.environment_name,
            required_executable_paths=required_executable_paths,
        )
    finally:
        if temporary_state:
            state.close()


def _release_baseline_allowed(
    *,
    selection_mode: str | None,
    verification_scope: VerificationScope,
    exit_code: int,
    checkout_stable: bool,
    aggregate: Mapping[str, Any] | None,
) -> bool:
    if selection_mode not in {"bootstrap", "full"} or exit_code != 0 or not checkout_stable or aggregate is None:
        return False
    if verification_scope != VerificationScope.RELEASE_BASELINE:
        return False
    cleanup = aggregate.get("cleanup")
    containment = aggregate.get("containment")
    return bool(
        aggregate.get("complete_corpus_covered") is True
        and aggregate.get("terminal_green") is True
        and aggregate.get("external_addopts_neutralized") is True
        and aggregate.get("external_plugins_neutralized") is True
        and aggregate.get("closed_world_collection") is True
        and isinstance(cleanup, Mapping)
        and cleanup.get("complete") is True
        and isinstance(containment, Mapping)
        and containment.get("complete") is True
    )


def _finalize_preflight_failure(
    run: VerifyRun,
    *,
    started_at: float,
    tier: str,
    head: str | None,
    verification_scope: VerificationScope,
    diagnosis: str,
    exit_code: int,
    message: str,
    use_json: bool,
    mutation_monitor: CheckoutMutationMonitor | None = None,
    initial_worktree_fingerprint: str | None = None,
) -> int:
    """Persist one normalized failed invocation before pytest can start."""
    final_head = _git_head()
    try:
        final_worktree_fingerprint = worktree_fingerprint(ROOT) if mutation_monitor is not None else "unavailable"
    except Exception:
        final_worktree_fingerprint = "unavailable"
    mutation_observation = (
        _finish_active_checkout_mutation_monitor(mutation_monitor) if mutation_monitor is not None else None
    )
    checkout_diagnosis: str | None = None
    if mutation_monitor is None:
        checkout_diagnosis = "preflight_failed_before_checkout_monitor"
    elif (
        head is None
        or final_head is None
        or initial_worktree_fingerprint in {None, "unavailable"}
        or final_worktree_fingerprint == "unavailable"
        or mutation_observation is None
        or mutation_observation.unavailable
    ):
        checkout_diagnosis = "checkout_fingerprint_unavailable"
    elif (
        final_head != head or mutation_observation.changed or final_worktree_fingerprint != initial_worktree_fingerprint
    ):
        checkout_diagnosis = "checkout_changed_during_verification"

    duration_s = round(time.monotonic() - started_at, 2)
    artifacts = run.start_step(label="verify preflight", cmd=[])
    step = run.finish_step(
        step_id=artifacts.step_id,
        result={
            "duration_s": duration_s,
            "exit": exit_code,
            "diagnosis": diagnosis,
            "error": message,
            "checkout_diagnosis": checkout_diagnosis,
        },
    )
    payload = run.finish(
        exit_code=exit_code,
        duration_s=duration_s,
        diagnosis=diagnosis,
        verification_scope=verification_scope.value,
        release_baseline_allowed=False,
        final_git_head=final_head,
        final_worktree_fingerprint=final_worktree_fingerprint,
        checkout_mutation_path=(mutation_observation.observed_path if mutation_observation is not None else None),
        checkout_diagnosis=checkout_diagnosis,
    )
    history_entry = {
        **payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "final_git_head": final_head,
        "tier": tier,
        "checkout_root": str(ROOT.resolve()),
        "worktree_fingerprint": initial_worktree_fingerprint,
        "final_worktree_fingerprint": final_worktree_fingerprint,
        "steps": [step] if step is not None else [],
        "total_duration_s": duration_s,
        "exit_code": exit_code,
        "verification_scope": verification_scope.value,
        "release_baseline_allowed": False,
        "diagnosis": diagnosis,
    }
    _save_history(history_entry)
    if use_json:
        _print_json(history_entry)
    sys.stderr.write(f"verify: {message}\n")
    _notify(
        _format_completion_notification(
            exit_code=exit_code,
            total_duration=duration_s,
            step_results=history_entry["steps"],
        )
    )
    return exit_code


def _main(argv: list[str] | None = None) -> int:
    global _ACTIVE_VERIFY_RUN
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(description="Run the local verification baseline.")
    parser.add_argument("--quick", action="store_true", help="Skip pytest and run only fast local gates.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete pytest correctness corpus (excluding performance benchmarks).",
    )
    parser.add_argument("--full", action="store_true", help="Alias for --all.")
    parser.add_argument("--commit", action="store_true", help="Pre-commit tier: format + lint + mypy only.")
    parser.add_argument(
        "--lab",
        action="store_true",
        help="Run the native pytest-testmon lifecycle plus verification-lab checks.",
    )
    parser.add_argument("--history", action="store_true", help="Print last 10 verify runs and exit.")
    parser.add_argument("--json", action="store_true", default=None, help="Write structured JSON to stdout.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    _anchor_verification_paths()
    if args.history:
        try:
            assert_polylogue_matches_checkout(ROOT, context="devtools verify")
        except CheckoutImportMismatchError as exc:
            sys.stderr.write(f"verify: {exc}\n")
            return 125
        _print_history()
        return 0

    full_requested = bool(args.all or args.full)
    use_json = args.json if args.json is not None else not sys.stdout.isatty()
    tier = (
        "commit"
        if args.commit
        else "quick"
        if args.quick
        else "full"
        if full_requested
        else "lab"
        if args.lab
        else "testmon"
    )
    head = _git_head()
    pytest_enabled = not (args.quick or args.commit)
    managed_pytest_enabled = pytest_enabled or args.lab
    planned_scope = _planned_verification_scope(
        args,
        testmon_mode="full" if full_requested else None,
    )
    verify_run = VerifyRun(
        tier=tier,
        argv=list(sys.argv[1:] if argv is None else argv),
        git_head=head,
    )
    _ACTIVE_VERIFY_RUN = _ActiveVerifyRun(
        run=verify_run,
        started_at=started_at,
        verification_scope=planned_scope,
        head=head,
    )

    optimization_level = _python_optimization_level()
    if managed_pytest_enabled and optimization_level > 0:
        return _finalize_preflight_failure(
            verify_run,
            started_at=started_at,
            tier=tier,
            head=head,
            verification_scope=planned_scope,
            diagnosis="optimized_python_interpreter",
            exit_code=125,
            message=(
                "Python optimization disables verification assertions; "
                f"refusing managed pytest at optimization level {optimization_level}"
            ),
            use_json=bool(use_json),
        )

    try:
        fingerprint = assert_polylogue_matches_checkout(ROOT, context="devtools verify")
    except CheckoutImportMismatchError as exc:
        return _finalize_preflight_failure(
            verify_run,
            started_at=started_at,
            tier=tier,
            head=head,
            verification_scope=planned_scope,
            diagnosis="checkout_import_mismatch",
            exit_code=125,
            message=str(exc),
            use_json=bool(use_json),
        )
    polylogue_import_path = fingerprint.polylogue_import_path
    environment_fingerprint = fingerprint.as_dict()
    verify_run.update_checkout_provenance(
        polylogue_import_path=str(polylogue_import_path),
        environment_fingerprint=environment_fingerprint,
    )
    sys.stderr.write(f"verify: polylogue package → {polylogue_import_path}\n")

    mutation_monitor = CheckoutMutationMonitor(ROOT)
    _start_active_checkout_mutation_monitor(mutation_monitor)
    checkout_fingerprint = worktree_fingerprint(ROOT)
    _finish_active_checkout_mutation_monitor(mutation_monitor)
    mutation_monitor = CheckoutMutationMonitor(ROOT)
    _start_active_checkout_mutation_monitor(mutation_monitor)
    assert _ACTIVE_VERIFY_RUN is not None
    _ACTIVE_VERIFY_RUN.initial_worktree_fingerprint = checkout_fingerprint
    verify_run.update_checkout_provenance(worktree_fingerprint=checkout_fingerprint)

    base_commit = _git_commit("origin/master") if pytest_enabled else None
    if pytest_enabled and (base_commit is None or head is None):
        return _finalize_preflight_failure(
            verify_run,
            started_at=started_at,
            tier=tier,
            head=head,
            verification_scope=planned_scope,
            diagnosis="native_git_authority_unavailable",
            exit_code=125,
            message="cannot resolve immutable Git refs for native affected-test authority.",
            use_json=bool(use_json),
            mutation_monitor=mutation_monitor,
            initial_worktree_fingerprint=checkout_fingerprint,
        )

    relevant_paths: tuple[str, ...] = ()
    required_executable_paths: tuple[str, ...] = ()
    preparation_required_executable_paths: tuple[str, ...] = ()
    runtime_data_paths: tuple[str, ...] = ()
    preparation: NativeTestmonPreparation | None = None
    testmon_mode: str | None = None
    native_pytest_environment = _native_pytest_environment(force_release_profile=full_requested)
    preparation_mutation_observation: CheckoutMutationObservation | None = None
    if pytest_enabled:
        assert base_commit is not None
        assert head is not None
        try:
            relevant_paths = _changed_test_relevant_paths(base_commit, head)
            change_impact = classify_native_testmon_changes(ROOT, relevant_paths)
            preparation_required_executable_paths = change_impact.executable_paths
            required_executable_paths = tuple(
                path for path in preparation_required_executable_paths if (ROOT / path).is_file()
            )
            runtime_data_paths = change_impact.runtime_data_paths
            preparation = prepare_native_testmon_environment(
                ROOT,
                required_executable_paths=preparation_required_executable_paths,
                pytest_profile=_pytest_profile(),
                pytest_environment=native_pytest_environment,
            )
            if (
                preparation.selection_mode == "bootstrap"
                and native_pytest_environment["HYPOTHESIS_PROFILE"] != "default"
            ):
                native_pytest_environment = _native_pytest_environment(force_release_profile=True)
                preparation = prepare_native_testmon_environment(
                    ROOT,
                    required_executable_paths=preparation_required_executable_paths,
                    pytest_profile=_pytest_profile(),
                    pytest_environment=native_pytest_environment,
                )
            assert _ACTIVE_VERIFY_RUN is not None
            _ACTIVE_VERIFY_RUN.owned_native_testmon_state = _open_owned_native_testmon_state(ROOT)
        except NativeTestmonDeadlineError as exc:
            return _finalize_preflight_failure(
                verify_run,
                started_at=started_at,
                tier=tier,
                head=head,
                verification_scope=planned_scope,
                diagnosis="verify_invocation_deadline_exceeded",
                exit_code=124,
                message=str(exc),
                use_json=bool(use_json),
                mutation_monitor=mutation_monitor,
                initial_worktree_fingerprint=checkout_fingerprint,
            )
        except (NativeTestmonRepairError, PytestResourceError) as exc:
            return _finalize_preflight_failure(
                verify_run,
                started_at=started_at,
                tier=tier,
                head=head,
                verification_scope=planned_scope,
                diagnosis="native_testmon_preparation_failed",
                exit_code=125,
                message=f"native pytest-testmon preparation failed: {exc}",
                use_json=bool(use_json),
                mutation_monitor=mutation_monitor,
                initial_worktree_fingerprint=checkout_fingerprint,
            )
        testmon_mode = "full" if full_requested or runtime_data_paths else preparation.selection_mode
        if preparation.removed_paths:
            sys.stderr.write(
                "verify: repaired invalid native pytest-testmon state by removing only "
                + ", ".join(str(path) for path in preparation.removed_paths)
                + "\n"
            )
        if preparation.copied_from is not None:
            sys.stderr.write(f"verify: copied matching native pytest-testmon DB from {preparation.copied_from}\n")
        elif preparation.selection_mode == "bootstrap":
            sys.stderr.write("verify: native pytest-testmon environment is empty; plain verify will build it\n")
        if runtime_data_paths and not full_requested:
            sys.stderr.write(
                "verify: changed package runtime data is outside Python tracing; running the complete corpus: "
                + ", ".join(runtime_data_paths)
                + "\n"
            )

    planned_scope = _planned_verification_scope(args, testmon_mode=testmon_mode)
    assert _ACTIVE_VERIFY_RUN is not None
    _ACTIVE_VERIFY_RUN.verification_scope = planned_scope

    if not use_json:
        sys.stderr.write("verify: running local verification baseline\n")
    if pytest_enabled:
        _warn_low_memory()

    try:
        steps = build_verify_steps(
            quick=bool(args.quick),
            commit=bool(args.commit),
            lab=bool(args.lab),
            testmon_mode=testmon_mode or "affected",
            testmon_environment=preparation.environment_name if preparation is not None else "",
        )
    except (PytestResourceError, ValueError) as exc:
        return _finalize_preflight_failure(
            verify_run,
            started_at=started_at,
            tier=tier,
            head=head,
            verification_scope=planned_scope,
            diagnosis="pytest_resource_preflight_failed",
            exit_code=125,
            message=str(exc),
            use_json=bool(use_json),
            mutation_monitor=mutation_monitor,
            initial_worktree_fingerprint=checkout_fingerprint,
        )

    # Git probes and native testmon preparation can refresh the index as part
    # of their own read path. Retain that interval's observation: a tracked
    # file can be edited and restored while the graph is prepared, which makes
    # any resulting selection unsuitable as exact-head authority.
    preparation_mutation_observation = _finish_active_checkout_mutation_monitor(mutation_monitor)
    mutation_monitor = CheckoutMutationMonitor(ROOT)
    _start_active_checkout_mutation_monitor(mutation_monitor)
    step_results: list[dict[str, Any]] = []
    exit_code = 0
    native_graph_touched = False
    for label, cmd in steps:
        if label.startswith("pytest"):
            _warn_low_memory()
        rc, elapsed, metadata = _run(label, cmd, run=verify_run)
        step_result: dict[str, Any] = {"name": label, "duration_s": round(elapsed, 2), "exit": rc}
        step_result.update(metadata)
        if label.startswith("pytest native parallel"):
            step_result["semantic_lane"] = "parallel"
            native_graph_touched = True
        elif label.startswith("pytest native serial"):
            step_result["semantic_lane"] = "serial"
            native_graph_touched = True
        step_results.append(step_result)
        if rc == 0:
            continue
        if exit_code == 0 or rc in {2, 3, 4, 124, 125, 130}:
            exit_code = rc
        if label.startswith("pytest native parallel") and not _native_lane_failure_requires_stop(step_result):
            continue
        if label.startswith("pytest") or rc == 130 or _stop_after_failed_step(label):
            break

    assert mutation_monitor is not None
    final_head = _git_head()
    final_checkout_fingerprint = worktree_fingerprint(ROOT)
    mutation_observation = _finish_active_checkout_mutation_monitor(mutation_monitor)
    checkout_stable = True
    checkout_fingerprint_unavailable = (
        head is None
        or final_head is None
        or "unavailable"
        in {
            checkout_fingerprint,
            final_checkout_fingerprint,
        }
    )
    if (
        checkout_fingerprint_unavailable
        or mutation_observation.unavailable
        or preparation_mutation_observation.unavailable
    ):
        checkout_stable = False
        diagnosis = (
            "checkout_fingerprint_unavailable"
            if checkout_fingerprint_unavailable
            else "checkout_mutation_monitor_unavailable"
        )
    elif (
        final_head != head
        or preparation_mutation_observation.changed
        or mutation_observation.changed
        or final_checkout_fingerprint != checkout_fingerprint
    ):
        checkout_stable = False
        diagnosis = "checkout_changed_during_verification"
    else:
        diagnosis = None
    if diagnosis is not None:
        stability_step = {
            "name": "checkout stability",
            "duration_s": 0.0,
            "exit": 125,
            "diagnosis": diagnosis,
            "initial_git_head": head,
            "final_git_head": final_head,
            "initial_worktree_fingerprint": checkout_fingerprint,
            "final_worktree_fingerprint": final_checkout_fingerprint,
            "transient_checkout_mutation": (preparation_mutation_observation.changed or mutation_observation.changed),
            "checkout_mutation_path": (
                preparation_mutation_observation.observed_path or mutation_observation.observed_path
            ),
        }
        step_results.append(stability_step)
        exit_code = 125
        sys.stderr.write("verify: checkout contents were not stable for exact-head evidence.\n")
        if native_graph_touched:
            try:
                removed = remove_invalid_native_testmon_state(ROOT)
            except NativeTestmonRepairError as exc:
                stability_step["testmon_cleanup_error"] = str(exc)
            else:
                stability_step["testmon_cleanup_paths"] = [str(path) for path in removed]

    native_state = None
    if preparation is not None:
        native_state = _native_environment_after_run(
            preparation,
            required_executable_paths=required_executable_paths,
        )
        if not native_state.valid:
            graph_step = {
                "name": "pytest native graph validation",
                "duration_s": 0.0,
                "exit": 5,
                "diagnosis": "native_testmon_graph_invalid",
                "reason": native_state.reason,
                "missing_executable_paths": list(native_state.missing_executable_paths),
            }
            step_results.append(graph_step)
            if exit_code == 0:
                exit_code = 5
            if native_state.missing_executable_paths:
                sys.stderr.write(
                    "verify: changed executable modules have no runtime dependency edge: "
                    + ", ".join(native_state.missing_executable_paths)
                    + "\n"
                )
    _close_active_native_testmon_state()

    total_duration = round(time.monotonic() - started_at, 2)
    pytest_aggregate: dict[str, Any] | None = None
    native_environment = native_state.environment if native_state is not None else None
    if preparation is not None:
        pytest_aggregate = aggregate_native_testmon_run(
            ROOT,
            steps=step_results,
            environment_name=preparation.environment_name,
            corpus_nodeids=native_environment.nodeids if native_environment is not None else (),
            environment_status=native_state.status if native_state is not None else "unavailable",
            environment_reason=native_state.reason if native_state is not None else "post-run inspection unavailable",
            selection_mode=testmon_mode or "affected",
            invocation_duration_s=total_duration,
        )

    # Finalization can outlast the last pytest lane, so wall_s is taken here
    # rather than from the lanes alone.
    finalized_duration = round(time.monotonic() - started_at, 2)
    if finalized_duration > total_duration:
        total_duration = finalized_duration
    if pytest_aggregate is not None:
        pytest_aggregate["wall_s"] = total_duration

    release_baseline_allowed = _release_baseline_allowed(
        selection_mode=testmon_mode,
        verification_scope=planned_scope,
        exit_code=exit_code,
        checkout_stable=checkout_stable,
        aggregate=pytest_aggregate,
    )
    verification_scope = planned_scope
    if testmon_mode == "affected":
        release_baseline_allowed = False

    checkout_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in reversed(step_results)
            if step.get("name") == "checkout stability" and isinstance(step.get("diagnosis"), str)
        ),
        None,
    )
    pytest_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in reversed(step_results)
            if str(step.get("name", "")).startswith("pytest") and step.get("exit") != 0
        ),
        None,
    )
    run_diagnosis = checkout_diagnosis or pytest_diagnosis

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
        "verification_scope": verification_scope.value,
        "release_baseline_allowed": release_baseline_allowed,
    }
    if preparation is not None:
        history_entry["testmon_environment"] = {
            "name": preparation.environment_name,
            "selection_mode": testmon_mode,
            "copied_from": str(preparation.copied_from) if preparation.copied_from is not None else None,
            "required_executable_paths": list(required_executable_paths),
            "bootstrap_trigger_paths": list(preparation_required_executable_paths),
            "runtime_data_paths": list(runtime_data_paths),
        }
    if pytest_aggregate is not None:
        history_entry["pytest_aggregate"] = pytest_aggregate
    if run_diagnosis is not None:
        history_entry["diagnosis"] = run_diagnosis

    finalized_payload = verify_run.finish(
        exit_code=exit_code,
        duration_s=total_duration,
        diagnosis=run_diagnosis,
        verification_scope=verification_scope.value,
        release_baseline_allowed=release_baseline_allowed,
        final_git_head=final_head,
        final_worktree_fingerprint=final_checkout_fingerprint,
        checkout_mutation_path=(preparation_mutation_observation.observed_path or mutation_observation.observed_path),
        checkout_diagnosis=checkout_diagnosis,
        pytest_aggregate=pytest_aggregate,
    )
    history_entry["pytest_aggregate"] = finalized_payload["pytest_aggregate"]
    if use_json:
        _print_json(history_entry)
    elif exit_code == 0:
        flags = _compare_against_last(step_results)
        sys.stderr.write(f"\nverify: all checks passed ({total_duration:.1f}s total)")
        if flags:
            sys.stderr.write("\n")
            for flag in flags:
                sys.stderr.write(flag + "\n")
        else:
            sys.stderr.write("\n")
    else:
        sys.stderr.write(f"\nverify: FAILED ({total_duration:.1f}s); fix before pushing\n")

    _save_history(history_entry)

    if exit_code == 0:
        _stamp_head()
    else:
        _notify(
            _format_completion_notification(
                exit_code=exit_code,
                total_duration=total_duration,
                step_results=step_results,
            )
        )
    return exit_code


def _finalize_verify_runner_exception(
    active: _ActiveVerifyRun,
    exc: BaseException,
    *,
    use_json: bool,
) -> int:
    """Leave typed, durable failed evidence when verification orchestration raises."""
    interrupted = isinstance(exc, KeyboardInterrupt)
    diagnosis = "verify_interrupted" if interrupted else "verify_runner_exception"
    exit_code = 130 if interrupted else 125
    run = active.run
    try:
        final_head = _git_head()
    except Exception:
        final_head = None
    try:
        final_worktree_fingerprint = worktree_fingerprint(ROOT)
    except Exception:
        final_worktree_fingerprint = "unavailable"
    mutation_observation = None
    if active.mutation_monitor is not None:
        try:
            mutation_observation = _finish_active_checkout_mutation_monitor(active.mutation_monitor)
        except Exception:
            mutation_observation = None
    _close_active_native_testmon_state()
    run.finish_interrupted_steps(
        exit_code=exit_code,
        diagnosis=diagnosis,
        termination_reason="operator_interrupt" if interrupted else "runner_exception",
    )
    if (
        active.head is None
        or final_head is None
        or active.initial_worktree_fingerprint in {None, "unavailable"}
        or final_worktree_fingerprint == "unavailable"
        or mutation_observation is None
        or mutation_observation.unavailable
    ):
        checkout_diagnosis = "checkout_fingerprint_unavailable"
    elif (
        final_head != active.head
        or mutation_observation.changed
        or final_worktree_fingerprint != active.initial_worktree_fingerprint
    ):
        checkout_diagnosis = "checkout_changed_during_verification"
    else:
        checkout_diagnosis = None
    payload = run.finish(
        exit_code=exit_code,
        duration_s=time.monotonic() - active.started_at,
        diagnosis=diagnosis,
        verification_scope=active.verification_scope.value,
        release_baseline_allowed=False,
        final_git_head=final_head,
        final_worktree_fingerprint=final_worktree_fingerprint,
        checkout_mutation_path=(mutation_observation.observed_path if mutation_observation is not None else None),
        checkout_diagnosis=checkout_diagnosis,
    )
    payload["exception_type"] = type(exc).__name__
    payload["error"] = str(exc)
    _save_history(payload)
    if use_json:
        _print_json(payload)
    sys.stderr.write(f"verify: unexpected runner exception: {exc}\n")
    return exit_code


@finalize_checkout_mutation_monitors
def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_VERIFY_RUN
    _ACTIVE_VERIFY_RUN = None
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    native_pytest_enabled = not any(flag in raw_argv for flag in ("--quick", "--commit", "--history"))
    lock = _native_testmon_lifecycle_lock(ROOT) if native_pytest_enabled else contextlib.nullcontext()
    try:
        with lock:
            try:
                return _main(argv)
            except KeyboardInterrupt as exc:
                if _ACTIVE_VERIFY_RUN is None:
                    raise
                return _finalize_verify_runner_exception(
                    _ACTIVE_VERIFY_RUN,
                    exc,
                    use_json="--json" in raw_argv,
                )
            except Exception as exc:
                if _ACTIVE_VERIFY_RUN is None:
                    raise
                return _finalize_verify_runner_exception(
                    _ACTIVE_VERIFY_RUN,
                    exc,
                    use_json="--json" in raw_argv,
                )
            finally:
                _close_active_native_testmon_state()
                _ACTIVE_VERIFY_RUN = None
    except PytestResourceError as exc:
        sys.stderr.write(f"verify: {exc}\n")
        return 125
