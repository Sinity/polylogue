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
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from devtools.verify_runs import (
    CURRENT_CONTAINMENT_PATH,
    CURRENT_EVENTS_DIR,
    CURRENT_POSTMORTEM_PATH,
    CURRENT_RESOURCES_PATH,
    PytestStepArtifacts,
    ResourceSampler,
    VerifyRun,
    classify_pytest_result,
    cleanup_managed_pytest_basetemp,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    latest_event_from_paths,
    merge_worker_events,
    normalize_pytest_basetemp_env,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]

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


HISTORY_PATH = Path(".cache/verify-history.jsonl")
TESTMON_DATA = Path(".cache/testmon/testmondata")
TESTMON_SEED_STAMP = Path(".cache/testmon/seed.json")
TESTMON_SEED_ATTEMPT = Path(".cache/testmon/seed-attempt.json")
TESTMON_SEED_PROTOCOL_VERSION = 2
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
DEFAULT_TESTMON_WORKERS = "4"


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


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_latest_pytest_event(path: Path = PYTEST_EVENTS_PATH) -> dict[str, Any] | None:
    """Return the latest valid pytest event from the live JSONL ledger."""
    if path == PYTEST_EVENTS_PATH:
        return latest_event_from_paths(PYTEST_EVENTS_DIR, PYTEST_EVENTS_PATH)
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
    if "-n" in cmd:
        index = cmd.index("-n")
        if index + 1 < len(cmd):
            metadata["pytest_workers"] = cmd[index + 1]
    else:
        metadata["pytest_workers"] = "unset"
    if "--testmon" in cmd:
        metadata["pytest_selection"] = "testmon-noselect" if "--testmon-noselect" in cmd else "testmon"
    else:
        metadata["pytest_selection"] = "full"
    return metadata


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
    latest_event = _read_latest_pytest_event()
    if latest_event is not None:
        payload["latest_test_event"] = {
            key: latest_event[key]
            for key in ("event", "nodeid", "when", "outcome", "duration_s", "updated_at")
            if key in latest_event
        }
        if latest_event.get("event") == "test_started" and isinstance(latest_event.get("nodeid"), str):
            payload["current_test_nodeid"] = latest_event["nodeid"]
    PYTEST_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PYTEST_PROGRESS_PATH.with_name(f"{PYTEST_PROGRESS_PATH.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(PYTEST_PROGRESS_PATH)


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
    runner_subreaper_enabled = enable_child_subreaper()
    preserved_runner_descendants = tuple(descendant_process_identities(os.getpid()))
    receipt_path = (
        artifacts.containment_path
        if artifacts is not None
        else Path(env.get("POLYLOGUE_PYTEST_CONTAINMENT_PATH", str(Path.cwd() / PYTEST_CONTAINMENT_PATH)))
    )
    launch = build_supervisor_launch(
        cmd,
        owner_pid=os.getpid(),
        timeout_s=timeout_s,
        term_grace_s=term_grace_s,
        receipt_path=receipt_path,
        run_id=run.run_id if run is not None else env.get("POLYLOGUE_PYTEST_RUN_ID"),
        env=env,
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
    initial_event = _read_latest_pytest_event()
    last_progress_marker: str | None = initial_event.get("updated_at") if initial_event is not None else None
    last_progress_at = last_sample
    seen_any_progress_event = initial_event is not None

    def _refresh_progress_marker(at: float, latest: dict[str, Any] | None = None) -> None:
        nonlocal last_progress_marker, last_progress_at, seen_any_progress_event
        if latest is None:
            latest = _read_latest_pytest_event()
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
                latest_event = _read_latest_pytest_event()
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
                )
            sample_now = time.monotonic()
            if (
                sampler is not None
                and resource_interval_s > 0
                and sample_now - last_resource_sample >= resource_interval_s
            ):
                sampler.sample(event="sample")
                last_resource_sample = sample_now
            if process.poll() is not None and not selector.get_map():
                break
    except BaseException:
        if process.poll() is None:
            _request_supervisor_termination(process, launch, reason="pytest runner interrupted")
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
        )
    _write_pytest_output(stdout, stderr)
    if artifacts is not None:
        artifacts.stdout_path.write_text(stdout, encoding="utf-8")
        artifacts.stderr_path.write_text(stderr, encoding="utf-8")
        artifacts.output_path.write_text(stdout + stderr, encoding="utf-8")
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
    if is_pytest:
        _clear_pytest_report(cmd)
    artifacts = run.start_step(label=label, cmd=cmd) if run is not None else None
    env = _subprocess_env()
    pytest_tmpfs = False
    basetemp_cleanup: Path | None = None
    if is_pytest:
        pytest_tmpfs = env.get("POLYLOGUE_PYTEST_TMPFS") == "1"
        if label.startswith("pytest seed-testmon"):
            # A complete corpus is currently ~16K nodes. Preserve the whole
            # selection in the attempt receipt so interrupted seeds can prove
            # eventual coverage instead of relying on a 500-node sample.
            env["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] = "50000"
        if run is not None and artifacts is not None:
            env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        result = _run_pytest_with_heartbeat(cmd, cwd=cwd, env=env, t0=t0, run=run, artifacts=artifacts)
        basetemp_cleanup = cleanup_managed_pytest_basetemp(
            root=ROOT,
            run_id=env.get("POLYLOGUE_PYTEST_RUN_ID", ""),
            env=env,
        )
        if artifacts is not None:
            merge_worker_events(artifacts.events_dir, artifacts.events_merged_path)
            with contextlib.suppress(FileNotFoundError):
                shutil.copyfile(PYTEST_PROGRESS_PATH, artifacts.progress_path)
    else:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - t0
    metadata: dict[str, Any] = {}
    if artifacts is not None:
        metadata["run_id"] = run.run_id if run is not None else None
        metadata["artifact_dir"] = str(artifacts.step_dir.relative_to(Path.cwd()))
    if is_pytest:
        metadata.update(_pytest_command_metadata(cmd))
        metadata["heartbeat_s"] = _pytest_heartbeat_interval()
        metadata["timeout_s"] = _pytest_timeout_s()
        metadata["stall_timeout_s"] = _pytest_stall_timeout_s()
        metadata["term_grace_s"] = _pytest_term_grace_s()
        metadata["resource_interval_s"] = _pytest_resource_interval_s()
        metadata["pytest_tmpfs"] = pytest_tmpfs
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
        if artifacts is not None and artifacts.resources_path.exists():
            sample_count = 0
            peak_rss = 0
            peak_pss: int | None = None
            peak_process_count = 0
            with artifacts.resources_path.open(encoding="utf-8") as resource_handle:
                for line in resource_handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample_count += 1
                    peak_rss = max(peak_rss, int(row.get("tree_rss_kb") or 0))
                    if row.get("tree_pss_kb") is not None:
                        peak_pss = max(peak_pss or 0, int(row["tree_pss_kb"]))
                    peak_process_count = max(peak_process_count, int(row.get("process_count") or 0))
            if sample_count:
                resource_summary = {
                    "resource_sample_count": sample_count,
                    "peak_tree_rss_kb": peak_rss,
                    "peak_tree_rss_mb": round(peak_rss / 1024, 1),
                    "peak_tree_pss_kb": peak_pss,
                    "peak_tree_pss_mb": round(peak_pss / 1024, 1) if peak_pss is not None else None,
                    "peak_process_count": peak_process_count,
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
        metadata["diagnosis"] = diagnosis
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
                **resource_summary,
            }
            artifacts.postmortem_path.write_text(json.dumps(postmortem, indent=2, ensure_ascii=False) + "\n")
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
    if result.returncode == 0:
        sys.stderr.write(f"ok ({elapsed:.1f}s)\n")
    else:
        sys.stderr.write(f"FAILED ({elapsed:.1f}s)\n")
        if result.stdout.strip():
            sys.stderr.write(result.stdout + "\n")
        if result.stderr.strip():
            sys.stderr.write(result.stderr + "\n")
    if run is not None and artifacts is not None:
        run.finish_step(
            step_id=artifacts.step_id, result={"duration_s": round(elapsed, 2), "exit": result.returncode, **metadata}
        )
    return result.returncode, elapsed, metadata


def _subprocess_env() -> dict[str, str]:
    env = normalize_pytest_basetemp_env(os.environ)
    env["POLYLOGUE_ROOT"] = str(ROOT)
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pycache")
    TESTMON_DATA.parent.mkdir(parents=True, exist_ok=True)
    env["TESTMON_DATAFILE"] = str(TESTMON_DATA)
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(Path.cwd() / PYTEST_EVENTS_PATH)
    env["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(Path.cwd() / PYTEST_SELECTION_PATH)
    env["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(Path.cwd() / PYTEST_SUMMARY_PATH)
    return env


def _stop_after_failed_step(label: str) -> bool:
    return label.startswith("pytest") or label in {"lab smoke", "bench slo"}


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
                ("verify topology", _devtools_cmd("verify topology")),
                ("verify layering", _devtools_cmd("verify layering")),
                ("verify closure-matrix", _devtools_cmd("verify closure-matrix")),
                ("lab schema roundtrip", _devtools_cmd("lab schema roundtrip", "--all")),
                ("verify manifests", _devtools_cmd("verify manifests")),
                ("verify ci-workflows", _devtools_cmd("verify ci-workflows")),
                ("verify doc-commands", _devtools_cmd("verify doc-commands")),
                ("verify docs-coverage", _devtools_cmd("verify docs-coverage")),
                ("verify test-infra-currency", _devtools_cmd("verify test-infra-currency")),
                ("verify test-clock-hygiene", _devtools_cmd("verify test-clock-hygiene")),
                ("verify pytest-timeout-overrides", _devtools_cmd("verify pytest-timeout-overrides")),
                ("verify degrade-loudly", _devtools_cmd("verify degrade-loudly")),
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
            "--durations=10",
            f"--junitxml={_report_dir}/verify-latest.xml",
            "--json-report",
            "--json-report-omit=collectors,log,streams,warnings",
            f"--json-report-file={PYTEST_REPORT_PATH}",
            "-p",
            "devtools.pytest_progress_plugin",
        ]
        base_marker = f"not slow and {scale_marker_expr}" if skip_slow else scale_marker_expr
        if seed_testmon:
            pytest_cmd.extend(["-m", base_marker, "--testmon"])
            if resume_testmon_seed:
                pytest_cmd.append("--testmon-forceselect")
                label = "pytest seed-testmon (resume)"
            else:
                pytest_cmd.append("--testmon-noselect")
                label = "pytest seed-testmon"
            pytest_cmd.extend(_pytest_worker_args(default="4"))
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
                *_pytest_worker_args(default="4"),
            ]
            steps.append(("pytest full (parallel)", bulk_cmd))

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
            steps.append(("pytest load-sensitive (isolated)", isolated_cmd))
        else:
            default_workers = DEFAULT_TESTMON_WORKERS
            pytest_cmd.extend(["-m", base_marker, "--testmon", *_pytest_worker_args(default=default_workers)])
            pytest_cmd.append("--testmon-forceselect")
            label = "pytest testmon (broad)" if broad_testmon else "pytest testmon"
            steps.append((label, pytest_cmd))

    if lab:
        steps.append(("lab smoke", _devtools_cmd("lab smoke", "run", "archive-smoke", "--tier", "0")))
        steps.append(("bench slo", _devtools_cmd("bench slo", "--include-lab")))
        steps.append(("lab policy schema-versioning", _devtools_cmd("lab policy schema-versioning")))
        steps.append(("lab policy timestamp-doctrine", _devtools_cmd("lab policy timestamp-doctrine")))
        steps.append(("lab policy insight-honesty", _devtools_cmd("lab policy insight-honesty")))
        steps.append(("lab policy demo-packet-registry", _devtools_cmd("lab policy demo-packet-registry")))
        steps.append(("lab policy demo-tour-freshness", _devtools_cmd("lab policy demo-tour-freshness")))
        steps.append(("lab policy docs-drift", _devtools_cmd("lab policy docs-drift")))
        steps.append(("lab policy backlog-hygiene", _devtools_cmd("lab policy backlog-hygiene")))
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


_BROAD_TESTMON_CHANGED_PATHS = {
    "pyproject.toml",
    "tests/conftest.py",
}


def _changed_paths() -> set[str]:
    changed: set[str] = set()
    commands = (
        ["git", "diff", "--name-only", "HEAD", "--"],
        ["git", "diff", "--name-only", "origin/master...HEAD", "--"],
    )
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            changed.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return changed


def _default_testmon_is_broad_change() -> bool:
    """Return true when affected-test selection should be treated as broad."""
    return bool(_changed_paths() & _BROAD_TESTMON_CHANGED_PATHS)


def _changed_executable_paths() -> tuple[str, ...]:
    """Return changed paths whose behavior should select at least one test."""
    roots = ("polylogue/", "devtools/", "tests/", "packaging/")
    exact = {"pyproject.toml", "uv.lock"}
    return tuple(sorted(path for path in _changed_paths() if path in exact or path.startswith(roots)))


def _testmon_preflight(*, seed_testmon: bool, full_pytest: bool, quick: bool, commit: bool) -> str | None:
    if quick or commit or seed_testmon or full_pytest:
        return None
    seed_message = (
        "verify: pytest-testmon is not seeded; run `devtools verify --seed-testmon` "
        "to create .cache/testmon/testmondata and .cache/testmon/seed.json "
        "before using the default affected-test path.\n"
    )
    if not TESTMON_DATA.exists() or not TESTMON_SEED_STAMP.exists():
        return seed_message
    try:
        stamp = json.loads(TESTMON_SEED_STAMP.read_text())
    except (OSError, json.JSONDecodeError):
        return (
            "verify: pytest-testmon seed stamp is unreadable; run `devtools verify --seed-testmon` "
            "to refresh .cache/testmon/testmondata and .cache/testmon/seed.json.\n"
        )
    if not isinstance(stamp, dict):
        return (
            "verify: pytest-testmon seed stamp has an invalid shape; run `devtools verify --seed-testmon` "
            "to refresh .cache/testmon/testmondata and .cache/testmon/seed.json.\n"
        )
    if stamp.get("protocol_version") != TESTMON_SEED_PROTOCOL_VERSION or stamp.get("status") != "complete":
        return (
            "verify: pytest-testmon has no validated complete seed receipt; run "
            "`devtools verify --seed-testmon` to resume or rebuild the dependency baseline.\n"
        )
    current_head = _git_head()
    stamped_head = stamp.get("git_head")
    if current_head is not None and stamped_head != current_head:
        sys.stderr.write(
            "verify: pytest-testmon seed was recorded for a different git head; "
            "continuing with the existing dependency database and recording affected-test evidence.\n"
        )
    if stamp.get("testmon_data") != _file_fingerprint(TESTMON_DATA):
        sys.stderr.write(
            "verify: pytest-testmon database changed after the seed stamp; "
            "continuing because testmon updates its dependency database during normal affected runs.\n"
        )
    return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worktree_fingerprint() -> str:
    """Fingerprint tracked content plus the untracked-path inventory."""
    digest = hashlib.sha256()
    for command in (
        ["git", "status", "--porcelain=v1", "-z"],
        ["git", "diff", "--binary", "HEAD", "--"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"
        if result.returncode != 0:
            return "unavailable"
        digest.update(result.stdout)
        digest.update(b"\0")
    return digest.hexdigest()


def _testmon_seed_identity(*, git_head: str | None, skip_slow: bool, lab: bool) -> dict[str, Any]:
    return {
        "git_head": git_head,
        "worktree_fingerprint": _worktree_fingerprint(),
        "python": sys.version,
        "skip_slow": skip_slow,
        "lab": lab,
    }


def _read_testmon_seed_attempt() -> dict[str, Any] | None:
    payload = _read_json_artifact(TESTMON_SEED_ATTEMPT)
    return payload if isinstance(payload, dict) else None


def _testmon_seed_can_resume(identity: Mapping[str, Any]) -> bool:
    attempt = _read_testmon_seed_attempt()
    if attempt is None or not TESTMON_DATA.exists():
        return False
    expected = attempt.get("expected_nodeids")
    return (
        attempt.get("protocol_version") == TESTMON_SEED_PROTOCOL_VERSION
        and attempt.get("status") in {"running", "incomplete"}
        and attempt.get("identity") == dict(identity)
        and isinstance(expected, list)
        and bool(expected)
        and all(isinstance(nodeid, str) for nodeid in expected)
    )


def _prepare_testmon_seed_attempt(
    *,
    identity: Mapping[str, Any],
    run: VerifyRun,
    resume: bool,
) -> dict[str, Any]:
    prior = _read_testmon_seed_attempt() if resume else None
    expected = prior.get("expected_nodeids", []) if prior is not None else []
    payload = {
        "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
        "status": "running",
        "identity": dict(identity),
        "resume": resume,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "artifact_dir": str(run.relative_run_dir),
        "testmon_data_before": _file_fingerprint(TESTMON_DATA),
    }
    TESTMON_SEED_STAMP.unlink(missing_ok=True)
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    return payload


def _testmon_database_state(expected_nodeids: Sequence[str]) -> dict[str, Any]:
    if not TESTMON_DATA.exists():
        return {
            "recorded_count": 0,
            "failed_count": 0,
            "missing_nodeids": list(expected_nodeids),
            "failed_nodeids": [],
            "error": "missing",
        }
    try:
        with sqlite3.connect(TESTMON_DATA) as conn:
            rows = conn.execute(
                """
                SELECT current.test_name, current.failed
                FROM test_execution AS current
                JOIN (
                    SELECT test_name, MAX(id) AS latest_id
                    FROM test_execution
                    GROUP BY test_name
                ) AS latest ON latest.latest_id = current.id
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {
            "recorded_count": 0,
            "failed_count": 0,
            "missing_nodeids": list(expected_nodeids),
            "failed_nodeids": [],
            "error": str(exc),
        }
    recorded = {str(name): bool(failed) for name, failed in rows}
    expected = set(expected_nodeids)
    failed = sorted(nodeid for nodeid in expected if recorded.get(nodeid) is True)
    return {
        "recorded_count": len(recorded),
        "failed_count": sum(recorded.values()),
        "missing_nodeids": sorted(expected - recorded.keys()),
        "failed_nodeids": failed,
        "error": None,
    }


def _failed_nodeids_from_events(path: Path) -> list[str]:
    failed: set[str] = set()
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    event = json.loads(line)
                    if event.get("event") == "test_report" and event.get("outcome") == "failed":
                        nodeid = event.get("nodeid")
                        if isinstance(nodeid, str):
                            failed.add(nodeid)
    except OSError:
        return []
    return sorted(failed)


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
    failed_events: list[str] = []
    if pytest_step is not None:
        artifact_dir_raw = pytest_step.get("artifact_dir")
        if isinstance(artifact_dir_raw, str):
            artifact_dir = Path(artifact_dir_raw)
            selection_payload = _read_json_artifact(artifact_dir / "selection.json")
            if isinstance(selection_payload, dict):
                selection = selection_payload
            failed_events = _failed_nodeids_from_events(artifact_dir / "events.jsonl")

    expected_raw = prepared.get("expected_nodeids") if prepared.get("resume") else selection.get("selected_nodeids")
    expected = [str(nodeid) for nodeid in expected_raw] if isinstance(expected_raw, list) else []
    omitted = int(selection.get("selected_nodeids_omitted") or 0)
    database = _testmon_database_state(expected)
    complete = (
        exit_code == 0
        and bool(expected)
        and (bool(prepared.get("resume")) or omitted == 0)
        and database["error"] is None
        and not database["missing_nodeids"]
        and not database["failed_nodeids"]
        and not failed_events
    )
    payload = {
        **dict(prepared),
        "status": "complete" if complete else "incomplete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest() if expected else None,
        "selection": {
            key: selection.get(key)
            for key in (
                "selected_count",
                "deselected_count",
                "selected_nodeids_omitted",
                "deselected_nodeids_omitted",
                "collection_duration_s",
            )
        },
        "database": database,
        "failed_event_nodeids": failed_events,
        "testmon_data": _file_fingerprint(TESTMON_DATA),
        "pytest_step": dict(pytest_step) if pytest_step is not None else None,
    }
    _atomic_write_json(TESTMON_SEED_ATTEMPT, payload)
    if complete:
        stamp = {
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "complete",
            "timestamp": payload["finished_at"],
            "git_head": dict(prepared["identity"]).get("git_head"),
            "identity": prepared["identity"],
            "expected_count": payload["expected_count"],
            "expected_digest": payload["expected_digest"],
            "testmon_data": payload["testmon_data"],
            "database": {
                "recorded_count": database["recorded_count"],
                "failed_count": database["failed_count"],
            },
            "run_id": payload["run_id"],
            "artifact_dir": payload["artifact_dir"],
        }
        _atomic_write_json(TESTMON_SEED_STAMP, stamp)
    return payload


# ── main ────────────────────────────────────────────────────────────


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

    head = _git_head()
    t0 = time.monotonic()
    verify_run = VerifyRun(tier=tier, argv=list(sys.argv[1:] if argv is None else argv), git_head=head)
    seed_identity: dict[str, Any] | None = None
    resume_testmon_seed = False
    prepared_seed_attempt: dict[str, Any] | None = None
    if args.seed_testmon:
        seed_identity = _testmon_seed_identity(
            git_head=head,
            skip_slow=bool(args.skip_slow),
            lab=bool(args.lab),
        )
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
    steps = build_verify_steps(
        quick=bool(args.quick),
        commit=bool(args.commit),
        lab=bool(args.lab),
        skip_slow=bool(args.skip_slow),
        seed_testmon=bool(args.seed_testmon),
        resume_testmon_seed=resume_testmon_seed,
        full_pytest=full_pytest,
        broad_testmon=_default_testmon_is_broad_change(),
    )

    step_results: list[dict[str, Any]] = []

    for label, cmd in steps:
        if label.startswith("pytest"):
            _warn_low_memory()  # check again right before the heavy step
        rc, elapsed, metadata = _run(label, cmd, run=verify_run)
        if rc == 0 and label in {"pytest testmon", "pytest testmon (broad)"} and metadata.get("selected_count") == 0:
            executable_paths = _changed_executable_paths()
            if executable_paths:
                rc = 5
                metadata["diagnosis"] = "zero_testmon_selection_for_executable_change"
                metadata["zero_selection_changed_paths"] = list(executable_paths)
                sys.stderr.write(
                    "verify: pytest-testmon selected zero tests for executable changes; "
                    "refresh the seed or repair dependency capture: " + ", ".join(executable_paths) + "\n"
                )
        step_result: dict[str, Any] = {"name": label, "duration_s": round(elapsed, 2), "exit": rc}
        step_result.update(metadata)
        step_results.append(step_result)
        if rc != 0:
            exit_code = rc
            if _stop_after_failed_step(label):
                break

    seed_receipt: dict[str, Any] | None = None
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

    total_duration = round(time.monotonic() - t0, 2)

    # Build history entry.
    history_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "tier": tier,
        "run_id": verify_run.run_id,
        "artifact_dir": str(verify_run.relative_run_dir),
        "steps": step_results,
        "total_duration_s": total_duration,
        "exit_code": exit_code,
    }
    pytest_diagnosis = next(
        (
            str(step["diagnosis"])
            for step in step_results
            if str(step.get("name", "")).startswith("pytest") and "diagnosis" in step
        ),
        None,
    )
    if pytest_diagnosis is not None:
        history_entry["diagnosis"] = pytest_diagnosis
    if seed_receipt is not None:
        history_entry["testmon_seed"] = {
            "status": seed_receipt["status"],
            "resume": seed_receipt["resume"],
            "expected_count": seed_receipt["expected_count"],
            "attempt_path": str(TESTMON_SEED_ATTEMPT),
            "stamp_path": str(TESTMON_SEED_STAMP) if seed_receipt["status"] == "complete" else None,
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
    verify_run.finish(exit_code=exit_code, duration_s=total_duration, diagnosis=pytest_diagnosis)
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
