"""Pre-push and pre-PR verification baseline.

Runs the checks that CI will enforce, locally and fast. Exit 0 means
the branch is ready to push; non-zero means fix before pushing.

Tiers:
  --commit   Pre-commit tier: ruff format + check + mypy (~3s warm).
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devtools.verify_runs import (
    CURRENT_EVENTS_DIR,
    CURRENT_POSTMORTEM_PATH,
    CURRENT_RESOURCES_PATH,
    PytestStepArtifacts,
    ResourceSampler,
    VerifyRun,
    classify_pytest_result,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    latest_event_from_paths,
    merge_worker_events,
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


def _shm_free_gb() -> float | None:
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return None
    return (stat.f_bavail * stat.f_frsize) / 1024 / 1024 / 1024


def _enable_tmpfs_for_broad_pytest(label: str, env: dict[str, str]) -> bool:
    if not (label == "pytest seed-testmon" or label == "pytest testmon (broad)" or label.startswith("pytest full")):
        return False
    if env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT") or env.get("POLYLOGUE_PYTEST_TMPFS"):
        return False
    mem = _check_available_memory()
    shm_free = _shm_free_gb()
    if mem is None or shm_free is None:
        return False
    available_gb = mem[0] / 1024 / 1024
    if available_gb < 10.0 or shm_free < 8.0:
        return False
    env["POLYLOGUE_PYTEST_TMPFS"] = "1"
    return True


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
PYTEST_JUNIT_REPORT_DIR = Path(".cache/test-reports")
PYTEST_JUNIT_REPORT_PATH = PYTEST_JUNIT_REPORT_DIR / "verify-latest.xml"
PYTEST_PROGRESS_PATH = PYTEST_REPORT_DIR / "current-pytest-progress.json"
PYTEST_EVENTS_PATH = PYTEST_REPORT_DIR / "current-pytest-events.jsonl"
PYTEST_EVENTS_DIR = CURRENT_EVENTS_DIR
PYTEST_SELECTION_PATH = PYTEST_REPORT_DIR / "current-pytest-selection.json"
PYTEST_SUMMARY_PATH = PYTEST_REPORT_DIR / "current-pytest-summary.json"
PYTEST_OUTPUT_PATH = PYTEST_REPORT_DIR / "current-pytest-output.log"
PYTEST_HEARTBEAT_ENV = "POLYLOGUE_VERIFY_HEARTBEAT_S"
PYTEST_TIMEOUT_ENV = "POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S"
PYTEST_STALL_TIMEOUT_ENV = "POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S"
PYTEST_RESOURCE_INTERVAL_ENV = "POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S"
DEFAULT_PYTEST_HEARTBEAT_S = 30.0
DEFAULT_PYTEST_TIMEOUT_S = 45 * 60.0
DEFAULT_PYTEST_STALL_TIMEOUT_S = 10 * 60.0
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


def _pytest_resource_interval_s() -> float:
    return _float_env(PYTEST_RESOURCE_INTERVAL_ENV, DEFAULT_PYTEST_RESOURCE_INTERVAL_S)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


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
    resource_interval_s = _pytest_resource_interval_s()
    sys.stderr.write(f"\n    command: {shlex.join(cmd)}\n")
    sys.stderr.flush()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
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
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    output_bytes = {"stdout": 0, "stderr": 0}
    last_cpu = _process_cpu_seconds(process.pid)
    last_sample = time.monotonic()
    last_output = last_sample
    last_resource_sample = last_sample
    termination_reason: str | None = None
    while True:
        now = time.monotonic()
        elapsed = now - t0
        idle = now - last_output
        if timeout_s > 0 and elapsed >= timeout_s:
            termination_reason = f"pytest runtime exceeded {timeout_s:g}s"
        elif stall_timeout_s > 0 and idle >= stall_timeout_s:
            termination_reason = f"pytest produced no output for {stall_timeout_s:g}s"
        if termination_reason is not None:
            _terminate_process_group(process)
            break

        deadlines: list[float] = []
        if heartbeat_s > 0:
            deadlines.append(heartbeat_s)
        if timeout_s > 0:
            deadlines.append(max(timeout_s - elapsed, 0.0))
        if stall_timeout_s > 0:
            deadlines.append(max(stall_timeout_s - idle, 0.0))
        timeout = min(deadlines) if deadlines else None
        events = selector.select(timeout=timeout)
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
                    _write_pytest_progress(
                        event="output",
                        cmd=cmd,
                        started_at=t0,
                        pid=process.pid,
                        idle_s=0.0,
                        output_bytes=output_bytes,
                        status=_process_status(process.pid),
                        run_id=run.run_id if run is not None else None,
                        artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
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
            if latest_event is not None:
                event = latest_event.get("event")
                nodeid = latest_event.get("nodeid")
                node_text = f", latest={event}:{nodeid}" if isinstance(event, str) and isinstance(nodeid, str) else ""
            else:
                node_text = ""
            sys.stderr.write(
                f"    still running: pid={process.pid}, elapsed={sample_now - t0:.0f}s, "
                f"idle={sample_now - last_output:.0f}s{state_text}{cpu_text}{rss_text}{node_text}\n"
            )
            sys.stderr.flush()
            _write_pytest_progress(
                event="heartbeat",
                cmd=cmd,
                started_at=t0,
                pid=process.pid,
                elapsed_s=sample_now - t0,
                idle_s=sample_now - last_output,
                output_bytes=output_bytes,
                status=status,
                cpu_pct=cpu_pct,
                run_id=run.run_id if run is not None else None,
                artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
            )
        sample_now = time.monotonic()
        if sampler is not None and resource_interval_s > 0 and sample_now - last_resource_sample >= resource_interval_s:
            sampler.sample(event="sample")
            last_resource_sample = sample_now
        if process.poll() is not None and not selector.get_map():
            break

    for stream in (process.stdout, process.stderr):
        with contextlib.suppress(OSError):
            remaining = stream.read()
        if remaining:
            stream_name = "stdout" if stream is process.stdout else "stderr"
            output[stream_name].append(remaining)
            output_bytes[stream_name] += len(remaining)
    stdout = b"".join(output["stdout"]).decode(errors="replace")
    stderr = b"".join(output["stderr"]).decode(errors="replace")
    _write_pytest_output(stdout, stderr)
    if artifacts is not None:
        artifacts.stdout_path.write_text(stdout, encoding="utf-8")
        artifacts.stderr_path.write_text(stderr, encoding="utf-8")
        artifacts.output_path.write_text(stdout + stderr, encoding="utf-8")
    resource_summary: dict[str, Any] = {}
    if sampler is not None:
        sampler.sample(event="finished" if termination_reason is None else "terminated")
        resource_summary = sampler.summary()
    if termination_reason is not None:
        _write_pytest_progress(
            event="terminated",
            cmd=cmd,
            started_at=t0,
            pid=process.pid,
            returncode=124,
            output_bytes=output_bytes,
            termination_reason=termination_reason,
            run_id=run.run_id if run is not None else None,
            artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
            resources=resource_summary,
        )
        stderr = f"{stderr}\nverify: {termination_reason}; terminated pytest process group {process.pid}\n"
        return subprocess.CompletedProcess(cmd, 124, stdout, stderr)
    _write_pytest_progress(
        event="finished",
        cmd=cmd,
        started_at=t0,
        pid=process.pid,
        returncode=process.returncode or 0,
        output_bytes=output_bytes,
        run_id=run.run_id if run is not None else None,
        artifact_dir=str(artifacts.step_dir) if artifacts is not None else None,
        resources=resource_summary,
    )
    return subprocess.CompletedProcess(cmd, process.returncode or 0, stdout, stderr)


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
    if is_pytest:
        pytest_tmpfs = _enable_tmpfs_for_broad_pytest(label, env)
        if run is not None and artifacts is not None:
            env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        result = _run_pytest_with_heartbeat(cmd, cwd=cwd, env=env, t0=t0, run=run, artifacts=artifacts)
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
        resource_summary: dict[str, Any] = {}
        if artifacts is not None and artifacts.resources_path.exists():
            resource_rows = [
                json.loads(line)
                for line in artifacts.resources_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if resource_rows:
                peak_rss = max(int(row.get("tree_rss_kb") or 0) for row in resource_rows)
                peak_pss_values = [
                    int(row["tree_pss_kb"]) for row in resource_rows if row.get("tree_pss_kb") is not None
                ]
                resource_summary = {
                    "resource_sample_count": len(resource_rows),
                    "peak_tree_rss_kb": peak_rss,
                    "peak_tree_rss_mb": round(peak_rss / 1024, 1),
                    "peak_tree_pss_kb": max(peak_pss_values) if peak_pss_values else None,
                    "peak_tree_pss_mb": round(max(peak_pss_values) / 1024, 1) if peak_pss_values else None,
                    "peak_process_count": max(int(row.get("process_count") or 0) for row in resource_rows),
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
    env = os.environ.copy()
    env["POLYLOGUE_ROOT"] = str(ROOT)
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pycache")
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(Path.cwd() / PYTEST_EVENTS_PATH)
    env["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(Path.cwd() / PYTEST_SELECTION_PATH)
    env["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(Path.cwd() / PYTEST_SUMMARY_PATH)
    return env


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
                ("verify-topology", _devtools_cmd("verify-topology")),
                ("verify-layering", _devtools_cmd("verify-layering")),
                ("verify-closure-matrix", _devtools_cmd("verify-closure-matrix")),
                ("verify-schema-roundtrip", _devtools_cmd("verify-schema-roundtrip", "--all")),
                ("verify-manifests", _devtools_cmd("verify-manifests")),
                ("verify-ci-workflows", _devtools_cmd("verify-ci-workflows")),
                ("verify-doc-commands", _devtools_cmd("verify-doc-commands")),
                ("verify-lane-assertions", _devtools_cmd("verify-lane-assertions")),
                ("verify-test-infra-currency", _devtools_cmd("verify-test-infra-currency")),
                ("verify-test-clock-hygiene", _devtools_cmd("verify-test-clock-hygiene")),
            ]
        )

    if not quick and not commit:
        _report_dir = PYTEST_JUNIT_REPORT_DIR
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
            "-p",
            "devtools.pytest_progress_plugin",
        ]
        base_marker = f"not slow and {scale_marker_expr}" if skip_slow else scale_marker_expr
        if seed_testmon:
            pytest_cmd.extend(["-m", base_marker, "--testmon", "--testmon-noselect", *_pytest_worker_args(default="4")])
            steps.append(("pytest seed-testmon", pytest_cmd))
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
            default_workers = "4" if broad_testmon else "0"
            pytest_cmd.extend(["-m", base_marker, "--testmon", *_pytest_worker_args(default=default_workers)])
            pytest_cmd.append("--testmon-forceselect")
            label = "pytest testmon (broad)" if broad_testmon else "pytest testmon"
            steps.append((label, pytest_cmd))

    if lab:
        steps.append(("lab scenario", _devtools_cmd("lab-scenario", "run", "archive-smoke", "--tier", "0")))
        steps.append(("verify-slos", _devtools_cmd("verify-slos", "--include-lab")))
        steps.append(("verify-schema-upgrade-lane", _devtools_cmd("verify-schema-upgrade-lane")))
        steps.append(("verify-test-coverage-contracts", _devtools_cmd("verify-test-coverage-contracts")))
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


def _default_testmon_is_broad_change() -> bool:
    """Return true when affected-test selection should be treated as broad."""
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
    return bool(changed & _BROAD_TESTMON_CHANGED_PATHS)


def _testmon_preflight(*, seed_testmon: bool, full_pytest: bool, quick: bool, commit: bool) -> str | None:
    if quick or commit or seed_testmon or full_pytest:
        return None
    seed_message = (
        "verify: pytest-testmon is not seeded; run `devtools verify --seed-testmon` "
        "to create .testmondata and .cache/testmon/seed.json before using the default affected-test path.\n"
    )
    if not TESTMON_DATA.exists() or not TESTMON_SEED_STAMP.exists():
        return seed_message
    try:
        stamp = json.loads(TESTMON_SEED_STAMP.read_text())
    except (OSError, json.JSONDecodeError):
        return (
            "verify: pytest-testmon seed stamp is unreadable; run `devtools verify --seed-testmon` "
            "to refresh .testmondata and .cache/testmon/seed.json.\n"
        )
    if not isinstance(stamp, dict):
        return (
            "verify: pytest-testmon seed stamp has an invalid shape; run `devtools verify --seed-testmon` "
            "to refresh .testmondata and .cache/testmon/seed.json.\n"
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
        broad_testmon=_default_testmon_is_broad_change(),
    )

    step_results: list[dict[str, Any]] = []

    for label, cmd in steps:
        if label.startswith("pytest"):
            _warn_low_memory()  # check again right before the heavy step
        rc, elapsed, metadata = _run(label, cmd, run=verify_run)
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
        if args.seed_testmon:
            _write_testmon_seed_stamp(history_entry)

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
