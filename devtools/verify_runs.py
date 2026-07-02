"""Durable verification-run artifacts and resource sampling.

The top-level ``devtools verify`` history is intentionally compact. This
module owns the heavier per-run evidence: per-step stdout/stderr, pytest
selection and event streams, resource samples, and postmortem classification.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERIFY_CACHE = Path(".cache/verify")
VERIFY_RUNS_DIR = VERIFY_CACHE / "runs"
CURRENT_RUN_PATH = VERIFY_CACHE / "current-run.json"
CURRENT_RESOURCES_PATH = VERIFY_CACHE / "current-pytest-resources.jsonl"
CURRENT_POSTMORTEM_PATH = VERIFY_CACHE / "current-pytest-postmortem.json"
CURRENT_EVENTS_DIR = VERIFY_CACHE / "current-pytest-events"
DEFAULT_BASETEMP_SIZE_SAMPLE_INTERVAL_S = 15.0
BASETEMP_SIZE_SAMPLE_INTERVAL_ENV = "POLYLOGUE_VERIFY_BASETEMP_SIZE_INTERVAL_S"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(*, tier: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_tier = re.sub(r"[^A-Za-z0-9_.-]+", "-", tier).strip("-") or "verify"
    return f"{stamp}-{safe_tier}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _process_alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def _current_owner_is_other_live_run(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "running":
        return False
    owner_pid = payload.get("owner_pid")
    if not isinstance(owner_pid, int) or owner_pid == os.getpid():
        return False
    return _process_alive(owner_pid)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower() or "step"


def git_dirty() -> bool:
    try:
        result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(result.stdout.strip())


@dataclass(frozen=True)
class PytestStepArtifacts:
    step_id: str
    step_dir: Path
    stdout_path: Path
    stderr_path: Path
    output_path: Path
    progress_path: Path
    events_dir: Path
    events_merged_path: Path
    selection_path: Path
    summary_path: Path
    resources_path: Path
    postmortem_path: Path


class VerifyRun:
    """A filesystem-backed verification run ledger."""

    def __init__(self, *, tier: str, argv: list[str], git_head: str | None, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.run_id = make_run_id(tier=tier)
        self.run_dir = self.root / VERIFY_RUNS_DIR / self.run_id
        self._payload: dict[str, Any] = {
            "run_id": self.run_id,
            "tier": tier,
            "argv": list(argv),
            "git_head": git_head,
            "git_dirty": git_dirty(),
            "owner_pid": os.getpid(),
            "started_at": utc_now(),
            "status": "running",
            "steps": [],
            "artifact_dir": str(VERIFY_RUNS_DIR / self.run_id),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write()

    @property
    def relative_run_dir(self) -> Path:
        return VERIFY_RUNS_DIR / self.run_id

    def write(self) -> None:
        _write_json(self.run_dir / "run.json", self._payload)
        current_path = self.root / CURRENT_RUN_PATH
        if not _current_owner_is_other_live_run(current_path):
            _write_json(current_path, self._payload)

    def start_step(self, *, label: str, cmd: list[str]) -> PytestStepArtifacts:
        index = len(self._payload["steps"]) + 1
        step_id = f"{index:02d}-{_slug(label)}"
        step_dir = self.run_dir / "steps" / step_id
        artifacts = PytestStepArtifacts(
            step_id=step_id,
            step_dir=step_dir,
            stdout_path=step_dir / "stdout.log",
            stderr_path=step_dir / "stderr.log",
            output_path=step_dir / "output.log",
            progress_path=step_dir / "progress.json",
            events_dir=step_dir / "events",
            events_merged_path=step_dir / "events.jsonl",
            selection_path=step_dir / "selection.json",
            summary_path=step_dir / "summary.json",
            resources_path=step_dir / "resources.jsonl",
            postmortem_path=step_dir / "postmortem.json",
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self._payload["steps"].append(
            {
                "step_id": step_id,
                "name": label,
                "cmd": list(cmd),
                "status": "running",
                "started_at": utc_now(),
                "artifact_dir": str(self.relative_run_dir / "steps" / step_id),
            }
        )
        self.write()
        return artifacts

    def finish_step(self, *, step_id: str, result: dict[str, Any]) -> None:
        for step in self._payload["steps"]:
            if step.get("step_id") == step_id:
                step.update(result)
                step["finished_at"] = utc_now()
                step["status"] = "success" if result.get("exit") == 0 else "failed"
                break
        self.write()

    def finish(self, *, exit_code: int, duration_s: float, diagnosis: str | None = None) -> dict[str, Any]:
        self._payload["finished_at"] = utc_now()
        self._payload["duration_s"] = round(duration_s, 2)
        self._payload["exit_code"] = int(exit_code)
        self._payload["status"] = "success" if exit_code == 0 else "failed"
        if diagnosis:
            self._payload["diagnosis"] = diagnosis
        self.write()
        return dict(self._payload)


def env_for_pytest_step(env: dict[str, str], *, run: VerifyRun, artifacts: PytestStepArtifacts) -> dict[str, str]:
    updated = dict(env)
    updated["POLYLOGUE_VERIFY_RUN_ID"] = run.run_id
    updated["POLYLOGUE_PYTEST_RUN_ID"] = run.run_id
    updated["POLYLOGUE_PYTEST_EVENTS_DIR"] = str(artifacts.events_dir)
    updated["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(artifacts.events_merged_path)
    updated["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(artifacts.selection_path)
    updated["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(artifacts.summary_path)
    return updated


def copy_current_pytest_artifacts(root: Path, artifacts: PytestStepArtifacts, *, legacy_paths: dict[str, Path]) -> None:
    for key, target in legacy_paths.items():
        source = getattr(artifacts, key)
        target_abs = root / target
        with contextlib.suppress(FileNotFoundError):
            target_abs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target_abs)
    if artifacts.events_dir.exists():
        current_events_dir = root / CURRENT_EVENTS_DIR
        if current_events_dir.exists():
            shutil.rmtree(current_events_dir)
        shutil.copytree(artifacts.events_dir, current_events_dir)
    with contextlib.suppress(FileNotFoundError):
        shutil.copyfile(artifacts.resources_path, root / CURRENT_RESOURCES_PATH)
    with contextlib.suppress(FileNotFoundError):
        shutil.copyfile(artifacts.postmortem_path, root / CURRENT_POSTMORTEM_PATH)


def merge_worker_events(events_dir: Path, merged_path: Path) -> int:
    if not events_dir.exists():
        return 0
    rows: list[dict[str, Any]] = []
    for path in sorted(events_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    rows.sort(key=lambda row: str(row.get("updated_at", "")))
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def latest_event_from_paths(*paths: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            for event_file in path.glob("*.jsonl"):
                candidates.extend(_read_last_jsonl(event_file, limit=1))
        else:
            candidates.extend(_read_last_jsonl(path, limit=1))
    if not candidates:
        return None
    candidates.sort(key=lambda row: str(row.get("updated_at", "")))
    return candidates[-1]


def _read_last_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        with contextlib.suppress(json.JSONDecodeError):
            rows.append(json.loads(line))
    return rows


def _proc_children() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            raw = stat_path.read_text()
            pid = int(stat_path.parent.name)
            ppid = int(raw.rsplit(") ", 1)[1].split()[1])
            children.setdefault(ppid, []).append(pid)
    return children


def process_tree(root_pid: int) -> list[int]:
    children = _proc_children()
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return sorted(seen)


def _status_values(pid: int) -> dict[str, int | str | None]:
    result: dict[str, int | str | None] = {"state": None, "rss_kb": 0}
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        if line.startswith("State:"):
            result["state"] = line.split(":", 1)[1].strip()
        elif line.startswith("VmRSS:"):
            with contextlib.suppress(ValueError, IndexError):
                result["rss_kb"] = int(line.split()[1])
    return result


def _pss_kb(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("Pss:"):
            with contextlib.suppress(ValueError, IndexError):
                return int(line.split()[1])
    return None


def _cpu_seconds(pid: int) -> float | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw.rsplit(") ", 1)[1].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (float(fields[11]) + float(fields[12])) / float(ticks)
    except (OSError, ValueError, IndexError):
        return None


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.split()[0])
    return values


def _pressure(kind: str) -> dict[str, float]:
    result: dict[str, float] = {}
    with contextlib.suppress(OSError):
        for line in Path(f"/proc/pressure/{kind}").read_text().splitlines():
            parts = line.split()
            prefix = parts[0]
            for part in parts[1:]:
                key, value = part.split("=", 1)
                if key == "avg10":
                    result[f"{prefix}_avg10"] = float(value)
    return result


def _fs_usage(path: Path) -> dict[str, int] | None:
    with contextlib.suppress(OSError):
        stat = os.statvfs(path)
        return {
            "used_kb": int(((stat.f_blocks - stat.f_bfree) * stat.f_frsize) / 1024),
            "free_kb": int((stat.f_bavail * stat.f_frsize) / 1024),
        }
    return None


def _dir_size_kb(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    try:
        for item in path.rglob("*"):
            with contextlib.suppress(OSError):
                if item.is_file():
                    total += item.stat().st_size
    except OSError:
        return None
    return int(total / 1024)


def checkout_hash(root: Path) -> str:
    return hashlib.sha1(str(root).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def pytest_basetemp_path(*, root: Path, run_id: str, env: dict[str, str]) -> Path:
    configured = env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if configured:
        scratch_root = Path(configured)
    elif env.get("POLYLOGUE_PYTEST_TMPFS") == "1" and Path("/dev/shm").is_dir():
        scratch_root = Path("/dev/shm")
    else:
        scratch_root = Path("/realm/tmp/polylogue-pytest")
    return scratch_root / f"pytest-polylogue-{checkout_hash(root)}-{run_id}"


def cleanup_managed_pytest_basetemp(*, root: Path, run_id: str, env: dict[str, str]) -> Path | None:
    """Remove the managed per-run pytest basetemp after pytest has exited.

    The pytest sessionfinish hook intentionally does not delete xdist
    basetemps while workers/reporters may still be flushing.  The supervisor is
    outside that teardown window, so it can reclaim tmpfs-backed broad-run
    basetemps immediately instead of waiting for the next pytest startup sweep.
    """

    basetemp = pytest_basetemp_path(root=root, run_id=run_id, env=env)
    if not basetemp.name.startswith("pytest-polylogue-") or "-seeded-" in basetemp.name:
        return None
    with contextlib.suppress(OSError):
        if basetemp.exists():
            shutil.rmtree(basetemp, ignore_errors=True)
            return basetemp
    return None


class ResourceSampler:
    """Samples host and process-tree resources for one subprocess tree."""

    def __init__(self, *, root_pid: int, run_id: str, root: Path, env: dict[str, str], output_path: Path) -> None:
        self.root_pid = root_pid
        self.run_id = run_id
        self.root = root
        self.env = env
        self.output_path = output_path
        self.sample_count = 0
        self.peak_rss_kb = 0
        self.peak_pss_kb: int | None = None
        self.peak_process_count = 0
        self.last_sample: dict[str, Any] | None = None
        self._basetemp = pytest_basetemp_path(root=root, run_id=run_id, env=env)
        self._basetemp_size_interval_s = _basetemp_size_sample_interval_s(env)
        self._last_basetemp_size_sample_at: float | None = None
        self._last_basetemp_size_kb: int | None = None

    def _sample_basetemp_size_kb(self, *, event: str) -> int | None:
        """Return basetemp size without recursively walking it every sample."""
        if self._basetemp_size_interval_s <= 0:
            return None
        now = time.monotonic()
        should_sample = (
            self._last_basetemp_size_sample_at is None
            or self._last_basetemp_size_kb is None
            or event in {"started", "finished"}
            or now - self._last_basetemp_size_sample_at >= self._basetemp_size_interval_s
        )
        if should_sample:
            self._last_basetemp_size_kb = _dir_size_kb(self._basetemp)
            self._last_basetemp_size_sample_at = now
        return self._last_basetemp_size_kb

    def sample(self, *, event: str) -> dict[str, Any]:
        pids = process_tree(self.root_pid)
        processes: list[dict[str, Any]] = []
        total_rss = 0
        total_pss = 0
        pss_available = False
        total_cpu = 0.0
        for pid in pids:
            status = _status_values(pid)
            rss = int(status.get("rss_kb") or 0)
            pss = _pss_kb(pid)
            cpu = _cpu_seconds(pid)
            total_rss += rss
            if pss is not None:
                pss_available = True
                total_pss += pss
            if cpu is not None:
                total_cpu += cpu
            processes.append(
                {
                    "pid": pid,
                    "state": status.get("state"),
                    "rss_kb": rss,
                    "pss_kb": pss,
                    "cpu_s": cpu,
                }
            )
        meminfo = _meminfo()
        sample: dict[str, Any] = {
            "updated_at": utc_now(),
            "event": event,
            "root_pid": self.root_pid,
            "process_count": len(pids),
            "tree_rss_kb": total_rss,
            "tree_pss_kb": total_pss if pss_available else None,
            "tree_cpu_s": round(total_cpu, 4),
            "host_mem_available_kb": meminfo.get("MemAvailable"),
            "host_mem_total_kb": meminfo.get("MemTotal"),
            "host_swap_free_kb": meminfo.get("SwapFree"),
            "host_swap_total_kb": meminfo.get("SwapTotal"),
            "pressure_cpu": _pressure("cpu"),
            "pressure_io": _pressure("io"),
            "pressure_memory": _pressure("memory"),
            "shm": _fs_usage(Path("/dev/shm")),
            "basetemp": str(self._basetemp),
            "basetemp_size_kb": self._sample_basetemp_size_kb(event=event),
            "top_processes": sorted(processes, key=lambda row: int(row.get("rss_kb") or 0), reverse=True)[:8],
        }
        self.sample_count += 1
        self.peak_rss_kb = max(self.peak_rss_kb, total_rss)
        if pss_available:
            self.peak_pss_kb = max(self.peak_pss_kb or 0, total_pss)
        self.peak_process_count = max(self.peak_process_count, len(pids))
        self.last_sample = sample
        _append_jsonl(self.output_path, sample)
        return sample

    def summary(self) -> dict[str, Any]:
        return {
            "resource_sample_count": self.sample_count,
            "peak_tree_rss_kb": self.peak_rss_kb,
            "peak_tree_rss_mb": round(self.peak_rss_kb / 1024, 1),
            "peak_tree_pss_kb": self.peak_pss_kb,
            "peak_tree_pss_mb": round(self.peak_pss_kb / 1024, 1) if self.peak_pss_kb is not None else None,
            "peak_process_count": self.peak_process_count,
            "last_resource_sample": self.last_sample,
        }


def _basetemp_size_sample_interval_s(env: dict[str, str]) -> float:
    raw = env.get(BASETEMP_SIZE_SAMPLE_INTERVAL_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BASETEMP_SIZE_SAMPLE_INTERVAL_S
    with contextlib.suppress(ValueError):
        return max(0.0, float(raw))
    return DEFAULT_BASETEMP_SIZE_SAMPLE_INTERVAL_S


def classify_pytest_result(
    *,
    returncode: int,
    termination_reason: str | None,
    report_present: bool,
    summary: dict[str, Any] | None,
    progress_event: str | None,
) -> str:
    if termination_reason:
        if "runtime exceeded" in termination_reason:
            return "pytest_timeout"
        if "produced no output" in termination_reason:
            return "pytest_stall_timeout"
        return "pytest_terminated"
    exitstatus = summary.get("exitstatus") if summary else None
    if returncode == 0:
        return "pytest_passed" if report_present else "pytest_passed_report_missing"
    if returncode < 0:
        if exitstatus == 0:
            return (
                "report_missing_after_sessionfinish_success" if not report_present else "external_sigterm_after_success"
            )
        return "external_signal"
    if progress_event == "terminated":
        return "pytest_terminated"
    return "pytest_failed"
