"""Durable verification-run artifacts and resource sampling.

The top-level ``devtools verify`` history is intentionally compact. This
module owns the heavier per-run evidence: per-step stdout/stderr, pytest
selection and event streams, resource samples, and postmortem classification.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import watchfiles

from polylogue.core.metrics import read_cgroup_memory_headroom_bytes

VERIFY_CACHE = Path(".cache/verify")
VERIFY_RUNS_DIR = VERIFY_CACHE / "runs"
_XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME", "").strip()
DEVTOOLS_STATE_DIR = (
    (Path(_XDG_STATE_HOME) if _XDG_STATE_HOME else Path.home() / ".local" / "state") / "polylogue" / "devtools"
)
VERIFY_HISTORY_PATH = DEVTOOLS_STATE_DIR / "verify-history.jsonl"
CURRENT_RUN_PATH = VERIFY_CACHE / "current-run.json"
VERIFICATION_INVOCATION_ID_ENV = "POLYLOGUE_VERIFICATION_INVOCATION_ID"
VERIFICATION_RECEIPT_PATH_ENV = "POLYLOGUE_VERIFICATION_RECEIPT_PATH"
CURRENT_RESOURCES_PATH = VERIFY_CACHE / "current-pytest-resources.jsonl"
CURRENT_POSTMORTEM_PATH = VERIFY_CACHE / "current-pytest-postmortem.json"
CURRENT_CONTAINMENT_PATH = VERIFY_CACHE / "current-pytest-containment.json"
CURRENT_STATISTICS_PATH = VERIFY_CACHE / "current-pytest-statistics.json"
PYTEST_CANONICAL_REPORT_NAME = "pytest-report.json"
CURRENT_EVENTS_DIR = VERIFY_CACHE / "current-pytest-events"
DEFAULT_BASETEMP_SIZE_SAMPLE_INTERVAL_S = 15.0
DEFAULT_TMPFS_SIZE_SAMPLE_INTERVAL_S = 2.0
BASETEMP_SIZE_SAMPLE_INTERVAL_ENV = "POLYLOGUE_VERIFY_BASETEMP_SIZE_INTERVAL_S"
PYTEST_TMPFS_MAX_MB_ENV = "POLYLOGUE_PYTEST_TMPFS_MAX_MB"
DEFAULT_PYTEST_TMPFS_MAX_MB = 512
MAX_PYTEST_TMPFS_MAX_MB = 2048
MIN_PYTEST_AVAILABLE_KB = 1024 * 1024
PYTEST_HOST_RESERVE_KB = 512 * 1024
MIN_PYTEST_TMPFS_BUDGET_KB = 64 * 1024
MAX_ADAPTIVE_PYTEST_WORKERS = 12
PYTEST_BASETEMP_MIN_FREE_MB_ENV = "POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB"
DEFAULT_PYTEST_BASETEMP_MIN_FREE_MB = 1024
PYTEST_BASETEMP_REQUIRED_MB_ENV = "POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB"
PYTEST_EXPLICIT_BASETEMP_ENV = "POLYLOGUE_PYTEST_EXPLICIT_BASETEMP"
PYTEST_MEMORY_ENVELOPE_WORKERS = 4
PYTEST_MEMORY_ENVELOPE_PSS_KB = 4_353_168
PYTEST_MEMORY_ENVELOPE_TMPFS_KB = 1_472_636
PYTEST_MEMORY_ENVELOPE_CGROUP_BYTES = 6_278_623_232
PYTEST_BASETEMP_PEAK_KB = 1_522 * 1024
PYTEST_PROCESS_MEMORY_PER_WORKER_KB = 768 * 1024
PYTEST_FOCUSED_CONTROLLER_MEMORY_KB = 256 * 1024
PYTEST_PROCESS_MEMORY_FIXED_KB = max(
    0,
    PYTEST_MEMORY_ENVELOPE_PSS_KB - PYTEST_MEMORY_ENVELOPE_WORKERS * PYTEST_PROCESS_MEMORY_PER_WORKER_KB,
)
PYTEST_CGROUP_OVERHEAD_FLOOR_KB = max(
    0,
    (PYTEST_MEMORY_ENVELOPE_CGROUP_BYTES + 1023) // 1024
    - PYTEST_MEMORY_ENVELOPE_PSS_KB
    - PYTEST_MEMORY_ENVELOPE_TMPFS_KB,
)


class PytestResourceError(RuntimeError):
    """Raised when the host cannot safely start a managed pytest run."""


def _trailing_history_record(descriptor: int, *, end: int) -> tuple[int, bytes]:
    """Read only the final unterminated JSONL record and its start offset."""
    cursor = end
    suffix: list[bytes] = []
    while cursor > 0:
        start = max(0, cursor - 64 * 1024)
        os.lseek(descriptor, start, os.SEEK_SET)
        chunk = os.read(descriptor, cursor - start)
        delimiter = chunk.rfind(b"\n")
        if delimiter >= 0:
            return start + delimiter + 1, chunk[delimiter + 1 :] + b"".join(reversed(suffix))
        suffix.append(chunk)
        cursor = start
    return 0, b"".join(reversed(suffix))


def append_verify_history(entry: Mapping[str, Any], *, path: Path = VERIFY_HISTORY_PATH) -> None:
    """Append one complete invocation to the cross-worktree run history.

    ``O_APPEND`` plus an advisory lock keeps concurrent worktrees from
    overwriting or interleaving their records, including short writes.
    Detailed artifacts remain checkout-local; this history is the compact
    durable index used to find and compare them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(entry), ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        end = os.lseek(descriptor, 0, os.SEEK_END)
        if end:
            os.lseek(descriptor, end - 1, os.SEEK_SET)
            if os.read(descriptor, 1) != b"\n":
                trailing_start, trailing = _trailing_history_record(descriptor, end=end)
                try:
                    json.loads(trailing)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    os.ftruncate(descriptor, trailing_start)
                else:
                    # A complete JSON record can lose only its framing newline
                    # during an interrupted append. Preserve it before adding
                    # the next durable record.
                    os.lseek(descriptor, 0, os.SEEK_END)
                    os.write(descriptor, b"\n")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("verification history append made no progress")
            remaining = remaining[written:]
    finally:
        os.close(descriptor)


def pytest_command_worker_request(cmd: Sequence[str]) -> str | None:
    """Return the last xdist worker request from a final pytest command."""
    request: str | None = None
    for index, argument in enumerate(cmd):
        if argument in {"-n", "--numprocesses"}:
            if index + 1 < len(cmd):
                request = cmd[index + 1]
        elif argument.startswith("--numprocesses="):
            request = argument.removeprefix("--numprocesses=")
        elif argument.startswith("-n") and len(argument) > 2:
            request = argument[2:].removeprefix("=")
    return request


def worktree_fingerprint(root: Path | None = None) -> str:
    """Fingerprint tracked changes plus exact non-ignored untracked content."""
    checkout_root = (root or Path.cwd()).resolve()
    digest = hashlib.sha256()
    for command in (
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        ["git", "diff", "--binary", "HEAD", "--"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, timeout=30, cwd=checkout_root)
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"
        if result.returncode != 0:
            return "unavailable"
        digest.update(result.stdout)
        digest.update(b"\0")
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=30,
            cwd=checkout_root,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if untracked.returncode != 0:
        return "unavailable"
    for raw_path in sorted(path for path in untracked.stdout.split(b"\0") if path):
        try:
            path_text = os.fsdecode(raw_path)
            path = checkout_root / path_text
            mode = path.lstat().st_mode
            digest.update(raw_path)
            digest.update(b"\0")
            if stat.S_ISLNK(mode):
                digest.update(b"symlink\0")
                digest.update(os.fsencode(os.readlink(path)))
            elif stat.S_ISREG(mode):
                digest.update(b"file\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                digest.update(f"mode:{stat.S_IFMT(mode):o}".encode())
            digest.update(b"\0")
        except OSError:
            return "unavailable"
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckoutMutationObservation:
    """Whether an exact-head verification interval observed a checkout write."""

    changed: bool
    unavailable: bool
    observed_path: str | None = None


class CheckoutMutationMonitor:
    """Fail closed when watchfiles cannot observe the checkout interval.

    Endpoint hashes establish the state of the checkout, while this monitor
    records writes that occur and are later reverted before the final sample.
    Watches exclude verifier-owned disposable directories so receipts do not
    invalidate themselves.
    """

    _IGNORED_TOP_LEVEL = frozenset(
        {
            ".cache",
            ".git",
            ".hypothesis",
            ".local",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
        }
    )
    _WATCH_START_TIMEOUT_S = 1.0
    _WATCH_SETTLE_S = 0.2
    _WATCH_RUST_TIMEOUT_MS = 25
    _POLLING_DISABLED_VALUES = frozenset({"false", "disable", "disabled"})

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._changed = False
        self._observed_path: str | None = None
        self._unavailable = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        """Start and prove the portable interval watcher before verification."""
        if self._polling_backend_requested():
            with self._state_lock:
                self._unavailable = True
            self._ready.set()
            return
        self._thread = threading.Thread(target=self._watch, name="checkout-mutation-monitor", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self._WATCH_START_TIMEOUT_S):
            with self._state_lock:
                self._unavailable = True
            self._stop.set()
            self._thread.join(timeout=self._WATCH_START_TIMEOUT_S)

    def finish(self) -> CheckoutMutationObservation:
        """Stop monitoring only after the caller took its final fingerprint."""
        # The final fingerprint is already sampled. Give watchfiles one short
        # backend turn to surface any event emitted before that sample, then
        # stop the generator cleanly through its portable stop event.
        self._stop.wait(self._WATCH_SETTLE_S)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                with self._state_lock:
                    self._unavailable = True
        with self._state_lock:
            return CheckoutMutationObservation(
                changed=self._changed,
                unavailable=self._unavailable,
                observed_path=self._observed_path,
            )

    def _watch(self) -> None:
        try:
            for changes in watchfiles.watch(
                *self._watched_directories(),
                watch_filter=None,
                debounce=0,
                step=1,
                stop_event=self._stop,
                rust_timeout=self._WATCH_RUST_TIMEOUT_MS,
                yield_on_timeout=True,
                raise_interrupt=False,
                force_polling=False,
                recursive=False,
            ):
                # An empty timeout batch proves the backend initialized before
                # a verification command starts, closing the startup race.
                self._ready.set()
                for _change, raw_path in changes:
                    self._record_change(Path(raw_path))
                    if self._changed or self._unavailable:
                        return
            if not self._stop.is_set():
                with self._state_lock:
                    self._unavailable = True
        except Exception:
            with self._state_lock:
                self._unavailable = True
        finally:
            self._ready.set()

    @classmethod
    def _polling_backend_requested(cls) -> bool:
        """Reject watchfiles modes that cannot witness every interval mutation."""
        forced = os.getenv("WATCHFILES_FORCE_POLLING")
        if forced:
            return forced.lower() not in cls._POLLING_DISABLED_VALUES
        uname = platform.uname()
        return uname.system.lower() == "linux" and "microsoft-standard" in uname.release.lower()

    def _watched_directories(self) -> list[Path]:
        """Watch existing source directories shallowly and omit disposable trees."""
        directories: list[Path] = []
        for current, child_directories, _files in os.walk(self.root):
            current_path = Path(current)
            child_directories[:] = [child for child in child_directories if child not in self._IGNORED_TOP_LEVEL]
            directories.append(current_path)
        return directories

    def _record_change(self, candidate: Path) -> None:
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            relative = candidate.relative_to(self.root)
        except ValueError:
            return
        if self._path_is_ignored(relative):
            return
        with self._state_lock:
            self._changed = True
            self._observed_path = relative.as_posix()

    def _path_is_ignored(self, relative: Path) -> bool:
        if any(part in self._IGNORED_TOP_LEVEL for part in relative.parts):
            return True
        try:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", "--no-index", "--", relative.as_posix()],
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            with self._state_lock:
                self._unavailable = True
            return True
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        with self._state_lock:
            self._unavailable = True
        return True


@dataclass(frozen=True)
class PytestRuntimePolicy:
    """One start-time resource decision for a managed pytest run."""

    available_kb: int
    tmpfs_budget_mb: int
    workers: int
    memory_full_avg10: float
    basetemp_root: str | None = None
    basetemp_label: str | None = None
    basetemp_required_mb: int | None = None
    basetemp_free_mb: int | None = None
    tmpfs_predicted_mb: int | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "available_mb": round(self.available_kb / 1024),
            "tmpfs_budget_mb": self.tmpfs_budget_mb,
            "workers": self.workers,
            "memory_full_avg10": self.memory_full_avg10,
            "basetemp_root": self.basetemp_root,
            "basetemp_label": self.basetemp_label,
            "basetemp_required_mb": self.basetemp_required_mb,
            "basetemp_free_mb": self.basetemp_free_mb,
            "tmpfs_predicted_mb": self.tmpfs_predicted_mb,
        }


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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 4)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50_s": _percentile(values, 0.50),
        "p95_s": _percentile(values, 0.95),
        "p99_s": _percentile(values, 0.99),
        "max_s": round(max(values), 4) if values else None,
        "sum_s": round(sum(values), 4) if values else 0.0,
    }


def aggregate_pytest_statistics(
    step_dir: Path,
    *,
    command: list[Any] | tuple[Any, ...] = (),
    step_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce the append-only pytest evidence into one durable step summary.

    The raw event stream remains authoritative for forensics.  This summary is
    deliberately derived and replaceable: it makes repeated performance work
    cheap without introducing a second source of truth for test outcomes.
    """
    events: list[dict[str, Any]] = []
    events_path = step_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                if isinstance(row, dict):
                    events.append(row)

    phases: dict[str, list[float]] = {"setup": [], "call": [], "teardown": []}
    outcomes: dict[str, int] = {}
    phase_outcomes: dict[str, dict[str, int]] = {"setup": {}, "call": {}, "teardown": {}}
    nodes: set[str] = set()
    workers: set[str] = set()
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for row in events:
        worker = row.get("worker_id")
        if isinstance(worker, str):
            workers.add(worker)
        nodeid = row.get("nodeid")
        if isinstance(nodeid, str) and nodeid:
            nodes.add(nodeid)
        event = row.get("event")
        if event != "test_report" or not isinstance(nodeid, str) or not nodeid:
            continue
        when = row.get("when")
        if when not in phases:
            continue
        key = (nodeid, when)
        prior = reports.get(key)
        # xdist sends the worker's original report to the controller. Prefer
        # the worker event when both arrive, while accepting old/controller-only
        # artifacts produced before that forwarding copy was suppressed.
        if prior is None or (prior.get("worker_id") == "controller" and row.get("worker_id") != "controller"):
            reports[key] = row

    canonical_outcomes: dict[str, str] = {}
    canonical_report_present = False
    canonical_report_path = step_dir / PYTEST_CANONICAL_REPORT_NAME
    if canonical_report_path.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            canonical_report = json.loads(canonical_report_path.read_text(encoding="utf-8"))
            canonical_tests = canonical_report.get("tests") if isinstance(canonical_report, dict) else None
            if isinstance(canonical_tests, list):
                canonical_report_present = True
                for test in canonical_tests:
                    if not isinstance(test, dict):
                        continue
                    nodeid = test.get("nodeid")
                    outcome = test.get("outcome")
                    if not isinstance(nodeid, str) or not nodeid or not isinstance(outcome, str):
                        continue
                    nodes.add(nodeid)
                    canonical_outcomes[nodeid] = outcome
                    for when in phases:
                        phase = test.get(when)
                        if not isinstance(phase, dict) or (nodeid, when) in reports:
                            continue
                        phase_outcome = phase.get("outcome")
                        duration = phase.get("duration")
                        reports[(nodeid, when)] = {
                            "nodeid": nodeid,
                            "when": when,
                            "outcome": phase_outcome,
                            "duration_s": duration,
                            "worker_id": "canonical-report",
                        }

    reports_by_node: dict[str, dict[str, dict[str, Any]]] = {}
    for (nodeid, when), row in reports.items():
        reports_by_node.setdefault(nodeid, {})[when] = row
        duration = row.get("duration_s")
        if isinstance(duration, (int, float)):
            phases[when].append(float(duration))
        outcome = row.get("outcome")
        if isinstance(outcome, str):
            bucket = phase_outcomes[when]
            bucket[outcome] = bucket.get(outcome, 0) + 1

    for nodeid in nodes:
        node_reports = reports_by_node.get(nodeid, {})
        setup = node_reports.get("setup", {}).get("outcome")
        call = node_reports.get("call", {}).get("outcome")
        teardown = node_reports.get("teardown", {}).get("outcome")
        canonical_outcome = canonical_outcomes.get(nodeid)
        if canonical_outcome is not None:
            terminal = canonical_outcome
        elif setup == "failed" or teardown == "failed":
            terminal = "error"
        elif isinstance(call, str):
            terminal = call
        elif setup in {"skipped", "xfailed", "xpassed"}:
            terminal = str(setup)
        elif teardown in {"skipped", "xfailed", "xpassed"}:
            terminal = str(teardown)
        else:
            # A test may have emitted its start event just before an interrupt
            # or forced containment cleanup. Keep that missing terminal phase
            # visible so outcome totals still account for every started node.
            terminal = "interrupted"
        outcomes[terminal] = outcomes.get(terminal, 0) + 1

    resources: list[dict[str, Any]] = []
    resources_path = step_dir / "resources.jsonl"
    if resources_path.exists():
        for line in resources_path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                if isinstance(row, dict):
                    resources.append(row)
    explicit_worker_count: int | None = None
    worker_request = pytest_command_worker_request([str(value) for value in command])
    if worker_request is not None:
        with contextlib.suppress(ValueError):
            explicit_worker_count = int(worker_request)
    basetemp_sizes = [
        int(size_value) * 1024 for row in resources if isinstance((size_value := row.get("basetemp_size_kb")), int)
    ]
    basetemp_allocated = [
        int(allocated_value) * 1024
        for row in resources
        if isinstance((allocated_value := row.get("basetemp_allocated_kb")), int)
    ]
    containment: dict[str, Any] = {}
    containment_path = step_dir / "containment.json"
    if containment_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            raw = json.loads(containment_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                containment = raw

    parent_cleanup = (step_result or {}).get("basetemp_cleanup")
    return {
        "schema_version": 1,
        "canonical_report_status": "present" if canonical_report_present else "missing",
        "command": [str(value) for value in command],
        "node_count": len(nodes),
        "outcomes": outcomes,
        "phase_outcomes": phase_outcomes,
        "phases": {name: _distribution(values) for name, values in phases.items()},
        "xdist": {
            "worker_ids": sorted(workers),
            "worker_count": max(
                0,
                len(workers) - (1 if "controller" in workers else 0),
                explicit_worker_count or 0,
            ),
        },
        "storage": {
            "basetemp_logical_bytes_max": max(basetemp_sizes, default=None),
            "basetemp_allocated_bytes_max": max(basetemp_allocated, default=None),
            "basetemp_root": next(
                (row.get("basetemp") for row in reversed(resources) if isinstance(row.get("basetemp"), str)),
                None,
            ),
        },
        "resources": {
            "peak_tree_rss_kb": max(
                (int(row["tree_rss_kb"]) for row in resources if isinstance(row.get("tree_rss_kb"), int)),
                default=None,
            ),
            "peak_tree_pss_kb": max(
                (int(row["tree_pss_kb"]) for row in resources if isinstance(row.get("tree_pss_kb"), int)),
                default=None,
            ),
            "peak_cgroup_memory_bytes": max(
                (
                    int(row["cgroup_memory_peak_bytes"])
                    for row in resources
                    if isinstance(row.get("cgroup_memory_peak_bytes"), int)
                ),
                default=None,
            ),
        },
        "cleanup": {
            "complete": True
            if isinstance(parent_cleanup, str) and parent_cleanup
            else containment.get("tmpfs_cleanup_complete"),
            "termination_reason": containment.get("termination_reason"),
            "escalated_to_sigkill": containment.get("escalated_to_sigkill"),
            "exit_code": containment.get("exit_code", (step_result or {}).get("exit")),
        },
    }


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=5, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(result.stdout.strip())


def git_head(cwd: Path | None = None) -> str | None:
    """Return the current checkout HEAD, or ``None`` when git is unavailable.

    Verification artifacts are evidence, not a reason to fail the user's test
    run. Keep the probe bounded and nullable so non-git archives, missing git
    executables, and wedged git commands preserve honest ``null`` metadata.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


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
    containment_path: Path
    statistics_path: Path


class VerifyRun:
    """A filesystem-backed verification run ledger."""

    def __init__(
        self,
        *,
        tier: str,
        argv: list[str],
        git_head: str | None,
        root: Path | None = None,
        polylogue_import_path: str | None = None,
        environment_fingerprint: Mapping[str, Any] | None = None,
        worktree_fingerprint: str | None = None,
    ) -> None:
        self.root = root or Path.cwd()
        self.run_id = make_run_id(tier=tier)
        self.run_dir = self.root / VERIFY_RUNS_DIR / self.run_id
        self._payload: dict[str, Any] = {
            "run_id": self.run_id,
            "tier": tier,
            "argv": list(argv),
            "git_head": git_head,
            "git_dirty": git_dirty(self.root),
            # Receipt for the worktree-import hazard (devtools/checkout_guard.py):
            # the resolved `polylogue` package path this run actually used, so a
            # wrong-tree run is visible after the fact from the run artifact
            # even where the live preflight already refused for an in-process
            # caller and this fired for a different process boundary.
            "polylogue_import_path": polylogue_import_path,
            "environment_fingerprint": dict(environment_fingerprint) if environment_fingerprint is not None else None,
            "worktree_fingerprint": worktree_fingerprint,
            # A VerifyRun can be constructed by maintenance/test helpers that
            # do not have a checkout fingerprint. Keep its current-run marker
            # attributable to this checkout either way.
            "checkout_root": str(self.root.resolve()),
            "owner_pid": os.getpid(),
            "started_at": utc_now(),
            "status": "running",
            "steps": [],
            "artifact_dir": str(VERIFY_RUNS_DIR / self.run_id),
        }
        invocation_id = os.environ.get(VERIFICATION_INVOCATION_ID_ENV)
        if invocation_id:
            self._payload["invocation_id"] = invocation_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write()

    @property
    def relative_run_dir(self) -> Path:
        return VERIFY_RUNS_DIR / self.run_id

    def write(self) -> None:
        _write_json(self.run_dir / "run.json", self._payload)
        invocation_receipt = os.environ.get(VERIFICATION_RECEIPT_PATH_ENV)
        if invocation_receipt:
            _write_json(Path(invocation_receipt), self._payload)
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
            containment_path=step_dir / "containment.json",
            statistics_path=step_dir / "statistics.json",
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

    def finish_step(self, *, step_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """Finalize one step and return its durable compact representation."""
        for step in self._payload["steps"]:
            if step.get("step_id") == step_id:
                step.update(result)
                step["finished_at"] = utc_now()
                step["status"] = "success" if result.get("exit") == 0 else "failed"
                step_dir = self.run_dir / "steps" / step_id
                if not str(step.get("name", "")).startswith("pytest"):
                    break
                # An interrupted runner never returns through the normal
                # post-subprocess merge. Fold shards here, before every
                # aggregation path, so completed worker evidence survives.
                with contextlib.suppress(OSError):
                    merge_worker_events(step_dir / "events", step_dir / "events.jsonl")
                statistics_path = step_dir / "statistics.json"
                with contextlib.suppress(OSError, ValueError):
                    statistics = aggregate_pytest_statistics(
                        step_dir,
                        command=step.get("cmd", []),
                        step_result=result,
                    )
                    _write_json(statistics_path, statistics)
                    with contextlib.suppress(OSError):
                        shutil.copyfile(statistics_path, self.root / CURRENT_STATISTICS_PATH)
                    step["statistics_path"] = str(self.relative_run_dir / "steps" / step_id / "statistics.json")
                    # Keep the compact aggregate in the cross-worktree history
                    # itself. The detailed artifact path is checkout-local and
                    # may disappear when a merged lane is cleaned up.
                    step["statistics"] = statistics
                break
        self.write()
        return next((dict(step) for step in self._payload["steps"] if step.get("step_id") == step_id), None)

    def finish_interrupted_steps(self, *, exit_code: int, diagnosis: str) -> None:
        """Close any open step when the outer runner receives Ctrl-C."""
        for step in self._payload["steps"]:
            if step.get("status") == "running":
                self.finish_step(
                    step_id=str(step["step_id"]),
                    result={
                        "duration_s": None,
                        "exit": exit_code,
                        "diagnosis": diagnosis,
                        "termination_reason": "operator_interrupt",
                    },
                )

    def finish(
        self,
        *,
        exit_code: int,
        duration_s: float,
        diagnosis: str | None = None,
        verification_scope: str | None = None,
        release_baseline_allowed: bool | None = None,
        terminal_authorization: str | None = None,
        final_worktree_fingerprint: str | None = None,
        checkout_mutation_path: str | None = None,
    ) -> dict[str, Any]:
        self._payload["finished_at"] = utc_now()
        self._payload["duration_s"] = round(duration_s, 2)
        self._payload["exit_code"] = int(exit_code)
        self._payload["status"] = "success" if exit_code == 0 else "failed"
        if diagnosis:
            self._payload["diagnosis"] = diagnosis
        if final_worktree_fingerprint is not None:
            self._payload["final_worktree_fingerprint"] = final_worktree_fingerprint
        if checkout_mutation_path is not None:
            self._payload["checkout_mutation_path"] = checkout_mutation_path
        if verification_scope is not None:
            self._payload["verification_scope"] = verification_scope
            self._payload["release_baseline_allowed"] = release_baseline_allowed
            self._payload["terminal_authorization"] = terminal_authorization
        self.write()
        return dict(self._payload)


def env_for_pytest_step(env: dict[str, str], *, run: VerifyRun, artifacts: PytestStepArtifacts) -> dict[str, str]:
    updated = dict(env)
    # The merge-gate invocation receipt belongs to the top-level devtools
    # process. Pytest and any nested harness commands must not inherit the
    # token and overwrite that receipt with a child run.
    updated.pop(VERIFICATION_INVOCATION_ID_ENV, None)
    updated.pop(VERIFICATION_RECEIPT_PATH_ENV, None)
    updated["POLYLOGUE_VERIFY_RUN_ID"] = run.run_id
    updated["POLYLOGUE_PYTEST_RUN_ID"] = run.run_id
    updated["POLYLOGUE_PYTEST_EVENTS_DIR"] = str(artifacts.events_dir)
    updated["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(artifacts.events_merged_path)
    updated["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(artifacts.selection_path)
    updated["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(artifacts.summary_path)
    updated["POLYLOGUE_PYTEST_CONTAINMENT_PATH"] = str(artifacts.containment_path)
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
    with contextlib.suppress(FileNotFoundError):
        shutil.copyfile(artifacts.containment_path, root / CURRENT_CONTAINMENT_PATH)
    with contextlib.suppress(FileNotFoundError):
        shutil.copyfile(artifacts.statistics_path, root / CURRENT_STATISTICS_PATH)


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


def _smaps_rollup_kb(pid: int) -> dict[str, int]:
    path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator or key not in {"Pss", "Pss_Anon", "Pss_File", "SwapPss"}:
            continue
        with contextlib.suppress(ValueError, IndexError):
            values[key] = int(raw.split()[0])
    return values


def _process_io_bytes(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path(f"/proc/{pid}/io").read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator or key not in {"read_bytes", "write_bytes", "cancelled_write_bytes"}:
            continue
        with contextlib.suppress(ValueError):
            values[key] = int(raw.strip())
    return values


def _cgroup_path(pid: int) -> str | None:
    with contextlib.suppress(OSError):
        for row in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            parts = row.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                return parts[2]
    return None


def _cgroup_int(path: str | None, name: str) -> int | None:
    if path is None:
        return None
    with contextlib.suppress(OSError, ValueError):
        return int((Path("/sys/fs/cgroup") / path.lstrip("/") / name).read_text().strip())
    return None


def _cgroup_io_bytes(path: str | None) -> dict[str, int] | None:
    if path is None:
        return None
    totals = {"rbytes": 0, "wbytes": 0}
    try:
        rows = (Path("/sys/fs/cgroup") / path.lstrip("/") / "io.stat").read_text().splitlines()
    except OSError:
        return None
    for row in rows:
        for item in row.split()[1:]:
            key, _, value = item.partition("=")
            if key in totals:
                with contextlib.suppress(ValueError):
                    totals[key] += int(value)
    return totals


def _process_identity(pid: int) -> str:
    """Return a PID-reuse-safe identity for one sampled process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return f"{pid}:{fields[19]}"
    except (OSError, IndexError):
        return f"{pid}:unknown"


def _process_environ_value(pid: int, key: str) -> str | None:
    """Read one same-user process environment value for role attribution."""
    try:
        rows = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = key.encode() + b"="
    for row in rows:
        if row.startswith(prefix):
            return row[len(prefix) :].decode(errors="replace") or None
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


def _dir_usage_kb(path: Path) -> tuple[int | None, int | None]:
    """Measure apparent and allocated bytes owned by one basetemp tree."""
    if not path.exists():
        return None, None
    logical_total = 0
    allocated_total = 0
    try:
        for item in path.rglob("*"):
            with contextlib.suppress(OSError):
                item_stat = item.lstat()
                if not stat.S_ISDIR(item_stat.st_mode):
                    logical_total += item_stat.st_size
                    allocated_total += item_stat.st_blocks * 512
    except OSError:
        return None, None
    return int(logical_total / 1024), int(allocated_total / 1024)


def checkout_hash(root: Path) -> str:
    return hashlib.sha1(str(root).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


DEFAULT_PYTEST_BASETEMP_ROOT = Path("/realm/tmp/polylogue-pytest")
_CLOUD_PYTEST_BASETEMP_ROOT = Path("/tmp/polylogue-pytest")
PYTEST_TMPFS_ROOT = Path("/dev/shm")
_PYTEST_BASETEMP_CLAIM_PREFIX = ".polylogue-pytest-claim-"


def _is_beneath(path: Path, root: Path) -> bool:
    """Return whether *path* resolves within *root*, including *root* itself."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def pytest_basetemp_claim_path(basetemp: Path, *, kind: str) -> Path:
    """Return the durable, adjacent claim path for one pytest basetemp.

    Pytest lazily clears an explicit ``--basetemp`` before first use, so an
    ownership record inside that tree cannot survive normal initialization.
    Claims live beside the tree and are keyed by its canonical filesystem
    path, not by a reusable basename.  A configured symlink and an explicit
    real-path spelling must therefore serialize through the same claim.
    """
    try:
        canonical = basetemp.resolve()
    except OSError:
        canonical = basetemp.absolute()
    digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:20]
    return canonical.parent / f"{_PYTEST_BASETEMP_CLAIM_PREFIX}{kind}-{digest}"


def clear_managed_pytest_basetemp_claim(basetemp: Path) -> None:
    """Remove the durable claim after a managed run's tree is reclaimed."""
    with contextlib.suppress(OSError):
        pytest_basetemp_claim_path(basetemp, kind="managed").unlink()


def _try_acquire_pytest_basetemp_claim_lock(basetemp: Path) -> TextIO | None:
    """Acquire the adjacent claim lock without waiting on another pytest run."""
    lock_path = pytest_basetemp_claim_path(basetemp, kind="lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError:
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        handle.close()
        return None
    return handle


def managed_pytest_basetemp_owner_alive(basetemp: Path) -> bool | None:
    """Return whether a positive managed claim still names a live process."""
    try:
        raw_identity = pytest_basetemp_claim_path(basetemp, kind="managed").read_text(encoding="utf-8").strip()
        raw_pid, separator, raw_start_ticks = raw_identity.partition(":")
        pid = int(raw_pid)
        start_ticks = int(raw_start_ticks) if separator else None
    except (OSError, ValueError):
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        current_start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError):
        return False
    return start_ticks is None or current_start_ticks == start_ticks


def pytest_tmpfs_budget_exceeded(sample: Mapping[str, Any], *, budget_kb: int) -> bool:
    """Return whether a sampled basetemp exceeds its tmpfs allocation cap.

    ``st_size`` remains forensic evidence because sparse files can expose a
    large logical extent. Tmpfs capacity is consumed by allocated blocks, so
    the admission limit must use that physical measure.
    """
    allocated_kb = sample.get("basetemp_allocated_kb")
    return isinstance(allocated_kb, int) and allocated_kb > budget_kb


def normalize_pytest_basetemp_env(env: Mapping[str, str]) -> dict[str, str]:
    """Keep cloud pytest defaults from escaping a workstation scratch volume.

    ``.claude/settings.json`` supplies ``/tmp/polylogue-pytest`` for cloud
    sandboxes. Agent subprocesses can inherit that setting on the workstation;
    remove only that known cloud default so the managed adaptive tmpfs policy
    can apply. Preserve other custom roots and retain the cloud setting when
    the workstation scratch mount is absent.
    """

    normalized = dict(env)
    configured = normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if configured == str(_CLOUD_PYTEST_BASETEMP_ROOT) and DEFAULT_PYTEST_BASETEMP_ROOT.parent.is_dir():
        normalized.pop("POLYLOGUE_PYTEST_BASETEMP_ROOT", None)
    return normalized


def force_managed_pytest_scratch(env: Mapping[str, str]) -> dict[str, str]:
    """Route an unsupervised nested pytest away from tmpfs.

    Preserve a genuine disk-backed operator root. An inherited root beneath
    ``/dev/shm`` is not a safe override for a subprocess whose parent does not
    sample or terminate it at the managed tmpfs cap.
    """
    normalized = normalize_pytest_basetemp_env(env)
    configured_root = normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if configured_root is not None and _is_beneath(Path(configured_root), PYTEST_TMPFS_ROOT):
        normalized.pop("POLYLOGUE_PYTEST_BASETEMP_ROOT", None)
    normalized["POLYLOGUE_PYTEST_TMPFS"] = "0"
    return normalized


def _pytest_process_memory_reserve_kb(workers: int) -> int:
    """Keep the measured controller floor while scaling worker processes."""
    return PYTEST_PROCESS_MEMORY_FIXED_KB + workers * PYTEST_PROCESS_MEMORY_PER_WORKER_KB


def _pytest_cgroup_overhead_reserve_kb(workers: int) -> int:
    """Keep the measured residual floor, then scale it above four workers."""
    scale_workers = max(workers, PYTEST_MEMORY_ENVELOPE_WORKERS)
    return (
        PYTEST_CGROUP_OVERHEAD_FLOOR_KB * scale_workers + PYTEST_MEMORY_ENVELOPE_WORKERS - 1
    ) // PYTEST_MEMORY_ENVELOPE_WORKERS


def _pytest_non_tmpfs_memory_reserve_kb(workers: int, *, full_suite: bool = True) -> int:
    if not full_suite:
        focused_process_kb = PYTEST_FOCUSED_CONTROLLER_MEMORY_KB + workers * PYTEST_PROCESS_MEMORY_PER_WORKER_KB
        focused_cgroup_overhead_kb = (
            PYTEST_CGROUP_OVERHEAD_FLOOR_KB * max(1, workers) + PYTEST_MEMORY_ENVELOPE_WORKERS - 1
        ) // PYTEST_MEMORY_ENVELOPE_WORKERS
        return focused_process_kb + focused_cgroup_overhead_kb + PYTEST_HOST_RESERVE_KB
    return (
        _pytest_process_memory_reserve_kb(workers)
        + _pytest_cgroup_overhead_reserve_kb(workers)
        + PYTEST_HOST_RESERVE_KB
    )


def _default_pytest_workers(*, available_kb: int, cpu_cap: int, memory_full_avg10: float) -> int:
    """Choose a pool that the same launch-time envelope can admit.

    Prefer the largest pool that preserves the measured aggregate basetemp in
    memory. If even one worker cannot do that, retain the largest pool that can
    start with the minimum tmpfs allowance so placement can reroute the shared
    basetemp to scratch.
    """

    maximum = min(MAX_ADAPTIVE_PYTEST_WORKERS, cpu_cap)

    def largest_with(tmpfs_reserve_kb: int) -> int | None:
        return next(
            (
                workers
                for workers in range(maximum, 0, -1)
                if _pytest_non_tmpfs_memory_reserve_kb(workers) + tmpfs_reserve_kb <= available_kb
            ),
            None,
        )

    workers = largest_with(PYTEST_BASETEMP_PEAK_KB)
    if workers is None:
        workers = largest_with(MIN_PYTEST_TMPFS_BUDGET_KB) or 1
    if memory_full_avg10 >= 2.0:
        return max(1, workers // 4)
    if memory_full_avg10 >= 0.5:
        return max(1, workers // 2)
    return workers


def adaptive_pytest_runtime_policy(
    *,
    available_kb: int | None = None,
    memory_full_avg10: float | None = None,
    cpu_count: int | None = None,
    shm_free_kb: int | None = None,
    worker_count: int | None = None,
    full_suite: bool = True,
) -> PytestRuntimePolicy:
    """Size tmpfs and xdist from one measured resource envelope.

    The tmpfs cap is an admission limit, so it must be derived from capacity
    the host can actually reserve.  A percentage of ``MemAvailable`` made a
    well-provisioned host advertise a 1.5 GiB cap to a full pytest run that
    had 15 GiB free, then the supervisor correctly killed the run at that
    artificial limit.  Reserve the command's known workers and host headroom
    from the measured four-worker process/cgroup/tmpfs envelope first.

    The four-worker PSS measurement contains a controller and supervisor that
    do not disappear at one worker. Preserve the remainder after the existing
    768 MiB marginal worker allowance as a fixed process component. Keep the
    measured residual cgroup charge as a floor through four workers and scale
    it above the observed concurrency. Reserve the separately observed 1,521.6
    MiB basetemp peak at its next whole-MiB ceiling because it is one aggregate
    run tree, independent of how xdist partitions the tests.
    """
    if available_kb is None:
        available_kb = _meminfo().get("MemAvailable")
        cgroup_headroom_bytes = read_cgroup_memory_headroom_bytes()
        if available_kb is not None and cgroup_headroom_bytes is not None:
            available_kb = min(available_kb, cgroup_headroom_bytes // 1024)
    if available_kb is None:
        raise PytestResourceError("cannot read MemAvailable; refusing an unbudgeted pytest run")
    if available_kb < MIN_PYTEST_AVAILABLE_KB:
        raise PytestResourceError(
            f"only {available_kb / (1024 * 1024):.2f} GiB memory available; managed pytest requires at least 1.00 GiB"
        )

    if memory_full_avg10 is None:
        memory_full_avg10 = _pressure("memory").get("full_avg10", 0.0)
    if shm_free_kb is None:
        shm = _fs_usage(PYTEST_TMPFS_ROOT)
        shm_free_kb = shm.get("free_kb", 0) if shm is not None else 0

    logical_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    cpu_cap = max(1, logical_cpus // 2)
    if worker_count is not None:
        if worker_count < 0:
            raise PytestResourceError(f"invalid pytest worker count {worker_count}")
        reserved_workers = worker_count
    else:
        reserved_workers = _default_pytest_workers(
            available_kb=available_kb,
            cpu_cap=cpu_cap,
            memory_full_avg10=memory_full_avg10,
        )

    fixed_reserve_kb = _pytest_non_tmpfs_memory_reserve_kb(reserved_workers, full_suite=full_suite)
    tmpfs_predicted_kb = PYTEST_BASETEMP_PEAK_KB if full_suite else None
    tmpfs_floor_kb = MIN_PYTEST_TMPFS_BUDGET_KB if shm_free_kb >= MIN_PYTEST_TMPFS_BUDGET_KB else 0
    if fixed_reserve_kb + tmpfs_floor_kb > available_kb:
        raise PytestResourceError(
            "cannot reserve measured pytest cgroup memory and host headroom "
            f"(available={available_kb / 1024:.0f} MiB, workers={reserved_workers}, "
            f"required={(fixed_reserve_kb + tmpfs_floor_kb) / 1024:.0f} MiB)"
        )
    memory_safe_tmpfs_kb = available_kb - fixed_reserve_kb
    safe_shm_budget_kb = int(shm_free_kb * 0.80)
    tmpfs_budget_kb = min(MAX_PYTEST_TMPFS_MAX_MB * 1024, safe_shm_budget_kb, memory_safe_tmpfs_kb)
    tmpfs_budget_mb = int(tmpfs_budget_kb / 1024)

    return PytestRuntimePolicy(
        available_kb=available_kb,
        tmpfs_budget_mb=tmpfs_budget_mb,
        workers=reserved_workers,
        memory_full_avg10=memory_full_avg10,
        tmpfs_predicted_mb=(tmpfs_predicted_kb + 1023) // 1024 if tmpfs_predicted_kb is not None else None,
    )


def _tmpfs_admission_refusal(
    *,
    kind: str,
    path: Path,
    declared_demand_kb: int,
    safe_budget_kb: int,
    headroom_kb: int,
) -> PytestResourceError:
    """Describe all failed tmpfs admission constraints in one refusal."""
    free_kb = _headroom_kb(path)
    required_headroom_kb = headroom_kb + max(declared_demand_kb, safe_budget_kb)
    available = f"{free_kb / 1024:.0f} MiB" if free_kb is not None else "unknown"
    return PytestResourceError(
        f"{kind} pytest basetemp declared demand exceeds its safe adaptive tmpfs budget "
        f"({path}: declared demand={declared_demand_kb / 1024:.0f} MiB, "
        f"safe tmpfs budget={safe_budget_kb / 1024:.0f} MiB, "
        f"available filesystem space={available}, "
        f"required filesystem headroom={required_headroom_kb / 1024:.0f} MiB)"
    )


def apply_managed_pytest_runtime_policy(
    env: Mapping[str, str], *, worker_count: int | None = None, full_suite: bool = True
) -> tuple[dict[str, str], PytestRuntimePolicy | None]:
    """Place broad runs on scratch and focused runs on bounded tmpfs by default.

    Also runs the basetemp disk-headroom preflight
    (:func:`resolve_pytest_basetemp_root`) so a starved basetemp location is
    refused here — loudly, with the candidates checked — before pytest ever
    starts, instead of surfacing as an ``OSError: [Errno 28]`` deep inside an
    unrelated command minutes or hours later.
    """
    normalized = normalize_pytest_basetemp_env(env)
    explicit_basetemp = normalized.get(PYTEST_EXPLICIT_BASETEMP_ENV)
    default_full_suite_scratch = (
        full_suite
        and explicit_basetemp is None
        and not normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
        and "POLYLOGUE_PYTEST_TMPFS" not in normalized
    )
    configured_root = normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    configured_tmpfs = configured_root is not None and _is_beneath(Path(configured_root), PYTEST_TMPFS_ROOT)
    explicit_tmpfs = explicit_basetemp is not None and _is_beneath(Path(explicit_basetemp), PYTEST_TMPFS_ROOT)
    manages_tmpfs = (
        explicit_tmpfs
        or configured_tmpfs
        or (explicit_basetemp is None and configured_root is None and normalized.get("POLYLOGUE_PYTEST_TMPFS") != "0")
    )
    policy = adaptive_pytest_runtime_policy(
        worker_count=worker_count,
        shm_free_kb=None if manages_tmpfs else 0,
        full_suite=full_suite,
    )
    rejected_candidates: tuple[str, ...] = ()
    if full_suite and policy.tmpfs_predicted_mb is not None:
        normalized.setdefault(PYTEST_BASETEMP_REQUIRED_MB_ENV, str(policy.tmpfs_predicted_mb))
    if manages_tmpfs:
        normalized["POLYLOGUE_PYTEST_TMPFS"] = "1"
        normalized.setdefault(PYTEST_TMPFS_MAX_MB_ENV, str(policy.tmpfs_budget_mb))
        effective_tmpfs_budget_kb = pytest_tmpfs_budget_kb(normalized)
        policy_tmpfs_budget_kb = policy.tmpfs_budget_mb * 1024
        if effective_tmpfs_budget_kb is not None and effective_tmpfs_budget_kb > policy_tmpfs_budget_kb:
            normalized[PYTEST_TMPFS_MAX_MB_ENV] = str(policy.tmpfs_budget_mb)
            effective_tmpfs_budget_kb = pytest_tmpfs_budget_kb(normalized)
        required_basetemp_kb = pytest_basetemp_required_kb(normalized)
        if (
            required_basetemp_kb is not None
            and effective_tmpfs_budget_kb is not None
            and effective_tmpfs_budget_kb < required_basetemp_kb
        ):
            if explicit_tmpfs:
                path = Path(explicit_basetemp or PYTEST_TMPFS_ROOT)
                raise _tmpfs_admission_refusal(
                    kind="explicit",
                    path=path,
                    declared_demand_kb=required_basetemp_kb,
                    safe_budget_kb=effective_tmpfs_budget_kb,
                    headroom_kb=pytest_basetemp_min_free_kb(normalized),
                )
            if configured_tmpfs:
                # The configured tmpfs root has become unsafe for this run.
                # Leaving it in place would make the resolver select it even
                # though tmpfs has just been disabled, without its cap.
                rejected_candidates = (
                    str(
                        _tmpfs_admission_refusal(
                            kind="configured",
                            path=Path(configured_root or PYTEST_TMPFS_ROOT),
                            declared_demand_kb=required_basetemp_kb,
                            safe_budget_kb=effective_tmpfs_budget_kb,
                            headroom_kb=pytest_basetemp_min_free_kb(normalized),
                        )
                    ),
                )
                normalized.pop("POLYLOGUE_PYTEST_BASETEMP_ROOT", None)
            normalized["POLYLOGUE_PYTEST_TMPFS"] = "0"
    if default_full_suite_scratch:
        # Broad-suite demand grows with the fixture universe and has exceeded
        # the supervised 2 GiB ceiling while tests were still progressing.
        # Keep that ceiling for explicit tmpfs runs; use NVMe for the default
        # broad route instead of guessing the next aggregate peak.
        normalized["POLYLOGUE_PYTEST_TMPFS"] = "0"
    if explicit_basetemp is not None:
        selected_root = Path(explicit_basetemp)
        selected_label = "explicit"
        free_kb = _headroom_kb(selected_root)
        required_kb = pytest_basetemp_required_kb(normalized)
        min_free_kb = max(pytest_basetemp_min_free_kb(normalized), required_kb or 0)
        explicit_required_kb = min_free_kb
        if _is_beneath(selected_root, PYTEST_TMPFS_ROOT) and normalized.get("POLYLOGUE_PYTEST_TMPFS") == "1":
            explicit_required_kb = pytest_basetemp_min_free_kb(normalized) + max(
                required_kb or 0,
                pytest_tmpfs_budget_kb(normalized) or 0,
            )
        if free_kb is None or free_kb < explicit_required_kb:
            raise PytestResourceError(
                "explicit pytest basetemp does not have enough free space "
                f"({selected_root}: {free_kb / 1024:.0f} MiB free, need >= {explicit_required_kb / 1024:.0f} MiB)"
                if free_kb is not None
                else f"explicit pytest basetemp is unreachable: {selected_root}"
            )
    else:
        selected_root, selected_label = resolve_pytest_basetemp_root(
            normalized, rejected_candidates=rejected_candidates
        )
        free_kb = _headroom_kb(selected_root)
    if (
        explicit_basetemp is None
        and not normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
        and selected_root != PYTEST_TMPFS_ROOT
    ):
        normalized["POLYLOGUE_PYTEST_BASETEMP_ROOT"] = str(selected_root)
        normalized["POLYLOGUE_PYTEST_TMPFS"] = "0"
    required_kb = pytest_basetemp_required_kb(normalized)
    policy = replace(
        policy,
        basetemp_root=str(selected_root),
        basetemp_label=selected_label,
        basetemp_required_mb=round(required_kb / 1024) if required_kb is not None else None,
        basetemp_free_mb=round(free_kb / 1024) if free_kb is not None else None,
    )
    return normalized, policy


def adaptive_pytest_worker_count(env: Mapping[str, str]) -> int:
    """Return an explicit worker override or the current adaptive count."""
    configured = env.get("POLYLOGUE_PYTEST_WORKERS", "").strip()
    if configured:
        try:
            return max(0, int(configured))
        except ValueError as exc:
            raise PytestResourceError(f"invalid POLYLOGUE_PYTEST_WORKERS={configured!r}") from exc
    return adaptive_pytest_runtime_policy().workers


def _is_tmpfs_dir(path: Path) -> bool:
    """True when ``path`` is a directory with the sticky bit set (tmpfs mounts)."""
    try:
        return path.is_dir() and bool(stat.S_ISVTX & path.stat().st_mode)
    except OSError:
        return False


def _existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path("/")


def _headroom_kb(path: Path) -> int | None:
    """Free space (KiB) at the deepest existing ancestor of ``path``."""
    usage = _fs_usage(_existing_ancestor(path))
    return usage.get("free_kb") if usage is not None else None


def pytest_basetemp_min_free_kb(env: Mapping[str, str]) -> int:
    """Required free-space headroom (KiB) a basetemp candidate must clear."""
    raw = env.get(PYTEST_BASETEMP_MIN_FREE_MB_ENV, "").strip()
    if raw:
        with contextlib.suppress(ValueError):
            return max(0, int(raw)) * 1024
    return DEFAULT_PYTEST_BASETEMP_MIN_FREE_MB * 1024


def pytest_basetemp_required_kb(env: Mapping[str, str]) -> int | None:
    """Return declared peak basetemp demand for this run, when known."""
    raw = env.get(PYTEST_BASETEMP_REQUIRED_MB_ENV, "").strip()
    if not raw:
        return None
    try:
        value_mb = int(raw)
        if value_mb < 0:
            raise PytestResourceError(f"invalid {PYTEST_BASETEMP_REQUIRED_MB_ENV}={raw!r}")
        return value_mb * 1024
    except ValueError as exc:
        raise PytestResourceError(f"invalid {PYTEST_BASETEMP_REQUIRED_MB_ENV}={raw!r}") from exc


def pytest_basetemp_known_roots(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Every root this or a past policy revision could have placed a basetemp under.

    Used for stale-basetemp sweeps so leftovers survive a placement-policy
    change instead of leaking silently on a root the current run no longer
    chooses.
    """
    roots = [PYTEST_TMPFS_ROOT, DEFAULT_PYTEST_BASETEMP_ROOT, _CLOUD_PYTEST_BASETEMP_ROOT]
    if env is not None:
        configured = env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
        if configured:
            roots.append(Path(configured))
    return tuple(dict.fromkeys(root for root in roots if root.is_dir()))


def _describe_candidate(path: Path, label: str, free_kb: int | None, min_free_kb: int) -> str:
    if free_kb is None:
        return f"{path} ({label}): free space unknown (path unreachable)"
    return f"{path} ({label}): {free_kb / 1024:.0f} MiB free, need >= {min_free_kb / 1024:.0f} MiB"


def _basetemp_refusal(checked: list[str], min_free_kb: int) -> PytestResourceError:
    lines = "\n".join(f"  - {line}" for line in checked)
    return PytestResourceError(
        "no pytest basetemp location has enough free space "
        f"(headroom requirement: {min_free_kb / 1024:.0f} MiB, set via "
        f"{PYTEST_BASETEMP_MIN_FREE_MB_ENV} to override):\n"
        f"{lines}\n"
        "Free space by removing stale runs, e.g.:\n"
        "  rm -rf /dev/shm/pytest-polylogue-* /realm/tmp/polylogue-pytest/pytest-polylogue-* "
        "/tmp/polylogue-pytest/pytest-polylogue-*\n"
        "(never remove *-seeded-* directories while another run may still be building or using them)"
    )


def resolve_pytest_basetemp_root(
    env: Mapping[str, str], *, rejected_candidates: tuple[str, ...] = ()
) -> tuple[Path, str]:
    """Pick the ONE basetemp root pytest will use this run.

    Single resolution order, shared by ``tests/conftest.py`` (direct pytest
    invocations) and the ``devtools test``/``devtools verify`` preflight
    (subprocess invocations), so there is exactly one placement policy instead
    of two that can silently disagree:

    1. ``POLYLOGUE_PYTEST_BASETEMP_ROOT``, if explicitly set. A known cloud
       sandbox default leaking onto a workstation with ``/realm`` mounted is
       stripped first (:func:`normalize_pytest_basetemp_env`); anything left
       is a genuine override and is still headroom-checked, never silently
       downgraded.
    2. ``/dev/shm`` (tmpfs) when ``POLYLOGUE_PYTEST_TMPFS`` is not disabled
       and it has enough free space — fast, and the deliberate default.
    3. ``/realm/tmp/polylogue-pytest`` (NVMe scratch) when ``/realm/tmp`` is
       mounted and has enough free space.
    4. ``/tmp/polylogue-pytest`` — only reachable when ``/realm/tmp`` is not
       mounted at all (a real cloud sandbox), never as a low-space fallback
       on a workstation where a small ``/tmp`` tmpfs is the whole problem.

    Raises :class:`PytestResourceError` naming every candidate checked, its
    free space, and the requirement, instead of letting an unrelated command
    fail three layers away with a bare ``OSError: [Errno 28]``.
    """
    required_kb = pytest_basetemp_required_kb(env)
    min_free_kb = max(pytest_basetemp_min_free_kb(env), required_kb or 0)
    normalized = normalize_pytest_basetemp_env(env)
    checked = list(rejected_candidates)

    configured = normalized.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if configured:
        root = Path(configured)
        free_kb = _headroom_kb(root)
        configured_required_kb = min_free_kb
        if _is_beneath(root, PYTEST_TMPFS_ROOT) and normalized.get("POLYLOGUE_PYTEST_TMPFS") == "1":
            budget_kb = pytest_tmpfs_budget_kb(normalized)
            headroom_kb = pytest_basetemp_min_free_kb(normalized)
            configured_required_kb = headroom_kb + max(required_kb or 0, budget_kb or 0)
        if free_kb is not None and free_kb >= configured_required_kb:
            return root, "configured"
        checked.append(_describe_candidate(root, "configured", free_kb, configured_required_kb))
        raise _basetemp_refusal(checked, configured_required_kb)

    if normalized.get("POLYLOGUE_PYTEST_TMPFS", "1") != "0":
        shm = PYTEST_TMPFS_ROOT
        if _is_tmpfs_dir(shm):
            free_kb = _headroom_kb(shm)
            budget_kb = pytest_tmpfs_budget_kb(normalized)
            # The budget is a cap for the run, not the amount of free space we
            # can safely consume. Keep the normal headroom in addition to the
            # cap so a bounded run cannot fill /dev/shm and strand unrelated
            # processes before the supervisor notices.
            headroom_kb = pytest_basetemp_min_free_kb(normalized)
            declared_demand_kb = max(required_kb or 0, budget_kb or 0)
            tmpfs_required_kb = headroom_kb + declared_demand_kb
            demand_fits_budget = free_kb is not None and free_kb >= tmpfs_required_kb
            if demand_fits_budget:
                return shm, "tmpfs opt-in"
            checked.append(_describe_candidate(shm, "tmpfs opt-in", free_kb, tmpfs_required_kb))

    if DEFAULT_PYTEST_BASETEMP_ROOT.parent.is_dir():
        free_kb = _headroom_kb(DEFAULT_PYTEST_BASETEMP_ROOT)
        if free_kb is not None and free_kb >= min_free_kb:
            return DEFAULT_PYTEST_BASETEMP_ROOT, "scratch"
        checked.append(_describe_candidate(DEFAULT_PYTEST_BASETEMP_ROOT, "scratch", free_kb, min_free_kb))
    else:
        # /realm is not mounted at all: a genuine cloud sandbox, where the
        # small-tmpfs-/tmp problem this policy exists to avoid does not apply.
        free_kb = _headroom_kb(_CLOUD_PYTEST_BASETEMP_ROOT)
        if free_kb is not None and free_kb >= min_free_kb:
            return _CLOUD_PYTEST_BASETEMP_ROOT, "disk fallback"
        checked.append(_describe_candidate(_CLOUD_PYTEST_BASETEMP_ROOT, "disk fallback", free_kb, min_free_kb))

    raise _basetemp_refusal(checked, min_free_kb)


def pytest_basetemp_path(*, root: Path, run_id: str, env: dict[str, str]) -> Path:
    """Path this run's basetemp lives (or lived) at, for monitoring/cleanup.

    Called both before a pytest subprocess starts (resource sampling) and
    after it exits (own-basetemp cleanup) — by that point the space preflight
    in :func:`apply_managed_pytest_runtime_policy` already ran, so a headroom
    refusal here would just be noise for a monitoring/cleanup path. Fall back
    to the top placement candidate, ignoring headroom, rather than raising.
    """
    explicit = env.get(PYTEST_EXPLICIT_BASETEMP_ENV)
    if explicit:
        return Path(explicit)
    try:
        scratch_root, _label = resolve_pytest_basetemp_root(env)
    except PytestResourceError:
        configured = env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
        if configured:
            scratch_root = Path(configured)
        elif env.get("POLYLOGUE_PYTEST_TMPFS", "1") != "0" and _is_tmpfs_dir(PYTEST_TMPFS_ROOT):
            scratch_root = PYTEST_TMPFS_ROOT
        elif DEFAULT_PYTEST_BASETEMP_ROOT.parent.is_dir():
            scratch_root = DEFAULT_PYTEST_BASETEMP_ROOT
        else:
            scratch_root = _CLOUD_PYTEST_BASETEMP_ROOT
    return scratch_root / f"pytest-polylogue-{checkout_hash(root)}-{run_id}"


def pytest_tmpfs_budget_kb(env: Mapping[str, str]) -> int | None:
    """Return the bounded per-run tmpfs budget shared by all pytest workers."""
    explicit = env.get(PYTEST_EXPLICIT_BASETEMP_ENV)
    configured_root = explicit or env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    configured_tmpfs = configured_root is not None and _is_beneath(Path(configured_root), PYTEST_TMPFS_ROOT)
    if env.get("POLYLOGUE_PYTEST_TMPFS") != "1" or (configured_root is not None and not configured_tmpfs):
        return None
    raw = env.get(PYTEST_TMPFS_MAX_MB_ENV, str(DEFAULT_PYTEST_TMPFS_MAX_MB))
    with contextlib.suppress(ValueError):
        requested_mb = int(raw)
        bounded_mb = max(64, min(requested_mb, MAX_PYTEST_TMPFS_MAX_MB))
        return bounded_mb * 1024
    return DEFAULT_PYTEST_TMPFS_MAX_MB * 1024


def cleanup_managed_pytest_basetemp(*, root: Path, run_id: str, env: dict[str, str]) -> Path | None:
    """Remove the managed per-run pytest basetemp after pytest has exited.

    The pytest sessionfinish hook intentionally does not delete xdist
    basetemps while workers/reporters may still be flushing.  The supervisor is
    outside that teardown window, so it can reclaim tmpfs-backed broad-run
    basetemps immediately instead of waiting for the next pytest startup sweep.
    """

    if env.get(PYTEST_EXPLICIT_BASETEMP_ENV):
        return None
    basetemp = pytest_basetemp_path(root=root, run_id=run_id, env=env)
    if not basetemp.name.startswith("pytest-polylogue-") or "-seeded-" in basetemp.name:
        return None
    claim_lock = _try_acquire_pytest_basetemp_claim_lock(basetemp)
    if claim_lock is None:
        # A successor with the same inherited run id owns this path.  Leave
        # both its claim and its fixture tree for that invocation to finish.
        return None
    try:
        owner_alive = managed_pytest_basetemp_owner_alive(basetemp)
        if owner_alive is True:
            return None
        # A serial pytest child may already have reclaimed this exact run-owned
        # directory in sessionfinish. That is a completed cleanup, not an absent
        # receipt for the durable summary to misclassify.
        if not basetemp.exists():
            if owner_alive is False:
                clear_managed_pytest_basetemp_claim(basetemp)
            return basetemp
        # Reclaim only a positively claimed tree whose owner is confirmed dead.
        # An unknown claim/tree may be caller-owned or belong to a newer runner.
        if owner_alive is not False:
            return None
        with contextlib.suppress(OSError):
            if basetemp.exists():
                shutil.rmtree(basetemp)
                if not basetemp.exists():
                    clear_managed_pytest_basetemp_claim(basetemp)
                    return basetemp
    finally:
        with contextlib.suppress(OSError):
            claim_lock.close()
    return None


def _pytest_event_worker_ids(events_dir: Path | None) -> dict[int, str]:
    """Recover xdist worker identities emitted after process exec.

    ``PYTEST_XDIST_WORKER`` is not guaranteed to appear in ``/proc``'s
    exec-time environment.  The progress plugin emits a session-start event
    from inside each worker, which is the authoritative identity for the
    supervisor sampler.
    """
    if events_dir is None or not events_dir.is_dir():
        return {}
    identities: dict[int, str] = {}
    for path in events_dir.glob("*.jsonl"):
        with contextlib.suppress(OSError, UnicodeDecodeError):
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(line)
                    pid = payload.get("pid")
                    worker_id = payload.get("worker_id")
                    if isinstance(pid, int) and isinstance(worker_id, str) and worker_id != "controller":
                        identities[pid] = worker_id
    return identities


class ResourceSampler:
    """Samples host and process-tree resources for one subprocess tree."""

    def __init__(self, *, root_pid: int, run_id: str, root: Path, env: dict[str, str], output_path: Path) -> None:
        self.root_pid = root_pid
        self.run_id = run_id
        self.root = root
        self.env = env
        self.output_path = output_path
        self.events_dir = Path(env["POLYLOGUE_PYTEST_EVENTS_DIR"]) if env.get("POLYLOGUE_PYTEST_EVENTS_DIR") else None
        self.sample_count = 0
        self.peak_rss_kb = 0
        self.peak_pss_kb: int | None = None
        self.peak_anon_pss_kb: int | None = None
        self.peak_file_pss_kb: int | None = None
        self.peak_swap_pss_kb: int | None = None
        self.peak_process_count = 0
        self.last_sample: dict[str, Any] | None = None
        self.first_sample: dict[str, Any] | None = None
        self._process_io_high_water: dict[str, dict[str, int]] = {}
        self._basetemp = pytest_basetemp_path(root=root, run_id=run_id, env=env)
        self._basetemp_size_interval_s = _basetemp_size_sample_interval_s(env)
        self._last_basetemp_size_sample_at: float | None = None
        self._last_basetemp_size_kb: int | None = None
        self._last_basetemp_allocated_kb: int | None = None

    def _sample_basetemp_sizes(self, *, event: str) -> tuple[int | None, int | None]:
        """Return basetemp size without recursively walking it every sample."""
        if self._basetemp_size_interval_s <= 0:
            return None, None
        now = time.monotonic()
        should_sample = (
            self._last_basetemp_size_sample_at is None
            or self._last_basetemp_size_kb is None
            or event in {"started", "finished"}
            or now - self._last_basetemp_size_sample_at >= self._basetemp_size_interval_s
        )
        if should_sample:
            self._last_basetemp_size_kb, self._last_basetemp_allocated_kb = _dir_usage_kb(self._basetemp)
            self._last_basetemp_size_sample_at = now
        return self._last_basetemp_size_kb, self._last_basetemp_allocated_kb

    def sample(self, *, event: str) -> dict[str, Any]:
        pids = process_tree(self.root_pid)
        processes: list[dict[str, Any]] = []
        total_rss = 0
        total_pss = 0
        pss_available = False
        total_anon_pss = 0
        anon_pss_available = False
        total_file_pss = 0
        file_pss_available = False
        total_swap_pss = 0
        swap_pss_available = False
        total_cpu = 0.0
        xdist_worker_count = 0
        xdist_uninterruptible_count = 0
        event_worker_ids = _pytest_event_worker_ids(self.events_dir)
        for pid in pids:
            status = _status_values(pid)
            rss = int(status.get("rss_kb") or 0)
            smaps = _smaps_rollup_kb(pid)
            pss = smaps.get("Pss")
            anon_pss = smaps.get("Pss_Anon")
            file_pss = smaps.get("Pss_File")
            swap_pss = smaps.get("SwapPss")
            cpu = _cpu_seconds(pid)
            process_identity = _process_identity(pid)
            worker_id = _process_environ_value(pid, "PYTEST_XDIST_WORKER") or event_worker_ids.get(pid)
            if worker_id is not None:
                xdist_worker_count += 1
                if str(status.get("state") or "").startswith("D"):
                    xdist_uninterruptible_count += 1
            process_io = _process_io_bytes(pid)
            io_high_water = self._process_io_high_water.setdefault(process_identity, {})
            for key, value in process_io.items():
                io_high_water[key] = max(io_high_water.get(key, 0), value)
            total_rss += rss
            if pss is not None:
                pss_available = True
                total_pss += pss
            if anon_pss is not None:
                anon_pss_available = True
                total_anon_pss += anon_pss
            if file_pss is not None:
                file_pss_available = True
                total_file_pss += file_pss
            if swap_pss is not None:
                swap_pss_available = True
                total_swap_pss += swap_pss
            if cpu is not None:
                total_cpu += cpu
            processes.append(
                {
                    "pid": pid,
                    "process_identity": process_identity,
                    "xdist_worker_id": worker_id,
                    "state": status.get("state"),
                    "rss_kb": rss,
                    "pss_kb": pss,
                    "anon_pss_kb": anon_pss,
                    "file_pss_kb": file_pss,
                    "swap_pss_kb": swap_pss,
                    **process_io,
                    "cpu_s": cpu,
                }
            )
        cumulative_io = {
            key: sum(values.get(key, 0) for values in self._process_io_high_water.values())
            for key in ("read_bytes", "write_bytes", "cancelled_write_bytes")
        }
        meminfo = _meminfo()
        cgroup_path = _cgroup_path(self.root_pid)
        cgroup_io = _cgroup_io_bytes(cgroup_path)
        basetemp_logical_kb, basetemp_allocated_kb = self._sample_basetemp_sizes(event=event)
        sample: dict[str, Any] = {
            "updated_at": utc_now(),
            "event": event,
            "root_pid": self.root_pid,
            "process_count": len(pids),
            "tree_rss_kb": total_rss,
            "tree_pss_kb": total_pss if pss_available else None,
            "tree_anon_pss_kb": total_anon_pss if anon_pss_available else None,
            "tree_file_pss_kb": total_file_pss if file_pss_available else None,
            "tree_swap_pss_kb": total_swap_pss if swap_pss_available else None,
            "tree_read_bytes": cumulative_io["read_bytes"],
            "tree_write_bytes": cumulative_io["write_bytes"],
            "tree_cancelled_write_bytes": cumulative_io["cancelled_write_bytes"],
            "tree_cpu_s": round(total_cpu, 4),
            "xdist_worker_count": xdist_worker_count,
            "xdist_uninterruptible_count": xdist_uninterruptible_count,
            "all_xdist_workers_uninterruptible": (
                xdist_worker_count > 0 and xdist_worker_count == xdist_uninterruptible_count
            ),
            "host_mem_available_kb": meminfo.get("MemAvailable"),
            "host_mem_total_kb": meminfo.get("MemTotal"),
            "host_swap_free_kb": meminfo.get("SwapFree"),
            "host_swap_total_kb": meminfo.get("SwapTotal"),
            "cgroup_path": cgroup_path,
            "cgroup_memory_current_bytes": _cgroup_int(cgroup_path, "memory.current"),
            "cgroup_memory_peak_bytes": _cgroup_int(cgroup_path, "memory.peak"),
            "cgroup_memory_swap_current_bytes": _cgroup_int(cgroup_path, "memory.swap.current"),
            "cgroup_read_bytes": cgroup_io.get("rbytes") if cgroup_io else None,
            "cgroup_write_bytes": cgroup_io.get("wbytes") if cgroup_io else None,
            "pressure_cpu": _pressure("cpu"),
            "pressure_io": _pressure("io"),
            "pressure_memory": _pressure("memory"),
            "shm": _fs_usage(Path("/dev/shm")),
            "basetemp": str(self._basetemp),
            "basetemp_size_kb": basetemp_logical_kb,
            "basetemp_allocated_kb": basetemp_allocated_kb,
            "top_processes": sorted(processes, key=lambda row: int(row.get("rss_kb") or 0), reverse=True)[:8],
        }
        self.sample_count += 1
        self.peak_rss_kb = max(self.peak_rss_kb, total_rss)
        if pss_available:
            self.peak_pss_kb = max(self.peak_pss_kb or 0, total_pss)
        if anon_pss_available:
            self.peak_anon_pss_kb = max(self.peak_anon_pss_kb or 0, total_anon_pss)
        if file_pss_available:
            self.peak_file_pss_kb = max(self.peak_file_pss_kb or 0, total_file_pss)
        if swap_pss_available:
            self.peak_swap_pss_kb = max(self.peak_swap_pss_kb or 0, total_swap_pss)
        self.peak_process_count = max(self.peak_process_count, len(pids))
        if self.first_sample is None:
            self.first_sample = sample
        self.last_sample = sample
        _append_jsonl(self.output_path, sample)
        return sample

    def summary(self) -> dict[str, Any]:
        first = self.first_sample or {}
        last = self.last_sample or {}

        def delta(key: str) -> int | None:
            first_value = first.get(key)
            last_value = last.get(key)
            if not isinstance(first_value, int) or not isinstance(last_value, int):
                return None
            return max(0, last_value - first_value)

        return {
            "resource_sample_count": self.sample_count,
            "peak_tree_rss_kb": self.peak_rss_kb,
            "peak_tree_rss_mb": round(self.peak_rss_kb / 1024, 1),
            "peak_tree_pss_kb": self.peak_pss_kb,
            "peak_tree_pss_mb": round(self.peak_pss_kb / 1024, 1) if self.peak_pss_kb is not None else None,
            "peak_tree_anon_pss_kb": self.peak_anon_pss_kb,
            "peak_tree_file_pss_kb": self.peak_file_pss_kb,
            "peak_tree_swap_pss_kb": self.peak_swap_pss_kb,
            "start_tree_swap_pss_kb": first.get("tree_swap_pss_kb"),
            "final_tree_swap_pss_kb": last.get("tree_swap_pss_kb"),
            "tree_swap_pss_delta_kb": delta("tree_swap_pss_kb"),
            "tree_read_bytes": last.get("tree_read_bytes"),
            "tree_write_bytes": last.get("tree_write_bytes"),
            "tree_cancelled_write_bytes": last.get("tree_cancelled_write_bytes"),
            "tree_read_bytes_delta": delta("tree_read_bytes"),
            "tree_write_bytes_delta": delta("tree_write_bytes"),
            "tree_cancelled_write_bytes_delta": delta("tree_cancelled_write_bytes"),
            "cgroup_path": first.get("cgroup_path") or last.get("cgroup_path"),
            "peak_cgroup_memory_bytes": max(
                (
                    value
                    for value in (first.get("cgroup_memory_peak_bytes"), last.get("cgroup_memory_peak_bytes"))
                    if isinstance(value, int)
                ),
                default=None,
            ),
            "final_cgroup_memory_current_bytes": last.get("cgroup_memory_current_bytes"),
            "final_cgroup_memory_swap_current_bytes": last.get("cgroup_memory_swap_current_bytes"),
            "cgroup_read_bytes_delta": delta("cgroup_read_bytes"),
            "cgroup_write_bytes_delta": delta("cgroup_write_bytes"),
            "peak_process_count": self.peak_process_count,
            "last_xdist_worker_count": last.get("xdist_worker_count"),
            "last_xdist_uninterruptible_count": last.get("xdist_uninterruptible_count"),
            "last_all_xdist_workers_uninterruptible": last.get("all_xdist_workers_uninterruptible"),
            "first_resource_sample": self.first_sample,
            "last_resource_sample": self.last_sample,
        }


def _basetemp_size_sample_interval_s(env: dict[str, str]) -> float:
    raw = env.get(BASETEMP_SIZE_SAMPLE_INTERVAL_ENV)
    if raw is None or raw.strip() == "":
        if pytest_tmpfs_budget_kb(env) is not None:
            return DEFAULT_TMPFS_SIZE_SAMPLE_INTERVAL_S
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


def xdist_uninterruptible_stall_reason(
    sample: Mapping[str, Any], *, started_at: float | None, now: float, timeout_s: float
) -> str | None:
    """Classify an xdist process tree wedged in uninterruptible I/O sleep.

    A heartbeat or controller output is not proof that tests are progressing:
    all workers can be blocked in SQLite/filesystem I/O while the controller
    remains alive.  This helper deliberately requires every observed xdist
    worker to be in ``D`` state and a complete stall interval before returning
    a termination reason.
    """
    if sample.get("all_xdist_workers_uninterruptible") is not True:
        return None
    if started_at is None or timeout_s <= 0 or now - started_at < timeout_s:
        return None
    worker_count = sample.get("xdist_worker_count")
    return (
        "pytest xdist workers remained in uninterruptible I/O sleep for "
        f"{now - started_at:.0f}s ({worker_count} workers; likely SQLite/filesystem stall)"
    )
