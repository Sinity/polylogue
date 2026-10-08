"""Typed local receipts for semantic verification commands.

AgentCTL owns jobs, deadlines, process trees, temporary storage, and checkout
lifecycle. This module records only the verifier facts Polylogue can state:
what ran, the decoded pytest outcome, and the resulting
scope.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os as os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devtools.agent_env import runtime_env
from devtools.checkout_identity import checkout_identity
from devtools.pytest_evidence import evaluate_pytest_evidence
from devtools.pytest_suite_cost_plugin import (
    RUN_RECEIPT_NAME as SUITE_COST_RUN_RECEIPT_NAME,
)
from devtools.pytest_suite_cost_plugin import SUITE_COST_DIR_ENV, summarize_step_receipts
from devtools.testmon_provision import TESTMON_DATA_RELPATH
from devtools.verification_result import INTERRUPTED_DIAGNOSES


def environment_fingerprint(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Identity that separates a product failure from a poisoned environment."""
    root = root or Path.cwd()
    environ = os.environ if env is None else env
    executable = Path(sys.executable).resolve()
    return {
        "checkout_root": str(root.absolute()),
        "python_executable": str(executable),
        "python_environment": str(Path(environ.get("VIRTUAL_ENV", executable.parent)).absolute()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "harness": environ.get("POLYLOGUE_VERIFY_HARNESS", "devtools"),
    }


VERIFY_CACHE = Path(".cache/verify")
VERIFY_RUNS_DIR = VERIFY_CACHE / "runs"
VERIFY_HISTORY_PATH = VERIFY_CACHE / "history.jsonl"
VERIFY_HISTORY_PATH_ENV = "POLYLOGUE_VERIFY_HISTORY_PATH"
# Established managed addresses must also refuse writes from shells opened
# before the operator environment started declaring additional retirements.
RETIRED_MANAGED_HISTORY_PATHS = (
    "/realm/activity/dev/polylogue/verify-history.jsonl",
    "/realm/activity/development/polylogue/verify-history.jsonl",
    "/realm/projects/polylogue/source/development/verify-history.jsonl",
)
VERIFY_EVIDENCE_PATH = VERIFY_CACHE / "evidence.jsonl"
VERIFY_EVIDENCE_PATH_ENV = "POLYLOGUE_VERIFICATION_EVIDENCE_PATH"
CURRENT_RUN_PATH = VERIFY_CACHE / "current-run.json"
CURRENT_STATISTICS_PATH = VERIFY_CACHE / "current-pytest-statistics.json"
CURRENT_EVENTS_DIR = VERIFY_CACHE / "current-pytest-events"
PYTEST_CANONICAL_REPORT_NAME = "pytest-report.json"


def _validate_history_write_path(path: Path, env: Mapping[str, str] | None = None) -> Path:
    """Refuse established managed retirements and additional configured paths."""
    environ = os.environ if env is None else env
    absolute = Path(os.path.abspath(path.expanduser()))
    retired = environ.get("POLYLOGUE_RETIRED_VERIFY_HISTORY_PATHS", "")
    for value in (*RETIRED_MANAGED_HISTORY_PATHS, *retired.split(os.pathsep)):
        if not value:
            continue
        old = Path(value).expanduser()
        if not old.is_absolute():
            raise ValueError("POLYLOGUE_RETIRED_VERIFY_HISTORY_PATHS requires absolute paths")
        if absolute == Path(os.path.abspath(old)):
            raise ValueError(
                "verification history destination is retired; set POLYLOGUE_VERIFY_HISTORY_PATH "
                "to the current owner-declared destination or an unrelated custom path"
            )
    return path


def verify_history_path(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    """Resolve shared verification history from explicit config or XDG state."""
    environ = os.environ if env is None else env
    configured = environ.get(VERIFY_HISTORY_PATH_ENV)
    if configured:
        path = Path(configured).expanduser()
        return _validate_history_write_path(path if path.is_absolute() else (root or Path.cwd()) / path, environ)
    configured_state_home = environ.get("XDG_STATE_HOME")
    state_home = (
        Path(configured_state_home).expanduser()
        if configured_state_home
        else Path(environ.get("HOME", "~")).expanduser() / ".local" / "state"
    )
    if not state_home.is_absolute():
        state_home = Path(environ.get("HOME", "~")).expanduser() / ".local" / "state"
    return state_home / "polylogue" / "verify" / "history.jsonl"


SUCCESSFUL_VERIFY_DETAIL_LIMIT = 8
FAILED_VERIFY_DETAIL_LIMIT = 12
FAILED_VERIFY_DETAIL_MAX_AGE_S = 7 * 24 * 60 * 60
FAILED_VERIFY_DETAIL_MAX_BYTES = 64 * 1024 * 1024
_RETENTION_LOCK_NAME = ".retention.lock"
_DETAIL_NODE_BUDGET = 100_000
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_APPEND_LOCK = threading.Lock()


@dataclass
class _NodeBudget:
    remaining: int

    def consume(self) -> bool:
        self.remaining -= 1
        return self.remaining >= 0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(*, tier: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_tier = re.sub(r"[^A-Za-z0-9_.-]+", "-", tier).strip("-") or "verify"
    return f"{stamp}-{safe_tier}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _absolute_path(path: Path) -> Path:
    """Make a lexical absolute path without resolving any symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _open_pinned_dir(path: Path) -> int:
    """Open every directory component with O_NOFOLLOW."""
    absolute = _absolute_path(path)
    parts = tuple(part for part in absolute.parts if part not in ("", os.sep))
    fd = os.open(os.sep, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    try:
        for part in parts:
            next_fd = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _mkdir_pinned(path: Path, mode: int = 0o700) -> None:
    """Create directory components without following an existing symlink."""
    absolute = _absolute_path(path)
    parts = tuple(part for part in absolute.parts if part not in ("", os.sep))
    fd = os.open(os.sep, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    try:
        for part in parts:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, mode, dir_fd=fd)
            next_fd = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = _open_pinned_dir(path)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_retention_lock(verify_dir: Path, *, nonblocking: bool) -> int | None:
    directory_fd = _open_pinned_dir(verify_dir)
    try:
        fd = os.open(
            _RETENTION_LOCK_NAME,
            os.O_RDWR | os.O_CREAT | _O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except BaseException:
        os.close(directory_fd)
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    except BlockingIOError:
        os.close(fd)
        os.close(directory_fd)
        return None
    except BaseException:
        os.close(fd)
        os.close(directory_fd)
        raise
    try:
        named = os.stat(_RETENTION_LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
        opened = os.fstat(fd)
        if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("verification retention lock pathname was replaced while locked")
    except BaseException:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise
    finally:
        os.close(directory_fd)
    return fd


def _close_retention_lock(fd: int) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _read_only_git_env() -> dict[str, str]:
    return {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            env=_read_only_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        # A tree whose status cannot be read is not a proven-clean tree.
        return True
    return bool((result.stdout or "").strip())


def git_head(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=cwd, env=_read_only_git_env()
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def git_worktree_content_sha256(cwd: Path, *, paths: Sequence[bytes] | None = None) -> str | None:
    """Hash Git-visible worktree paths and their execution-time content."""
    try:
        if paths is None:
            listed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                capture_output=True,
                timeout=30,
                cwd=cwd,
                env=_read_only_git_env(),
            )
            if listed.returncode != 0:
                return None
            paths = listed.stdout.split(b"\0")
        digest = hashlib.sha256()
        paths = sorted(set(paths) - {b""})
        for raw_path in paths:
            path = cwd / os.fsdecode(raw_path)
            digest.update(len(raw_path).to_bytes(8, "big"))
            digest.update(raw_path)
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                digest.update(b"missing\0")
                continue
            if stat.S_ISLNK(mode):
                digest.update(b"symlink\0")
                target = os.fsencode(os.readlink(path))
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            elif stat.S_ISREG(mode):
                digest.update(b"executable\0" if mode & 0o111 else b"file\0")
                file_digest = hashlib.sha256()
                size = 0
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        file_digest.update(chunk)
                        size += len(chunk)
                digest.update(size.to_bytes(8, "big"))
                digest.update(file_digest.digest())
            else:
                return None
        return digest.hexdigest()
    except (OSError, subprocess.TimeoutExpired):
        return None


@dataclass(frozen=True)
class PytestStepArtifacts:
    step_id: str
    step_dir: Path
    output_path: Path
    progress_path: Path
    events_dir: Path
    events_merged_path: Path
    selection_path: Path
    summary_path: Path
    statistics_path: Path


class VerifyRun:
    """Filesystem-backed receipt for one local semantic verification run."""

    def __init__(
        self,
        *,
        tier: str,
        argv: list[str],
        git_head: str | None,
        root: Path | None = None,
        mirror_current: bool = True,
        agentctl_operation: str | None = None,
    ) -> None:
        self.root = root or Path.cwd()
        self.mirror_current = mirror_current
        self.run_id = make_run_id(tier=tier)
        self.run_dir = self.root / VERIFY_RUNS_DIR / self.run_id
        self._payload: dict[str, Any] = {
            "run_id": self.run_id,
            "tier": tier,
            "argv": list(argv),
            "git_head": git_head,
            # A cited receipt names what it tested: a run on the default
            # branch tested the base, not a change (devtools/checkout_identity.py).
            "git_branch": checkout_identity(self.root).branch,
            "git_dirty": git_dirty(self.root),
            "started_at": utc_now(),
            "status": "running",
            "steps": [],
            "artifact_dir": str(VERIFY_RUNS_DIR / self.run_id),
        }
        if tier == "focused-test":
            self._payload["git_worktree_content_sha256"] = None
            self._payload["worktree_capture_source"] = None
        # These are opaque execution identities.  They are provenance only;
        # semantic status is still decided by this verifier.
        if agentctl_operation is not None:
            for field, variable in (
                ("agentctl_job_id", "AGENTCTL_JOB_ID"),
                ("agentctl_correlation_id", "AGENTCTL_CORRELATION_ID"),
            ):
                value = runtime_env(variable)
                if value:
                    self._payload[field] = value
        # Kept on every receipt so later classification does not depend on a
        # live interpreter or a reconstructed shell environment.
        self._payload["environment_fingerprint"] = environment_fingerprint(root=self.root)
        # Static gates start and finish steps from several threads at once.
        self._lock = threading.RLock()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write()

    @property
    def relative_run_dir(self) -> Path:
        return VERIFY_RUNS_DIR / self.run_id

    def write(self) -> None:
        with self._lock:
            _write_json(self.run_dir / "run.json", self._payload)
            if self.mirror_current:
                _write_json(self.root / CURRENT_RUN_PATH, self._payload)

    def declare_workload(self, spec: Mapping[str, Any]) -> None:
        """Persist the complete intended plan before any step is admitted or run.

        The terminal ``workload_receipt`` binds observations to this same
        spec, so an interrupted, refused, or abandoned run still names the plan
        it was executing rather than only the steps it reached.
        """
        self._payload["workload_spec"] = dict(spec)
        self.write()

    @property
    def recorded_git_dirty(self) -> bool:
        """The start-of-run dirty state; ``git_dirty`` fails closed to ``True``."""
        return bool(self._payload.get("git_dirty", True))

    def record_execution_environment_key(self, key: str) -> None:
        """Bind the run to the caller environment that shaped its execution."""
        self._payload["execution_environment_key"] = key

    def record_execution_worktree(self, provenance: Mapping[str, Any]) -> None:
        for key in ("git_head", "git_branch", "git_dirty", "git_worktree_content_sha256"):
            self._payload[key] = provenance.get(key)
        self._payload["worktree_capture_source"] = provenance.get("capture_source")

    def record_selection(
        self,
        *,
        selection_mode: str,
        graph_status: str,
        graph_reason: str,
        full_rerun_cause: str | None = None,
        graph_recorded_tests: int | None = None,
        graph_source_dependencies: int | None = None,
        seed_source: str | None = None,
        seed_source_mtime_ns: int | None = None,
        selection_reason: str | None = None,
        selected_count: int | None = None,
        estimated_seconds: float | None = None,
        admission: Mapping[str, Any] | None = None,
    ) -> None:
        self._payload["testmon_selection"] = {
            "selection_mode": selection_mode,
            "graph_status": graph_status,
            "graph_reason": graph_reason,
            "full_rerun_cause": full_rerun_cause,
            "graph_recorded_tests": graph_recorded_tests,
            "graph_source_dependencies": graph_source_dependencies,
            "seed_source": seed_source,
            "seed_source_mtime_ns": seed_source_mtime_ns,
            # Why the selection is bounded or empty; what a reader of a run
            # with no pytest step needs to accept it.
            "selection_reason": selection_reason,
            "selected_count": selected_count,
            "estimated_seconds": estimated_seconds,
            "admission": dict(admission) if admission is not None else None,
        }
        self.write()

    def start_step(self, *, label: str, cmd: list[str]) -> PytestStepArtifacts:
        with self._lock:
            return self._start_step(label=label, cmd=cmd)

    def _start_step(self, *, label: str, cmd: list[str]) -> PytestStepArtifacts:
        step_id = f"{len(self._payload['steps']) + 1:02d}-{_slug(label)}"
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        artifacts = PytestStepArtifacts(
            step_id,
            step_dir,
            step_dir / "output.log",
            step_dir / "progress.json",
            step_dir / "events",
            step_dir / "events.jsonl",
            step_dir / "selection.json",
            step_dir / "summary.json",
            step_dir / "statistics.json",
        )
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

    def finish_step(self, *, step_id: str, result: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            return self._finish_step(step_id=step_id, result=result)

    def _finish_step(self, *, step_id: str, result: Mapping[str, Any]) -> dict[str, Any] | None:
        for step in self._payload["steps"]:
            if step.get("step_id") != step_id:
                continue
            finalized = dict(result)
            statistics: dict[str, Any] | None = None
            if str(step.get("name", "")).startswith("pytest"):
                step_dir = self.run_dir / "steps" / step_id
                raw_exit = result.get("exit")
                finalized["process_exit"] = raw_exit
                # Keep the execution reason when pytest could not start or was
                # interrupted, independently of any evidence delivery fault.
                explicit_terminal = result.get("diagnosis") in {
                    "oom_killed",
                    "focused_test_runner_exception",
                    "pytest_interrupted",
                    "pytest_slot_unavailable",
                    "execution_source_unavailable",
                    "verification_interrupted",
                }
                phase = "aggregation"
                try:
                    statistics = aggregate_pytest_statistics(step_dir, command=step.get("cmd", []), step_result=result)
                    phase = "statistics_publication"
                    _write_json(step_dir / "statistics.json", statistics)
                    finalized["statistics_path"] = str(self.relative_run_dir / "steps" / step_id / "statistics.json")
                    if self.mirror_current:
                        phase = "statistics_mirror"
                        shutil.copyfile(step_dir / "statistics.json", self.root / CURRENT_STATISTICS_PATH)
                except (OSError, ValueError, TypeError) as exc:
                    finalized["evidence_error"] = {
                        "phase": phase,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    if raw_exit == 0:
                        finalized["exit"] = 1
                    if not explicit_terminal:
                        finalized["diagnosis"] = "pytest_evidence_unavailable"
                    if statistics is not None:
                        # Measured outcomes remain available, but failed delivery
                        # cannot advertise eligible verification evidence.
                        statistics = {
                            **statistics,
                            "ok": False,
                            "ordinary_eligible": False,
                            "diagnosis": "pytest_evidence_unavailable",
                        }
                else:
                    if not explicit_terminal and raw_exit == 0 and not statistics.get("ok"):
                        finalized["exit"] = 5 if statistics.get("diagnosis") == "pytest_no_tests_selected" else 1
                    if not explicit_terminal:
                        finalized["diagnosis"] = str(statistics.get("diagnosis") or "pytest_no_evidence")
                if statistics is not None:
                    finalized["statistics"] = statistics
            # Publish a terminal step only after its evidence publication outcome
            # is known. The run-local receipt retains faults in either phase.
            step.update(finalized)
            step["finished_at"] = utc_now()
            step["status"] = "success" if finalized.get("exit") == 0 else "failed"
            self.write()
            return dict(step)
        return None

    def finish_interrupted_steps(self, *, exit_code: int, diagnosis: str, termination_reason: str) -> None:
        for step in tuple(self._payload["steps"]):
            if step.get("status") == "running":
                self.finish_step(
                    step_id=str(step["step_id"]),
                    result={
                        "duration_s": None,
                        "exit": exit_code,
                        "diagnosis": diagnosis,
                        "termination_reason": termination_reason,
                    },
                )

    def finish(
        self,
        *,
        exit_code: int,
        duration_s: float,
        diagnosis: str | None = None,
        verification_scope: str | None = None,
        final_git_head: str | None = None,
        pytest_aggregate: Mapping[str, Any] | None = None,
        workload_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._payload.update(
            {
                "finished_at": utc_now(),
                "duration_s": round(duration_s, 2),
                "exit_code": int(exit_code),
                "status": "success" if exit_code == 0 else "failed",
                "final_git_head": final_git_head,
                "final_git_dirty": git_dirty(self.root),
            }
        )
        if diagnosis is not None:
            self._payload["diagnosis"] = diagnosis
        if verification_scope is not None:
            self._payload["verification_scope"] = verification_scope
        if pytest_aggregate is not None:
            self._payload["pytest_aggregate"] = dict(pytest_aggregate)
        else:
            self._payload["pytest_aggregate"] = {
                "selection_mode": "focused" if self._payload["tier"] == "focused-test" else "none"
            }
        if workload_receipt is not None:
            self._payload["workload_receipt"] = dict(workload_receipt)
        # The suite's dominant cost -- how many archive tiers this run built and
        # what it wrote -- is read from the same receipt as its test outcomes,
        # so a change meant to reduce it is compared without a second tool. The
        # per-worker detail stays in each step's suite-cost directory.
        suite_cost = summarize_step_receipts(
            path
            for step in self._payload["steps"]
            if step.get("artifact_dir")
            for path in (
                self.root / str(step["artifact_dir"]) / "suite-cost" / SUITE_COST_RUN_RECEIPT_NAME,
                self.root / str(step["artifact_dir"]) / "suite-cost-rerun" / SUITE_COST_RUN_RECEIPT_NAME,
            )
        )
        if suite_cost is not None:
            self._payload["suite_cost"] = suite_cost
        self.write()
        return dict(self._payload)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "step"


def pytest_step_run_id(run_id: str, step_id: str) -> str:
    index = step_id.split("-", 1)[0]
    return f"{run_id}-s{index}" if index.isdigit() else f"{run_id}-{step_id}"


def env_for_pytest_step(
    env: dict[str, str], *, run: VerifyRun, artifacts: PytestStepArtifacts, testmon: bool = True
) -> dict[str, str]:
    updated = dict(env)
    # testmon opens the datafile directly; without its directory the session
    # dies in an internal error rather than a test result.
    datafile = run.root / TESTMON_DATA_RELPATH
    if testmon:
        datafile.parent.mkdir(parents=True, exist_ok=True)
    # Every managed run records its own archive-construction and write cost,
    # beside the artifacts the rest of the receipt is read from. Set by the
    # step rather than inherited, because the queue reduces the submitting
    # client's environment and an ambient switch never reaches the run.
    suite_cost_dir = env.get(SUITE_COST_DIR_ENV, "").strip() or str(artifacts.step_dir / "suite-cost")
    updated.update(
        {
            SUITE_COST_DIR_ENV: suite_cost_dir,
            "POLYLOGUE_VERIFY_RUN_ID": run.run_id,
            # Scratch archives never need durability; see connection_profile.
            "POLYLOGUE_SQLITE_SYNCHRONOUS": "OFF",
            "POLYLOGUE_PYTEST_RUN_ID": pytest_step_run_id(run.run_id, artifacts.step_id),
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(artifacts.events_dir),
            "POLYLOGUE_PYTEST_EVENTS_PATH": str(artifacts.events_merged_path),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(artifacts.selection_path),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(artifacts.summary_path),
            **({"TESTMON_DATAFILE": str(datafile)} if testmon else {}),
        }
    )
    if not testmon:
        updated.pop("TESTMON_DATAFILE", None)
    return updated


def copy_current_pytest_artifacts(
    root: Path, artifacts: PytestStepArtifacts, *, legacy_paths: Mapping[str, Path]
) -> None:
    for key, target in legacy_paths.items():
        with contextlib.suppress(FileNotFoundError):
            source = getattr(artifacts, key)
            destination = root / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    if artifacts.events_dir.exists():
        destination = root / CURRENT_EVENTS_DIR
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(artifacts.events_dir, destination)


def merge_worker_events(events_dir: Path, merged_path: Path) -> int:
    try:
        with os.scandir(events_dir) as entries:
            paths = sorted(Path(entry.path) for entry in entries if entry.name.endswith(".jsonl"))
    except FileNotFoundError:
        return 0
    if not paths:
        return 0
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_pytest_events(path))
    rows.sort(key=lambda row: str(row.get("updated_at", "")))
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def aggregate_pytest_statistics(
    step_dir: Path, *, command: Sequence[object] = (), step_result: Mapping[str, object] = {}
) -> dict[str, Any]:
    report = _read_terminal_json(step_dir / PYTEST_CANONICAL_REPORT_NAME)
    selection = _read_terminal_json(step_dir / "selection.json") or {}
    # Absence is evaluated by the evidence predicate; an existing unreadable
    # summary is a read failure, not evidence that pytest omitted its summary.
    summary = _read_terminal_json(step_dir / "summary.json")
    outcomes: dict[str, int] = {}
    for test in (report or {}).get("tests", []):
        if isinstance(test, Mapping):
            outcome = str(test.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    event_path = step_dir / "events.jsonl"
    worker_events_dir = step_dir / "events"
    merge_worker_events(worker_events_dir, event_path)
    try:
        event_rows = list(_read_pytest_events(event_path))
    except FileNotFoundError:
        event_rows = []
    if not outcomes:
        for event in event_rows:
            if isinstance(event.get("outcome"), str):
                outcome = event["outcome"]
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
    raw_exit = step_result.get("exit")
    evidence = evaluate_pytest_evidence(
        report=report,
        selection=selection,
        summary=summary,
        events=event_rows,
        exit_code=raw_exit if isinstance(raw_exit, int) and not isinstance(raw_exit, bool) else 125,
        collection_only=any(str(part) == "--collect-only" for part in command),
    )
    evidence["outcomes"] = outcomes
    return {
        "command": [str(part) for part in command],
        "exit": step_result.get("exit"),
        "report_status": "present" if report is not None else "missing",
        "canonical_report_status": "present" if report is not None else "missing",
        "selected_count": selection.get("selected_count"),
        "deselected_count": selection.get("deselected_count"),
        "summary_exitstatus": None if summary is None else summary.get("exitstatus"),
        "event_count": len(event_rows),
        **evidence,
    }


def _read_terminal_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"pytest evidence must be a JSON object: {path}")
    return payload


def _read_pytest_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"pytest event must be a JSON object: {path}")
            yield event


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else {}


def pytest_command_worker_request(cmd: Sequence[str]) -> str | None:
    for index, argument in enumerate(cmd):
        if argument in {"-n", "--numprocesses"} and index + 1 < len(cmd):
            return cmd[index + 1]
        if argument.startswith("-n") and len(argument) > 2:
            return argument[2:].removeprefix("=")
        if argument.startswith("--numprocesses="):
            return argument.split("=", 1)[1]
    return None


def configured_pytest_worker_request(env: Mapping[str, str]) -> int | None:
    raw = env.get("POLYLOGUE_PYTEST_WORKERS")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _read_json_pinned(path: Path) -> dict[str, Any]:
    parent_fd = _open_pinned_dir(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        with contextlib.suppress(OSError):
            os.close(fd)
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_history_pinned(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each well-formed history row, one line at a time.

    The shared history is append-only and grows with every run on the host,
    so no caller holds it whole: materializing it was the verifier's largest
    allocation, held for the length of the run.
    """
    try:
        parent_fd = _open_pinned_dir(path.parent)
    except FileNotFoundError:
        return
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
        except FileNotFoundError:
            return
    finally:
        os.close(parent_fd)
    try:
        handle = os.fdopen(fd, "r", encoding="utf-8")
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    with handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _tree_size_without_links(root: Path, *, budget: _NodeBudget | None = None, depth: int = 0) -> tuple[int, bool]:
    """Measure a run tree while treating links and special nodes as corrupt."""
    budget = budget or _NodeBudget(_DETAIL_NODE_BUDGET)
    if depth > 256:
        return 0, False
    total = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not budget.consume():
                    return total, False
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    return total, False
                if stat.S_ISDIR(info.st_mode):
                    nested, safe = _tree_size_without_links(Path(entry.path), budget=budget, depth=depth + 1)
                    total += nested
                    if not safe:
                        return total, False
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    return total, False
    except OSError:
        return total, False
    return total, True


def _remove_tree_at(parent_fd: int, name: str, *, budget: _NodeBudget | None = None) -> None:
    """Remove one run directory through pinned descriptors only."""
    budget = budget or _NodeBudget(_DETAIL_NODE_BUDGET)
    if not budget.consume():
        raise RuntimeError("verification detail deletion exceeded bounded node budget")
    info = os.lstat(name, dir_fd=parent_fd)
    if stat.S_ISLNK(info.st_mode):
        raise OSError("refusing to delete a symlinked verification detail tree")
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    child_fd = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=parent_fd)
    try:
        with os.scandir(child_fd) as entries:
            for entry in entries:
                if not budget.consume():
                    raise RuntimeError("verification detail deletion exceeded bounded node budget")
                child_info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(child_info.st_mode):
                    raise OSError("refusing to traverse a symlinked verification detail node")
                if stat.S_ISDIR(child_info.st_mode):
                    _remove_tree_at(child_fd, entry.name, budget=budget)
                elif stat.S_ISREG(child_info.st_mode):
                    os.unlink(entry.name, dir_fd=child_fd)
                else:
                    raise OSError("refusing to delete an unsupported verification detail node")
        os.fchmod(child_fd, os.fstat(child_fd).st_mode | stat.S_IWUSR)
    finally:
        os.close(child_fd)
    os.fchmod(parent_fd, os.fstat(parent_fd).st_mode | stat.S_IWUSR)
    os.rmdir(name, dir_fd=parent_fd)


def _history_path_for_root(root: Path, history_path: Path | None) -> tuple[Path, Path]:
    root = _absolute_path(root)
    history = _absolute_path(
        history_path
        if history_path is not None and history_path.is_absolute()
        else root / (history_path or VERIFY_HISTORY_PATH)
    )
    return root, history


def append_verify_history(entry: Mapping[str, Any], *, path: Path | None = None) -> None:
    """Append a compact semantic row, never the private run invocation."""
    _append_jsonl(_semantic_history_row(entry), path=path or verify_history_path())


def _append_jsonl(entry: Mapping[str, Any], *, path: Path) -> None:
    _append_jsonl_batch((entry,), path=path)


def _append_jsonl_batch(entries: Iterable[Mapping[str, Any]], *, path: Path) -> None:
    # flock is process-scoped on some Unix implementations and is therefore
    # insufficient to serialize threads in one verifier process. Keep both
    # guards around the identity scan and every append in this batch.
    with _APPEND_LOCK:
        _append_jsonl_batch_locked(entries, path=path)


def _append_jsonl_batch_locked(entries: Iterable[Mapping[str, Any]], *, path: Path) -> None:
    path = _validate_history_write_path(_absolute_path(path))
    _mkdir_pinned(path.parent)
    lock_fd = _open_retention_lock(path.parent, nonblocking=False)
    if lock_fd is None:
        raise RuntimeError("verification retention lock is busy")
    parent_fd = _open_pinned_dir(path.parent)
    try:
        # One scan per batch: a recovery backlog must not rescan the growing
        # lane for every missing receipt. Full receipt payloads stay streamed.
        published = {
            (row["run_id"], row.get("kind"))
            for row in _iter_history_pinned(path)
            if isinstance(row.get("run_id"), str) and (row.get("kind") is None or isinstance(row.get("kind"), str))
        }
        for entry in entries:
            run_id = entry.get("run_id")
            identity = (run_id, entry.get("kind"))
            if isinstance(run_id, str) and identity in published:
                continue
            fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600, dir_fd=parent_fd)
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(entry), ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                # Persist every row and directory entry before advancing the
                # batch: interruption leaves a durable prefix for recovery.
                os.fsync(parent_fd)
            finally:
                with contextlib.suppress(OSError):
                    os.close(fd)
            if isinstance(run_id, str):
                published.add(identity)
    finally:
        os.close(parent_fd)
        _close_retention_lock(lock_fd)


def _terminal_status(entry: Mapping[str, Any]) -> str:
    aggregate = entry.get("pytest_aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    reason = str(entry.get("termination_reason") or aggregate.get("termination_reason") or "")
    if entry.get("diagnosis") in INTERRUPTED_DIAGNOSES or reason in {
        "cancelled",
        "canceled",
        "timeout",
        "signal",
        "interrupted",
    }:
        return "cancelled" if "cancel" in reason else "interrupted"
    if entry.get("status") == "running":
        return "unknown"
    return "passed" if entry.get("exit_code") == 0 else "failed"


#: Slot-receipt fields that name checkout-local files. The durable lane
#: outlives the checkout, so a path there points at nothing and leaks layout.
_SLOT_RECEIPT_LOCAL_FIELDS = frozenset({"log_path"})


def _durable_slot_receipt(receipt: object) -> dict[str, Any] | None:
    if not isinstance(receipt, Mapping):
        return None
    return {key: value for key, value in receipt.items() if key not in _SLOT_RECEIPT_LOCAL_FIELDS}


def canonical_verification_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded, cross-source contract for one verifier run.

    This deliberately contains no argv, prompts, environment, log contents,
    or machine-local paths.  ``artifact_ref`` is an opaque local evidence
    handle; consumers must not interpret it as AgentCTL lifecycle state.
    """
    steps: list[dict[str, Any]] = []
    raw_steps = entry.get("steps")
    if isinstance(raw_steps, list):
        for raw in raw_steps:
            if not isinstance(raw, Mapping):
                continue
            step: dict[str, Any] = {
                "step_id": raw.get("step_id"),
                "name": raw.get("name"),
                "status": "running"
                if raw.get("status") == "running"
                else "passed"
                if raw.get("exit") == 0
                else "failed",
                "exit_code": raw.get("exit"),
                "process_exit": raw.get("process_exit"),
                "duration_s": raw.get("duration_s"),
                "diagnosis": raw.get("diagnosis"),
                "termination_reason": raw.get("termination_reason"),
                "termination_killer": raw.get("termination_killer"),
                "termination_unit": raw.get("termination_unit"),
                "pytest_slot_receipt": _durable_slot_receipt(raw.get("pytest_slot_receipt")),
                "artifact_ref": f"polylogue://verification/{entry.get('run_id')}/steps/{raw.get('step_id')}"
                if raw.get("step_id") is not None
                else None,
            }
            evidence_error = raw.get("evidence_error")
            if isinstance(evidence_error, Mapping):
                # The canonical contract excludes paths and log contents.
                step["evidence_error"] = {
                    key: evidence_error[key] for key in ("phase", "type") if key in evidence_error
                }
            steps.append({key: value for key, value in step.items() if value is not None})
    tree_unknown = entry.get("worktree_capture_source") == "unavailable"
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "polylogue.verification-receipt",
        "run_id": entry.get("run_id"),
        # An execution tree nobody captured is unknown: the checkout at
        # finalization is not evidence of what ran.
        "source_revision": None
        if tree_unknown
        or entry.get("git_dirty")
        or entry.get("final_git_dirty")
        # A head that moved under the run (or was never observed at the end)
        # names a revision nobody tested.
        or entry.get("final_git_head") != entry.get("git_head")
        else entry.get("git_head"),
        "status": _terminal_status(entry),
        "started_at": entry.get("started_at"),
        "finished_at": entry.get("finished_at"),
        "duration_s": entry.get("duration_s"),
        "steps": steps,
        "artifact_ref": f"polylogue://verification/{entry.get('run_id')}",
        "semantic_status": entry.get("status"),
        "git_dirty": None if tree_unknown else bool(entry.get("git_dirty") or entry.get("final_git_dirty")),
        "tier": entry.get("tier"),
        "verification_scope": entry.get("verification_scope"),
    }
    refs = {
        public_key: entry[internal_key]
        for public_key, internal_key in (
            ("job_id", "agentctl_job_id"),
            ("correlation_id", "agentctl_correlation_id"),
        )
        if isinstance(entry.get(internal_key), str) and entry[internal_key]
    }
    if refs:
        result["agentctl"] = refs
    aggregate = entry.get("pytest_aggregate")
    if isinstance(aggregate, Mapping):
        result["pytest"] = {
            key: aggregate[key]
            for key in (
                "selection_mode",
                "selected_union_count",
                "terminal_union_count",
                "terminal_green",
                "complete_corpus_covered",
                "outcomes",
            )
            if key in aggregate
        }
    selection = entry.get("testmon_selection")
    if isinstance(selection, Mapping):
        bounded_selection = {
            key: selection[key]
            for key in (
                "selection_mode",
                "graph_status",
                "graph_reason",
                "selected_count",
                "estimated_seconds",
                "selection_reason",
                "admission",
            )
            if key in selection
        }
        if bounded_selection:
            result["selection"] = bounded_selection
    diagnosis = entry.get("diagnosis")
    if isinstance(diagnosis, str):
        result["diagnosis"] = diagnosis
    return result


def _semantic_history_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    receipt = canonical_verification_receipt(entry)
    # Keep the small legacy columns used by `why` and retention readers.  The
    # canonical receipt is the cross-source join contract.
    row: dict[str, Any] = {
        key: entry[key]
        for key in (
            "run_id",
            "tier",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "exit_code",
            "diagnosis",
            "artifact_dir",
            "git_head",
            "git_dirty",
            "testmon_selection",
            "pytest_aggregate",
            "git_head",
            "tier",
            "verification_scope",
            "final_git_head",
            "final_git_dirty",
            "termination_reason",
        )
        if key in entry
    }
    raw_steps = entry.get("steps")
    if isinstance(raw_steps, list):
        row["steps"] = [
            {
                key: step[key]
                for key in (
                    "step_id",
                    "name",
                    "exit",
                    "status",
                    "diagnosis",
                    "termination_reason",
                    "termination_killer",
                    "termination_unit",
                )
                if key in step
            }
            for step in raw_steps
            if isinstance(step, Mapping)
        ]
    row["semantic_receipt"] = receipt
    row["receipt_schema_version"] = 1
    # Keep the join identity addressable on the history row as well as in the
    # versioned receipt.  This is provenance only; lifecycle state remains
    # owned by AgentCTL and semantic status remains verifier-owned.
    if "agentctl" in receipt:
        row["agentctl"] = receipt["agentctl"]
    return row


def verification_evidence_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the canonical evidence lane lives.

    The lane is the durable record of every verifier run, so it lives in the
    user's state directory rather than the checkout: a worktree, and with it
    ``.cache/verify``, is removed once its branch lands, while the run history
    must outlive it. ``POLYLOGUE_VERIFICATION_EVIDENCE_PATH`` relocates it.
    """
    source = os.environ if env is None else env
    configured = source.get(VERIFY_EVIDENCE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    state_home = source.get("XDG_STATE_HOME")
    base = Path(state_home) if state_home else Path(source.get("HOME", "~")).expanduser() / ".local" / "state"
    return base / "polylogue" / "verification" / "evidence.jsonl"


def append_verification_evidence(entry: Mapping[str, Any], *, path: Path | None = None) -> None:
    """Publish the same canonical receipt to the durable evidence lane."""
    _append_jsonl(canonical_verification_receipt(entry), path=path or verification_evidence_path())


def read_verification_evidence(path: Path) -> list[dict[str, Any]]:
    """Read only valid canonical rows for a Lynchpin-style projection."""
    return [
        row
        for row in _iter_history_pinned(_absolute_path(path))
        if row.get("kind") == "polylogue.verification-receipt" and row.get("schema_version") == 1
    ]


def prune_successful_verify_runs(
    *,
    root: Path,
    history_path: Path | None = None,
    max_successful: int = SUCCESSFUL_VERIFY_DETAIL_LIMIT,
    max_failed: int = FAILED_VERIFY_DETAIL_LIMIT,
    max_failed_age_s: float = FAILED_VERIFY_DETAIL_MAX_AGE_S,
    max_failed_bytes: int = FAILED_VERIFY_DETAIL_MAX_BYTES,
    now: float | None = None,
) -> dict[str, object]:
    """Bound terminal verification detail after history is durably appended.

    Successful details retain the newest ``max_successful`` runs. Failed,
    cancelled, and crashed details retain the newest run unconditionally, then
    at most ``max_failed`` recent runs within ``max_failed_age_s`` and
    ``max_failed_bytes``. The newest failure is the explicit exception to the
    byte, age, and count caps so a single run cannot erase the only diagnostic
    evidence. The append-only history remains the compact structured summary.
    Any symlink, malformed receipt, unsafe path, or active retention lock
    causes pruning to retain the affected evidence and report the refusal.

    A detail tree whose run has no durable history row is outside the bound by
    construction: its summary was never appended, so pruning it would destroy
    the only record of that run. Those are reported as
    ``orphaned_detail_run_ids`` so a tree count above the documented bound can
    be read as the backlog it is rather than as a bound that stopped running.
    """
    if max_successful < 0 or max_failed < 0 or max_failed_age_s < 0 or max_failed_bytes < 0:
        raise ValueError("successful verify detail limit must be non-negative")
    try:
        root, resolved_history = _history_path_for_root(root, history_path)
        root_fd = _open_pinned_dir(root)
        verify_dir = root / VERIFY_CACHE
        verify_fd = _open_pinned_dir(verify_dir)
        runs_root = root / VERIFY_RUNS_DIR
        runs_fd = _open_pinned_dir(runs_root)
    except (OSError, ValueError) as exc:
        return {
            "retained_run_ids": [],
            "retained_failure_run_ids": [],
            "pruned_run_ids": [],
            "history_durable": False,
            "refused": True,
            "reason": str(exc),
        }
    lock_fd = _open_retention_lock(verify_dir, nonblocking=True)
    if lock_fd is None:
        for fd in (runs_fd, verify_fd, root_fd):
            os.close(fd)
        return {
            "retained_run_ids": [],
            "retained_failure_run_ids": [],
            "pruned_run_ids": [],
            "history_durable": False,
            "retention_locked": True,
        }
    try:
        # Only the fields pruning reads are kept per run; the latest row for a
        # run still wins.
        durable: dict[str, dict[str, Any]] = {}
        try:
            for row in _iter_history_pinned(resolved_history):
                run_id = row.get("run_id")
                if isinstance(run_id, str) and row.get("status") != "running":
                    aggregate = row.get("pytest_aggregate")
                    durable[run_id] = {
                        "status": row.get("status"),
                        "diagnosis": row.get("diagnosis"),
                        "finished_at": row.get("finished_at"),
                        "pytest_aggregate": (
                            {"covered_by_run": aggregate.get("covered_by_run")}
                            if isinstance(aggregate, Mapping)
                            else aggregate
                        ),
                    }
        except (OSError, ValueError):
            return {
                "retained_run_ids": [],
                "retained_failure_run_ids": [],
                "pruned_run_ids": [],
                "history_durable": False,
                "refused": True,
            }
        if not durable:
            return {
                "retained_run_ids": [],
                "retained_failure_run_ids": [],
                "pruned_run_ids": [],
                "orphaned_detail_run_ids": [],
                "history_durable": False,
                "retention_locked": False,
            }

        candidates: list[dict[str, Any]] = []
        # Detail trees with no durable history row of their own. The bound
        # cannot reach them -- pruning one would destroy the only copy of its
        # cost evidence -- so they are counted and reported rather than
        # silently making the retained tree count look like a broken bound.
        orphaned_detail_run_ids: list[str] = []
        with os.scandir(runs_fd) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    continue
                run_dir = runs_root / entry.name
                try:
                    payload = _read_json_pinned(run_dir / "run.json")
                except (OSError, ValueError):
                    # A malformed or hostile detail tree is evidence, not a
                    # pruning candidate.
                    continue
                run_id = payload.get("run_id")
                history = durable.get(run_id) if isinstance(run_id, str) else None
                if (
                    not isinstance(run_id, str)
                    or entry.name != run_id
                    or history is None
                    or payload.get("status") != history.get("status")
                ):
                    if isinstance(run_id, str) and history is None:
                        orphaned_detail_run_ids.append(run_id)
                    continue
                finished_at = payload.get("finished_at")
                try:
                    finished_epoch = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError, OverflowError):
                    continue
                size, safe = _tree_size_without_links(run_dir)
                if not safe:
                    continue
                candidates.append(
                    {
                        "run_id": run_id,
                        "status": payload.get("status"),
                        "finished_at": str(finished_at),
                        "finished_epoch": finished_epoch,
                        "size": size,
                        "name": entry.name,
                    }
                )

        # A skipped complete run has no detail of its own; it must not spend
        # the successful-detail quota. The run a skip names as coverage stays
        # retained while one of the newest ``max_successful`` skips points at
        # it, so pins are bounded the way retained successes are rather than
        # accumulating with the append-only history.
        skipped: list[tuple[str, str, str]] = []
        for run_id, row in durable.items():
            if row.get("diagnosis") != "corpus_already_verified":
                continue
            aggregate = row.get("pytest_aggregate")
            covered = aggregate.get("covered_by_run") if isinstance(aggregate, Mapping) else None
            skipped.append((str(row.get("finished_at") or ""), run_id, covered if isinstance(covered, str) else ""))
        skipped.sort(reverse=True)
        skipped_ids = {run_id for _finished, run_id, _covered in skipped}
        pinned_ids = {covered for _finished, _run_id, covered in skipped[:max_successful] if covered}
        successes = sorted(
            (
                candidate
                for candidate in candidates
                if candidate["status"] == "success" and candidate["run_id"] not in skipped_ids
            ),
            key=lambda item: (item["finished_at"], item["run_id"]),
            reverse=True,
        )
        pinned = [
            candidate
            for candidate in candidates
            if candidate["run_id"] in pinned_ids and candidate not in successes[:max_successful]
        ]
        failures = sorted(
            (candidate for candidate in candidates if candidate["status"] != "success"),
            key=lambda item: (item["finished_at"], item["run_id"]),
            reverse=True,
        )
        retained = [candidate["run_id"] for candidate in successes[:max_successful]] + [
            candidate["run_id"] for candidate in pinned
        ]
        retained_failures: list[str] = []
        keep_failure_names: set[str] = set()
        if failures:
            newest = failures[0]
            retained_failures.append(newest["run_id"])
            keep_failure_names.add(newest["name"])
            used_bytes = newest["size"]
            current_time = time.time() if now is None else now
            for candidate in failures[1:]:
                if len(retained_failures) - 1 >= max_failed:
                    continue
                if current_time - candidate["finished_epoch"] > max_failed_age_s:
                    continue
                if used_bytes + candidate["size"] > max_failed_bytes:
                    continue
                retained_failures.append(candidate["run_id"])
                keep_failure_names.add(candidate["name"])
                used_bytes += candidate["size"]

        keep_names = (
            {candidate["name"] for candidate in successes[:max_successful]}
            | {candidate["name"] for candidate in pinned}
            | keep_failure_names
        )
        pruned: list[str] = []
        for candidate in candidates:
            if candidate["name"] in keep_names:
                continue
            try:
                _remove_tree_at(runs_fd, candidate["name"])
            except (OSError, RuntimeError, ValueError):
                continue
            pruned.append(candidate["run_id"])
        if pruned:
            os.fsync(runs_fd)
        return {
            "retained_run_ids": retained,
            "retained_failure_run_ids": retained_failures,
            "pruned_run_ids": pruned,
            "orphaned_detail_run_ids": sorted(orphaned_detail_run_ids),
            "history_durable": True,
            "retention_locked": False,
        }
    finally:
        _close_retention_lock(lock_fd)
        for fd in (runs_fd, verify_fd, root_fd):
            os.close(fd)


#: The diagnosis an abandoned run carries. Distinct from
#: ``verification_interrupted``, which is what the in-process signal handlers
#: write: that one proves a handler ran and unwound. This one says the
#: opposite -- the run's own process died without ever writing a terminal
#: state, and this receipt was reconciled from outside it.
ABANDONED_DIAGNOSIS = "verification_abandoned"
#: Where AgentCTL keeps the per-job outcome document this reconciler adopts.
_AGENTCTL_JOBS_RELPATH = Path("agentctl") / "jobs"


def _owning_pid(run_id: str) -> int | None:
    """The pid embedded in ``make_run_id``'s ``<stamp>-<tier>-<pid>-<uuid8>``.

    A tier may contain ``-`` (``focused-test``), so the pid is read from the
    right. None when the id does not carry one, which is the same answer as a
    live pid: nothing about this run can be concluded from outside it.
    """
    parts = run_id.rsplit("-", 2)
    if len(parts) != 3:
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    return pid if pid > 0 else None


def _process_is_live(pid: int) -> bool:
    """Whether ``pid`` still names a process.

    A pid this user may not signal is still a live process, so
    ``PermissionError`` is 'live'. Every uncertain answer is 'live': the cost
    of leaving a stranded receipt one cycle longer is a stale row, and the cost
    of the opposite is overwriting a running verification's own receipt.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _process_owns_receipt(pid: int, started_at: object, *, is_live: Callable[[int], bool]) -> bool:
    """Reject a reused PID whose current process started after this receipt."""
    if not is_live(pid):
        return False
    if not isinstance(started_at, str):
        return True
    try:
        receipt_time = datetime.fromisoformat(started_at).timestamp()
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields_after_comm = stat_line[stat_line.rfind(")") + 2 :].split()
        start_ticks = int(fields_after_comm[19])  # proc stat field 22, after pid/comm
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="ascii").splitlines()
            if line.startswith("btime ")
        )
        process_started = boot_time + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, IndexError):
        return True
    return process_started <= receipt_time + 1.0


def _agentctl_state_root(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    state_home = source.get("XDG_STATE_HOME")
    base = Path(state_home) if state_home else Path(source.get("HOME", "~")).expanduser() / ".local" / "state"
    return base / _AGENTCTL_JOBS_RELPATH


def read_agentctl_outcome(job_id: str, *, state_root: Path | None = None) -> dict[str, Any] | None:
    """The AgentCTL ``.outcome`` document for ``job_id``, when one exists.

    This is execution provenance, not semantic status: it says how the process
    ended, which is exactly the fact a receipt abandoned mid-run is missing.
    """
    if not job_id or "/" in job_id or job_id.startswith("."):
        return None
    root = _agentctl_state_root() if state_root is None else state_root
    try:
        document = json.loads((root / f"{job_id}.outcome").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _adopt_outcome(payload: dict[str, Any], outcome: Mapping[str, Any]) -> None:
    """Take the process-level ending AgentCTL recorded for this run.

    ``outcome`` is AgentCTL's own bucket and is coarse: an oom-kill and an
    ordinary non-zero exit are both ``"failed"``. ``systemd_result`` is the
    killer the unit recorded -- ``"oom-kill"``, ``"timeout"``, or null when the
    command simply exited -- and it is the fact a reader of an abandoned
    receipt actually needs. Dropping it is what made three differently-killed
    scheduled runs read as one undifferentiated failure (polylogue-yk0zz), and
    sent the reader to `journalctl` for something already on disk. The unit
    name comes along because it is what a journal query needs.
    """
    exit_code = outcome.get("exit_code")
    if isinstance(exit_code, int):
        payload["exit_code"] = exit_code
        payload["status"] = "success" if exit_code == 0 else "failed"
    ended = outcome.get("outcome")
    if isinstance(ended, str) and ended:
        payload["termination_reason"] = ended
    systemd_result = outcome.get("systemd_result")
    if isinstance(systemd_result, str) and systemd_result:
        payload["termination_killer"] = systemd_result
    unit = outcome.get("unit")
    if isinstance(unit, str) and unit:
        payload["termination_unit"] = unit
    payload["agentctl_outcome_adopted"] = True


def reconcile_abandoned_verify_runs(
    *,
    runs_root: Path,
    state_root: Path | None = None,
    is_live: Callable[[int], bool] = _process_is_live,
) -> list[dict[str, Any]]:
    """Give every stranded ``running`` receipt a terminal state, and say why.

    ``VerifyRun.finish`` is the only in-process exit from ``running``, and a
    SIGKILL (systemd-oomd, a hard cancel, a host that lost power) cannot reach
    it: the signal handlers in ``devtools.verify`` fire for SIGINT/SIGTERM and
    for nothing else. Without this, such a run stays ``running`` forever and
    every reader downstream propagates that as a legal state -- four
    consecutive scheduled runs were stranded that way by 2026-09-20.

    The owning pid is embedded in the run id, so liveness is decidable from the
    receipt alone. A run whose pid is still live is LEFT ALONE; only a receipt
    with no process behind it is reconciled, and any AgentCTL ``.outcome``
    recorded for the same job is adopted so the run reports how it actually
    ended rather than merely that it stopped.

    ``runs_root`` is the directory of run directories -- the one the caller is
    about to read -- so a reader with a relocated cache reconciles that cache
    and not the checkout's.

    Returns the payloads it rewrote, newest first by run id, so a caller can
    append history for them.
    """
    reconciled: list[dict[str, Any]] = []
    try:
        entries = sorted(runs_root.iterdir())
    except OSError:
        return reconciled
    for run_dir in entries:
        path = run_dir / "run.json"
        try:
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or run_dir.name != run_id:
            continue
        if payload.get("status") in {"success", "failed"} and payload.get("diagnosis") == ABANDONED_DIAGNOSIS:
            current_path = runs_root.parent / CURRENT_RUN_PATH.name
            current = _read_json(current_path)
            if current and current.get("run_id") == run_id and current != payload:
                _write_json(current_path, payload)
            reconciled.append(payload)
            continue
        if payload.get("status") != "running":
            continue
        pid = _owning_pid(run_id)
        if pid is None or _process_owns_receipt(pid, payload.get("started_at"), is_live=is_live):
            continue
        # The owner may have published its own verdict and exited between the
        # read above and the liveness check. Only a receipt still ``running``
        # once the owner is proven gone is abandoned; a terminal one is the
        # owner's authoritative result.
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("run_id") != run_id or payload.get("status") != "running":
            continue
        job_id = payload.get("agentctl_job_id")
        outcome = read_agentctl_outcome(job_id, state_root=state_root) if isinstance(job_id, str) else None
        if isinstance(job_id, str) and outcome is None:
            continue
        payload["status"] = "failed"
        payload["exit_code"] = None
        payload["diagnosis"] = ABANDONED_DIAGNOSIS
        payload["finished_at"] = utc_now()
        started_at = payload.get("started_at")
        if isinstance(started_at, str):
            with contextlib.suppress(ValueError):
                payload["duration_s"] = max(
                    0.0,
                    (
                        datetime.fromisoformat(payload["finished_at"]) - datetime.fromisoformat(started_at)
                    ).total_seconds(),
                )
        payload["abandoned_pid"] = pid
        if outcome is not None:
            _adopt_outcome(payload, outcome)
        for step in payload.get("steps") or ():
            if isinstance(step, dict) and step.get("status") == "running":
                step["status"] = "failed"
                step["exit"] = payload.get("exit_code")
                step["diagnosis"] = ABANDONED_DIAGNOSIS
                step["finished_at"] = payload["finished_at"]
        try:
            _write_json(path, payload)
        except OSError:
            continue
        current_path = runs_root.parent / CURRENT_RUN_PATH.name
        current = _read_json(current_path)
        if current and current.get("run_id") == run_id:
            _write_json(current_path, payload)
        reconciled.append(payload)
    reconciled.sort(key=lambda entry: str(entry.get("run_id")), reverse=True)
    return reconciled


def _checkout_root() -> Path:
    """The checkout this ``devtools`` package runs from."""
    return Path(__file__).resolve().parents[1]


def reconcile_and_record_verify_runs(
    *,
    runs_root: Path,
    state_root: Path | None = None,
    evidence_path: Path | None = None,
    history_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Close abandoned runs and recover interrupted terminal publication.

    Terminal run receipts remain authoritative while their details exist.
    History retains their canonical receipt after pruning, so evidence
    publication can also resume when those details have gone. Running receipts
    whose owner remains live are never published. Existing append locks make
    concurrent finish/recovery publication idempotent by run identity.
    """
    reconciled = reconcile_abandoned_verify_runs(runs_root=runs_root, state_root=state_root)
    cache = runs_root.parent
    # ``runs_root`` is ``<checkout>/.cache/verify/runs``.
    owner = runs_root.resolve().parents[2]
    checkout_cache = owner == _checkout_root()
    if history_path is None:
        if os.environ.get(VERIFY_HISTORY_PATH_ENV) or checkout_cache:
            history_path = verify_history_path(root=owner)
        else:
            history_path = cache / VERIFY_HISTORY_PATH.name
    if evidence_path is None:
        if os.environ.get(VERIFY_EVIDENCE_PATH_ENV) or checkout_cache:
            evidence_path = verification_evidence_path()
        else:
            evidence_path = cache / VERIFY_EVIDENCE_PATH.name

    def terminal_payloads() -> Iterator[dict[str, Any]]:
        try:
            for run_dir in runs_root.iterdir():
                with contextlib.suppress(OSError):
                    if not run_dir.is_dir() or run_dir.is_symlink():
                        continue
                    payload = _read_json(run_dir / "run.json")
                    if payload is None:
                        continue
                    run_id = payload.get("run_id")
                    if (
                        isinstance(run_id, str)
                        and run_id == run_dir.name
                        and payload.get("status") in {"success", "failed"}
                    ):
                        yield payload
        except OSError:
            return

    def evidence_receipts() -> Iterator[dict[str, Any]]:
        # History preserves the exact canonical receipt after detail pruning.
        for row in _iter_history_pinned(history_path):
            receipt = row.get("semantic_receipt")
            if (
                isinstance(receipt, dict)
                and receipt.get("kind") == "polylogue.verification-receipt"
                and receipt.get("schema_version") == 1
                and receipt.get("run_id") == row.get("run_id")
                and receipt.get("semantic_status") in {"success", "failed"}
            ):
                yield receipt
        # A failed history append must not block independent evidence recovery.
        for payload in terminal_payloads():
            yield canonical_verification_receipt(payload)

    with contextlib.suppress(OSError, ValueError):
        _append_jsonl_batch((_semantic_history_row(payload) for payload in terminal_payloads()), path=history_path)
    with contextlib.suppress(OSError, ValueError):
        _append_jsonl_batch(evidence_receipts(), path=evidence_path)
    return reconciled
