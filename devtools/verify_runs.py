"""Typed local receipts for semantic verification commands.

AgentCTL owns jobs, deadlines, process trees, temporary storage, and checkout
lifecycle. This module records only the verifier facts Polylogue can state:
what ran, the decoded pytest outcome, and the resulting
scope.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devtools.pytest_evidence import evaluate_pytest_evidence
from devtools.testmon_provision import TESTMON_DATA_RELPATH


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
VERIFY_EVIDENCE_PATH = VERIFY_CACHE / "evidence.jsonl"
VERIFY_EVIDENCE_PATH_ENV = "POLYLOGUE_VERIFICATION_EVIDENCE_PATH"
CURRENT_RUN_PATH = VERIFY_CACHE / "current-run.json"
CURRENT_STATISTICS_PATH = VERIFY_CACHE / "current-pytest-statistics.json"
CURRENT_EVENTS_DIR = VERIFY_CACHE / "current-pytest-events"
PYTEST_CANONICAL_REPORT_NAME = "pytest-report.json"
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
            "git_dirty": git_dirty(self.root),
            "started_at": utc_now(),
            "status": "running",
            "steps": [],
            "artifact_dir": str(VERIFY_RUNS_DIR / self.run_id),
        }
        # These are opaque execution identities.  They are provenance only;
        # semantic status is still decided by this verifier.
        if agentctl_operation is not None:
            for field, variable in (
                ("agentctl_job_id", "SINNIXD_JOB_ID"),
                ("agentctl_correlation_id", "SINNIXD_CORRELATION_ID"),
            ):
                value = os.environ.get(variable)
                if value:
                    self._payload[field] = value
        # Kept on every receipt so later classification does not depend on a
        # live interpreter or a reconstructed shell environment.
        self._payload["environment_fingerprint"] = environment_fingerprint(root=self.root)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write()

    @property
    def relative_run_dir(self) -> Path:
        return VERIFY_RUNS_DIR / self.run_id

    def write(self) -> None:
        _write_json(self.run_dir / "run.json", self._payload)
        if self.mirror_current:
            _write_json(self.root / CURRENT_RUN_PATH, self._payload)

    def record_selection(
        self, *, selection_mode: str, graph_status: str, graph_reason: str, full_rerun_cause: str | None = None
    ) -> None:
        self._payload["testmon_selection"] = {
            "selection_mode": selection_mode,
            "graph_status": graph_status,
            "graph_reason": graph_reason,
            "full_rerun_cause": full_rerun_cause,
        }
        self.write()

    def start_step(self, *, label: str, cmd: list[str]) -> PytestStepArtifacts:
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
        for step in self._payload["steps"]:
            if step.get("step_id") != step_id:
                continue
            finalized = dict(result)
            statistics: dict[str, Any] | None = None
            if str(step.get("name", "")).startswith("pytest"):
                step_dir = self.run_dir / "steps" / step_id
                with contextlib.suppress(OSError, ValueError):
                    statistics = aggregate_pytest_statistics(step_dir, command=step.get("cmd", []), step_result=result)
                explicit_terminal = result.get("diagnosis") in {
                    "focused_test_runner_exception",
                    "pytest_interrupted",
                    "verification_interrupted",
                }
                if statistics is not None:
                    raw_exit = result.get("exit")
                    finalized["process_exit"] = raw_exit
                    if not explicit_terminal and raw_exit == 0 and not statistics.get("ok"):
                        finalized["exit"] = 5 if statistics.get("diagnosis") == "pytest_no_tests_selected" else 1
                    if not explicit_terminal:
                        finalized["diagnosis"] = str(statistics.get("diagnosis") or "pytest_no_evidence")
            step.update(finalized)
            step["finished_at"] = utc_now()
            step["status"] = "success" if finalized.get("exit") == 0 else "failed"
            if str(step.get("name", "")).startswith("pytest"):
                step_dir = self.run_dir / "steps" / step_id
                if statistics is not None:
                    _write_json(step_dir / "statistics.json", statistics)
                    step["statistics"] = statistics
                    step["statistics_path"] = str(self.relative_run_dir / "steps" / step_id / "statistics.json")
                    if self.mirror_current:
                        shutil.copyfile(step_dir / "statistics.json", self.root / CURRENT_STATISTICS_PATH)
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
        self.write()
        return dict(self._payload)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "step"


def pytest_step_run_id(run_id: str, step_id: str) -> str:
    index = step_id.split("-", 1)[0]
    return f"{run_id}-s{index}" if index.isdigit() else f"{run_id}-{step_id}"


def env_for_pytest_step(env: dict[str, str], *, run: VerifyRun, artifacts: PytestStepArtifacts) -> dict[str, str]:
    updated = dict(env)
    # testmon opens the datafile directly; without its directory the session
    # dies in an internal error rather than a test result.
    datafile = run.root / TESTMON_DATA_RELPATH
    datafile.parent.mkdir(parents=True, exist_ok=True)
    updated.update(
        {
            "POLYLOGUE_VERIFY_RUN_ID": run.run_id,
            # Scratch archives never need durability; see connection_profile.
            "POLYLOGUE_SQLITE_SYNCHRONOUS": "OFF",
            "POLYLOGUE_PYTEST_RUN_ID": pytest_step_run_id(run.run_id, artifacts.step_id),
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(artifacts.events_dir),
            "POLYLOGUE_PYTEST_EVENTS_PATH": str(artifacts.events_merged_path),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(artifacts.selection_path),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(artifacts.summary_path),
            "TESTMON_DATAFILE": str(datafile),
        }
    )
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
    if not events_dir.exists():
        return 0
    rows: list[dict[str, Any]] = []
    for path in sorted(events_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    rows.sort(key=lambda row: str(row.get("updated_at", "")))
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def aggregate_pytest_statistics(
    step_dir: Path, *, command: Sequence[object] = (), step_result: Mapping[str, object] = {}
) -> dict[str, Any]:
    report = _read_json(step_dir / PYTEST_CANONICAL_REPORT_NAME)
    selection = _read_json(step_dir / "selection.json") or {}
    summary = _read_json(step_dir / "summary.json") or {}
    outcomes: dict[str, int] = {}
    for test in (report or {}).get("tests", []):
        if isinstance(test, Mapping):
            outcome = str(test.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    event_path = step_dir / "events.jsonl"
    worker_events_dir = step_dir / "events"
    event_count = (
        merge_worker_events(worker_events_dir, event_path)
        if worker_events_dir.is_dir() and any(worker_events_dir.glob("*.jsonl"))
        else len(event_path.read_text(encoding="utf-8", errors="replace").splitlines())
        if event_path.exists()
        else 0
    )
    event_rows: list[dict[str, Any]] = []
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                event = json.loads(line)
                if isinstance(event, dict):
                    event_rows.append(event)
    if not outcomes and event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                event = json.loads(line)
                if isinstance(event, dict) and isinstance(event.get("outcome"), str):
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
        "summary_exitstatus": summary.get("exitstatus"),
        "event_count": event_count,
        **evidence,
    }


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


def _read_history_pinned(path: Path) -> list[dict[str, Any]]:
    parent_fd = _open_pinned_dir(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            rows: list[dict[str, Any]] = []
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
            return rows
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


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
    try:
        history.relative_to(root)
    except ValueError as exc:
        raise ValueError("verification history must remain inside the repository root") from exc
    return root, history


def append_verify_history(entry: Mapping[str, Any], *, path: Path = VERIFY_HISTORY_PATH) -> None:
    """Append a compact semantic row, never the private run invocation."""
    _append_jsonl(_semantic_history_row(entry), path=path)


def _append_jsonl(entry: Mapping[str, Any], *, path: Path) -> None:
    # flock is process-scoped on some Unix implementations and is therefore
    # insufficient to serialize threads in one verifier process.  Keep the
    # OS lock for cross-process writers and add this small in-process guard.
    with _APPEND_LOCK:
        _append_jsonl_locked(entry, path=path)


def _append_jsonl_locked(entry: Mapping[str, Any], *, path: Path) -> None:
    path = _absolute_path(path)
    _mkdir_pinned(path.parent)
    lock_fd = _open_retention_lock(path.parent, nonblocking=False)
    if lock_fd is None:
        raise RuntimeError("verification retention lock is busy")
    parent_fd = _open_pinned_dir(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(entry), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # The history record is the authority that permits detail
            # pruning. Persist its directory entry before returning to the
            # caller that immediately starts pruning.
            os.fsync(parent_fd)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
    finally:
        os.close(parent_fd)
        _close_retention_lock(lock_fd)


def _terminal_status(entry: Mapping[str, Any]) -> str:
    aggregate = entry.get("pytest_aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    reason = str(entry.get("termination_reason") or aggregate.get("termination_reason") or "")
    if entry.get("diagnosis") == "verification_interrupted" or reason:
        return "cancelled" if "cancel" in reason else "interrupted"
    if entry.get("status") == "running":
        return "unknown"
    return "passed" if entry.get("exit_code") == 0 else "failed"


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
                "duration_s": raw.get("duration_s"),
                "diagnosis": raw.get("diagnosis"),
                "artifact_ref": f"polylogue://verification/{entry.get('run_id')}/steps/{raw.get('step_id')}"
                if raw.get("step_id") is not None
                else None,
            }
            # A step that went green only because its failures passed alone is
            # not the same evidence as a step that never failed. The receipt
            # names the flakes so a reader can tell the two apart.
            rerun = raw.get("rerun")
            if isinstance(rerun, Mapping):
                flaky = [str(nodeid) for nodeid in rerun.get("flaky") or ()]
                if flaky:
                    step["flaky"] = flaky
                    step["flaky_count"] = len(flaky)
            steps.append({key: value for key, value in step.items() if value is not None})
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "polylogue.verification-receipt",
        "run_id": entry.get("run_id"),
        "source_revision": entry.get("final_git_head") or entry.get("git_head"),
        "status": _terminal_status(entry),
        "started_at": entry.get("started_at"),
        "finished_at": entry.get("finished_at"),
        "duration_s": entry.get("duration_s"),
        "steps": steps,
        "artifact_ref": f"polylogue://verification/{entry.get('run_id')}",
        "semantic_status": entry.get("status"),
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
        )
        if key in entry
    }
    row["semantic_receipt"] = receipt
    row["receipt_schema_version"] = 1
    return row


def append_verification_evidence(entry: Mapping[str, Any], *, path: Path | None = None) -> None:
    """Publish the same canonical receipt to the configured evidence lane."""
    configured = os.environ.get(VERIFY_EVIDENCE_PATH_ENV)
    target = path or (Path(configured) if configured else VERIFY_EVIDENCE_PATH)
    _append_jsonl(canonical_verification_receipt(entry), path=target)


def read_verification_evidence(path: Path) -> list[dict[str, Any]]:
    """Read only valid canonical rows for a Lynchpin-style projection."""
    return [
        row
        for row in _read_history_pinned(_absolute_path(path))
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
        try:
            history_rows = _read_history_pinned(resolved_history)
        except (OSError, ValueError):
            return {
                "retained_run_ids": [],
                "retained_failure_run_ids": [],
                "pruned_run_ids": [],
                "history_durable": False,
                "refused": True,
            }
        durable: dict[str, dict[str, Any]] = {}
        for row in history_rows:
            run_id = row.get("run_id")
            if isinstance(run_id, str) and row.get("status") != "running":
                durable[run_id] = row

        candidates: list[dict[str, Any]] = []
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
                if len(retained_failures) >= max_failed:
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
            "history_durable": True,
            "retention_locked": False,
        }
    finally:
        _close_retention_lock(lock_fd)
        for fd in (runs_fd, verify_fd, root_fd):
            os.close(fd)
