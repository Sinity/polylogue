"""Typed local receipts for semantic verification commands.

AgentCTL owns jobs, deadlines, process trees, temporary storage, and checkout
lifecycle. This module records only the verifier facts Polylogue can state:
what ran, the decoded pytest outcome, testmon selection, and the resulting
scope.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERIFY_CACHE = Path(".cache/verify")
VERIFY_RUNS_DIR = VERIFY_CACHE / "runs"
VERIFY_HISTORY_PATH = VERIFY_CACHE / "history.jsonl"
CURRENT_RUN_PATH = VERIFY_CACHE / "current-run.json"
CURRENT_STATISTICS_PATH = VERIFY_CACHE / "current-pytest-statistics.json"
CURRENT_EVENTS_DIR = VERIFY_CACHE / "current-pytest-events"
PYTEST_CANONICAL_REPORT_NAME = "pytest-report.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(*, tier: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_tier = re.sub(r"[^A-Za-z0-9_.-]+", "-", tier).strip("-") or "verify"
    return f"{stamp}-{safe_tier}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_only_git_env() -> dict[str, str]:
    return {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True, timeout=5, cwd=cwd, env=_read_only_git_env()
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(result.stdout.strip())


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
        self,
        *,
        selection_mode: str,
        state_status: str,
        state_reason: str,
        missing_executable_paths: Sequence[str] = (),
        runtime_data_paths: Sequence[str] = (),
        environment_digest: str | None = None,
    ) -> None:
        self._payload["testmon_selection"] = {
            "selection_mode": selection_mode,
            "state_status": state_status,
            "state_reason": state_reason,
            "missing_executable_paths": list(missing_executable_paths),
            "runtime_data_paths": list(runtime_data_paths),
            "environment_digest": environment_digest,
        }
        self.write()

    @property
    def testmon_selection(self) -> dict[str, Any] | None:
        selection = self._payload.get("testmon_selection")
        return dict(selection) if isinstance(selection, dict) else None

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
            step.update(result)
            step["finished_at"] = utc_now()
            step["status"] = "success" if result.get("exit") == 0 else "failed"
            if str(step.get("name", "")).startswith("pytest"):
                step_dir = self.run_dir / "steps" / step_id
                with contextlib.suppress(OSError, ValueError):
                    statistics = aggregate_pytest_statistics(step_dir, command=step.get("cmd", []), step_result=result)
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
    updated.update(
        {
            "POLYLOGUE_VERIFY_RUN_ID": run.run_id,
            "POLYLOGUE_PYTEST_RUN_ID": pytest_step_run_id(run.run_id, artifacts.step_id),
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(artifacts.events_dir),
            "POLYLOGUE_PYTEST_EVENTS_PATH": str(artifacts.events_merged_path),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(artifacts.selection_path),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(artifacts.summary_path),
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
    selection = _read_json(step_dir / "selection.json")
    summary = _read_json(step_dir / "summary.json")
    outcomes: dict[str, int] = {}
    for test in report.get("tests", []):
        if isinstance(test, Mapping):
            outcome = str(test.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    event_path = step_dir / "events.jsonl"
    if not outcomes and event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                event = json.loads(line)
                if isinstance(event, dict) and isinstance(event.get("outcome"), str):
                    outcome = event["outcome"]
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "command": [str(part) for part in command],
        "exit": step_result.get("exit"),
        "report_status": "present" if report else "missing",
        "canonical_report_status": "present" if report else "missing",
        "outcomes": outcomes,
        "selected_count": selection.get("selected_count"),
        "deselected_count": selection.get("deselected_count"),
        "summary_exitstatus": summary.get("exitstatus"),
        "event_count": merge_worker_events(step_dir / "events", event_path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
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


def append_verify_history(entry: Mapping[str, Any], *, path: Path = VERIFY_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=False, sort_keys=True) + "\n")
