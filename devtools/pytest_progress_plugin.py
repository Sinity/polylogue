"""Incremental pytest event and selection ledgers for ``devtools verify``.

The pytest-json-report and JUnit artifacts are written at session end. When a
long verify run is killed for timeout or output stall, those reports may never
flush. This plugin writes one JSON object per completed test call so the
operator still has node-level failure evidence after an interrupted run.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

_EVENTS_ENV = "POLYLOGUE_PYTEST_EVENTS_PATH"
_EVENTS_DIR_ENV = "POLYLOGUE_PYTEST_EVENTS_DIR"
_SELECTION_ENV = "POLYLOGUE_PYTEST_SELECTION_PATH"
_SUMMARY_ENV = "POLYLOGUE_PYTEST_SUMMARY_PATH"
_SELECTION_NODEID_LIMIT_ENV = "POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"
_DESELECTED_NODEIDS_SAMPLE: list[str] = []
_DESELECTED_COUNT = 0
_SELECTED_COUNT = 0
_SLOWEST_REPORTS: list[dict[str, Any]] = []
_RECORDED_REPORT_KEYS: set[tuple[int, str, str, str, float]] = set()
_COLLECTION_STARTED_AT: float | None = None
_COLLECTION_DURATION_S: float | None = None
_CONTROLLER_COLLECTION_PAYLOAD: dict[str, Any] | None = None
_SLOW_REPORT_LIMIT = 20
_DEFAULT_SELECTION_NODEID_LIMIT = 500
_COLLECTION_FACT_SUFFIX = ".collection.json"
_ARTIFACT_ENV_NAMES = (_EVENTS_ENV, _EVENTS_DIR_ENV, _SELECTION_ENV, _SUMMARY_ENV)


@dataclass
class _SessionState:
    deselected_nodeids_sample: list[str]
    deselected_count: int
    selected_count: int
    slowest_reports: list[dict[str, Any]]
    recorded_report_keys: set[tuple[int, str, str, str, float]]
    collection_started_at: float | None
    collection_duration_s: float | None
    controller_collection_payload: dict[str, Any] | None
    artifact_environment: dict[str, str | None]


_SESSION_STATE_STACK: list[_SessionState] = []


def _capture_session_state() -> _SessionState:
    return _SessionState(
        deselected_nodeids_sample=list(_DESELECTED_NODEIDS_SAMPLE),
        deselected_count=_DESELECTED_COUNT,
        selected_count=_SELECTED_COUNT,
        slowest_reports=list(_SLOWEST_REPORTS),
        recorded_report_keys=set(_RECORDED_REPORT_KEYS),
        collection_started_at=_COLLECTION_STARTED_AT,
        collection_duration_s=_COLLECTION_DURATION_S,
        controller_collection_payload=(
            dict(_CONTROLLER_COLLECTION_PAYLOAD) if _CONTROLLER_COLLECTION_PAYLOAD else None
        ),
        artifact_environment={name: os.environ.get(name) for name in _ARTIFACT_ENV_NAMES},
    )


def _restore_session_state(state: _SessionState) -> None:
    global _COLLECTION_STARTED_AT, _COLLECTION_DURATION_S, _CONTROLLER_COLLECTION_PAYLOAD
    global _DESELECTED_COUNT, _SELECTED_COUNT
    _DESELECTED_NODEIDS_SAMPLE[:] = state.deselected_nodeids_sample
    _DESELECTED_COUNT = state.deselected_count
    _SELECTED_COUNT = state.selected_count
    _SLOWEST_REPORTS[:] = state.slowest_reports
    _RECORDED_REPORT_KEYS.clear()
    _RECORDED_REPORT_KEYS.update(state.recorded_report_keys)
    _COLLECTION_STARTED_AT = state.collection_started_at
    _COLLECTION_DURATION_S = state.collection_duration_s
    _CONTROLLER_COLLECTION_PAYLOAD = state.controller_collection_payload
    for name, value in state.artifact_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _reset_session_state() -> None:
    global _COLLECTION_STARTED_AT, _COLLECTION_DURATION_S, _CONTROLLER_COLLECTION_PAYLOAD
    global _DESELECTED_COUNT, _SELECTED_COUNT
    _DESELECTED_NODEIDS_SAMPLE.clear()
    _DESELECTED_COUNT = 0
    _SELECTED_COUNT = 0
    _SLOWEST_REPORTS.clear()
    _RECORDED_REPORT_KEYS.clear()
    _COLLECTION_STARTED_AT = None
    _COLLECTION_DURATION_S = None
    _CONTROLLER_COLLECTION_PAYLOAD = None


def _isolate_nested_artifact_destinations() -> None:
    """Give an in-process nested pytest invocation its own durable evidence."""
    raw_candidates = [os.environ.get(name) for name in _ARTIFACT_ENV_NAMES]
    base = next((Path(value).parent for value in raw_candidates if value), None)
    if base is None:
        return
    root = base / f"nested-pytest-{os.getpid()}-{uuid.uuid4().hex}"
    if os.environ.get(_EVENTS_DIR_ENV):
        os.environ[_EVENTS_DIR_ENV] = str(root / "events")
        os.environ.pop(_EVENTS_ENV, None)
    elif os.environ.get(_EVENTS_ENV):
        os.environ[_EVENTS_ENV] = str(root / "events.jsonl")
        os.environ.pop(_EVENTS_DIR_ENV, None)
    if os.environ.get(_SELECTION_ENV):
        os.environ[_SELECTION_ENV] = str(root / "selection.json")
    if os.environ.get(_SUMMARY_ENV):
        os.environ[_SUMMARY_ENV] = str(root / "summary.json")


def _selection_nodeid_limit() -> int:
    raw = os.environ.get(_SELECTION_NODEID_LIMIT_ENV)
    if raw is None:
        return _DEFAULT_SELECTION_NODEID_LIMIT
    with contextlib.suppress(ValueError):
        return max(0, int(raw))
    return _DEFAULT_SELECTION_NODEID_LIMIT


def _write_event(payload: dict[str, Any]) -> None:
    raw_dir = os.environ.get(_EVENTS_DIR_ENV)
    raw_path = os.environ.get(_EVENTS_ENV)
    if not raw_dir and not raw_path:
        return
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": os.environ.get("POLYLOGUE_VERIFY_RUN_ID"),
        "worker_id": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        "pid": os.getpid(),
        **payload,
    }
    if raw_dir:
        worker_id = str(payload["worker_id"]).replace("/", "-")
        path = Path(raw_dir) / f"{worker_id}-{os.getpid()}.jsonl"
    else:
        path = Path(str(raw_path))
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_selection(payload: dict[str, Any]) -> None:
    raw_path = os.environ.get(_SELECTION_ENV)
    if not raw_path:
        return
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": os.environ.get("POLYLOGUE_VERIFY_RUN_ID"),
        "worker_id": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        "pid": os.getpid(),
        **payload,
    }
    path = Path(raw_path)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)


def _write_worker_collection_fact(payload: dict[str, Any]) -> None:
    """Publish one worker-local collection fact for controller aggregation."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    raw_dir = os.environ.get(_EVENTS_DIR_ENV)
    if not worker_id or not raw_dir:
        return
    path = Path(raw_dir) / f"{worker_id.replace('/', '-')}-{os.getpid()}{_COLLECTION_FACT_SUFFIX}"
    payload = {"worker_id": worker_id, "pid": os.getpid(), **payload}
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)


def _worker_collection_payloads(events_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read worker collection facts in a stable order for the controller."""
    if events_dir is None:
        raw_dir = os.environ.get(_EVENTS_DIR_ENV)
        if not raw_dir:
            return []
        events_dir = Path(raw_dir)
    payloads: list[tuple[str, int, str, dict[str, Any]]] = []
    for path in events_dir.glob(f"*{_COLLECTION_FACT_SUFFIX}"):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            payload = json.loads(path.read_text(encoding="utf-8"))
            worker_id = payload.get("worker_id")
            pid = payload.get("pid")
            if isinstance(worker_id, str) and isinstance(pid, int):
                payloads.append((worker_id, pid, path.name, payload))
    return [payload for _worker_id, _pid, _name, payload in sorted(payloads)]


def _collection_payload() -> dict[str, Any]:
    """Return this process's complete collection fact."""
    limit = _selection_nodeid_limit()
    payload: dict[str, Any] = {
        "selected_count": _SELECTED_COUNT,
        "deselected_count": _DESELECTED_COUNT,
        "selected_nodeids": [],
        "selected_node_markers": {},
        "selected_nodeids_omitted": _SELECTED_COUNT,
        "deselected_nodeids": list(_DESELECTED_NODEIDS_SAMPLE),
        "deselected_nodeids_omitted": max(0, _DESELECTED_COUNT - len(_DESELECTED_NODEIDS_SAMPLE)),
        "nodeid_sample_limit": limit,
    }
    if _COLLECTION_DURATION_S is not None:
        payload["collection_duration_s"] = _COLLECTION_DURATION_S
    return payload


def merge_worker_collection_payloads(events_dir: Path | None = None) -> dict[str, Any] | None:
    """Choose one canonical xdist collection set and the slowest wall time."""
    payloads = _worker_collection_payloads(events_dir)
    if not payloads:
        return None
    merged = dict(payloads[0])
    durations = [payload.get("collection_duration_s") for payload in payloads]
    numeric_durations = [duration for duration in durations if isinstance(duration, (int, float))]
    if numeric_durations:
        merged["collection_duration_s"] = max(numeric_durations)
    return merged


def _write_summary(payload: dict[str, Any]) -> None:
    raw_path = os.environ.get(_SUMMARY_ENV)
    if not raw_path:
        return
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": os.environ.get("POLYLOGUE_VERIFY_RUN_ID"),
        "worker_id": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        "pid": os.getpid(),
        **payload,
    }
    path = Path(raw_path)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)


def _remember_report(payload: dict[str, Any]) -> None:
    _SLOWEST_REPORTS.append(payload)
    _SLOWEST_REPORTS.sort(key=lambda item: float(item.get("duration_s", 0.0)), reverse=True)
    del _SLOWEST_REPORTS[_SLOW_REPORT_LIMIT:]


def _durable_report_outcome(report: Any, outcome: str) -> str:
    """Preserve pytest's xfail semantics in the append-only event ledger."""
    if not getattr(report, "wasxfail", None):
        return outcome
    if outcome == "skipped":
        return "xfailed"
    if outcome == "passed":
        return "xpassed"
    return outcome


@pytest.hookimpl
def pytest_sessionstart(session: Any) -> None:
    """Reset per-session ledgers when tests invoke pytest in-process."""
    del session
    _SESSION_STATE_STACK.append(_capture_session_state())
    _reset_session_state()
    if len(_SESSION_STATE_STACK) > 1:
        _isolate_nested_artifact_destinations()
    # The worker environment is assigned after process exec, so it is not
    # reliably visible through /proc/<pid>/environ.  Emit the identity from
    # inside the worker for the supervisor's process-state sampler.
    _write_event({"event": "session_started"})


@pytest.hookimpl
def pytest_collection(session: Any) -> None:
    """Record collection start so selected-test runs expose import/collection cost."""
    del session
    global _COLLECTION_STARTED_AT
    _COLLECTION_STARTED_AT = time.monotonic()
    _write_event({"event": "collection_started"})


@pytest.hookimpl
def pytest_deselected(items: list[Any]) -> None:
    """Track deselected node IDs so the final selection artifact explains scope."""
    global _DESELECTED_COUNT
    limit = _selection_nodeid_limit()
    _DESELECTED_COUNT += len(items)
    remaining = max(0, limit - len(_DESELECTED_NODEIDS_SAMPLE))
    if remaining:
        _DESELECTED_NODEIDS_SAMPLE.extend(str(getattr(item, "nodeid", item)) for item in items[:remaining])


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session: Any, config: Any, items: list[Any]) -> None:
    """Write the final selected test set after pytest/testmon deselection."""
    del config
    global _COLLECTION_DURATION_S, _CONTROLLER_COLLECTION_PAYLOAD, _SELECTED_COUNT
    if _COLLECTION_STARTED_AT is not None:
        _COLLECTION_DURATION_S = round(time.monotonic() - _COLLECTION_STARTED_AT, 4)
    _SELECTED_COUNT = len(items)
    limit = _selection_nodeid_limit()
    selected_nodeids = [str(getattr(item, "nodeid", item)) for item in items[:limit]]
    # Marker metadata is a compact routing index, not a node-id sample. Keep
    # it complete so seed sharding can isolate load-sensitive/TUI nodes even
    # when the human-readable node-id sample is capped at 500 entries.
    selected_node_markers = {
        str(getattr(item, "nodeid", item)): sorted(
            {str(mark.name) for mark in getattr(item, "iter_markers", lambda: ())()}
        )
        for item in items
    }
    payload = _collection_payload()
    payload.update(
        {
            "selected_nodeids": selected_nodeids,
            "selected_node_markers": selected_node_markers,
            "selected_nodeids_omitted": max(0, _SELECTED_COUNT - len(selected_nodeids)),
        }
    )
    if os.environ.get("PYTEST_XDIST_WORKER"):
        _write_worker_collection_fact(payload)
    else:
        _CONTROLLER_COLLECTION_PAYLOAD = dict(payload)
        _write_selection(payload)
    _write_event(
        {
            "event": "collection_finished",
            "selected_count": _SELECTED_COUNT,
            "deselected_count": _DESELECTED_COUNT,
            **({"duration_s": _COLLECTION_DURATION_S} if _COLLECTION_DURATION_S is not None else {}),
        }
    )


@pytest.hookimpl
def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    """Append one event when pytest starts running a test node."""
    _write_event(
        {
            "event": "test_started",
            "nodeid": nodeid,
            "location": [location[0], location[1], location[2]],
        }
    )


@pytest.hookimpl
def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    """Append one event when pytest finishes running a test node."""
    _write_event(
        {
            "event": "test_finished",
            "nodeid": nodeid,
            "location": [location[0], location[1], location[2]],
        }
    )


@pytest.hookimpl
def _record_phase_report(report: Any, *, write_event: bool = True) -> None:
    """Append one phase report so slow setup/call/teardown remains visible."""
    when = str(getattr(report, "when", ""))
    nodeid = str(getattr(report, "nodeid", ""))
    outcome = _durable_report_outcome(report, str(getattr(report, "outcome", "")))
    duration = float(getattr(report, "duration", 0.0) or 0.0)
    report_key = (id(report), when, nodeid, outcome, duration)
    if report_key in _RECORDED_REPORT_KEYS:
        return
    _RECORDED_REPORT_KEYS.add(report_key)
    if when not in {"setup", "call", "teardown"}:
        return
    payload = {
        "event": "test_report",
        "nodeid": str(getattr(report, "nodeid", "")),
        "when": when,
        "outcome": outcome,
        "duration_s": round(duration, 4),
    }
    if payload["outcome"] == "failed":
        payload["longrepr"] = str(getattr(report, "longrepr", ""))
    _remember_report(payload)
    if write_event:
        _write_event(payload)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Capture the phase report before other reporting plugins transform it."""
    del item, call
    outcome = yield
    _record_phase_report(outcome.get_result())


@pytest.hookimpl
def pytest_runtest_logreport(report: Any) -> None:
    """Retain the direct/log-hook fallback used by older pytest plugins/tests."""
    # xdist forwards each worker's report to the controller. The worker has
    # already written the authoritative shard event through makereport. Keep
    # its timing in the controller's summary, but do not duplicate the ledger.
    if not os.environ.get("PYTEST_XDIST_WORKER") and getattr(report, "worker_id", None):
        _record_phase_report(report, write_event=False)
        return
    _record_phase_report(report)


@pytest.hookimpl
def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Write a compact post-run diagnosis artifact independent of pytest-json-report."""
    del session
    try:
        # Worker processes have their own in-memory slowest lists. The controller
        # receives the forwarded timings and is the only writer for the shared
        # summary path, so an empty worker summary cannot overwrite it.
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return
        collection_payload = (
            merge_worker_collection_payloads() or _CONTROLLER_COLLECTION_PAYLOAD or _collection_payload()
        )
        _write_selection(collection_payload)
        payload: dict[str, Any] = {
            "exitstatus": int(exitstatus),
            "selected_count": collection_payload["selected_count"],
            "deselected_count": collection_payload["deselected_count"],
            "slowest_reports": list(_SLOWEST_REPORTS),
        }
        if "collection_duration_s" in collection_payload:
            payload["collection_duration_s"] = collection_payload["collection_duration_s"]
        _write_summary(payload)
    finally:
        if _SESSION_STATE_STACK:
            _restore_session_state(_SESSION_STATE_STACK.pop())
