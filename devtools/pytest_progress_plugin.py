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
_COLLECTION_STARTED_AT: float | None = None
_COLLECTION_DURATION_S: float | None = None
_SLOW_REPORT_LIMIT = 20
_DEFAULT_SELECTION_NODEID_LIMIT = 500


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


@pytest.hookimpl
def pytest_sessionstart(session: Any) -> None:
    """Reset per-session ledgers when tests invoke pytest in-process."""
    del session
    global _COLLECTION_STARTED_AT, _COLLECTION_DURATION_S, _DESELECTED_COUNT, _SELECTED_COUNT
    _DESELECTED_NODEIDS_SAMPLE.clear()
    _DESELECTED_COUNT = 0
    _SELECTED_COUNT = 0
    _SLOWEST_REPORTS.clear()
    _COLLECTION_STARTED_AT = None
    _COLLECTION_DURATION_S = None


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
    global _COLLECTION_DURATION_S, _SELECTED_COUNT
    if _COLLECTION_STARTED_AT is not None:
        _COLLECTION_DURATION_S = round(time.monotonic() - _COLLECTION_STARTED_AT, 4)
    _SELECTED_COUNT = len(items)
    limit = _selection_nodeid_limit()
    selected_nodeids = [str(getattr(item, "nodeid", item)) for item in items[:limit]]
    payload: dict[str, Any] = {
        "selected_count": _SELECTED_COUNT,
        "deselected_count": _DESELECTED_COUNT,
        "selected_nodeids": selected_nodeids,
        "selected_nodeids_omitted": max(0, _SELECTED_COUNT - len(selected_nodeids)),
        "deselected_nodeids": list(_DESELECTED_NODEIDS_SAMPLE),
        "deselected_nodeids_omitted": max(0, _DESELECTED_COUNT - len(_DESELECTED_NODEIDS_SAMPLE)),
        "nodeid_sample_limit": limit,
    }
    if _COLLECTION_DURATION_S is not None:
        payload["collection_duration_s"] = _COLLECTION_DURATION_S
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
def pytest_runtest_logreport(report: Any) -> None:
    """Append one phase report so slow setup/call/teardown remains visible."""
    when = str(getattr(report, "when", ""))
    outcome = str(getattr(report, "outcome", ""))
    if when not in {"setup", "call", "teardown"}:
        return
    payload = {
        "event": "test_report",
        "nodeid": str(getattr(report, "nodeid", "")),
        "when": when,
        "outcome": outcome,
        "duration_s": round(float(getattr(report, "duration", 0.0) or 0.0), 4),
    }
    if payload["outcome"] == "failed":
        payload["longrepr"] = str(getattr(report, "longrepr", ""))
    _remember_report(payload)
    _write_event(payload)


@pytest.hookimpl
def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Write a compact post-run diagnosis artifact independent of pytest-json-report."""
    del session
    payload: dict[str, Any] = {
        "exitstatus": int(exitstatus),
        "selected_count": _SELECTED_COUNT,
        "deselected_count": _DESELECTED_COUNT,
        "slowest_reports": list(_SLOWEST_REPORTS),
    }
    if _COLLECTION_DURATION_S is not None:
        payload["collection_duration_s"] = _COLLECTION_DURATION_S
    _write_summary(payload)
