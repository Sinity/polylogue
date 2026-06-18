"""Incremental pytest event ledger for ``devtools verify``.

The pytest-json-report and JUnit artifacts are written at session end. When a
long verify run is killed for timeout or output stall, those reports may never
flush. This plugin writes one JSON object per completed test call so the
operator still has node-level failure evidence after an interrupted run.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

_EVENTS_ENV = "POLYLOGUE_PYTEST_EVENTS_PATH"


def _write_event(payload: dict[str, Any]) -> None:
    raw_path = os.environ.get(_EVENTS_ENV)
    if not raw_path:
        return
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    path = Path(raw_path)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    """Append one test-call event when configured by ``devtools verify``."""
    when = str(getattr(report, "when", ""))
    outcome = str(getattr(report, "outcome", ""))
    if when != "call" and outcome not in {"failed", "error"}:
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
    _write_event(payload)
