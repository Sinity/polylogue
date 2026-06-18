"""Incremental pytest event ledger for ``devtools verify``.

The pytest-json-report and JUnit artifacts are written at session end. When a
long verify run is killed for timeout or output stall, those reports may never
flush. This plugin writes one JSON object per completed test call so the
operator still has node-level failure evidence after an interrupted run.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EVENTS_ENV = "POLYLOGUE_PYTEST_EVENTS_PATH"


def pytest_runtest_logreport(report: Any) -> None:
    """Append one test-call event when configured by ``devtools verify``."""
    if getattr(report, "when", None) != "call":
        return
    raw_path = os.environ.get(_EVENTS_ENV)
    if not raw_path:
        return
    payload = {
        "event": "test_report",
        "updated_at": datetime.now(UTC).isoformat(),
        "nodeid": str(getattr(report, "nodeid", "")),
        "outcome": str(getattr(report, "outcome", "")),
        "duration_s": round(float(getattr(report, "duration", 0.0) or 0.0), 4),
    }
    if payload["outcome"] == "failed":
        payload["longrepr"] = str(getattr(report, "longrepr", ""))
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
