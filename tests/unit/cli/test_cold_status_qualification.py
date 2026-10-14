"""Cold timing admits real EMPTY serving, never daemon absence or another owner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.infra.cli_cold_start import assert_served_empty_status


def _served_empty(root: Path) -> dict[str, Any]:
    return {
        "source": "daemon",
        "daemon_liveness": True,
        "ok": False,
        "archive_root": str(root),
        "active_archive_root_matches_configured": True,
        "sessions": 0,
        "messages": 0,
        "status_snapshot": {"state": "fresh", "frame_changed": False, "frame_error": None, "refresh_error": None},
        "storage": {"identity": {"process_id": 123}},
        "raw_materialization_readiness": {"raw_artifact_count": 0, "available": False},
    }


def test_cold_status_qualifies_empty_serving_without_claiming_health(tmp_path: Path) -> None:
    payload = _served_empty(tmp_path)
    assert_served_empty_status(payload, root=tmp_path, daemon_pid=123)
    assert payload["ok"] is False
    assert payload["raw_materialization_readiness"]["available"] is False


@pytest.mark.parametrize(
    "fault", ["absent", "stale", "foreign_pid", "foreign_archive", "populated", "restarted", "local"]
)
def test_cold_status_rejects_unqualified_timing_payload(tmp_path: Path, fault: str) -> None:
    payload = deepcopy(_served_empty(tmp_path))
    if fault == "absent":
        payload["daemon_liveness"] = False
        payload["status_snapshot"]["state"] = "unavailable"
    elif fault == "stale":
        payload["status_snapshot"]["state"] = "stale"
    elif fault == "foreign_pid":
        payload["storage"]["identity"]["process_id"] = 456
    elif fault == "foreign_archive":
        payload["archive_root"] = str(tmp_path / "foreign")
    elif fault == "populated":
        payload["raw_materialization_readiness"]["raw_artifact_count"] = 1
    elif fault == "restarted":
        payload["status_snapshot"]["frame_changed"] = True
    else:
        payload["source"] = "direct"
    with pytest.raises(AssertionError):
        assert_served_empty_status(payload, root=tmp_path, daemon_pid=123)
