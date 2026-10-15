"""Cold timing admits real EMPTY serving, never daemon absence or another owner."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.infra.cli_cold_start import assert_served_empty_status, stop_owned_process_group
from tests.infra.frozen_clock import FrozenClock


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


@pytest.mark.uses_real_clock
def test_cold_status_cleanup_stops_group_after_leader_exit(tmp_path: Path) -> None:
    child_ready = tmp_path / "child-ready"
    child = "import pathlib,time; pathlib.Path(__import__('sys').argv[1]).touch(); time.sleep(120)"
    leader = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); "
        "p=pathlib.Path(sys.argv[1]); "
        "exec('while not p.exists(): time.sleep(0.01)')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader, str(child_ready)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    proof: dict[str, object] = {}
    try:
        assert process.wait(timeout=10) == 0
        assert child_ready.exists()
        os.killpg(process.pid, 0)  # The descendant still occupies the owned group.
        stop_owned_process_group(process, proof)
        assert proof["daemon_returncode"] == 0
        assert proof["owned_process_group_id"] == process.pid
        assert proof["owned_process_group_stopped"] is True
        assert proof.get("forced_cleanup") is not True
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _compact_populated_status(frozen_clock: FrozenClock) -> dict[str, Any]:
    from polylogue.cli.commands.status import _compact_status_payload

    return _compact_status_payload(
        {
            "ok": True,
            "daemon_liveness": True,
            "status_snapshot": {
                "state": "fresh",
                "captured_at": frozen_clock.now().isoformat(),
                "age_s": 0,
                "frame": "synthetic-frame",
                "current_frame": "synthetic-frame",
                "frame_changed": False,
            },
            "raw_failure_lifecycle_available": True,
            "raw_failure_lifecycle_state": "healthy",
            "raw_parse_failures": 0,
            "raw_validation_failures": 0,
            "raw_unexplained_failures": 0,
            "raw_materialization_readiness": {"raw_artifact_count": 7},
            "raw_frontier_integrity": {
                "available": True,
                "overall_status": "healthy",
                "broken_head_status": "healthy",
                "broken_head_count": 0,
                "broken_head_checked_count": 7,
                "broken_head_samples": [],
                "broken_head_reason": "",
                "missing_source_raw_status": "healthy",
                "missing_source_raw_count": 0,
                "missing_source_raw_samples": [],
                "missing_source_raw_reason": "",
                "cursor_ahead_status": "healthy",
                "cursor_ahead_count": 0,
                "cursor_ahead_checked_count": 5,
                "cursor_head_comparison_count": 5,
                "cursor_ahead_comparison_count": 0,
                "cursor_ahead_samples": [],
                "cursor_authority_gap_count": 0,
                "cursor_authority_gap_samples": [],
                "cursor_authority_deferred_count": 0,
                "cursor_ahead_reason": "",
            },
        },
        source="daemon",
    )


@pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
def test_populated_cold_driver_accepts_the_actual_compact_status_shape(frozen_clock: FrozenClock) -> None:
    from tests.benchmarks.test_cli_cold_start import _require_healthy_status

    payload = _compact_populated_status(frozen_clock)
    assert payload["ok"] is True
    assert not any(key.endswith("_samples") for key in payload["raw_frontier_integrity"])
    _require_healthy_status(payload)


@pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "unavailable",
        "missing_count",
        "violation",
        "gap",
        "negative_count",
        "bool_count",
        "bool_available",
        "wrong_raw_count",
        "bool_raw_count",
        "fewer_comparisons",
        "unchecked_comparisons",
        "ahead_comparison",
        "stale_frame",
    ],
)
def test_populated_cold_driver_rejects_unproved_compact_status(frozen_clock: FrozenClock, fault: str) -> None:
    from tests.benchmarks.test_cli_cold_start import _require_healthy_status

    payload = _compact_populated_status(frozen_clock)
    assert payload["ok"] is True
    frontier = payload["raw_frontier_integrity"]
    if fault == "unknown":
        frontier["cursor_ahead_status"] = "unknown"
    elif fault == "unavailable":
        frontier["available"] = False
    elif fault == "missing_count":
        del frontier["cursor_ahead_checked_count"]
    elif fault == "violation":
        frontier["missing_source_raw_count"] = 1
    elif fault == "gap":
        frontier["cursor_authority_gap_count"] = 1
    elif fault == "negative_count":
        frontier["cursor_ahead_checked_count"] = -1
    elif fault == "bool_count":
        frontier["cursor_ahead_checked_count"] = True
    elif fault == "bool_available":
        frontier["available"] = 1
    elif fault == "wrong_raw_count":
        payload["raw_materialization_readiness"]["raw_artifact_count"] = 8
    elif fault == "bool_raw_count":
        payload["raw_materialization_readiness"]["raw_artifact_count"] = True
    elif fault == "fewer_comparisons":
        frontier["cursor_head_comparison_count"] = 4
    elif fault == "unchecked_comparisons":
        frontier["cursor_ahead_checked_count"] = 0
    elif fault == "ahead_comparison":
        frontier["cursor_ahead_comparison_count"] = 1
    else:
        payload["status_snapshot"]["frame_changed"] = True
    with pytest.raises((AssertionError, KeyError)):
        _require_healthy_status(payload)
