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
