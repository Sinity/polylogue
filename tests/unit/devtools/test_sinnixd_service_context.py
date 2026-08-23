from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from devtools.sinnixd_service_context import require_declared_service_context, terminate_process_group

_JOB_ID = "123e4567-e89b-42d3-a456-426614174000"


def _environment(operation: str = "dev_loop_proof") -> dict[str, str]:
    return {
        "SINNIXD_JOB_ID": _JOB_ID,
        "SINNIXD_PROJECT_ID": "polylogue",
        "SINNIXD_OPERATION": operation,
    }


def test_service_context_requires_matching_unit_and_declared_operation() -> None:
    environment = _environment()
    unit = "sinnixd-job-123e4567-e89b-42d3-a456-426614174000.service"

    assert (
        require_declared_service_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: f"0::/user.slice/user-1000.slice/user@1000.service/agent.slice/{unit}\n",
            unit_environment_reader=lambda observed: {**environment, "observed_unit": observed},
        )
        == unit
    )


def test_forged_environment_without_matching_cgroup_fails_before_launch() -> None:
    environment = _environment()

    with pytest.raises(ValueError, match="matching transient unit"):
        require_declared_service_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/shell.scope\n",
            unit_environment_reader=lambda _unit: environment,
        )


def test_matching_cgroup_with_wrong_unit_operation_fails() -> None:
    environment = _environment()
    unit = "sinnixd-job-123e4567-e89b-42d3-a456-426614174000.service"

    with pytest.raises(ValueError, match="transient unit does not match"):
        require_declared_service_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: f"0::/agent.slice/{unit}\n",
            unit_environment_reader=lambda _unit: {**environment, "SINNIXD_OPERATION": "other"},
        )


@pytest.mark.uses_real_clock("waits for a real process group and its child to exit")
def test_terminate_process_group_reaps_descendants(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); open(sys.argv[1], 'w').write(str(child.pid)); time.sleep(60)",
            str(child_pid),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 2
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        descendant = int(child_pid.read_text(encoding="utf-8"))
        terminate_process_group(process, timeout_s=0.2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(descendant, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        assert process.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
    finally:
        terminate_process_group(process, timeout_s=0.2)
