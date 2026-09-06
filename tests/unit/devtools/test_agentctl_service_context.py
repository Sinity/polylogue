from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from devtools.agentctl_service_context import (
    require_declared_operation_context,
    terminate_process_group,
)

_JOB_ID = "polylogue-dev_loop_proof-8e3c63a7"
#: (environment prefix, pool slice) for the current runtime and the older one.
_FAMILIES = [("AGENTCTL_", "agentctl-interactive.slice"), ("SINNIXD_", "sinnixd-pueue-interactive.slice")]
_UNIT = "agentctl-interactive-polylogue-dev_loop_proof-8e3c63a7-0123456789ab.service"


def _environment(prefix: str = "AGENTCTL_", operation: str = "dev_loop_proof") -> dict[str, str]:
    return {
        f"{prefix}JOB_ID": _JOB_ID,
        f"{prefix}PROJECT_ID": "polylogue",
        f"{prefix}OPERATION": operation,
    }


def _cgroup(slice_name: str) -> str:
    return f"0::/user.slice/user-1000.slice/user@1000.service/{slice_name}/{_UNIT}\n"


@pytest.mark.parametrize(("prefix", "slice_name"), _FAMILIES)
def test_operation_context_requires_the_pool_and_the_declared_operation(prefix: str, slice_name: str) -> None:
    assert (
        require_declared_operation_context(
            "dev_loop_proof",
            environment=_environment(prefix),
            cgroup_reader=lambda: _cgroup(slice_name),
        )
        == _JOB_ID
    )


def test_forged_environment_without_the_pool_cgroup_fails_before_launch() -> None:
    with pytest.raises(ValueError, match="not inside the interactive pool"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=_environment(),
            cgroup_reader=lambda: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/shell.scope\n",
        )


def test_another_pool_is_not_the_declared_one() -> None:
    with pytest.raises(ValueError, match="not inside the interactive pool"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=_environment(),
            cgroup_reader=lambda: _cgroup("agentctl-pytest.slice"),
        )


def test_the_pool_is_the_operation_declaration() -> None:
    assert (
        require_declared_operation_context(
            "pytest_focused",
            pool="pytest",
            environment=_environment(operation="pytest_focused"),
            cgroup_reader=lambda: _cgroup("agentctl-pytest.slice"),
        )
        == _JOB_ID
    )


def test_matching_cgroup_with_another_operation_fails() -> None:
    with pytest.raises(ValueError, match="does not match the declared operation"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=_environment(operation="other"),
            cgroup_reader=lambda: _cgroup("agentctl-interactive.slice"),
        )


def test_another_project_fails() -> None:
    environment = _environment() | {"AGENTCTL_PROJECT_ID": "sinex"}

    with pytest.raises(ValueError, match="does not belong to this project"):
        require_declared_operation_context(
            "dev_loop_proof", environment=environment, cgroup_reader=lambda: _cgroup("agentctl-interactive.slice")
        )


def test_a_missing_job_id_fails() -> None:
    environment = _environment()
    del environment["AGENTCTL_JOB_ID"]

    with pytest.raises(ValueError, match="no job id"):
        require_declared_operation_context(
            "dev_loop_proof", environment=environment, cgroup_reader=lambda: _cgroup("agentctl-interactive.slice")
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


def test_terminate_process_group_leaves_final_reap_to_systemd_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, signal.Signals]] = []

    class StubbornProcess:
        pid = 4242

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            raise subprocess.TimeoutExpired(cmd=["stubborn"], timeout=timeout)

    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    terminate_process_group(cast(Any, StubbornProcess()), timeout_s=0.01)

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
