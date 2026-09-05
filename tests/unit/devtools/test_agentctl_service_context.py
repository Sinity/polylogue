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

_JOB_ID = "123e4567-e89b-42d3-a456-426614174000"
#: (environment prefix, transient unit prefix) for the current runtime and the older one.
_FAMILIES = [("AGENTCTL_", "agentctl"), ("SINNIXD_", "sinnixd")]


def _environment(prefix: str = "AGENTCTL_", operation: str = "dev_loop_proof") -> dict[str, str]:
    return {
        f"{prefix}JOB_ID": _JOB_ID,
        f"{prefix}PROJECT_ID": "polylogue",
        f"{prefix}OPERATION": operation,
    }


@pytest.mark.parametrize(("prefix", "unit_prefix"), _FAMILIES)
def test_operation_context_requires_matching_unit_and_declared_operation(prefix: str, unit_prefix: str) -> None:
    environment = _environment(prefix)
    unit = f"{unit_prefix}-job-{_JOB_ID}.service"

    assert (
        require_declared_operation_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: f"0::/user.slice/user-1000.slice/user@1000.service/agent.slice/{unit}\n",
            unit_exec_start_reader=lambda observed: (
                "{ path=/nix/store/capture/bin/sinnix-capture ; "
                "argv[]=/nix/store/capture/bin/sinnix-capture --log-path /private/job.log -- "
                "/run/current-system/sw/bin/env -i HOME=/home/sinity "
                f"{prefix}JOB_ID={_JOB_ID} {prefix}OPERATION=dev_loop_proof "
                f"{prefix}PROJECT_ID=polylogue nix develop --accept-flake-config --command "
                "python -m devtools.dev_loop_service --json ; status=0/0 ; }"
                if observed == unit
                else ""
            ),
        )
        == unit
    )


def test_forged_environment_without_matching_cgroup_fails_before_launch() -> None:
    environment = _environment()

    with pytest.raises(ValueError, match="matching transient unit"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/shell.scope\n",
            unit_exec_start_reader=lambda _unit: "",
        )


@pytest.mark.parametrize(("prefix", "unit_prefix"), _FAMILIES)
def test_matching_cgroup_with_wrong_unit_operation_fails(prefix: str, unit_prefix: str) -> None:
    environment = _environment(prefix)
    unit = f"{unit_prefix}-job-{_JOB_ID}.service"

    with pytest.raises(ValueError, match="transient unit does not match"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: f"0::/agent.slice/{unit}\n",
            unit_exec_start_reader=lambda _unit: (
                "/run/current-system/sw/bin/env -i "
                f"{prefix}JOB_ID={_JOB_ID} {prefix}PROJECT_ID=polylogue {prefix}OPERATION=other"
            ),
        )


def test_unit_must_declare_the_family_the_environment_supplied() -> None:
    """A legacy-named unit cannot vouch for an AGENTCTL_* environment, or vice versa."""
    environment = _environment("AGENTCTL_")
    unit = f"sinnixd-job-{_JOB_ID}.service"

    with pytest.raises(ValueError, match="transient unit does not match"):
        require_declared_operation_context(
            "dev_loop_proof",
            environment=environment,
            cgroup_reader=lambda: f"0::/agent.slice/{unit}\n",
            unit_exec_start_reader=lambda _unit: (
                f"/run/current-system/sw/bin/env -i SINNIXD_JOB_ID={_JOB_ID} "
                "SINNIXD_PROJECT_ID=polylogue SINNIXD_OPERATION=dev_loop_proof"
            ),
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
