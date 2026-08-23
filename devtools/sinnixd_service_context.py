"""Local defense-in-depth checks for declared Sinnixd service commands.

Sinnixd remains the authority for admission, exact-head validation, and leases.
These checks only ensure a private project command did not escape the transient
unit Sinnixd created for that already-authorized job.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID

_CGROUP_PATH = Path("/proc/self/cgroup")
_PROJECT_ID = "polylogue"


def _unit_name(job_id: str) -> str:
    try:
        return f"sinnixd-job-{UUID(job_id)}.service"
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Sinnixd job identity must be a UUID") from error


def _current_cgroup(read_text: Callable[[], str]) -> str:
    for line in read_text().splitlines():
        _hierarchy, separator, path = line.partition("::")
        if separator and path:
            return path
    raise ValueError("Sinnixd service context requires a unified cgroup path")


def _unit_environment(unit: str) -> Mapping[str, str]:
    completed = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=Environment", "--value"],
        capture_output=True,
        check=False,
        text=True,
        timeout=2,
    )
    if completed.returncode != 0:
        raise ValueError("Sinnixd service context could not inspect its transient unit")
    return dict(item.split("=", 1) for item in shlex.split(completed.stdout) if "=" in item)


def require_declared_service_context(
    operation: str,
    *,
    environment: Mapping[str, str] | None = None,
    cgroup_reader: Callable[[], str] | None = None,
    unit_environment_reader: Callable[[str], Mapping[str, str]] | None = None,
) -> str:
    """Require the declared operation's actual transient unit before launch.

    Environment variables identify the expected service, but are not trusted on
    their own. The current cgroup must name the matching Sinnixd unit and the
    user manager must report the same job, project, and operation for that unit.
    This does not replace Sinnixd's own exact-head and lease validation.
    """
    env = os.environ if environment is None else environment
    job_id = env.get("SINNIXD_JOB_ID", "")
    unit = _unit_name(job_id)
    expected = {
        "SINNIXD_JOB_ID": job_id,
        "SINNIXD_PROJECT_ID": _PROJECT_ID,
        "SINNIXD_OPERATION": operation,
    }
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("Sinnixd service context does not match the declared operation")
    read_cgroup = cgroup_reader or (lambda: _CGROUP_PATH.read_text(encoding="utf-8"))
    cgroup = _current_cgroup(read_cgroup)
    if f"/agent.slice/{unit}" not in cgroup:
        raise ValueError("Sinnixd service context is not inside the matching transient unit")
    read_unit_environment = unit_environment_reader or _unit_environment
    unit_environment = read_unit_environment(unit)
    if any(unit_environment.get(key) != value for key, value in expected.items()):
        raise ValueError("Sinnixd transient unit does not match the declared operation")
    return unit


def terminate_process_group(process: subprocess.Popen[Any], *, timeout_s: float = 2.0) -> None:
    """Terminate a launcher and all children in its dedicated process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=timeout_s)
    else:
        time.sleep(timeout_s)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait(timeout=timeout_s)
