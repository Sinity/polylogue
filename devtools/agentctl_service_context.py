"""Local defense-in-depth checks for declared runtime operations.

The Sinnix runtime (agentctl) remains the authority for admission, exact-head
validation, and leases. These checks only ensure a private project command is
running as the declared operation, inside the pool the runtime placed that
already-authorized job in.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

from devtools.agent_env import inside_pool_cgroup, runtime_env

_PROJECT_ID = "polylogue"
#: The pool every declared proof operation runs in (``.agentctl/project.toml``).
DEFAULT_POOL = "interactive"


def require_declared_operation_context(
    operation: str,
    *,
    pool: str = DEFAULT_POOL,
    environment: Mapping[str, str] | None = None,
    cgroup_reader: Callable[[], str] | None = None,
) -> str:
    """Require the declared operation's runtime context before launch; returns the job id.

    The runtime exports the job id, project and operation into the job's
    environment and places the job in ``agentctl-<pool>.slice``. The
    environment names the expected job; the cgroup is what an ordinary shell
    invocation cannot supply. Identity is read as ``AGENTCTL_*`` with
    ``SINNIXD_*`` fallback. This does not replace the runtime's exact-head or
    lease validation.
    """
    env = os.environ if environment is None else environment
    job_id = runtime_env("AGENTCTL_JOB_ID", env)
    if not job_id:
        raise ValueError("runtime service context has no job id")
    if runtime_env("AGENTCTL_PROJECT_ID", env) != _PROJECT_ID:
        raise ValueError("runtime service context does not belong to this project")
    if runtime_env("AGENTCTL_OPERATION", env) != operation:
        raise ValueError("runtime service context does not match the declared operation")
    if not inside_pool_cgroup(pool, cgroup_reader=cgroup_reader):
        raise ValueError(f"runtime service context is not inside the {pool} pool")
    return job_id


def terminate_process_group(process: subprocess.Popen[Any], *, timeout_s: float = 2.0) -> None:
    """Boundedly signal a process group; systemd remains final cgroup reaper."""
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
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=timeout_s)
