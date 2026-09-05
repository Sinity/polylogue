"""Runtime environment names and bounds on test execution inside agent jobs.

The Sinnix runtime (agentctl) exports ``AGENTCTL_*`` variables into every job
it starts and places the job in ``agentctl-<pool>.slice``; older hosts export
the same values as ``SINNIXD_*`` (the pool as ``SINNIXD_QUEUE_POOL``) and use
``sinnixd-pueue-<pool>.slice``. Every consumer reads the names through the
helpers here so the fallback lives in one place.

An agent lane runs in the agent pool. Tests for a lane run exactly once, as
the declared ``verify_affected`` job in the host's single-worker pytest pool;
a lane running the affected or complete tier itself doubles that load outside
admission. Focused runs stay available, bounded to a few workers.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

RUNTIME_ENV_PREFIX = "AGENTCTL_"
LEGACY_RUNTIME_ENV_PREFIX = "SINNIXD_"
#: Variables whose legacy name is not the mechanical prefix swap.
_LEGACY_RUNTIME_ENV_NAMES = {"AGENTCTL_POOL": "SINNIXD_QUEUE_POOL"}
#: The runtime's transient job unit, ``<prefix>-job-<uuid>.service``.
RUNTIME_UNIT_PREFIXES = ("agentctl", "sinnixd")

AGENT_PRINCIPAL_ENV = "AGENTCTL_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"
QUEUE_WORKER_ENV = "AGENTCTL_QUEUE_WORKER"
QUEUE_POOL_ENV = "AGENTCTL_POOL"
QUEUE_WORKER_VALUE = "1"
PYTEST_POOL = "pytest"
#: Every operation declaring ``pool = "pytest"`` in ``.agentctl/project.toml``,
#: plus the ``test`` operation the pytest slot's own launch document names. A
#: queue runner that does not export the pool leaves this the only classifier,
#: and a pytest-pool operation missing here re-queues into the single-slot
#: group its own worker already occupies, which cannot drain.
PYTEST_WORKER_OPERATIONS = frozenset({"test", "verify_affected", "verify_all"})
_CGROUP_PATH = Path("/proc/self/cgroup")
_AGENT_CGROUP_SLICES = frozenset({"agent.slice", "agentctl-agent.slice", "sinnixd-pueue-agent.slice"})
_PYTEST_CGROUP_SLICES = frozenset({"agentctl-pytest.slice", "sinnixd-pueue-pytest.slice"})


def runtime_env_names(variable: str) -> tuple[str, str]:
    """``(AGENTCTL_<name>, SINNIXD_<name>)`` for an ``AGENTCTL_<name>`` variable."""
    if not variable.startswith(RUNTIME_ENV_PREFIX):
        raise ValueError(f"{variable} is not an {RUNTIME_ENV_PREFIX}* runtime variable")
    legacy = _LEGACY_RUNTIME_ENV_NAMES.get(variable)
    if legacy is None:
        legacy = LEGACY_RUNTIME_ENV_PREFIX + variable.removeprefix(RUNTIME_ENV_PREFIX)
    return (variable, legacy)


def runtime_env(variable: str, env: Mapping[str, str] | None = None) -> str | None:
    """The runtime variable ``AGENTCTL_<name>``, or its ``SINNIXD_*`` name when only that is set."""
    source = os.environ if env is None else env
    for candidate in runtime_env_names(variable):
        value = source.get(candidate)
        if value:
            return value
    return None


def _inside_cgroup(cgroup_reader: Callable[[], str] | None, slices: frozenset[str]) -> bool:
    read_cgroup = cgroup_reader or (lambda: _CGROUP_PATH.read_text(encoding="utf-8"))
    try:
        cgroup = read_cgroup()
    except OSError:
        return False
    for line in cgroup.splitlines():
        _hierarchy, separator, path = line.partition("::")
        if separator and any(part in slices for part in PurePosixPath(path).parts):
            return True
    return False


def _inside_agent_cgroup(cgroup_reader: Callable[[], str] | None) -> bool:
    return _inside_cgroup(cgroup_reader, _AGENT_CGROUP_SLICES)


def _inside_pytest_cgroup(cgroup_reader: Callable[[], str] | None) -> bool:
    return _inside_cgroup(cgroup_reader, _PYTEST_CGROUP_SLICES)


def inside_agent_job(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    return runtime_env(AGENT_PRINCIPAL_ENV, env) == AGENT_PRINCIPAL or _inside_agent_cgroup(cgroup_reader)


def inside_declared_pytest_worker(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    """Whether a queue worker is the declared pytest operation itself.

    The queue worker inherits the lane's principal, so principal identity is
    not sufficient to classify the process. The queue marker and job identity
    come from the runtime's queue runner; the pytest slice binds that identity
    to the pool declared by the operation.
    """
    if runtime_env(QUEUE_WORKER_ENV, env) != QUEUE_WORKER_VALUE or not runtime_env("AGENTCTL_JOB_ID", env):
        return False
    declared_pool = runtime_env(QUEUE_POOL_ENV, env)
    if declared_pool is not None:
        if declared_pool != PYTEST_POOL:
            return False
    elif runtime_env("AGENTCTL_OPERATION", env) not in PYTEST_WORKER_OPERATIONS:
        return False
    return _inside_pytest_cgroup(cgroup_reader)


def agent_worker_cap(
    requested: int | None,
    env: Mapping[str, str],
    *,
    cgroup_reader: Callable[[], str] | None = None,
) -> int | None:
    """The worker count a focused run may use; unchanged outside agent jobs."""
    if not inside_agent_job(env, cgroup_reader=cgroup_reader):
        return requested
    if requested is None:
        return AGENT_MAX_PYTEST_WORKERS
    # Zero is a configuration, not an absent one: it asks for no xdist at all,
    # which is always within the cap.
    return min(max(requested, 0), AGENT_MAX_PYTEST_WORKERS)


def refuse_verify_tier(
    argv: list[str], env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None
) -> str | None:
    """Why a ``devtools verify`` invocation is refused inside an agent job, or None."""
    if (
        inside_declared_pytest_worker(env, cgroup_reader=cgroup_reader)
        or not inside_agent_job(env, cgroup_reader=cgroup_reader)
        or "--quick" in argv
    ):
        return None
    return (
        "devtools verify: test tiers do not run inside agent jobs; the lane's tests run once as "
        "`agentctl job start <project> verify_affected --workspace <workspace>`. "
        "Use `devtools verify --quick` for the static gates and `devtools test <selection>` for focused runs."
    )


def refuse_bare_pytest(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> str | None:
    """Why a pytest session is refused inside an agent job, or None."""
    if (
        inside_declared_pytest_worker(env, cgroup_reader=cgroup_reader)
        or not inside_agent_job(env, cgroup_reader=cgroup_reader)
        or env.get(HARNESS_RUN_ENV)
    ):
        return None
    return "pytest: bare pytest does not run inside agent jobs; use `devtools test <selection>`"
