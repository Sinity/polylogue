"""Runtime environment names and bounds on test execution inside agent jobs.

The Sinnix runtime (agentctl) exports ``AGENTCTL_*`` variables into every job
it starts and places the job in ``agentctl-<pool>.slice``; older hosts export
the same values as ``SINNIXD_*`` (the pool as ``SINNIXD_QUEUE_POOL``) and use
``sinnixd-pueue-<pool>.slice``. Every consumer reads the names through the
helpers here so the fallback lives in one place.

An agent lane runs in the agent pool. Tests for a lane run once, as the pull
request's hosted ``verify`` check; a lane running the affected or complete
tier itself doubles that load outside admission. Focused runs stay available
through the pytest pool, bounded to a few workers.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

RUNTIME_ENV_PREFIX = "AGENTCTL_"
LEGACY_RUNTIME_ENV_PREFIX = "SINNIXD_"
#: Variables whose legacy name is not the mechanical prefix swap.
_LEGACY_RUNTIME_ENV_NAMES = {"AGENTCTL_POOL": "SINNIXD_QUEUE_POOL"}

AGENT_PRINCIPAL_ENV = "AGENTCTL_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"
#: The pool the runtime placed the job in.
QUEUE_POOL_ENV = "AGENTCTL_POOL"
PYTEST_POOL = "pytest"
_CGROUP_PATH = Path("/proc/self/cgroup")


def pool_slices(pool: str) -> frozenset[str]:
    """The slices the runtime places a ``pool`` job in: current and older host."""
    return frozenset({f"agentctl-{pool}.slice", f"sinnixd-pueue-{pool}.slice"})


_AGENT_CGROUP_SLICES = frozenset({"agent.slice"}) | pool_slices("agent")
_PYTEST_CGROUP_SLICES = pool_slices(PYTEST_POOL)


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


def inside_pool_cgroup(pool: str, *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    """Whether this process's cgroup is under the slice of the runtime's ``pool``."""
    return _inside_cgroup(cgroup_reader, pool_slices(pool))


def _inside_agent_cgroup(cgroup_reader: Callable[[], str] | None) -> bool:
    return _inside_cgroup(cgroup_reader, _AGENT_CGROUP_SLICES)


def _inside_pytest_cgroup(cgroup_reader: Callable[[], str] | None) -> bool:
    return _inside_cgroup(cgroup_reader, _PYTEST_CGROUP_SLICES)


def inside_agent_job(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    return runtime_env(AGENT_PRINCIPAL_ENV, env) == AGENT_PRINCIPAL or _inside_agent_cgroup(cgroup_reader)


def declared_pool(env: Mapping[str, str]) -> str | None:
    """The pool the runtime says this job runs in, or None outside a job."""
    return runtime_env(QUEUE_POOL_ENV, env)


def inside_pytest_pool(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    """Whether this process holds the host's pytest slot: its job runs in the pytest pool.

    Ownership is the pool the runtime placed the job in -- the pool it
    exported, or the pool slice in the cgroup. A job id, a principal or an
    operation name is never ownership: a lane inherits all three and still
    has to queue.
    """
    return declared_pool(env) == PYTEST_POOL or _inside_pytest_cgroup(cgroup_reader)


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
        inside_pytest_pool(env, cgroup_reader=cgroup_reader)
        or not inside_agent_job(env, cgroup_reader=cgroup_reader)
        or "--quick" in argv
    ):
        return None
    return (
        "devtools verify: test tiers do not run inside agent jobs; the lane's tests run once as the "
        "pull request's hosted `verify` check (by hand: "
        "`agentctl job start polylogue verify_affected --workspace <workspace>`). "
        "Use `devtools verify --quick` for the static gates and `devtools test <selection>` for focused runs."
    )


def refuse_bare_pytest(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> str | None:
    """Why a pytest session is refused inside an agent job, or None."""
    if (
        inside_pytest_pool(env, cgroup_reader=cgroup_reader)
        or not inside_agent_job(env, cgroup_reader=cgroup_reader)
        or env.get(HARNESS_RUN_ENV)
    ):
        return None
    return (
        "pytest: bare pytest does not run inside agent jobs; "
        "use `devtools test <selection>`, which queues on the host's pytest slot"
    )
