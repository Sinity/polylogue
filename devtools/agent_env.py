"""Bounds on test execution inside agent jobs.

An agent lane runs under sinnixd's ``sinnixd-pueue-agent.slice``.
Tests for a lane run exactly once, as the declared ``verify_affected`` job
in the host's single-worker pytest pool; a lane running the affected or
complete tier itself doubles that load outside admission. Focused runs stay
available, bounded to a few workers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

#: The queue runner exports ``AGENTCTL_<NAME>``; the earlier daemon exported
#: ``SINNIXD_<NAME>``. Every read accepts both, newest first.
RUNTIME_ENV_PREFIXES = ("AGENTCTL_", "SINNIXD_")
AGENT_PRINCIPAL_ENV = "AGENTCTL_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"
QUEUE_WORKER_ENV = "AGENTCTL_QUEUE_WORKER"
QUEUE_POOL_ENV = "AGENTCTL_QUEUE_POOL"


def runtime_env(env: Mapping[str, str], name: str) -> str | None:
    """Read a queue-runner variable under any of its prefixes.

    ``name`` is the bare suffix (``"JOB_ID"``) or a full name under either
    prefix; the newest prefix wins when both are set.
    """
    suffix = name
    for prefix in RUNTIME_ENV_PREFIXES:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            break
    for prefix in RUNTIME_ENV_PREFIXES:
        value = env.get(prefix + suffix)
        if value is not None:
            return value
    return None


QUEUE_WORKER_VALUE = "1"
PYTEST_POOL = "pytest"
#: Every operation declaring ``pool = "pytest"`` in ``.agentctl/project.toml``,
#: plus the ``test`` operation the pytest slot's own launch document names. A
#: queue runner that does not export ``SINNIXD_QUEUE_POOL`` leaves this the only
#: classifier, and a pytest-pool operation missing here re-queues into the
#: single-slot group its own worker already occupies, which cannot drain.
PYTEST_WORKER_OPERATIONS = frozenset({"test", "verify_affected", "verify_all"})
_CGROUP_PATH = Path("/proc/self/cgroup")
_AGENT_CGROUP_SLICES = frozenset({"agent.slice", "sinnixd-pueue-agent.slice"})
_PYTEST_CGROUP_SLICES = frozenset({"sinnixd-pueue-pytest.slice"})


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
    return runtime_env(env, AGENT_PRINCIPAL_ENV) == AGENT_PRINCIPAL or _inside_agent_cgroup(cgroup_reader)


def inside_declared_pytest_worker(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    """Whether a queue worker is the declared pytest operation itself.

    The queue worker inherits the lane's principal, so principal identity is
    not sufficient to classify the process. The queue marker and job identity
    come from ``sinnixd-queue-run``; the pytest slice binds that identity to the
    pool declared by the operation.
    """
    if runtime_env(env, QUEUE_WORKER_ENV) != QUEUE_WORKER_VALUE or not runtime_env(env, "JOB_ID"):
        return False
    declared_pool = runtime_env(env, QUEUE_POOL_ENV)
    if declared_pool is not None:
        if declared_pool != PYTEST_POOL:
            return False
    elif runtime_env(env, "OPERATION") not in PYTEST_WORKER_OPERATIONS:
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
