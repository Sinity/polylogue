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

AGENT_PRINCIPAL_ENV = "SINNIXD_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"
_CGROUP_PATH = Path("/proc/self/cgroup")
_AGENT_CGROUP_SLICES = frozenset({"agent.slice", "sinnixd-pueue-agent.slice"})


def _inside_agent_cgroup(cgroup_reader: Callable[[], str] | None) -> bool:
    read_cgroup = cgroup_reader or (lambda: _CGROUP_PATH.read_text(encoding="utf-8"))
    try:
        cgroup = read_cgroup()
    except OSError:
        return False
    for line in cgroup.splitlines():
        _hierarchy, separator, path = line.partition("::")
        if separator and any(part in _AGENT_CGROUP_SLICES for part in PurePosixPath(path).parts):
            return True
    return False


def inside_agent_job(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    return env.get(AGENT_PRINCIPAL_ENV) == AGENT_PRINCIPAL or _inside_agent_cgroup(cgroup_reader)


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
    if not inside_agent_job(env, cgroup_reader=cgroup_reader) or "--quick" in argv:
        return None
    return (
        "devtools verify: test tiers do not run inside agent jobs; the lane's tests run once as "
        "`agentctl job start <project> verify_affected --workspace <workspace>`. "
        "Use `devtools verify --quick` for the static gates and `devtools test <selection>` for focused runs."
    )


def refuse_bare_pytest(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> str | None:
    """Why a pytest session is refused inside an agent job, or None."""
    if not inside_agent_job(env, cgroup_reader=cgroup_reader) or env.get(HARNESS_RUN_ENV):
        return None
    return "pytest: bare pytest does not run inside agent jobs; use `devtools test <selection>`"
