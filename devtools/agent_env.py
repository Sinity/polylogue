"""Runtime environment names and bounds on test execution inside agent jobs.

The Sinnix runtime exports ``AGENTCTL_*`` variables into every job it starts;
older hosts still export the same values as ``SINNIXD_*``. Every consumer reads
them through :func:`runtime_env` so the fallback lives in one place.

An agent lane runs with ``AGENTCTL_PRINCIPAL=agent-control``.
Tests for a lane run exactly once, as the declared ``verify_affected`` job
in the host's single-worker pytest pool; a lane running the affected or
complete tier itself doubles that load outside admission. Focused runs stay
available, bounded to a few workers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

RUNTIME_ENV_PREFIX = "AGENTCTL_"
LEGACY_RUNTIME_ENV_PREFIX = "SINNIXD_"
#: The runtime's transient job unit, ``<prefix>-job-<uuid>.service``.
RUNTIME_UNIT_PREFIXES = ("agentctl", "sinnixd")

AGENT_PRINCIPAL_ENV = "AGENTCTL_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"


def runtime_env_names(variable: str) -> tuple[str, str]:
    """``(AGENTCTL_<name>, SINNIXD_<name>)`` for an ``AGENTCTL_<name>`` variable."""
    if not variable.startswith(RUNTIME_ENV_PREFIX):
        raise ValueError(f"{variable} is not an {RUNTIME_ENV_PREFIX}* runtime variable")
    return (variable, LEGACY_RUNTIME_ENV_PREFIX + variable.removeprefix(RUNTIME_ENV_PREFIX))


def runtime_env(variable: str, env: Mapping[str, str] | None = None) -> str | None:
    """The runtime variable ``AGENTCTL_<name>``, or ``SINNIXD_<name>`` when only that is set."""
    source = os.environ if env is None else env
    for candidate in runtime_env_names(variable):
        value = source.get(candidate)
        if value:
            return value
    return None


def inside_agent_job(env: Mapping[str, str]) -> bool:
    return runtime_env(AGENT_PRINCIPAL_ENV, env) == AGENT_PRINCIPAL


def agent_worker_cap(requested: int | None, env: Mapping[str, str]) -> int | None:
    """The worker count a focused run may use; unchanged outside agent jobs."""
    if not inside_agent_job(env):
        return requested
    if requested is None:
        return AGENT_MAX_PYTEST_WORKERS
    # Zero is a configuration, not an absent one: it asks for no xdist at all,
    # which is always within the cap.
    return min(max(requested, 0), AGENT_MAX_PYTEST_WORKERS)


def refuse_verify_tier(argv: list[str], env: Mapping[str, str]) -> str | None:
    """Why a ``devtools verify`` invocation is refused inside an agent job, or None."""
    if not inside_agent_job(env) or "--quick" in argv:
        return None
    return (
        "devtools verify: test tiers do not run inside agent jobs; the lane's tests run once as "
        "`agentctl job start <project> verify_affected --workspace <workspace>`. "
        "Use `devtools verify --quick` for the static gates and `devtools test <selection>` for focused runs."
    )


def refuse_bare_pytest(env: Mapping[str, str]) -> str | None:
    """Why a pytest session is refused inside an agent job, or None."""
    if not inside_agent_job(env) or env.get(HARNESS_RUN_ENV):
        return None
    return "pytest: bare pytest does not run inside agent jobs; use `devtools test <selection>`"
