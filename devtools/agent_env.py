"""Bounds on test execution inside agent jobs.

An agent lane runs under sinnixd with ``SINNIXD_PRINCIPAL=agent-control``.
Tests for a lane run exactly once, as the declared ``verify_affected`` job
in the host's single-worker pytest pool; a lane running the affected or
complete tier itself doubles that load outside admission. Focused runs stay
available, bounded to a few workers.
"""

from __future__ import annotations

from collections.abc import Mapping

AGENT_PRINCIPAL_ENV = "SINNIXD_PRINCIPAL"
AGENT_PRINCIPAL = "agent-control"
AGENT_MAX_PYTEST_WORKERS = 2
HARNESS_RUN_ENV = "POLYLOGUE_PYTEST_RUN_ID"


def inside_agent_job(env: Mapping[str, str]) -> bool:
    return env.get(AGENT_PRINCIPAL_ENV) == AGENT_PRINCIPAL


def agent_worker_cap(requested: int | None, env: Mapping[str, str]) -> int | None:
    """The worker count a focused run may use; unchanged outside agent jobs."""
    if not inside_agent_job(env):
        return requested
    if requested is None or requested < 1:
        return AGENT_MAX_PYTEST_WORKERS
    return min(requested, AGENT_MAX_PYTEST_WORKERS)


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
