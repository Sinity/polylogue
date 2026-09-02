from __future__ import annotations

from devtools import agent_env

AGENT = {agent_env.AGENT_PRINCIPAL_ENV: agent_env.AGENT_PRINCIPAL}


def test_outside_agent_jobs_nothing_changes() -> None:
    assert agent_env.agent_worker_cap(8, {}) == 8
    assert agent_env.refuse_verify_tier([], {}) is None
    assert agent_env.refuse_bare_pytest({}) is None


def test_agent_jobs_get_bounded_focused_runs_and_no_test_tiers() -> None:
    """Anti-vacuity: dropping any guard lets a lane run half the corpus outside admission."""
    assert agent_env.agent_worker_cap(8, AGENT) == agent_env.AGENT_MAX_PYTEST_WORKERS
    assert agent_env.agent_worker_cap(None, AGENT) == agent_env.AGENT_MAX_PYTEST_WORKERS
    assert agent_env.agent_worker_cap(1, AGENT) == 1
    assert agent_env.refuse_verify_tier(["--quick"], AGENT) is None
    assert agent_env.refuse_verify_tier([], AGENT) is not None
    assert agent_env.refuse_verify_tier(["--all"], AGENT) is not None
    assert agent_env.refuse_bare_pytest(AGENT) is not None
    assert agent_env.refuse_bare_pytest({**AGENT, agent_env.HARNESS_RUN_ENV: "run-1"}) is None
