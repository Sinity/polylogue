from __future__ import annotations

import pytest

from devtools import agent_env

AGENT = {agent_env.AGENT_PRINCIPAL_ENV: agent_env.AGENT_PRINCIPAL}
LEGACY_AGENT = {"SINNIXD_PRINCIPAL": agent_env.AGENT_PRINCIPAL}


def test_runtime_env_reads_the_new_name_first_and_falls_back_to_the_old() -> None:
    assert agent_env.runtime_env_names("AGENTCTL_JOB_ID") == ("AGENTCTL_JOB_ID", "SINNIXD_JOB_ID")
    assert agent_env.runtime_env("AGENTCTL_JOB_ID", {}) is None
    assert agent_env.runtime_env("AGENTCTL_JOB_ID", {"SINNIXD_JOB_ID": "old"}) == "old"
    assert agent_env.runtime_env("AGENTCTL_JOB_ID", {"AGENTCTL_JOB_ID": "new"}) == "new"
    assert agent_env.runtime_env("AGENTCTL_JOB_ID", {"AGENTCTL_JOB_ID": "new", "SINNIXD_JOB_ID": "old"}) == "new"
    assert agent_env.runtime_env("AGENTCTL_JOB_ID", {"AGENTCTL_JOB_ID": "", "SINNIXD_JOB_ID": "old"}) == "old"
    with pytest.raises(ValueError):
        agent_env.runtime_env("SINNIXD_JOB_ID", {})


def test_runtime_env_defaults_to_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTCTL_JOB_ID", raising=False)
    monkeypatch.setenv("SINNIXD_JOB_ID", "old")
    assert agent_env.runtime_env("AGENTCTL_JOB_ID") == "old"
    monkeypatch.setenv("AGENTCTL_JOB_ID", "new")
    assert agent_env.runtime_env("AGENTCTL_JOB_ID") == "new"


@pytest.mark.parametrize("agent", [AGENT, LEGACY_AGENT])
def test_either_principal_name_marks_an_agent_job(agent: dict[str, str]) -> None:
    assert agent_env.inside_agent_job(agent)


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
