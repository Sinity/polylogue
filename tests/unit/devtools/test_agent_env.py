from __future__ import annotations

from devtools import agent_env

AGENT = {agent_env.AGENT_PRINCIPAL_ENV: agent_env.AGENT_PRINCIPAL}
DEPLOYED_AGENT_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/sinnixd.slice/"
    "sinnixd-pueue.slice/sinnixd-pueue-agent.slice/run-p3557538-i192191396.scope\n"
)
OUTSIDE_CGROUP = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/shell.scope\n"


def outside_cgroup() -> str:
    return OUTSIDE_CGROUP


def deployed_agent_cgroup() -> str:
    return DEPLOYED_AGENT_CGROUP


def test_outside_agent_jobs_nothing_changes() -> None:
    assert agent_env.agent_worker_cap(8, {}, cgroup_reader=outside_cgroup) == 8
    assert agent_env.refuse_verify_tier([], {}, cgroup_reader=outside_cgroup) is None
    assert agent_env.refuse_bare_pytest({}, cgroup_reader=outside_cgroup) is None


def test_agent_jobs_get_bounded_focused_runs_and_no_test_tiers() -> None:
    """Anti-vacuity: dropping any guard lets a lane run half the corpus outside admission."""
    assert (
        agent_env.agent_worker_cap(8, AGENT, cgroup_reader=deployed_agent_cgroup) == agent_env.AGENT_MAX_PYTEST_WORKERS
    )
    assert (
        agent_env.agent_worker_cap(None, AGENT, cgroup_reader=deployed_agent_cgroup)
        == agent_env.AGENT_MAX_PYTEST_WORKERS
    )
    assert agent_env.agent_worker_cap(1, AGENT, cgroup_reader=deployed_agent_cgroup) == 1
    assert agent_env.refuse_verify_tier(["--quick"], AGENT, cgroup_reader=deployed_agent_cgroup) is None
    assert agent_env.refuse_verify_tier([], AGENT, cgroup_reader=deployed_agent_cgroup) is not None
    assert agent_env.refuse_verify_tier(["--all"], AGENT, cgroup_reader=deployed_agent_cgroup) is not None
    assert (
        agent_env.refuse_verify_tier(
            [], {**AGENT, "SINNIXD_OPERATION": "verify_affected"}, cgroup_reader=deployed_agent_cgroup
        )
        is None
    )
    assert (
        agent_env.refuse_verify_tier(
            ["--all"], {**AGENT, "SINNIXD_OPERATION": "verify_all"}, cgroup_reader=deployed_agent_cgroup
        )
        is None
    )
    assert (
        agent_env.refuse_verify_tier(
            ["--all"], {**AGENT, "SINNIXD_OPERATION": "verify_affected"}, cgroup_reader=deployed_agent_cgroup
        )
        is not None
    )
    assert agent_env.refuse_bare_pytest(AGENT, cgroup_reader=deployed_agent_cgroup) is not None
    assert (
        agent_env.refuse_bare_pytest({**AGENT, agent_env.HARNESS_RUN_ENV: "run-1"}, cgroup_reader=deployed_agent_cgroup)
        is None
    )


def test_deployed_lane_cgroup_rejects_direct_pytest_without_routing_environment() -> None:
    """Anti-vacuity: removing cgroup detection lets direct pytest bypass admission."""
    environment: dict[str, str] = {}

    assert agent_env.inside_agent_job(environment, cgroup_reader=deployed_agent_cgroup)
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=deployed_agent_cgroup) is not None
