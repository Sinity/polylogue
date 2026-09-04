from __future__ import annotations

from pathlib import Path

import tomllib

from devtools import agent_env
from tests.unit.devtools.cgroups import (
    deployed_agent_cgroup,
    deployed_pytest_cgroup,
    outside_cgroup,
    workflow_runner_cgroup,
)

AGENT = {agent_env.AGENT_PRINCIPAL_ENV: agent_env.AGENT_PRINCIPAL}


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


def test_declared_pytest_worker_is_not_refused_for_its_own_operation() -> None:
    environment = {
        **AGENT,
        "SINNIXD_JOB_ID": "verify-affected-1",
        "SINNIXD_OPERATION": "verify_affected",
        "SINNIXD_QUEUE_WORKER": "1",
    }

    assert agent_env.inside_agent_job(environment, cgroup_reader=deployed_pytest_cgroup)
    assert agent_env.inside_declared_pytest_worker(environment, cgroup_reader=deployed_pytest_cgroup)
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=deployed_pytest_cgroup) is None
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=deployed_pytest_cgroup) is None


def test_non_pytest_queue_worker_remains_agent_bound() -> None:
    environment = {
        **AGENT,
        "SINNIXD_JOB_ID": "agent-1",
        "SINNIXD_OPERATION": "lane",
        "SINNIXD_QUEUE_WORKER": "1",
    }

    assert not agent_env.inside_declared_pytest_worker(environment, cgroup_reader=deployed_agent_cgroup)
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=deployed_agent_cgroup) is not None


def test_the_workflow_runner_is_not_an_agent_job() -> None:
    """Hosted verification runs the test tier; widening the agent slices would refuse it.

    Anti-vacuity: adding the runner's ``sinnixd-work.slice`` to the agent slices
    makes every assertion here red.
    """
    environment = {"GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "33895265963", "CI": "true"}

    assert not agent_env.inside_agent_job(environment, cgroup_reader=workflow_runner_cgroup)
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=workflow_runner_cgroup) is None
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=workflow_runner_cgroup) is None
    assert agent_env.agent_worker_cap(8, environment, cgroup_reader=workflow_runner_cgroup) == 8


def test_every_declared_pytest_pool_operation_classifies_its_own_worker() -> None:
    """A pytest-pool worker that queues again waits on the slot it already occupies.

    Anti-vacuity: dropping any pytest-pool operation from
    ``PYTEST_WORKER_OPERATIONS`` makes this red, and that deadlock is what a
    queue runner not exporting ``SINNIXD_QUEUE_POOL`` would hit.
    """
    declarations = tomllib.loads(
        (Path(__file__).resolve().parents[3] / ".agentctl" / "project.toml").read_text(encoding="utf-8")
    )
    declared = {
        name for name, operation in declarations["operations"].items() if operation.get("pool") == agent_env.PYTEST_POOL
    }

    assert declared, "the project declares the pytest pool; an empty set would assert nothing"
    assert declared <= agent_env.PYTEST_WORKER_OPERATIONS, sorted(declared - agent_env.PYTEST_WORKER_OPERATIONS)
    for operation in sorted(declared):
        environment = {
            **AGENT,
            "SINNIXD_JOB_ID": f"{operation}-1",
            "SINNIXD_OPERATION": operation,
            "SINNIXD_QUEUE_WORKER": "1",
        }

        assert agent_env.inside_declared_pytest_worker(environment, cgroup_reader=deployed_pytest_cgroup), operation
