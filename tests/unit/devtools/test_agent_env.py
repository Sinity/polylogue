from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from devtools import agent_env
from tests.unit.devtools.cgroups import (
    AGENT_CGROUPS,
    PYTEST_CGROUPS,
    deployed_agent_cgroup,
    deployed_pytest_cgroup,
    outside_cgroup,
    reader,
)

AGENT = {agent_env.AGENT_PRINCIPAL_ENV: agent_env.AGENT_PRINCIPAL}
LEGACY_AGENT = {"SINNIXD_PRINCIPAL": agent_env.AGENT_PRINCIPAL}
#: The runtime's environment prefix for the current build and the older one.
PREFIXES = ["AGENTCTL_", "SINNIXD_"]


def _worker(prefix: str, operation: str, **extra: str) -> dict[str, str]:
    return {
        **AGENT,
        f"{prefix}JOB_ID": f"{operation}-1",
        f"{prefix}OPERATION": operation,
        f"{prefix}QUEUE_WORKER": "1",
        **extra,
    }


def test_runtime_env_reads_the_new_name_first_and_falls_back_to_the_old() -> None:
    assert agent_env.runtime_env_names("AGENTCTL_JOB_ID") == ("AGENTCTL_JOB_ID", "SINNIXD_JOB_ID")
    assert agent_env.runtime_env_names("AGENTCTL_POOL") == ("AGENTCTL_POOL", "SINNIXD_QUEUE_POOL")
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
    assert agent_env.inside_agent_job(agent, cgroup_reader=outside_cgroup)


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


@pytest.mark.parametrize("cgroup", AGENT_CGROUPS)
def test_every_agent_cgroup_family_rejects_direct_pytest_without_routing_environment(cgroup: str) -> None:
    """Anti-vacuity: removing cgroup detection lets direct pytest bypass admission."""
    environment: dict[str, str] = {}

    assert agent_env.inside_agent_job(environment, cgroup_reader=reader(cgroup))
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=reader(cgroup)) is not None


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("cgroup", PYTEST_CGROUPS)
def test_declared_pytest_worker_is_not_refused_for_its_own_operation(prefix: str, cgroup: str) -> None:
    environment = _worker(prefix, "verify_affected")

    assert agent_env.inside_agent_job(environment, cgroup_reader=reader(cgroup))
    assert agent_env.inside_declared_pytest_worker(environment, cgroup_reader=reader(cgroup))
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=reader(cgroup)) is None
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=reader(cgroup)) is None


@pytest.mark.parametrize("prefix", PREFIXES)
def test_non_pytest_queue_worker_remains_agent_bound(prefix: str) -> None:
    environment = _worker(prefix, "lane")

    assert not agent_env.inside_declared_pytest_worker(environment, cgroup_reader=deployed_agent_cgroup)
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=deployed_agent_cgroup) is not None


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_declared_pytest_pool_operation_classifies_its_own_worker(prefix: str) -> None:
    """A pytest-pool worker that queues again waits on the slot it already occupies.

    Anti-vacuity: dropping any pytest-pool operation from
    ``PYTEST_WORKER_OPERATIONS`` makes this red, and that deadlock is what a
    queue runner not exporting the pool would hit.
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
        environment = _worker(prefix, operation)

        assert agent_env.inside_declared_pytest_worker(environment, cgroup_reader=deployed_pytest_cgroup), operation
