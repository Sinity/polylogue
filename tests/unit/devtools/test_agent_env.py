from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from devtools import agent_env
from tests.unit.devtools.cgroups import (
    AGENT_CGROUPS,
    PYTEST_CGROUPS,
    deployed_agent_cgroup,
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


@pytest.mark.parametrize("cgroup", PYTEST_CGROUPS)
def test_a_job_in_the_pytest_pool_slice_is_not_refused(cgroup: str) -> None:
    """The slice is ownership on its own: a job whose environment says nothing still holds the slot."""
    environment = {**AGENT, "AGENTCTL_JOB_ID": "verify-affected-1", "AGENTCTL_OPERATION": "verify_affected"}

    assert agent_env.inside_pytest_pool({}, cgroup_reader=reader(cgroup))
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=reader(cgroup)) is None
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=reader(cgroup)) is None


@pytest.mark.parametrize("pool_variable", ["AGENTCTL_POOL", "SINNIXD_QUEUE_POOL"])
def test_the_exported_pytest_pool_is_ownership(pool_variable: str) -> None:
    environment = {**AGENT, pool_variable: "pytest"}

    assert agent_env.declared_pool(environment) == "pytest"
    assert agent_env.inside_pytest_pool(environment, cgroup_reader=outside_cgroup)
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=outside_cgroup) is None
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=outside_cgroup) is None


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("operation", ["lane", "verify_affected", "verify_all"])
def test_a_job_id_never_grants_the_pytest_slot(prefix: str, operation: str) -> None:
    """Anti-vacuity: classifying by job id or operation name lets a lane run pytest outside the pool."""
    environment = _worker(prefix, operation)

    assert not agent_env.inside_pytest_pool(environment, cgroup_reader=deployed_agent_cgroup)
    assert agent_env.refuse_bare_pytest(environment, cgroup_reader=deployed_agent_cgroup) is not None
    assert agent_env.refuse_verify_tier([], environment, cgroup_reader=deployed_agent_cgroup) is not None


@pytest.mark.parametrize("prefix", PREFIXES)
def test_a_declared_agent_pool_never_grants_the_pytest_slot(prefix: str) -> None:
    pool_variable = agent_env.runtime_env_names("AGENTCTL_POOL")[PREFIXES.index(prefix)]
    environment = _worker(prefix, "lane", **{pool_variable: "agent"})

    assert not agent_env.inside_pytest_pool(environment, cgroup_reader=deployed_agent_cgroup)


@pytest.mark.parametrize("cgroup", AGENT_CGROUPS)
def test_a_managed_lane_running_bare_pytest_is_told_the_route(cgroup: str) -> None:
    refusal = agent_env.refuse_bare_pytest({}, cgroup_reader=reader(cgroup))

    assert refusal is not None
    assert "devtools test" in refusal


def test_every_declared_pytest_pool_operation_is_owned_by_its_slice() -> None:
    """Every operation the descriptor puts in the pytest pool holds the slot by its cgroup alone."""
    declarations = tomllib.loads(
        (Path(__file__).resolve().parents[3] / ".agentctl" / "project.toml").read_text(encoding="utf-8")
    )
    declared = {
        name for name, operation in declarations["operations"].items() if operation.get("pool") == agent_env.PYTEST_POOL
    }

    assert {"pytest_focused", "verify_affected", "verify_all"} <= declared
    for cgroup in PYTEST_CGROUPS:
        assert agent_env.inside_pytest_pool({}, cgroup_reader=reader(cgroup))
