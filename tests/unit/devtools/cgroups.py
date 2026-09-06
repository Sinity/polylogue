"""The cgroup shapes ``devtools.agent_env`` classifies, as the deployment writes them.

The cgroup is what binds a queue worker's identity to the pool its operation
declared, so a test asserting that decision names the cgroup it means instead of
inheriting whichever one the test session happens to run under.

Two runtime generations are live: agentctl places a job in
``agentctl-<pool>.slice`` as a transient service; the older sinnixd build used
``sinnixd-pueue-<pool>.slice`` and a scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import agent_env

_USER_MANAGER = "0::/user.slice/user-1000.slice/user@1000.service/"

DEPLOYED_AGENT_CGROUP = (
    f"{_USER_MANAGER}agentctl.slice/agentctl-agent.slice/agentctl-agent-worker-0a1b2c3d4e5f.service\n"
)
DEPLOYED_PYTEST_CGROUP = (
    f"{_USER_MANAGER}agentctl.slice/agentctl-pytest.slice/agentctl-pytest-verify_affected-0a1b2c3d4e5f.service\n"
)
LEGACY_AGENT_CGROUP = (
    f"{_USER_MANAGER}sinnixd.slice/sinnixd-pueue.slice/sinnixd-pueue-agent.slice/run-p3557538-i192191396.scope\n"
)
LEGACY_PYTEST_CGROUP = (
    f"{_USER_MANAGER}sinnixd.slice/sinnixd-pueue.slice/sinnixd-pueue-pytest.slice/"
    "sinnixd-pueue-pytest-verify_affected-job.scope\n"
)
#: A session subagent's shell: the bare ``agent.slice`` of the desktop session.
SESSION_AGENT_CGROUP = f"{_USER_MANAGER}agent.slice/run-p2452956-i195278870.scope\n"
OUTSIDE_CGROUP = f"{_USER_MANAGER}app.slice/shell.scope\n"
WORKFLOW_RUNNER_CGROUP = "0::/agentctl.slice/agentctl-work.slice/github-runner-polylogue.service\n"

AGENT_CGROUPS = (DEPLOYED_AGENT_CGROUP, LEGACY_AGENT_CGROUP, SESSION_AGENT_CGROUP)
PYTEST_CGROUPS = (DEPLOYED_PYTEST_CGROUP, LEGACY_PYTEST_CGROUP)


def outside_cgroup() -> str:
    return OUTSIDE_CGROUP


def deployed_agent_cgroup() -> str:
    return DEPLOYED_AGENT_CGROUP


def deployed_pytest_cgroup() -> str:
    return DEPLOYED_PYTEST_CGROUP


def workflow_runner_cgroup() -> str:
    return WORKFLOW_RUNNER_CGROUP


def reader(cgroup: str):  # type: ignore[no-untyped-def]
    return lambda: cgroup


def stub_cgroup(cgroup: str, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the ambient cgroup read at ``cgroup``.

    ``devtools.pytest_slot.holds_pytest_slot`` takes no reader, so the file
    ``agent_env`` reads is what a test of the whole slot decision controls.
    """
    path = tmp_path / "cgroup"
    path.write_text(cgroup, encoding="utf-8")
    monkeypatch.setattr(agent_env, "_CGROUP_PATH", path)
