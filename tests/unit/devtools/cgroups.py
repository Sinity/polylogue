"""The cgroup shapes ``devtools.agent_env`` classifies, as the deployment writes them.

The cgroup is what binds a queue worker's identity to the pool its operation
declared, so a test asserting that decision names the cgroup it means instead of
inheriting whichever one the test session happens to run under.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import agent_env

DEPLOYED_AGENT_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/sinnixd.slice/"
    "sinnixd-pueue.slice/sinnixd-pueue-agent.slice/run-p3557538-i192191396.scope\n"
)
DEPLOYED_PYTEST_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/sinnixd.slice/"
    "sinnixd-pueue.slice/sinnixd-pueue-pytest.slice/"
    "sinnixd-pueue-pytest-verify_affected-job.scope\n"
)
OUTSIDE_CGROUP = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/shell.scope\n"
WORKFLOW_RUNNER_CGROUP = "0::/sinnixd.slice/sinnixd-work.slice/github-runner-polylogue.service\n"


def outside_cgroup() -> str:
    return OUTSIDE_CGROUP


def deployed_agent_cgroup() -> str:
    return DEPLOYED_AGENT_CGROUP


def deployed_pytest_cgroup() -> str:
    return DEPLOYED_PYTEST_CGROUP


def workflow_runner_cgroup() -> str:
    return WORKFLOW_RUNNER_CGROUP


def stub_cgroup(cgroup: str, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the ambient cgroup read at ``cgroup``.

    ``devtools.pytest_slot.holds_pytest_slot`` takes no reader, so the file
    ``agent_env`` reads is what a test of the whole slot decision controls.
    """
    path = tmp_path / "cgroup"
    path.write_text(cgroup, encoding="utf-8")
    monkeypatch.setattr(agent_env, "_CGROUP_PATH", path)
