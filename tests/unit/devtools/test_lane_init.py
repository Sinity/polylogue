from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from devtools import lane_init


def _write_minimal_uv_project(root: Path) -> None:
    (root / "polylogue").mkdir(parents=True)
    (root / "polylogue" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """\
[project]
name = "polylogue"
version = "0.0.0"
requires-python = ">=3.14"

[project.optional-dependencies]
dev-common = []
speed = []

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
""",
        encoding="utf-8",
    )


def test_recommended_workers_fair_share_floor_and_cap() -> None:
    assert lane_init.recommended_workers(16, cpu_count=24) == 1
    assert lane_init.recommended_workers(8, cpu_count=24) == 3
    assert lane_init.recommended_workers(2, cpu_count=24) == 4  # capped
    assert lane_init.recommended_workers(0, cpu_count=24) == 4  # clamped lanes
    assert lane_init.recommended_workers(64, cpu_count=24) == 1  # floor


def test_lane_record_shape_and_ledger_append(tmp_path: Path) -> None:
    record = lane_init.lane_record(
        worktree=tmp_path / "lane-x",
        branch="feature/x",
        base_sha="abc123def",
        beads=["polylogue-aaa", "polylogue-bbb"],
        venv=True,
        workers=2,
    )
    assert record["lane"] == "lane-x"
    assert record["status"] == "provisioned"
    assert record["beads"] == ["polylogue-aaa", "polylogue-bbb"]
    path = lane_init.append_ledger(tmp_path, record)
    lane_init.append_ledger(tmp_path, record)
    assert path == tmp_path / lane_init.LEDGER_RELPATH
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["branch"] == "feature/x"


def test_guard_check_rejects_a_true_foreign_environment(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    python = lane / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        "echo 'devtools workspace lane-init: import resolved outside lane: /foreign/polylogue/__init__.py' >&2\n"
        "exit 125\n"
    )
    python.chmod(0o755)
    error = lane_init._guard_check(lane)
    assert error is not None
    assert "lane checkout guard failed" in error
    assert "/foreign/polylogue/__init__.py" in error


def test_guard_check_runs_lane_interpreter_from_lane_root(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    python = lane / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        'if [ -n "$PYTHONPATH" ]; then\n'
        "  printf '%s\\n' /foreign/polylogue/__init__.py\n"
        "else\n"
        "  printf '%s/polylogue/__init__.py\\n' \"$PWD\"\n"
        "fi\n"
    )
    python.chmod(0o755)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("PYTHONPATH", "/foreign")
        assert lane_init._guard_check(lane) is None


def test_provision_venv_forces_the_lane_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = tmp_path / "lane"
    lane.mkdir()
    monkeypatch.setenv("VIRTUAL_ENV", "/coordinator/.venv")
    monkeypatch.setenv("PYTHONHOME", "/coordinator/pythonhome")
    monkeypatch.setenv("PYTHONPATH", "/coordinator/pythonpath")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/coordinator/.venv")
    captured: dict[str, object] = {}

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        captured.update(cmd=cmd, cwd=cwd, env=env)
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(lane_init, "_run", fake_run)

    assert lane_init._provision_venv(lane) is None
    assert captured["cwd"] == lane
    env = captured["env"]
    assert isinstance(env, dict)
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["UV_PROJECT_ENVIRONMENT"] == str(lane / ".venv")


def test_provision_venv_ignores_inherited_uv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A coordinator's UV_PROJECT must not install its editable source in a lane venv."""
    coordinator = tmp_path / "coordinator"
    lane = tmp_path / "lane"
    _write_minimal_uv_project(coordinator)
    _write_minimal_uv_project(lane)
    monkeypatch.setenv("UV_PROJECT", str(coordinator))
    monkeypatch.setenv("UV_WORKING_DIR", str(coordinator))

    assert lane_init._provision_venv(lane) is None


def test_main_verifies_a_new_lane_with_its_own_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production command must leave a main-checkout shell before verifying.

    Mutation proof: restoring the former ``sys.executable``/coordinator-cwd
    verification route makes the simulated checkout guard return 125 before
    lane-init can report the lane as ready.
    """
    coordinator = tmp_path / "main"
    lane = tmp_path / "lane"
    coordinator.mkdir()
    lane.mkdir()
    lane_python = lane / ".venv" / "bin" / "python"
    lane_python.parent.mkdir(parents=True)
    lane_python.touch()
    coordinator_python = coordinator / ".venv" / "bin" / "python"

    monkeypatch.setattr(lane_init, "repo_root", lambda: coordinator)
    monkeypatch.setattr(lane_init, "_ensure_worktree", lambda *_: None)
    monkeypatch.setattr(sys, "executable", str(coordinator_python))
    monkeypatch.setenv("VIRTUAL_ENV", str(coordinator / ".venv"))
    monkeypatch.setenv("PYTHONPATH", str(coordinator))
    monkeypatch.setattr(lane_init, "coordinator_root", lambda root: root)
    monkeypatch.setattr(lane_init, "append_ledger", lambda root, record: root / lane_init.LEDGER_RELPATH)

    events: list[str] = []

    def provision(worktree: Path) -> None:
        assert worktree == lane
        events.append("provision")
        return None

    def guard(worktree: Path) -> None:
        assert worktree == lane
        events.append("guard")
        return None

    def run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> CompletedProcess[str]:
        if "verify-worktree" in cmd:
            events.append("verify")
            if cmd[0] != str(lane_python) or cwd != lane or env is None:
                return CompletedProcess(cmd, 125, "", "guard: coordinator editable import\n")
            assert "VIRTUAL_ENV" not in env
            assert "PYTHONPATH" not in env
            assert env["UV_PROJECT_ENVIRONMENT"] == str(lane / ".venv")
            return CompletedProcess(cmd, 0, "", "")
        if cmd[:4] == ["git", "-C", str(lane), "rev-parse"]:
            return CompletedProcess(cmd, 0, "abcdef123\n", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(lane_init, "_provision_venv", provision)
    monkeypatch.setattr(lane_init, "_guard_check", guard)
    monkeypatch.setattr(lane_init, "_run", run)

    assert lane_init.main([str(lane), "--branch", "feature/test/lane-env"]) == 0
    assert events == ["provision", "guard", "verify"]
