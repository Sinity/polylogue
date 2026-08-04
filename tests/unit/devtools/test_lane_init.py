from __future__ import annotations

import json
import os
import subprocess
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


def test_main_provisions_and_verifies_a_real_lane_from_a_poisoned_coordinator(tmp_path: Path) -> None:
    """Run the full lane-init process boundary against a real linked worktree.

    The coordinator copy deliberately makes its verify-worktree module fail.
    The outer process also names that coordinator in every Python and uv
    routing variable. A successful run therefore proves that lane-init creates
    the linked worktree, provisions its venv, runs the shared checkout guard,
    invokes verify-worktree through the lane interpreter, and appends its
    ledger without using coordinator code or environment state.

    Mutation proof: restoring the former coordinator interpreter/environment
    route executes the coordinator canary and makes lane-init fail. Removing
    either Python or uv environment scrubbing causes the lane guard to resolve
    the coordinator package and return the asserted exit 125 below.
    """
    coordinator = tmp_path / "main"
    lane = tmp_path / "lane"
    project_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        ["git", "clone", "--shared", str(project_root), str(coordinator)],
        check=True,
        capture_output=True,
        text=True,
    )
    coordinator_python = coordinator / ".venv" / "bin" / "python"
    coordinator_python.parent.mkdir(parents=True)
    coordinator_python.symlink_to(Path(sys.executable))
    (coordinator / "devtools" / "verify_worktree.py").write_text(
        "raise SystemExit('coordinator verify-worktree must not run')\n",
        encoding="utf-8",
    )

    poisoned_env = os.environ | {
        "VIRTUAL_ENV": str(coordinator / ".venv"),
        "PYTHONPATH": str(coordinator),
        "UV_PROJECT": str(coordinator),
        "UV_WORKING_DIR": str(coordinator),
    }
    driver = "from devtools.lane_init import main; import sys; raise SystemExit(main(sys.argv[1:]))"
    result = subprocess.run(
        [
            str(coordinator_python),
            "-c",
            driver,
            str(lane),
            "--branch",
            "feature/test/real-lane",
            "--base",
            "HEAD",
        ],
        cwd=coordinator,
        env=poisoned_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lane ready:" in result.stdout
    assert "coordinator verify-worktree must not run" not in result.stderr
    assert (lane / ".git").is_file()
    lane_python = lane / ".venv" / "bin" / "python"
    assert lane_python.is_file()

    ledger = coordinator / lane_init.LEDGER_RELPATH
    record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert record["worktree"] == str(lane)
    assert record["branch"] == "feature/test/real-lane"
    assert record["venv"] is True

    foreign_guard = subprocess.run(
        [
            str(lane_python),
            "-P",
            "-c",
            (
                "from pathlib import Path\n"
                "import sys\n"
                "from devtools.checkout_guard import CheckoutImportMismatchError, assert_polylogue_matches_checkout\n"
                "try:\n"
                "    assert_polylogue_matches_checkout(Path.cwd(), context='poisoned lane')\n"
                "except CheckoutImportMismatchError as exc:\n"
                "    print(exc, file=sys.stderr)\n"
                "    raise SystemExit(125)\n"
            ),
        ],
        cwd=lane,
        env=poisoned_env,
        capture_output=True,
        text=True,
    )
    assert foreign_guard.returncode == 125
    assert "resolved OUTSIDE this checkout" in foreign_guard.stderr
    assert "give this checkout its own venv" in foreign_guard.stderr
