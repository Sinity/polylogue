from __future__ import annotations

import os
import shutil
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
dev = ["polylogue[dev-common,speed]"]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
""",
        encoding="utf-8",
    )


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


def test_lane_env_removes_the_inherited_venv_bin_from_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = tmp_path / "coordinator"
    lane = tmp_path / "lane"
    foreign_bin = coordinator / ".venv" / "bin"
    retained_bin = tmp_path / "tools"
    monkeypatch.setenv("VIRTUAL_ENV", str(coordinator / ".venv"))
    monkeypatch.setenv("PATH", os.pathsep.join((str(foreign_bin), str(retained_bin), str(foreign_bin))))

    env = lane_init.lane_subprocess_env(lane)

    assert env["PATH"] == str(retained_bin)
    assert "VIRTUAL_ENV" not in env
    assert env["UV_PROJECT_ENVIRONMENT"] == str(lane / ".venv")


def test_main_refuses_foreign_lane_init_implementation_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "lane"
    root.mkdir()
    monkeypatch.setattr(lane_init, "repo_root", lambda: root)
    monkeypatch.setattr(lane_init, "_implementation_root", lambda: tmp_path / "other-checkout")
    monkeypatch.setattr(lane_init, "_ensure_worktree", lambda *_args: pytest.fail("must not create a worktree"))

    assert lane_init.main([str(root / "child"), "--branch", "feature/test/lane"]) == 125
    assert "implementation belongs to a different checkout" in capsys.readouterr().err


def test_main_requires_branch_only_when_it_must_create_a_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(lane_init, "repo_root", lambda: root)
    monkeypatch.setattr(lane_init, "_implementation_root", lambda: root)
    monkeypatch.setattr(lane_init, "_ensure_worktree", lambda *_args: pytest.fail("must not create a worktree"))

    assert lane_init.main([str(root / "missing-lane")]) == 2
    assert "--branch is required when creating a worktree" in capsys.readouterr().err


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
    invokes verify-worktree through the lane interpreter without using
    coordinator code or environment state.

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


def test_public_lane_init_bootstraps_an_existing_lane_with_inherited_coordinator_venv(tmp_path: Path) -> None:
    """The public devtools command self-heals the one allowed foreign environment."""
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
    coordinator_python.write_text(
        f'#!/bin/sh\nexec {sys.executable!s} "$@"\n',
        encoding="utf-8",
    )
    coordinator_python.chmod(0o755)
    branch = "feature/test/public-lane-bootstrap"
    assert lane_init._ensure_worktree(coordinator, lane, branch, "HEAD") is None
    for relative_path in ("devtools/click_dispatch.py", "devtools/lane_init.py"):
        shutil.copy2(project_root / relative_path, lane / relative_path)

    poisoned_env = os.environ | {
        "VIRTUAL_ENV": str(coordinator / ".venv"),
        "PYTHONPATH": str(coordinator),
        "UV_PROJECT": str(coordinator),
        "UV_WORKING_DIR": str(coordinator),
    }
    result = subprocess.run(
        [
            str(coordinator_python),
            "-m",
            "devtools",
            "workspace",
            "lane-init",
            str(lane),
        ],
        cwd=lane,
        env=poisoned_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "lane ready:" in result.stdout
    assert (lane / ".venv" / "bin" / "python").is_file()

    rerouted = subprocess.run(
        [
            str(coordinator_python),
            "-m",
            "devtools",
            "workspace",
            "verify-worktree",
            str(lane),
            "--expect-branch",
            branch,
        ],
        cwd=lane,
        env=poisoned_env,
        capture_output=True,
        text=True,
    )
    assert rerouted.returncode == 0, rerouted.stdout + rerouted.stderr


# ---------------------------------------------------------------------------
# Interpreter pinning (polylogue-l218h)
# ---------------------------------------------------------------------------


def _fake_python(path: Path, base_executable: str) -> Path:
    """A stand-in venv python that reports a chosen ``sys._base_executable``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{base_executable}'\n")
    path.chmod(0o755)
    return path


def test_provision_venv_pins_the_coordinator_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: dropping ``--python`` from the uv command fails this.

    Without the pin, uv resolves an interpreter itself and was observed
    downloading a free-threaded CPython 3.14.5 for a coordinator running Nix
    CPython 3.14.4 -- a lane on which ``import hypothesis`` dies in
    ``sysconfig`` and no test can be collected at all.
    """
    lane = tmp_path / "lane"
    lane.mkdir()
    captured: dict[str, object] = {}

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        captured["cmd"] = list(cmd)
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(lane_init, "_run", fake_run)
    assert lane_init._provision_venv(lane, Path("/nix/store/abc/bin/python3.14t")) is None

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--python" in cmd
    assert cmd[cmd.index("--python") + 1] == "/nix/store/abc/bin/python3.14t"


def test_provision_venv_without_an_interpreter_omits_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An underivable interpreter must not become the literal string ``None``."""
    lane = tmp_path / "lane"
    lane.mkdir()
    captured: dict[str, object] = {}

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        captured["cmd"] = list(cmd)
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(lane_init, "_run", fake_run)
    assert lane_init._provision_venv(lane, None) is None
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--python" not in cmd


def test_provision_venv_uses_the_canonical_devshell_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lane sync must select the same dependency surface as the coordinator."""
    lane = tmp_path / "lane"
    lane.mkdir()
    captured: dict[str, object] = {}

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        captured["cmd"] = list(cmd)
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(lane_init, "_run", fake_run)
    assert lane_init._provision_venv(lane) is None

    command = captured["cmd"]
    assert isinstance(command, list)
    assert command[:4] == ["uv", "sync", "--extra", "dev"]
    assert command[4:] == []


def test_verify_candidate_helper_preserves_deliberate_profile_and_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools import verify

    monkeypatch.setenv("HYPOTHESIS_PROFILE", "ci")
    monkeypatch.setenv("POLYLOGUE_CI", "1")

    candidates = verify._native_pytest_environment_candidates()

    assert candidates == (
        {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"},
        {"HYPOTHESIS_PROFILE": "default", "POLYLOGUE_CI": "1"},
    )


def test_interpreter_guard_refuses_a_mismatched_lane_interpreter(tmp_path: Path) -> None:
    """RED TWIN: a lane built from another Python must be refused, loudly.

    ``uv sync --python`` can silently reuse a pre-existing ``.venv`` built
    from a different interpreter, so requesting the pin is not the same as
    getting it. Deleting the ``_interpreter_guard`` call from ``main`` makes
    the exact 2026-08-19 failure silent again.
    """
    expected = _fake_python(tmp_path / "coordinator" / "bin" / "python", "/irrelevant")
    _fake_python(tmp_path / "lane" / ".venv" / "bin" / "python", str(tmp_path / "other" / "bin" / "python3.14"))

    error = lane_init._interpreter_guard(tmp_path / "lane", expected)

    assert error is not None
    assert "does not match the coordinator's" in error
    assert str(tmp_path / "other" / "bin" / "python3.14") in error


def test_interpreter_guard_accepts_a_matching_lane_interpreter(tmp_path: Path) -> None:
    real = tmp_path / "shared" / "bin" / "python3.14t"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    _fake_python(tmp_path / "lane" / ".venv" / "bin" / "python", str(real))

    assert lane_init._interpreter_guard(tmp_path / "lane", real) is None


def test_interpreter_guard_reports_a_missing_lane_venv(tmp_path: Path) -> None:
    error = lane_init._interpreter_guard(tmp_path / "lane", Path(sys.executable))
    assert error is not None
    assert "no venv python" in error


def test_coordinator_base_interpreter_prefers_the_checkout_venv(tmp_path: Path) -> None:
    """Derived from the coordinator's venv, never a hardcoded store path.

    A pinned literal would rot on the next nixpkgs bump into exactly the
    mismatch the pin exists to prevent.
    """
    real = tmp_path / "store" / "bin" / "python3.14t"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    _fake_python(tmp_path / "root" / ".venv" / "bin" / "python", str(real))

    assert lane_init.coordinator_base_interpreter(tmp_path / "root") == real


def test_coordinator_base_interpreter_falls_back_to_the_running_interpreter(tmp_path: Path) -> None:
    """No coordinator venv (a fresh clone) must still yield a usable pin."""
    resolved = lane_init.coordinator_base_interpreter(tmp_path / "no-venv-here")
    assert resolved is not None
    assert resolved.exists()


# ---------------------------------------------------------------------------
# Interpreter-describing environment must not reach a lane (polylogue-l218h)
# ---------------------------------------------------------------------------


def test_lane_env_scrubs_interpreter_describing_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED TWIN: these describe a BUILD, so inheriting them is fatal, not untidy.

    A devshell exports ``_PYTHON_SYSCONFIGDATA_NAME`` for its own interpreter.
    Applied to a venv built from a different one, ``sysconfig`` resolves
    another build's configuration and pytest dies before collecting anything
    (``AttributeError: 'installed_base'`` or ``ModuleNotFoundError: No module
    named '_sysconfigdata_...'``, depending on which builds collide).
    Restoring any of these to the inherited set reproduces a lane that
    provisions cleanly and cannot run one test.
    """
    monkeypatch.setenv("_PYTHON_SYSCONFIGDATA_NAME", "_sysconfigdata_t_linux_x86_64-linux-gnu")
    monkeypatch.setenv("_PYTHON_HOST_PLATFORM", "linux-x86_64")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/coordinator/.cache/pycache")

    env = lane_init.lane_subprocess_env(tmp_path / "lane")

    assert "_PYTHON_SYSCONFIGDATA_NAME" not in env
    assert "_PYTHON_HOST_PLATFORM" not in env
    assert "PYTHONPYCACHEPREFIX" not in env


def test_lane_command_env_activates_only_the_sanitized_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = tmp_path / "lane"
    for key in lane_init._LANE_ENV_UNSETS:
        if key != "VIRTUAL_ENV":
            monkeypatch.setenv(key, f"inherited-{key}")
    monkeypatch.setenv("VIRTUAL_ENV", "/coordinator/.venv")
    monkeypatch.setenv("PATH", "/coordinator/.venv/bin:/usr/bin")

    env = lane_init.lane_command_env(lane)

    assert env["VIRTUAL_ENV"] == str(lane / ".venv")
    assert env["PATH"].split(os.pathsep)[0] == str(lane / ".venv" / "bin")
    assert "/coordinator/.venv/bin" not in env["PATH"].split(os.pathsep)
    for key in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT", "UV_WORKING_DIR", *lane_init._INTERPRETER_DESCRIBING_ENV):
        assert key not in env
    assert env["UV_PROJECT_ENVIRONMENT"] == str(lane / ".venv")
