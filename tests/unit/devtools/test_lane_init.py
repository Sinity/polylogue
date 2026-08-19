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
# Graph seeding reports warmth honestly (polylogue-l218h)
# ---------------------------------------------------------------------------


def _graph_with(path: Path, environments: Sequence[str]) -> Path:
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE environment (environment_name TEXT)")
        conn.executemany("INSERT INTO environment VALUES (?)", [(name,) for name in environments])
    return path


def test_graph_environments_reads_names_and_tolerates_junk(tmp_path: Path) -> None:
    populated = _graph_with(tmp_path / "graph.db", ["polylogue-aaa", "polylogue-bbb", "polylogue-aaa"])
    assert lane_init.graph_environments(populated) == {"polylogue-aaa", "polylogue-bbb"}

    assert lane_init.graph_environments(tmp_path / "absent.db") == set()

    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all")
    assert lane_init.graph_environments(junk) == set()

    empty = tmp_path / "empty.db"
    import sqlite3

    sqlite3.connect(empty).close()
    assert lane_init.graph_environments(empty) == set()


def test_seed_reports_cold_when_the_graph_lacks_this_lanes_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED TWIN: a stale seed must NOT advertise warmth.

    This is the exact 2026-08-19 shape: the copy succeeds and the graph is
    perfectly valid, but it holds only environments from before a dependency
    bump / conftest edit, so the lane's first verify is a complete-corpus
    bootstrap. Restoring the unconditional
    ``"seeded from main checkout (lane verifies start warm)"`` return makes
    this fail, because ``warm`` comes back True and the note claims warmth for
    a graph that cannot deliver it.
    """
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _graph_with(root / TESTMON_DATA_RELPATH, ["polylogue-stale-one", "polylogue-stale-two"])
    monkeypatch.setattr(lane_init, "lane_environment_digest", lambda _worktree: "polylogue-current")

    note, warm = lane_init._seed_testmon_graph(root, lane)

    assert warm is False
    assert "NONE matching this lane" in note
    assert "bootstrap" in note
    assert (lane / TESTMON_DATA_RELPATH).is_file()


def test_seed_reports_warm_only_on_a_real_environment_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _graph_with(root / TESTMON_DATA_RELPATH, ["polylogue-stale", "polylogue-current"])
    monkeypatch.setattr(lane_init, "lane_environment_digest", lambda _worktree: "polylogue-current")

    note, warm = lane_init._seed_testmon_graph(root, lane)

    assert warm is True
    assert "verifies start warm" in note


def test_seed_reports_cold_for_an_empty_or_absent_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"

    note, warm = lane_init._seed_testmon_graph(root, lane)
    assert warm is False
    assert "no coordinator graph" in note

    _graph_with(root / TESTMON_DATA_RELPATH, [])
    monkeypatch.setattr(lane_init, "lane_environment_digest", lambda _worktree: "polylogue-current")
    note, warm = lane_init._seed_testmon_graph(root, lane)
    assert warm is False
    assert "no environments" in note


def test_seed_reports_cold_when_the_lane_digest_cannot_be_computed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unverifiable lane must never be reported as warm."""
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _graph_with(root / TESTMON_DATA_RELPATH, ["polylogue-something"])
    monkeypatch.setattr(lane_init, "lane_environment_digest", lambda _worktree: None)

    note, warm = lane_init._seed_testmon_graph(root, lane)

    assert warm is False
    assert "warmth unverified" in note


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

    env = lane_init._lane_env(tmp_path / "lane")

    assert "_PYTHON_SYSCONFIGDATA_NAME" not in env
    assert "_PYTHON_HOST_PLATFORM" not in env
    assert "PYTHONPYCACHEPREFIX" not in env


def test_dispatch_env_lines_carry_the_unsets_and_the_ca_bundle(tmp_path: Path) -> None:
    """The lane's own venv, the fatal unsets, and the CA bundle gh/pr-scope need.

    ``_lane_env`` only sanitises lane-init's OWN subprocesses; an agent that
    opens the worktree later inherits the harness environment untouched, so
    the remedy has to be printed where a dispatcher reads it.
    """
    lane = tmp_path / "lane"
    lines = lane_init.dispatch_env_lines(lane)
    joined = "\n".join(lines)

    assert f"export VIRTUAL_ENV={lane / '.venv'}" in lines
    for key in lane_init._INTERPRETER_DESCRIBING_ENV:
        assert f"unset {key}" in lines
    assert "export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in lines
    assert "$VIRTUAL_ENV/bin:$PATH" in joined
