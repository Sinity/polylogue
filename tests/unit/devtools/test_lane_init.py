from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from devtools import lane_init, testmon_bootstrap, verify


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


def test_lane_warm_claim_is_revalidated_after_distribution_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real lane boundary refuses a graph after its distribution set drifts.

    The lane is provisioned and attested through its own interpreter, then
    ``lane_init.main`` seeds it from a certified coordinator graph.  A second
    lane-interpreter subprocess adds an installed distribution after that warm
    claim.  Verifier preparation must recompute the lane digest and go cold;
    a coordinator-process preparation or mocked attestation would miss this.
    """

    coordinator = tmp_path / "coordinator"
    lane = tmp_path / "lane"
    project_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        ["git", "clone", "--shared", str(project_root), str(coordinator)],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "default")
    monkeypatch.delenv("POLYLOGUE_CI", raising=False)
    branch = "feature/test/seeded-verifier"
    assert lane_init._ensure_worktree(coordinator, lane, branch, "HEAD") is None
    coordinator_python = Path(sys.executable)
    coordinator_interpreter = lane_init.coordinator_base_interpreter(coordinator)
    assert coordinator_interpreter is not None
    assert lane_init._provision_venv(lane, coordinator_interpreter) is None
    attestation = lane_init.lane_environment_attestation(lane)
    assert attestation is not None
    _certified_graph_with_names(coordinator, attestation.digests)

    poisoned_env = os.environ | {
        "VIRTUAL_ENV": str(coordinator / ".venv"),
        "PYTHONPATH": str(coordinator),
        "UV_PROJECT": str(coordinator),
        "UV_WORKING_DIR": str(coordinator),
    }
    main_driver = "from devtools.lane_init import main; import sys; raise SystemExit(main(sys.argv[1:]))"
    main_result = subprocess.run(
        [
            str(coordinator_python),
            "-c",
            main_driver,
            str(lane),
            "--branch",
            branch,
            "--base",
            "HEAD",
            "--json",
        ],
        cwd=coordinator,
        env=poisoned_env,
        capture_output=True,
        text=True,
    )
    assert main_result.returncode == 0, main_result.stdout + main_result.stderr

    record = json.loads((coordinator / lane_init.LEDGER_RELPATH).read_text(encoding="utf-8").splitlines()[-1])
    assert record["testmon_warm"] is True
    lane_python = lane / ".venv" / "bin" / "python"
    assert lane_python.is_file()
    drift_driver = (
        "from pathlib import Path; "
        "import sysconfig; "
        "root = Path(sysconfig.get_paths()['purelib']) / 'polylogue_lane_drift-0.0.0.dist-info'; "
        "root.mkdir(); "
        "(root / 'METADATA').write_text('Metadata-Version: 2.1\\nName: polylogue-lane-drift\\nVersion: 0.0.0\\n')"
    )
    drift_result = subprocess.run(
        [str(lane_python), "-P", "-c", drift_driver],
        cwd=lane,
        env=lane_init._lane_env(lane),
        capture_output=True,
        text=True,
    )
    assert drift_result.returncode == 0, drift_result.stdout + drift_result.stderr
    drift_attestation_driver = (
        "import json; "
        "from pathlib import Path; "
        "from devtools.lane_init import lane_environment_attestation; "
        "attestation = lane_environment_attestation(Path.cwd()); "
        "assert attestation is not None; "
        "print(json.dumps({'digests': list(attestation.digests)}))"
    )
    drift_attestation_result = subprocess.run(
        [str(lane_python), "-P", "-c", drift_attestation_driver],
        cwd=lane,
        env=lane_init._lane_env(lane),
        capture_output=True,
        text=True,
    )
    assert drift_attestation_result.returncode == 0, drift_attestation_result.stdout + drift_attestation_result.stderr
    drift_attestation = json.loads(drift_attestation_result.stdout)
    assert drift_attestation["digests"][0] != attestation.digests[0]
    preparation_driver = (
        "import json; "
        "from pathlib import Path; "
        "from devtools.testmon_bootstrap import prepare_native_testmon_environment; "
        "preparation = prepare_native_testmon_environment("
        "Path.cwd(), pytest_profile='correctness=complete', "
        "pytest_environment={'HYPOTHESIS_PROFILE': 'default', 'POLYLOGUE_CI': None}); "
        "print(json.dumps({'environment_name': preparation.environment_name, "
        "'selection_mode': preparation.selection_mode, "
        "'local_status': preparation.local_state.status, "
        "'copied_from': preparation.copied_from is not None}))"
    )
    preparation_result = subprocess.run(
        [str(lane_python), "-P", "-c", preparation_driver],
        cwd=lane,
        env=lane_init._lane_env(lane),
        capture_output=True,
        text=True,
    )
    assert preparation_result.returncode == 0, preparation_result.stdout + preparation_result.stderr
    preparation = json.loads(preparation_result.stdout)
    assert preparation["environment_name"] == drift_attestation["digests"][0]
    assert preparation["selection_mode"] == "bootstrap"
    assert preparation["local_status"] in {"absent", "invalid"}
    assert preparation["copied_from"] is False


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


def test_lane_environment_digest_uses_the_verify_digest_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "lane"
    python = lane / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    captured: dict[str, object] = {}

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        captured["cmd"] = list(cmd)
        return CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "pytest_profile": "correctness=complete",
                    "pytest_environment": {"HYPOTHESIS_PROFILE": "default", "POLYLOGUE_CI": None},
                    "digests": ["polylogue-current"],
                }
            )
            + "\n",
            "",
        )

    monkeypatch.setattr(lane_init, "_run", fake_run)

    assert lane_init.lane_environment_digest(lane) == "polylogue-current"
    command = captured["cmd"]
    assert isinstance(command, list)
    probe = command[-1]
    assert isinstance(probe, str)
    assert "_native_pytest_environment_candidates" in probe
    assert "initial.get('HYPOTHESIS_PROFILE')" not in probe


def test_lane_environment_attestation_includes_verify_default_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "lane"
    python = lane / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)

    def fake_run(
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        return CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "pytest_profile": "correctness=complete",
                    "pytest_environment": {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"},
                    "digests": ["polylogue-initial", "polylogue-default"],
                }
            ),
            "",
        )

    monkeypatch.setattr(lane_init, "_run", fake_run)
    attestation = lane_init.lane_environment_attestation(lane)

    assert attestation is not None
    assert attestation.digests == ("polylogue-initial", "polylogue-default")
    assert attestation.environment == {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"}


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
# Graph seeding reports warmth honestly (polylogue-l218h)
# ---------------------------------------------------------------------------


def _graph_with(path: Path, environments: Sequence[str]) -> Path:
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE environment (environment_name TEXT)")
        conn.executemany("INSERT INTO environment VALUES (?)", [(name,) for name in environments])
    return path


def _certified_graph_with(path: Path, environment_name: str) -> Path:
    return _certified_graph_with_names(path, [environment_name])


def _certified_graph_with_names(
    path: Path,
    environment_names: Sequence[str],
    recorded_test_name: str = "tests/test_recorded.py::test_recorded",
) -> Path:
    import testmon.db

    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH, write_certified_corpus

    data = path / TESTMON_DATA_RELPATH
    data.parent.mkdir(parents=True, exist_ok=True)
    recorded_test_path = recorded_test_name.split("::", 1)[0]
    (path / recorded_test_path).parent.mkdir(parents=True, exist_ok=True)
    (path / recorded_test_path).write_text("def test_recorded(): pass\n", encoding="utf-8")
    db = testmon.db.DB(str(data))
    try:
        con = db.con
        fingerprint_id = con.execute(
            "INSERT INTO file_fp (filename, method_checksums, mtime, fsha) VALUES (?, ?, ?, ?)",
            (recorded_test_path, b"", 0.0, ""),
        ).lastrowid
        for index, environment_name in enumerate(environment_names):
            environment_id = con.execute(
                "INSERT INTO environment (environment_name, system_packages, python_version) VALUES (?, ?, ?)",
                (environment_name, f"packages-{index}", "3.14"),
            ).lastrowid
            execution_id = con.execute(
                "INSERT INTO test_execution (environment_id, test_name, duration, failed, forced) VALUES (?, ?, ?, ?, ?)",
                (environment_id, recorded_test_name, 0.01, 0, 0),
            ).lastrowid
            con.execute(
                "INSERT INTO test_execution_file_fp (test_execution_id, fingerprint_id) VALUES (?, ?)",
                (execution_id, fingerprint_id),
            )
        con.commit()
    finally:
        db.con.close()

    for environment_name in set(environment_names):
        assert write_certified_corpus(path, environment_name, [recorded_test_name])
    return data


def _attestation(*digests: str, profile: str = "correctness=complete") -> lane_init.LaneEnvironmentAttestation:
    return lane_init.LaneEnvironmentAttestation(
        profile,
        (("HYPOTHESIS_PROFILE", "default"), ("POLYLOGUE_CI", None)),
        digests,
    )


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
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    _certified_graph_with(root, "polylogue-stale")

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is False
    assert "not attestable" in note
    assert "bootstrap" in note


def test_seed_reports_warm_only_on_a_real_environment_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    _certified_graph_with(root, "polylogue-current")

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is True
    assert "verifies start warm" in note


def test_seed_preserves_a_certified_destination_when_source_is_uncertified(tmp_path: Path) -> None:
    """Rerunning lane-init cannot replace a certified lane with stale source state."""
    import sqlite3

    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH, inspect_native_testmon_environment

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _certified_graph_with(root, "polylogue-current")
    _certified_graph_with_names(lane, ["polylogue-current", "lane-only"])
    with sqlite3.connect(root / TESTMON_DATA_RELPATH) as connection:
        connection.execute("DROP TABLE polylogue_certified_corpus")
        connection.commit()

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is True
    assert "preserved certified lane environment" in note
    assert inspect_native_testmon_environment(lane / TESTMON_DATA_RELPATH, environment_name="lane-only").valid


def test_seed_rejects_an_uncertified_source_before_publication(tmp_path: Path) -> None:
    """A structurally valid coordinator graph needs a certificate before copy."""
    import sqlite3

    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    _certified_graph_with(root, "polylogue-current")
    with sqlite3.connect(root / TESTMON_DATA_RELPATH) as connection:
        connection.execute("DROP TABLE polylogue_certified_corpus")
        connection.commit()

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is False
    assert "not certified" in note
    assert not (lane / TESTMON_DATA_RELPATH).exists()


def test_seed_uses_bound_source_for_certificate_read_after_public_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement immediately before source attestation cannot redirect the certificate read."""
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    source_data = _certified_graph_with_names(
        root,
        ["polylogue-current"],
        recorded_test_name="tests/test_original.py::test_original",
    )
    replacement_root = tmp_path / "replacement"
    replacement_data = _certified_graph_with_names(
        replacement_root,
        ["polylogue-current"],
        recorded_test_name="tests/test_replacement.py::test_replacement",
    )

    original_violation = testmon_bootstrap.certified_attestation_violation
    swapped = False

    def replace_before_certificate_read(
        repo_root: Path,
        *,
        environment_name: str,
        current_nodeids: Sequence[str],
        certificate_data_path: Path | None = None,
    ) -> str | None:
        nonlocal swapped
        if repo_root.resolve() == root.resolve() and certificate_data_path is not None and not swapped:
            swapped = True
            os.replace(replacement_data, source_data)
        return original_violation(
            repo_root,
            environment_name=environment_name,
            current_nodeids=current_nodeids,
            certificate_data_path=certificate_data_path,
        )

    monkeypatch.setattr(testmon_bootstrap, "certified_attestation_violation", replace_before_certificate_read)

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert swapped
    assert warm is True
    assert "verifies start warm" in note
    copied = testmon_bootstrap.inspect_native_testmon_environment(
        lane / testmon_bootstrap.TESTMON_DATA_RELPATH,
        environment_name="polylogue-current",
    )
    assert copied.valid
    assert copied.environment is not None
    assert copied.environment.nodeids == ("tests/test_original.py::test_original",)


def test_seed_preserves_a_certified_destination_when_source_primary_is_absent(tmp_path: Path) -> None:
    """A certified fallback lane survives an absent coordinator primary candidate."""
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH, inspect_native_testmon_environment

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _certified_graph_with(root, "unrelated-environment")
    _certified_graph_with(lane, "fallback-environment")

    note, warm = lane_init._seed_testmon_graph(
        root,
        lane,
        attestation=_attestation("primary-environment", "fallback-environment"),
    )

    assert warm is True
    assert "preserved certified lane environment" in note
    assert inspect_native_testmon_environment(
        lane / TESTMON_DATA_RELPATH,
        environment_name="fallback-environment",
    ).valid


def test_seed_does_not_preserve_a_fallback_when_destination_primary_is_invalid(tmp_path: Path) -> None:
    """An invalid primary blocks the fallback the verifier would reject."""
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH, inspect_native_testmon_environment

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _certified_graph_with(root, "unrelated-environment")
    _certified_graph_with_names(lane, ["primary-environment", "primary-environment", "fallback-environment"])

    note, warm = lane_init._seed_testmon_graph(
        root,
        lane,
        attestation=_attestation("primary-environment", "fallback-environment"),
    )

    assert warm is False
    assert "initial verify environment is invalid" in note
    assert inspect_native_testmon_environment(
        lane / TESTMON_DATA_RELPATH,
        environment_name="fallback-environment",
    ).valid


@pytest.mark.uses_real_clock("coordinates lane publication with verifier preparation")
def test_seed_serializes_with_verifier_publication_and_preserves_later_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verifier write cannot be overwritten by a concurrent lane seed."""
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    verifier_source = tmp_path / "verifier-source"
    lane.mkdir()
    _certified_graph_with(root, "lane-environment")
    _certified_graph_with(verifier_source, "verifier-environment")

    original_copy = testmon_bootstrap._atomic_copy_sqlite_database
    publication_started = threading.Event()
    release_publication = threading.Event()
    verifier_started = threading.Event()
    verifier_finished = threading.Event()
    first_publication = True
    publication_state_lock = threading.Lock()

    def block_lane_publication(*args: object, **kwargs: object) -> None:
        nonlocal first_publication
        with publication_state_lock:
            is_lane_publication = first_publication
            first_publication = False
        if is_lane_publication:
            publication_started.set()
            assert release_publication.wait(timeout=2)
        original_copy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(testmon_bootstrap, "_atomic_copy_sqlite_database", block_lane_publication)
    monkeypatch.setattr(
        testmon_bootstrap,
        "testmon_environment_digest",
        lambda checkout, **_kwargs: (
            "verifier-environment" if checkout.resolve() == lane.resolve() else "lane-environment"
        ),
    )
    monkeypatch.setattr(
        testmon_bootstrap,
        "linked_worktree_info",
        lambda checkout, **_kwargs: (True, verifier_source) if checkout.resolve() == lane.resolve() else None,
    )
    monkeypatch.setattr(
        verify,
        "linked_worktree_info",
        lambda checkout, **_kwargs: (True, verifier_source) if checkout.resolve() == lane.resolve() else None,
    )

    def verifier_prepare() -> testmon_bootstrap.NativeTestmonPreparation:
        verifier_started.set()
        try:
            with verify._native_testmon_lifecycle_lock(lane):
                return testmon_bootstrap.prepare_native_testmon_environment(lane)
        finally:
            verifier_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        lane_future = pool.submit(
            lane_init._seed_testmon_graph,
            root,
            lane,
            attestation=_attestation("lane-environment"),
        )
        assert publication_started.wait(timeout=2)
        verifier_future = pool.submit(verifier_prepare)
        assert verifier_started.wait(timeout=2)
        assert not verifier_finished.wait(timeout=0.05)
        release_publication.set()
        note, warm = lane_future.result(timeout=2)
        preparation = verifier_future.result(timeout=2)

    assert warm is True
    assert "verifies start warm" in note
    assert preparation.selection_mode == "affected"
    final_state = testmon_bootstrap.inspect_native_testmon_environment(
        lane / testmon_bootstrap.TESTMON_DATA_RELPATH,
        environment_name="verifier-environment",
    )
    assert final_state.valid


def test_seeded_primary_copy_stays_warm_when_lane_primary_state_is_invalid(tmp_path: Path) -> None:
    """Lane seeding and verifier preparation agree on a safe primary reuse."""
    from devtools.testmon_bootstrap import prepare_native_testmon_environment, testmon_environment_digest

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    digest_inputs = {"HYPOTHESIS_PROFILE": "default", "POLYLOGUE_CI": None}
    primary = testmon_environment_digest(
        root,
        pytest_profile="correctness=complete",
        pytest_environment=digest_inputs,
    )
    _certified_graph_with(root, primary)
    _certified_graph_with_names(lane, [primary, primary])

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation(primary))

    assert warm is True
    assert "verifies start warm" in note
    preparation = prepare_native_testmon_environment(
        lane,
        pytest_profile="correctness=complete",
        pytest_environment=digest_inputs,
    )
    assert preparation.environment_name == primary
    assert preparation.selection_mode == "affected"
    assert preparation.copied_from is None


def test_seed_accepts_verify_default_profile_fallback_when_certified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    _certified_graph_with(root, "polylogue-default")
    attestation = _attestation("polylogue-initial", "polylogue-default", profile="ci")

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=attestation)

    assert warm is True
    assert "verify default-profile fallback" in note


def test_seed_rejects_certified_fallback_after_an_invalid_initial_environment(
    tmp_path: Path,
) -> None:
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _certified_graph_with(root, "polylogue-default")
    _certified_graph_with_names(lane, ["polylogue-initial", "polylogue-initial"])

    note, warm = lane_init._seed_testmon_graph(
        root,
        lane,
        attestation=_attestation("polylogue-initial", "polylogue-default", profile="ci"),
    )

    assert warm is False
    assert "initial verify environment is invalid" in note
    assert "refusing the default-profile fallback" in note
    assert (lane / TESTMON_DATA_RELPATH).is_file()


def test_seed_then_verifier_stays_cold_after_invalid_initial_with_valid_default(tmp_path: Path) -> None:
    """An invalid primary must not authorize a different profile's graph."""
    from devtools.testmon_bootstrap import prepare_native_testmon_environment, testmon_environment_digest

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    initial_inputs = {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"}
    initial = testmon_environment_digest(
        root,
        pytest_profile="correctness=complete",
        pytest_environment=initial_inputs,
    )
    default = testmon_environment_digest(
        root,
        pytest_profile="correctness=complete",
        pytest_environment={"HYPOTHESIS_PROFILE": "default", "POLYLOGUE_CI": "1"},
    )
    _certified_graph_with(root, default)
    _certified_graph_with_names(lane, [initial, initial])

    note, warm = lane_init._seed_testmon_graph(
        root,
        lane,
        attestation=_attestation(initial, default),
    )

    assert warm is False
    assert "refusing the default-profile fallback" in note
    preparation = prepare_native_testmon_environment(
        lane,
        pytest_profile="correctness=complete",
        pytest_environment=initial_inputs,
    )
    assert preparation.environment_name == initial
    assert preparation.selection_mode == "bootstrap"
    assert preparation.copied_from is None


def test_seed_reports_cold_for_an_empty_or_absent_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()

    note, warm = lane_init._seed_testmon_graph(root, lane)
    assert warm is False
    assert "no coordinator graph" in note

    _graph_with(root / TESTMON_DATA_RELPATH, [])
    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))
    assert warm is False
    assert "not attestable" in note


def test_seed_reports_cold_when_the_lane_digest_cannot_be_computed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unverifiable lane must never be reported as warm."""
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    _certified_graph_with(root, "polylogue-something")
    note, warm = lane_init._seed_testmon_graph(root, lane)

    assert warm is False
    assert "normalized digest inputs" in note


def test_seed_rejects_a_minimal_fake_graph_even_with_a_matching_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    _graph_with(root / TESTMON_DATA_RELPATH, ["polylogue-current"])

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is False
    assert "not attestable" in note
    assert "initial verify environment is invalid" in note


def test_distribution_mutation_makes_a_seeded_graph_unattestable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed installed distribution cannot inherit a warm graph claim."""
    from devtools import testmon_bootstrap
    from devtools.testmon_bootstrap import testmon_environment_digest

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    lane.mkdir()
    digest_inputs = {"HYPOTHESIS_PROFILE": "default", "POLYLOGUE_CI": None}
    recorded = testmon_environment_digest(
        root,
        pytest_profile="correctness=complete",
        pytest_environment=digest_inputs,
    )
    _certified_graph_with(root, recorded)

    monkeypatch.setattr(testmon_bootstrap, "_installed_distributions", lambda: (("pytest", "mutated"),))
    unattestable = testmon_environment_digest(
        root,
        pytest_profile="correctness=complete",
        pytest_environment=digest_inputs,
    )
    assert unattestable != recorded
    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation(unattestable))

    assert warm is False
    assert "not attestable" in note
    assert "bootstrap" in note


@pytest.mark.parametrize("symlinked_parent", [".cache", ".cache/testmon"])
def test_seed_refuses_symlinked_cache_before_touching_external_sentinel(
    tmp_path: Path,
    symlinked_parent: str,
) -> None:
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    root = tmp_path / "root"
    lane = tmp_path / "lane"
    _certified_graph_with(root, "polylogue-current")
    external = tmp_path / "external"
    sentinel = external / "sentinel"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("do not touch", encoding="utf-8")
    parent = lane / symlinked_parent
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.symlink_to(external, target_is_directory=True)

    note, warm = lane_init._seed_testmon_graph(root, lane, attestation=_attestation("polylogue-current"))

    assert warm is False
    assert "unsafe owned testmon path" in note
    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    assert not (external / TESTMON_DATA_RELPATH.name).exists()


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
    lines = lane_init.dispatch_env_lines(lane, attestation=_attestation())
    joined = "\n".join(lines)

    assert f"export VIRTUAL_ENV={lane / '.venv'}" in lines
    for key in lane_init._INTERPRETER_DESCRIBING_ENV:
        assert f"unset {key}" in lines
    assert "export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in lines
    assert "$VIRTUAL_ENV/bin:$PATH" in joined


def test_dispatch_env_lines_bind_normalized_testmon_inputs(tmp_path: Path) -> None:
    lines = lane_init.dispatch_env_lines(
        tmp_path / "lane",
        attestation=_attestation("polylogue-initial", "polylogue-default", profile="correctness=complete"),
    )

    assert "export HYPOTHESIS_PROFILE=default" in lines
    assert "unset POLYLOGUE_CI" in lines
    assert "# testmon pytest profile: correctness=complete" in lines
    assert "# testmon fallback: HYPOTHESIS_PROFILE=default after a non-default bootstrap" in lines


def test_dispatch_env_lines_match_lane_env_sanitization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "lane"
    for key in lane_init._LANE_ENV_UNSETS:
        monkeypatch.setenv(key, f"inherited-{key}")
    attestation = lane_init.LaneEnvironmentAttestation(
        "correctness=complete",
        (("HYPOTHESIS_PROFILE", "ci"), ("POLYLOGUE_CI", "1")),
        ("polylogue-initial", "polylogue-default"),
    )

    sanitized = lane_init._lane_env(lane)
    lines = lane_init.dispatch_env_lines(lane, attestation=attestation)

    for key in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT", "UV_WORKING_DIR", *lane_init._INTERPRETER_DESCRIBING_ENV):
        assert key not in sanitized
        assert f"unset {key}" in lines
    assert sanitized["UV_PROJECT_ENVIRONMENT"] == str(lane / ".venv")
    assert f"export UV_PROJECT_ENVIRONMENT={lane / '.venv'}" in lines
    assert "export HYPOTHESIS_PROFILE=ci" in lines
    assert "export POLYLOGUE_CI=1" in lines
