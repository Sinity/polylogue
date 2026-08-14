from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import devtools.verify as verify
from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
    NativeTestmonRepairError,
    inspect_native_testmon_environment,
    prepare_native_testmon_environment,
)
from devtools.testmon_bootstrap import (
    testmon_environment_digest as _testmon_environment_digest,
)
from devtools.verify_runs import PYTEST_CANONICAL_REPORT_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[3]
pytestmark = [
    pytest.mark.uses_real_clock("coordinates real pytest subprocesses and an interrupt deadline"),
    pytest.mark.timeout(300),
]


@dataclass(frozen=True)
class LaneResult:
    completed: subprocess.CompletedProcess[str]
    artifact_dir: Path
    selection: dict[str, object]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _init_repo(root: Path, *, conftest: str = "") -> None:
    (root / "tests").mkdir(parents=True)
    (root / ".gitignore").write_text(
        ".artifacts/\n.benchmarks/\n.cache/\n.coverage*\n.pytest_cache/\n__pycache__/\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        """
[tool.pytest.ini_options]
addopts = "-p no:randomly"
cache_dir = ".cache/pytest"
markers = [
  "load_sensitive: serial native-testmon lane",
  "tui: Textual interaction category",
]
""".lstrip(),
        encoding="utf-8",
    )
    (root / "tests" / "conftest.py").write_text(conftest, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Polylogue Tests")


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def _pytest_environment(repo: Path) -> dict[str, str]:
    (repo / TESTMON_DATA_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("PYTEST_"):
            env.pop(key)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["TESTMON_DATAFILE"] = str(repo / TESTMON_DATA_RELPATH)
    env["PYTHONPATH"] = os.pathsep.join((str(repo), str(PROJECT_ROOT), env.get("PYTHONPATH", "")))
    env["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] = "50000"
    return env


def _run_lane(
    repo: Path,
    *,
    environment_name: str,
    mode: str,
    lane: str,
    workers: int = 0,
    timeout: float = 30,
    base_marker: str | None = None,
) -> LaneResult:
    artifact_dir = repo / ".artifacts" / f"{mode}-{lane}-{uuid.uuid4().hex}"
    artifact_dir.mkdir(parents=True)
    env = _pytest_environment(repo)
    env.update(
        {
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(artifact_dir / "events"),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(artifact_dir / "selection.json"),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(artifact_dir / "summary.json"),
        }
    )
    semantic_marker = "not load_sensitive" if lane == "parallel" else "load_sensitive"
    marker = semantic_marker if base_marker is None else f"({base_marker}) and ({semantic_marker})"
    selection = "--testmon-forceselect" if mode == "affected" else "--testmon-noselect"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--override-ini=addopts=",
        "--testmon",
        f"--testmon-env={environment_name}",
        selection,
        "-m",
        marker,
        "-p",
        "devtools.pytest_progress_plugin",
        "-p",
        "pytest-testmon",
        "-p",
        "pytest_jsonreport",
        "-p",
        "xdist",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={artifact_dir / PYTEST_CANONICAL_REPORT_NAME}",
        "-n",
        str(workers),
    ]
    completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
    selection_path = artifact_dir / "selection.json"
    if not selection_path.is_file():
        raise AssertionError(
            f"native pytest lane produced no selection artifact (returncode={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    return LaneResult(completed, artifact_dir, selection_payload)


def _run_plain_verify_corpus(
    repo: Path,
    *,
    mode: str,
    environment_name: str,
    base_marker: str | None = None,
) -> tuple[LaneResult, LaneResult]:
    parallel = _run_lane(
        repo,
        environment_name=environment_name,
        mode=mode,
        lane="parallel",
        workers=2,
        base_marker=base_marker,
    )
    serial = _run_lane(
        repo,
        environment_name=environment_name,
        mode=mode,
        lane="serial",
        base_marker=base_marker,
    )
    return parallel, serial


def _run_production_verify(
    repo: Path,
    *args: str,
    allow_rejection: bool = False,
    environment_overrides: dict[str, str] | None = None,
    interpreter_args: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run the production verifier orchestration against a tiny fixture corpus.

    The subprocess keeps the real native preparation, two-lane runner,
    containment, deadline, aggregate, invocation receipt, and XDG history.
    Only unrelated static gates are filtered so this fixture need not copy the
    entire Polylogue source tree.
    """
    state_root = repo.parent / f"{repo.name}-verify-state"
    receipt = state_root / "receipts" / f"{uuid.uuid4().hex}.json"
    invocation_id = uuid.uuid4().hex
    driver = """
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import devtools.verify as verify

root = Path(sys.argv[1]).resolve()
real_build = verify.build_verify_steps
real_env_for_pytest_step = verify.env_for_pytest_step
pytest_index = root / ".git" / "pytest-index"
shutil.copy2(root / ".git" / "index", pytest_index)

def native_steps_only(**kwargs):
    return [step for step in real_build(**kwargs) if step[0].startswith("pytest native")]

def fixture_env_for_pytest_step(env, **kwargs):
    child_env = real_env_for_pytest_step(env, **kwargs)
    child_env["GIT_INDEX_FILE"] = str(pytest_index)
    return child_env

verify.ROOT = root
verify.build_verify_steps = native_steps_only
verify.env_for_pytest_step = fixture_env_for_pytest_step
verify.assert_polylogue_matches_checkout = lambda *_args, **_kwargs: SimpleNamespace(
    polylogue_import_path=root / "polylogue" / "__init__.py",
    as_dict=lambda: {"checkout_root": str(root), "test_fixture": True},
)
os.chdir(root)
raise SystemExit(verify.main(sys.argv[2:]))
"""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT),
            "XDG_STATE_HOME": str(state_root / "xdg-state"),
            "POLYLOGUE_PYTEST_WORKERS": "1",
            "POLYLOGUE_VERIFICATION_INVOCATION_ID": invocation_id,
            "POLYLOGUE_VERIFICATION_RECEIPT_PATH": str(receipt),
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    if environment_overrides is not None:
        env.update(environment_overrides)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                *interpreter_args,
                "-c",
                driver,
                str(repo),
                *args,
                "--json",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_value: object = exc.stdout
        stderr_value: object = exc.stderr
        stdout = (
            stdout_value.decode(errors="replace")
            if isinstance(stdout_value, bytes)
            else stdout_value
            if isinstance(stdout_value, str)
            else ""
        )
        stderr = (
            stderr_value.decode(errors="replace")
            if isinstance(stderr_value, bytes)
            else stderr_value
            if isinstance(stderr_value, str)
            else ""
        )
        pytest.fail(f"production verify fixture timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")
    if not receipt.exists():
        pytest.fail(
            f"production verify wrote no invocation receipt\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"production verify emitted no JSON payload ({exc})\n"
            f"returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if completed.returncode == 125 and not allow_rejection:
        pytest.fail(
            f"production verify rejected the fixture checkout\npayload:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        persisted = json.loads(receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"production verify wrote invalid receipt JSON ({exc})\n"
            f"returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    assert persisted["invocation_id"] == invocation_id
    assert persisted["pytest_aggregate"] == payload["pytest_aggregate"]
    for authority_field in ("diagnosis", "exit_code", "release_baseline_allowed"):
        assert persisted.get(authority_field) == payload.get(authority_field)
    return completed, payload


def _selected(*results: LaneResult) -> set[str]:
    selected: set[str] = set()
    for result in results:
        raw_nodeids = result.selection.get("selected_nodeids")
        assert isinstance(raw_nodeids, list)
        assert all(isinstance(nodeid, str) for nodeid in raw_nodeids)
        selected.update(raw_nodeids)
    return selected


def test_empty_plain_verify_bootstraps_then_warm_verify_is_affected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        """
import pytest

def test_parallel_owner():
    from app import answer
    assert answer() == 42

@pytest.mark.load_sensitive
def test_serial_owner():
    from app import answer
    assert answer() == 42
""".lstrip(),
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")

    preparation = prepare_native_testmon_environment(repo)
    assert preparation.selection_mode == "bootstrap"
    first = _run_plain_verify_corpus(
        repo,
        mode=preparation.selection_mode,
        environment_name=preparation.environment_name,
    )
    assert [result.completed.returncode for result in first] == [0, 0]
    state = inspect_native_testmon_environment(
        repo / TESTMON_DATA_RELPATH,
        environment_name=preparation.environment_name,
        required_executable_paths=("app.py",),
    )
    assert state.valid
    assert state.environment is not None
    assert state.environment.corpus_count == 2

    warm = prepare_native_testmon_environment(repo, required_executable_paths=("app.py",))
    assert warm.selection_mode == "affected"
    second = _run_plain_verify_corpus(repo, mode="affected", environment_name=warm.environment_name)
    assert [result.completed.returncode for result in second] == [0, 0]
    assert _selected(*second) == set()


def test_production_plain_verify_owns_bootstrap_warm_selection_deadline_and_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    integration_dir = repo / "tests" / "integration"
    integration_dir.mkdir()
    (integration_dir / "test_app.py").write_text(
        """
import pytest

def test_parallel_owner():
    from polylogue.app import answer
    assert answer() == 42

@pytest.mark.load_sensitive
def test_serial_owner():
    from polylogue.app import answer
    assert answer() == 42
""".lstrip(),
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    first, bootstrap = _run_production_verify(repo)

    assert first.returncode == 0, f"{first.stderr}\n{json.dumps(bootstrap, indent=2, sort_keys=True)}"
    assert bootstrap["testmon_environment"]["selection_mode"] == "bootstrap"
    aggregate = bootstrap["pytest_aggregate"]
    assert aggregate["complete_corpus_covered"] is True
    assert aggregate["terminal_green"] is True
    assert aggregate["cleanup"] == {"complete": True}
    assert aggregate["containment"] == {"complete": True}
    assert aggregate["deadline"] == {"budget_s": 3600.0, "met": True}
    lane_steps = [step for step in bootstrap["steps"] if step.get("semantic_lane")]
    assert [step["semantic_lane"] for step in lane_steps] == ["parallel", "serial"]
    environments = {
        arg
        for step in lane_steps
        for arg in step["statistics"]["command"]
        if isinstance(arg, str) and arg.startswith("--testmon-env=")
    }
    assert environments == {f"--testmon-env={bootstrap['testmon_environment']['name']}"}
    lane_timeouts = [step["timeout_s"] for step in lane_steps]
    assert 0 < lane_timeouts[1] < lane_timeouts[0] <= 3600
    history_path = repo.parent / "repo-verify-state" / "xdg-state" / "polylogue" / "devtools" / "verify-history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert history[-1]["pytest_aggregate"] == aggregate

    second, warm = _run_production_verify(repo)

    assert second.returncode == 0, second.stderr
    assert warm["testmon_environment"]["selection_mode"] == "affected"
    assert warm["release_baseline_allowed"] is False
    assert warm["pytest_aggregate"]["selected_union_count"] == 0

    (package / "app.py").write_text("def answer() -> int:\n    return 0\n", encoding="utf-8")
    third, mutated = _run_production_verify(repo)

    assert third.returncode == 1
    assert mutated["testmon_environment"]["selection_mode"] == "affected"
    assert mutated["pytest_aggregate"]["terminal_union_count"] == 2
    assert mutated["release_baseline_allowed"] is False
    assert "assert 0 == 42" in third.stderr


def test_production_verify_all_grants_release_authority_after_complete_two_lane_run(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_release.py").write_text(
        "import pytest\n\n"
        "def test_parallel_release_owner():\n"
        "    from polylogue.app import answer\n"
        "    assert answer() == 42\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_release_owner():\n"
        "    from polylogue.app import answer\n"
        "    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    completed, payload = _run_production_verify(repo, "--all")

    assert completed.returncode == 0, completed.stderr
    assert payload["tier"] == "full"
    assert payload["testmon_environment"]["selection_mode"] == "full"
    assert payload["verification_scope"] == "release-baseline"
    assert payload["release_baseline_allowed"] is True
    assert payload["worktree_fingerprint"] == payload["final_worktree_fingerprint"]
    lanes = [step for step in payload["steps"] if step.get("semantic_lane")]
    assert [step["semantic_lane"] for step in lanes] == ["parallel", "serial"]
    assert [step["name"] for step in lanes] == ["pytest native parallel (full)", "pytest native serial (full)"]
    for step in lanes:
        assert "--testmon-noselect" in step["statistics"]["command"]
        assert "--testmon-forceselect" not in step["statistics"]["command"]
        assert "--override-ini=addopts=" in step["statistics"]["command"]
        assert step["external_addopts_neutralized"] is True
        assert step["external_plugins_neutralized"] is True
        assert step["closed_world_collection"] is True
    aggregate = payload["pytest_aggregate"]
    assert aggregate["external_addopts_neutralized"] is True
    assert aggregate["external_plugins_neutralized"] is True
    assert aggregate["closed_world_collection"] is True
    assert aggregate["selection_mode"] == "full"
    assert aggregate["environment"]["native_corpus_count"] == 2
    assert aggregate["corpus"]["count"] == 2
    assert aggregate["selected_union_count"] == 2
    assert aggregate["terminal_union_count"] == 2
    assert aggregate["missing_terminal_count"] == 0
    assert aggregate["complete_corpus_covered"] is True
    assert aggregate["terminal_green"] is True
    assert aggregate["cleanup"] == {"complete": True}
    assert aggregate["containment"] == {"complete": True}
    assert aggregate["deadline"] == {"budget_s": 3600.0, "met": True}


@pytest.mark.parametrize(("verify_args", "selection_mode"), [((), "bootstrap"), (("--all",), "full")])
def test_release_native_runs_override_a_reduced_hypothesis_profile(
    tmp_path: Path,
    verify_args: tuple[str, ...],
    selection_mode: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(
        repo,
        conftest=(
            "import os\n"
            "from hypothesis import settings\n\n"
            "settings.register_profile('default', max_examples=100)\n"
            "settings.register_profile('verify', max_examples=10)\n"
            "settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'default'))\n"
        ),
    )
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_hypothesis_profile.py").write_text(
        "import pytest\n"
        "from hypothesis import settings\n\n"
        "def test_parallel_release_profile_is_complete():\n"
        "    assert settings().max_examples == 100\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_release_profile_is_complete():\n"
        "    assert settings().max_examples == 100\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    completed, payload = _run_production_verify(
        repo,
        *verify_args,
        environment_overrides={"HYPOTHESIS_PROFILE": "verify"},
    )

    assert completed.returncode == 0, completed.stderr
    assert payload["testmon_environment"]["selection_mode"] == selection_mode
    assert payload["release_baseline_allowed"] is True


@pytest.mark.parametrize(
    ("environment_addopts", "configured_addopts"),
    [
        ("--setup-only", None),
        ("--collect-only", None),
        ("tests/test_release.py::test_parallel_body_must_run", None),
        ("--ignore-glob=tests/**", None),
        ("--ignore-glob tests/**", None),
        (None, "--setup-only --ignore-glob=tests/**"),
        ("-ra --strict-markers", None),
    ],
    ids=(
        "setup-only",
        "collect-only",
        "positional-node",
        "ignore-glob-equal",
        "ignore-glob-split",
        "configured",
        "harmless",
    ),
)
def test_production_verify_all_neutralizes_external_pytest_addopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_addopts: str | None,
    configured_addopts: str | None,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_release.py").write_text(
        "import pytest\n\n"
        "def test_parallel_body_must_run():\n"
        "    assert False, 'parallel body executed'\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_body_must_run():\n"
        "    assert False, 'serial body executed'\n",
        encoding="utf-8",
    )
    if configured_addopts is not None:
        config = repo / "pyproject.toml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                'addopts = "-p no:randomly"',
                f'addopts = "{configured_addopts}"',
            ),
            encoding="utf-8",
        )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")
    if environment_addopts is None:
        monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    else:
        monkeypatch.setenv("PYTEST_ADDOPTS", environment_addopts)

    completed, payload = _run_production_verify(repo, "--all")

    assert completed.returncode == 1
    assert payload["release_baseline_allowed"] is False
    aggregate = payload["pytest_aggregate"]
    assert aggregate["external_addopts_neutralized"] is True
    assert aggregate["external_plugins_neutralized"] is True
    assert aggregate["selected_union_count"] == 2
    assert aggregate["terminal_union_count"] == 2
    assert aggregate["outcomes"] == {"failed": 2}
    lanes = [step for step in payload["steps"] if step.get("semantic_lane")]
    assert [step["external_addopts_neutralized"] for step in lanes] == [True, True]
    assert [step["external_plugins_neutralized"] for step in lanes] == [True, True]
    assert all("--override-ini=addopts=" in step["statistics"]["command"] for step in lanes)
    assert "parallel body executed" in completed.stderr
    assert "serial body executed" in completed.stderr


def test_production_verify_all_drops_pythonpath_startup_injection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    release_source = (
        "import pytest\n\n"
        "def test_parallel_passes():\n"
        "    assert True\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_passes():\n"
        "    assert True\n\n"
        "def test_omitted_failure():\n"
        "    assert False, 'PYTHONPATH startup injection did not narrow execution'\n"
    )
    (repo / "tests" / "test_release.py").write_text(release_source, encoding="utf-8")
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")
    (repo / "sitecustomize.py").write_text(
        'import os\nos.environ["PYTEST_ADDOPTS"] = "-k passes"\nos.environ["HYPOTHESIS_PROFILE"] = "narrow"\n',
        encoding="utf-8",
    )

    completed, payload = _run_production_verify(repo, "--all")

    assert completed.returncode == 1
    assert payload["release_baseline_allowed"] is False
    aggregate = payload["pytest_aggregate"]
    assert aggregate["selected_union_count"] == 3
    assert aggregate["terminal_union_count"] == 3
    assert aggregate["outcomes"] == {"failed": 1, "passed": 2}
    assert aggregate["terminal_green"] is False
    assert "PYTHONPATH startup injection did not narrow execution" in completed.stderr


def test_plain_native_lane_environment_removes_ambient_pytest_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "ambient_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "ambient test identity")

    environment = _pytest_environment(tmp_path)

    assert {key for key in environment if key.startswith("PYTEST_")} == {"PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.mark.parametrize(
    ("environment_overrides", "interpreter_args", "verify_args"),
    [
        ({"PYTHONOPTIMIZE": "1"}, (), ("--all",)),
        ({}, ("-O",), ("--all",)),
        ({}, ("-OO",), ("--all",)),
        ({"PYTHONOPTIMIZE": "1"}, (), ("--quick", "--lab")),
        ({"PYTHONOPTIMIZE": "1"}, (), ("--commit", "--lab")),
    ],
    ids=("pythonoptimize", "dash-o", "dash-oo", "quick-lab", "commit-lab"),
)
def test_production_verify_rejects_optimized_managed_pytest_interpreter(
    tmp_path: Path,
    environment_overrides: dict[str, str],
    interpreter_args: tuple[str, ...],
    verify_args: tuple[str, ...],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "invariant.py").write_text(
        "def require_failure():\n    assert False, 'product assertion executed'\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_release.py").write_text(
        "import pytest\n"
        "from polylogue.invariant import require_failure\n\n"
        "def test_parallel_product_assertion():\n"
        "    require_failure()\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_lane():\n"
        "    require_failure()\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    completed, payload = _run_production_verify(
        repo,
        *verify_args,
        allow_rejection=True,
        environment_overrides=environment_overrides,
        interpreter_args=interpreter_args,
    )

    assert completed.returncode == 125
    assert payload["diagnosis"] == "optimized_python_interpreter"
    assert payload["exit_code"] == 125
    assert payload["release_baseline_allowed"] is False
    assert payload["pytest_aggregate"]["selection_mode"] == "none"
    assert "Python optimization disables verification assertions" in completed.stderr


@pytest.mark.parametrize("ambient_addopts", ["--collect-only", "--setup-only"])
def test_production_affected_verify_neutralizes_execution_suppressing_addopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_addopts: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    tests = repo / "tests" / "test_affected.py"
    tests.write_text(
        "import pytest\n\n"
        "def test_parallel_body():\n"
        "    assert True\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_body():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")
    monkeypatch.setenv("PYTEST_ADDOPTS", ambient_addopts)

    seeded, bootstrap = _run_production_verify(repo)

    assert seeded.returncode == 0, seeded.stderr
    assert bootstrap["testmon_environment"]["selection_mode"] == "bootstrap"
    tests.write_text(
        "import pytest\n\n"
        "def test_parallel_body():\n"
        "    assert False, 'affected parallel body executed'\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_body():\n"
        "    assert False, 'affected serial body executed'\n",
        encoding="utf-8",
    )

    completed, payload = _run_production_verify(repo)

    assert completed.returncode == 1
    assert payload["testmon_environment"]["selection_mode"] == "affected"
    assert payload["release_baseline_allowed"] is False
    aggregate = payload["pytest_aggregate"]
    assert aggregate["selected_union_count"] == 2
    assert aggregate["terminal_union_count"] == 2
    assert aggregate["outcomes"] == {"failed": 2}
    lanes = [step for step in payload["steps"] if step.get("semantic_lane")]
    assert [step["external_addopts_neutralized"] for step in lanes] == [True, True]
    assert [step["closed_world_collection"] for step in lanes] == [True, True]
    assert "affected parallel body executed" in completed.stderr
    assert "affected serial body executed" in completed.stderr


def test_production_verify_all_owns_complete_test_root_over_configured_testpaths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    narrowed = repo / "tests" / "narrowed"
    narrowed.mkdir()
    (narrowed / "test_owned.py").write_text(
        "import pytest\n\n"
        "def test_parallel_owned():\n"
        "    assert True\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_owned():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    outside = repo / "tests" / "outside"
    outside.mkdir()
    (outside / "test_omitted_failure.py").write_text(
        "class TestOmitted:\n"
        "    def test_must_not_be_omitted(self):\n"
        "        assert False, 'outside configured discovery executed'\n",
        encoding="utf-8",
    )
    (repo / "ambient_narrow.py").write_text(
        "def pytest_ignore_collect(collection_path, config):\n    return 'outside' in collection_path.parts\n",
        encoding="utf-8",
    )
    config = repo / "pyproject.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'cache_dir = ".cache/pytest"',
            'cache_dir = ".cache/pytest"\n'
            'testpaths = ["tests/narrowed"]\n'
            'python_files = ["test_owned.py"]\n'
            'python_classes = ["Owned"]\n'
            'python_functions = ["test_*_owned"]\n'
            'norecursedirs = ["outside"]',
        ),
        encoding="utf-8",
    )
    _commit_all(repo, "narrow discovery fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")
    monkeypatch.setenv("PYTEST_PLUGINS", "ambient_narrow")
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    completed, payload = _run_production_verify(repo, "--all")

    assert completed.returncode == 1
    assert payload["release_baseline_allowed"] is False
    aggregate = payload["pytest_aggregate"]
    assert aggregate["external_plugins_neutralized"] is True
    assert aggregate["closed_world_collection"] is True
    assert aggregate["corpus"]["count"] == 3
    assert aggregate["selected_union_count"] == 3
    assert aggregate["terminal_union_count"] == 3
    assert aggregate["outcomes"] == {"failed": 1, "passed": 2}
    assert aggregate["complete_corpus_covered"] is True
    assert aggregate["terminal_green"] is False
    lanes = [step for step in payload["steps"] if step.get("semantic_lane")]
    assert [step["external_plugins_neutralized"] for step in lanes] == [True, True]
    assert [step["closed_world_collection"] for step in lanes] == [True, True]
    assert all(step["statistics"]["command"].count("tests") == 1 for step in lanes)
    assert "outside configured discovery executed" in completed.stderr


def test_runtime_json_only_mutation_forces_complete_native_selection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    semantic = repo / "polylogue" / "archive" / "semantic"
    data = semantic / "data"
    data.mkdir(parents=True)
    for package in (repo / "polylogue", repo / "polylogue" / "archive", semantic):
        (package / "__init__.py").write_text("", encoding="utf-8")
    pricing_data = data / "litellm_model_prices.json"
    pricing_data.write_text('{"test-model": {"input_cost_per_token": 42}}\n', encoding="utf-8")
    (semantic / "pricing.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "def input_price() -> int:\n"
        "    path = Path(__file__).parent / 'data' / 'litellm_model_prices.json'\n"
        "    return int(json.loads(path.read_text(encoding='utf-8'))['test-model']['input_cost_per_token'])\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_pricing.py").write_text(
        "import pytest\n\n"
        "def test_parallel_pricing_owner():\n"
        "    from polylogue.archive.semantic.pricing import input_price\n"
        "    assert input_price() == 42\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_pricing_owner():\n"
        "    from polylogue.archive.semantic.pricing import input_price\n"
        "    assert input_price() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr
    assert bootstrap["testmon_environment"]["selection_mode"] == "bootstrap"

    pricing_data.write_text('{"test-model": {"input_cost_per_token": 0}}\n', encoding="utf-8")
    completed, mutated = _run_production_verify(repo)

    assert completed.returncode == 1
    assert mutated["testmon_environment"]["selection_mode"] == "full"
    assert mutated["testmon_environment"]["runtime_data_paths"] == [
        "polylogue/archive/semantic/data/litellm_model_prices.json"
    ]
    assert mutated["pytest_aggregate"]["selected_union_count"] == 2
    assert mutated["pytest_aggregate"]["terminal_union_count"] == 2
    assert mutated["release_baseline_allowed"] is False
    assert [step["semantic_lane"] for step in mutated["steps"] if step.get("semantic_lane")] == [
        "parallel",
        "serial",
    ]
    assert "assert 0 == 42" in completed.stderr


def test_production_verify_test_runtime_data_mutation_executes_and_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    data = repo / "tests" / "data" / "expected.txt"
    data.parent.mkdir()
    data.write_text("42\n", encoding="utf-8")
    (repo / "tests" / "test_data.py").write_text(
        "import pytest\n"
        "from pathlib import Path\n\n"
        "def expected() -> int:\n"
        "    return int((Path(__file__).parent / 'data' / 'expected.txt').read_text())\n\n"
        "def test_parallel_data_owner():\n"
        "    assert expected() == 42\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_data_owner():\n"
        "    assert expected() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr
    assert bootstrap["testmon_environment"]["selection_mode"] == "bootstrap"

    data.write_text("0\n", encoding="utf-8")
    completed, mutated = _run_production_verify(repo)

    assert completed.returncode == 1
    assert mutated["testmon_environment"]["selection_mode"] == "full"
    assert mutated["testmon_environment"]["runtime_data_paths"] == ["tests/data/expected.txt"]
    assert mutated["pytest_aggregate"]["selected_union_count"] == 2
    assert mutated["pytest_aggregate"]["terminal_union_count"] == 2
    assert "assert 0 == 42" in completed.stderr


def test_production_verify_deleted_module_rebuilds_and_fails_dependents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from polylogue.app import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr
    assert bootstrap["pytest_aggregate"]["selected_union_count"] == 1

    _git(repo, "rm", "polylogue/app.py")
    _commit_all(repo, "delete production module")
    completed, mutated = _run_production_verify(repo)

    assert completed.returncode == 1
    assert mutated["testmon_environment"]["selection_mode"] == "bootstrap"
    assert mutated["testmon_environment"]["required_executable_paths"] == []
    assert mutated["testmon_environment"]["bootstrap_trigger_paths"] == ["polylogue/app.py"]
    assert mutated["pytest_aggregate"]["selected_union_count"] == 1
    assert mutated["pytest_aggregate"]["terminal_union_count"] == 1
    assert "ModuleNotFoundError" in completed.stderr


def test_production_verify_moved_module_rebuilds_and_fails_dependents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from polylogue.app import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr
    assert bootstrap["pytest_aggregate"]["selected_union_count"] == 1

    _git(repo, "mv", "polylogue/app.py", "polylogue/renamed.py")
    _commit_all(repo, "move production module")
    completed, mutated = _run_production_verify(repo)

    assert completed.returncode == 1
    assert mutated["testmon_environment"]["selection_mode"] == "bootstrap"
    assert mutated["testmon_environment"]["required_executable_paths"] == ["polylogue/renamed.py"]
    assert mutated["testmon_environment"]["bootstrap_trigger_paths"] == [
        "polylogue/app.py",
        "polylogue/renamed.py",
    ]
    assert mutated["pytest_aggregate"]["selected_union_count"] == 1
    assert mutated["pytest_aggregate"]["terminal_union_count"] == 1
    assert "ModuleNotFoundError" in completed.stderr


def test_production_verify_deleted_module_with_updated_imports_rebuilds_successfully(tmp_path: Path) -> None:
    """Keeping the deleted path in post-run requirements makes the updated import fail."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "old.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (package / "app.py").write_text("from polylogue.old import answer\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from polylogue.app import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, _bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr

    _git(repo, "rm", "polylogue/old.py")
    (package / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    _commit_all(repo, "delete module and update imports")
    completed, rebuilt = _run_production_verify(repo)

    assert completed.returncode == 0, completed.stderr
    environment = rebuilt["testmon_environment"]
    assert environment["selection_mode"] == "bootstrap"
    assert environment["required_executable_paths"] == ["polylogue/app.py"]
    assert environment["bootstrap_trigger_paths"] == ["polylogue/app.py", "polylogue/old.py"]
    assert rebuilt["pytest_aggregate"]["terminal_green"] is True


def test_production_verify_moved_module_with_updated_imports_rebuilds_successfully(tmp_path: Path) -> None:
    """Keeping the old path as a graph requirement makes the updated move fail."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    package = repo / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "old.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from polylogue.old import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    seeded, _bootstrap = _run_production_verify(repo)
    assert seeded.returncode == 0, seeded.stderr

    _git(repo, "mv", "polylogue/old.py", "polylogue/renamed.py")
    (repo / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from polylogue.renamed import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(repo, "move module and update imports")
    completed, rebuilt = _run_production_verify(repo)

    assert completed.returncode == 0, completed.stderr
    environment = rebuilt["testmon_environment"]
    assert environment["selection_mode"] == "bootstrap"
    assert environment["required_executable_paths"] == ["polylogue/renamed.py", "tests/test_app.py"]
    assert environment["bootstrap_trigger_paths"] == [
        "polylogue/old.py",
        "polylogue/renamed.py",
        "tests/test_app.py",
    ]
    assert rebuilt["pytest_aggregate"]["terminal_green"] is True


def test_empty_linked_worktree_with_empty_main_self_bootstraps(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _init_repo(main)
    (main / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (main / "tests" / "test_app.py").write_text("def test_value():\n    from app import VALUE\n    assert VALUE == 1\n")
    _commit_all(main, "fixture")
    lane = tmp_path / "lane"
    _git(main, "worktree", "add", "-qb", "lane", str(lane))

    preparation = prepare_native_testmon_environment(lane)

    assert preparation.linked_worktree
    assert preparation.main_checkout == main
    assert preparation.copied_from is None
    assert preparation.selection_mode == "bootstrap"
    results = _run_plain_verify_corpus(lane, mode="bootstrap", environment_name=preparation.environment_name)
    assert [result.completed.returncode for result in results] == [0, 0]


def test_matching_main_copy_then_product_mutation_selects_and_fails_owner(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _init_repo(main)
    (main / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (main / "tests" / "test_app.py").write_text(
        "def test_answer():\n    from app import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    _commit_all(main, "fixture")
    main_preparation = prepare_native_testmon_environment(main)
    _run_plain_verify_corpus(main, mode="bootstrap", environment_name=main_preparation.environment_name)
    lane = tmp_path / "lane"
    _git(main, "worktree", "add", "-qb", "lane", str(lane))
    (lane / "app.py").write_text("def answer() -> int:\n    return 0\n", encoding="utf-8")

    preparation = prepare_native_testmon_environment(lane, required_executable_paths=("app.py",))

    assert preparation.selection_mode == "affected"
    assert preparation.copied_from == main / TESTMON_DATA_RELPATH
    result = _run_lane(
        lane,
        environment_name=preparation.environment_name,
        mode="affected",
        lane="parallel",
    )
    assert result.completed.returncode == 1
    assert result.selection["selected_nodeids"] == ["tests/test_app.py::test_answer"]
    assert "assert 0 == 42" in result.completed.stdout


def test_interrupted_bootstrap_native_state_resumes_failed_and_unfinished_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    started = repo / "started"
    (repo / "tests" / "test_a_completed.py").write_text(
        "def test_completed():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_b_failed.py").write_text(
        "def test_failed():\n    assert False\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_c_unfinished.py").write_text(
        f"""
import pathlib
import time

def test_unfinished():
    marker = pathlib.Path({str(started)!r})
    if marker.exists():
        return
    marker.write_text('started')
    time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")
    environment_name = _testmon_environment_digest(repo)
    env = _pytest_environment(repo)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "--testmon",
        f"--testmon-env={environment_name}",
        "--testmon-noselect",
        "tests",
    ]
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not started.exists():
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        raise AssertionError(f"interrupted fixture never started\nstdout:\n{stdout}\nstderr:\n{stderr}")
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=10)

    state = inspect_native_testmon_environment(
        repo / TESTMON_DATA_RELPATH,
        environment_name=environment_name,
    )
    assert state.valid
    resumed = _run_lane(repo, environment_name=environment_name, mode="affected", lane="parallel")
    assert resumed.completed.returncode == 1
    assert _selected(resumed) == {
        "tests/test_b_failed.py::test_failed",
        "tests/test_c_unfinished.py::test_unfinished",
    }
    assert "tests/test_a_completed.py::test_completed" not in _selected(resumed)


def test_node_add_delete_converges_without_a_custom_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    tests = repo / "tests" / "test_nodes.py"
    tests.write_text("def test_existing():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    preparation = prepare_native_testmon_environment(repo)
    _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=preparation.environment_name)

    tests.write_text("def test_existing():\n    assert True\n\ndef test_new():\n    assert True\n", encoding="utf-8")
    added = _run_plain_verify_corpus(repo, mode="affected", environment_name=preparation.environment_name)
    assert "tests/test_nodes.py::test_new" in _selected(*added)
    tests.write_text("def test_new():\n    assert True\n", encoding="utf-8")
    _run_plain_verify_corpus(repo, mode="affected", environment_name=preparation.environment_name)

    state = inspect_native_testmon_environment(
        repo / TESTMON_DATA_RELPATH,
        environment_name=preparation.environment_name,
    )
    assert state.valid
    assert state.environment is not None
    assert state.environment.nodeids == ("tests/test_nodes.py::test_new",)


def test_collection_only_executable_dependency_blocks_until_test_executes_it(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, conftest="from app import answer\n")
    (repo / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    test_file = repo / "tests" / "test_app.py"
    test_file.write_text("def test_unrelated():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    preparation = prepare_native_testmon_environment(repo)
    _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=preparation.environment_name)

    blind = inspect_native_testmon_environment(
        repo / TESTMON_DATA_RELPATH,
        environment_name=preparation.environment_name,
        required_executable_paths=("app.py",),
    )
    assert not blind.valid
    assert blind.missing_executable_paths == ("app.py",)

    (repo / "tests" / "conftest.py").write_text("", encoding="utf-8")
    test_file.write_text(
        "def test_owner():\n    from app import answer\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    repaired = prepare_native_testmon_environment(repo, required_executable_paths=("app.py",))
    assert repaired.selection_mode == "bootstrap"
    _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=repaired.environment_name)
    assert inspect_native_testmon_environment(
        repo / TESTMON_DATA_RELPATH,
        environment_name=repaired.environment_name,
        required_executable_paths=("app.py",),
    ).valid


def test_removed_environment_or_dependency_edge_invalidates_native_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("def test_value():\n    from app import VALUE\n    assert VALUE == 1\n")
    _commit_all(repo, "fixture")
    preparation = prepare_native_testmon_environment(repo)
    _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=preparation.environment_name)
    data = repo / TESTMON_DATA_RELPATH

    with sqlite3.connect(data) as connection:
        connection.execute("DELETE FROM environment WHERE environment_name = ?", (preparation.environment_name,))
    assert not inspect_native_testmon_environment(data, environment_name=preparation.environment_name).valid

    rebuilt = prepare_native_testmon_environment(repo)
    _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=rebuilt.environment_name)
    with sqlite3.connect(data) as connection:
        connection.execute(
            """
            DELETE FROM test_execution_file_fp
            WHERE fingerprint_id IN (SELECT id FROM file_fp WHERE filename = 'app.py')
            """
        )
    missing = inspect_native_testmon_environment(
        data,
        environment_name=rebuilt.environment_name,
        required_executable_paths=("app.py",),
    )
    assert not missing.valid
    assert missing.missing_executable_paths == ("app.py",)


def test_neutralized_environment_and_declared_plugin_identity_are_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    test_file = repo / "tests" / "test_identity.py"
    test_file.write_text("def test_initial():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")

    initial = prepare_native_testmon_environment(repo)
    assert initial.selection_mode == "bootstrap"
    assert [
        result.completed.returncode
        for result in _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=initial.environment_name)
    ] == [0, 0]

    monkeypatch.setenv("PYTEST_ADDOPTS", "--setup-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "ambient_plugin")
    environment_unchanged = prepare_native_testmon_environment(repo)
    assert environment_unchanged.selection_mode == "affected"
    assert environment_unchanged.environment_name == initial.environment_name
    test_file.write_text(
        "import pytest\n\n"
        "def test_parallel_body_must_run():\n"
        "    assert False, 'ambient pytest controls suppressed the parallel body'\n\n"
        "@pytest.mark.load_sensitive\n"
        "def test_serial_body_must_run():\n"
        "    assert False, 'ambient pytest controls suppressed the serial body'\n",
        encoding="utf-8",
    )
    ambient_results = _run_plain_verify_corpus(
        repo,
        mode="bootstrap",
        environment_name=environment_unchanged.environment_name,
    )
    assert [result.completed.returncode for result in ambient_results] == [1, 1]
    assert _selected(*ambient_results) == {
        "tests/test_identity.py::test_parallel_body_must_run",
        "tests/test_identity.py::test_serial_body_must_run",
    }
    monkeypatch.delenv("PYTEST_ADDOPTS")
    monkeypatch.delenv("PYTEST_PLUGINS")

    plugin = repo / "local_plugin.py"
    plugin.write_text(
        "import pytest\n\n@pytest.fixture\ndef native_identity():\n    return 'v1'\n",
        encoding="utf-8",
    )
    (repo / "tests" / "conftest.py").write_text('pytest_plugins = ("local_plugin",)\n', encoding="utf-8")
    test_file.write_text(
        "def test_initial():\n    assert True\n\ndef test_plugin(native_identity):\n    assert native_identity == 'v1'\n",
        encoding="utf-8",
    )
    plugin_changed = prepare_native_testmon_environment(repo)
    assert plugin_changed.selection_mode == "bootstrap"
    assert plugin_changed.environment_name != environment_unchanged.environment_name
    plugin_results = _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=plugin_changed.environment_name)
    assert [result.completed.returncode for result in plugin_results] == [0, 0]
    assert "tests/test_identity.py::test_plugin" in _selected(*plugin_results)

    plugin.write_text(
        "import pytest\n\n@pytest.fixture\ndef native_identity():\n    return 'v2'\n",
        encoding="utf-8",
    )
    test_file.write_text(test_file.read_text(encoding="utf-8").replace("'v1'", "'v2'"), encoding="utf-8")
    plugin_mutated = prepare_native_testmon_environment(repo)
    assert plugin_mutated.selection_mode == "bootstrap"
    assert plugin_mutated.environment_name != plugin_changed.environment_name
    assert [
        result.completed.returncode
        for result in _run_plain_verify_corpus(repo, mode="bootstrap", environment_name=plugin_mutated.environment_name)
    ] == [0, 0]


def test_production_verify_reports_stdout_when_json_payload_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_json_result(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        receipt = Path(environment["POLYLOGUE_VERIFICATION_RECEIPT_PATH"])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "not JSON", "verifier diagnostics")

    monkeypatch.setattr(subprocess, "run", invalid_json_result)

    with pytest.raises(pytest.fail.Exception) as failure:
        _run_production_verify(tmp_path)

    assert "production verify emitted no JSON payload" in str(failure.value)
    assert "stdout:\nnot JSON" in str(failure.value)
    assert "stderr:\nverifier diagnostics" in str(failure.value)


def test_production_verify_reports_stdout_when_receipt_json_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_receipt_result(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        receipt = Path(environment["POLYLOGUE_VERIFICATION_RECEIPT_PATH"])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("not JSON", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "{}", "verifier diagnostics")

    monkeypatch.setattr(subprocess, "run", invalid_receipt_result)

    with pytest.raises(pytest.fail.Exception) as failure:
        _run_production_verify(tmp_path)

    assert "production verify wrote invalid receipt JSON" in str(failure.value)
    assert "stdout:\n{}" in str(failure.value)
    assert "stderr:\nverifier diagnostics" in str(failure.value)


@pytest.mark.parametrize(
    "declaration",
    [
        "from plugin_config import plugin_names\n\npytest_plugins = plugin_names\n",
        'from plugin_config import plugin_names\n\nglobals()["pytest_plugins"] = plugin_names\n',
    ],
)
def test_production_verify_fails_closed_on_dynamic_pytest_plugins(tmp_path: Path, declaration: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "plugin_config.py").write_text('plugin_names = ("local_plugin",)\n', encoding="utf-8")
    (repo / "local_plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "conftest.py").write_text(declaration, encoding="utf-8")
    (repo / "tests" / "test_body.py").write_text("def test_body():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    origin = tmp_path / "origin.git"
    _git(origin.parent, "init", "--bare", "-q", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "branch", "-M", "master")
    _git(repo, "push", "-qu", "origin", "master")

    completed, payload = _run_production_verify(repo, allow_rejection=True)

    assert completed.returncode == 125
    assert payload["diagnosis"] == "native_testmon_preparation_failed"
    assert payload["release_baseline_allowed"] is False
    assert "pytest_plugins declaration must" in completed.stderr


def test_managed_native_launch_keeps_state_inode_bound_during_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tests" / "test_body.py").write_text("def test_body():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    prepare_native_testmon_environment(repo)
    external_cache = tmp_path / "external-cache"
    external_cache.mkdir()
    monkeypatch.setattr(verify, "ROOT", repo)
    monkeypatch.setattr(verify, "TESTMON_DATA", repo / TESTMON_DATA_RELPATH)

    def replace_parent(
        _label: str,
        _cmd: list[str],
        **kwargs: object,
    ) -> tuple[int, float, dict[str, object]]:
        bound_data = kwargs["native_testmon_data"]
        assert isinstance(bound_data, Path)
        (repo / ".cache").rename(repo / ".cache-owned")
        (repo / ".cache").symlink_to(external_cache, target_is_directory=True)
        bound_data.write_text("bound database", encoding="utf-8")
        return 0, 0.01, {}

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(verify, "_run_step", replace_parent)
        assert verify._run("pytest native parallel (affected)", ["pytest"])[0] == 0

    assert (repo / ".cache-owned" / "testmon" / "testmondata").read_text(encoding="utf-8") == "bound database"
    assert list(external_cache.iterdir()) == []


def test_managed_native_routes_reject_replaced_cache_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tests" / "test_body.py").write_text("def test_body():\n    assert True\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    preparation = prepare_native_testmon_environment(repo)
    owned_cache = repo / ".cache"
    owned_cache.rename(repo / ".cache-owned")
    external_cache = tmp_path / "external-cache"
    external_cache.mkdir()
    sentinel = external_cache / "sentinel"
    sentinel.write_text("external", encoding="utf-8")
    owned_cache.symlink_to(external_cache, target_is_directory=True)
    monkeypatch.setattr(verify, "ROOT", repo)
    monkeypatch.setattr(verify, "TESTMON_DATA", repo / TESTMON_DATA_RELPATH)

    with pytest.raises(NativeTestmonRepairError, match="refusing symlinked owned testmon parent"):
        verify._run("pytest native parallel (affected)", ["pytest"])
    with pytest.raises(NativeTestmonRepairError, match="refusing symlinked owned testmon parent"):
        verify._native_environment_after_run(preparation, required_executable_paths=())

    assert sentinel.read_text(encoding="utf-8") == "external"
    assert list(external_cache.iterdir()) == [sentinel]


def test_runtime_helper_mutation_stays_incremental_and_selects_owner(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    infra = repo / "tests" / "infra"
    infra.mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (infra / "__init__.py").write_text("", encoding="utf-8")
    helper = infra / "runtime_helper.py"
    helper.write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
    (repo / "tests" / "test_runtime_helper.py").write_text(
        """
import pytest

def test_runtime_helper_owner():
    from tests.infra.runtime_helper import answer
    assert answer() == 42

@pytest.mark.load_sensitive
def test_runtime_helper_serial_owner():
    from tests.infra.runtime_helper import answer
    assert answer() == 42
""".lstrip(),
        encoding="utf-8",
    )
    _commit_all(repo, "fixture")

    preparation = prepare_native_testmon_environment(repo)
    bootstrap = _run_plain_verify_corpus(
        repo,
        mode="bootstrap",
        environment_name=preparation.environment_name,
    )
    assert [result.completed.returncode for result in bootstrap] == [0, 0]

    helper.write_text("def answer() -> int:\n    return 0\n", encoding="utf-8")
    affected = prepare_native_testmon_environment(repo)
    assert affected.selection_mode == "affected"
    results = _run_plain_verify_corpus(repo, mode="affected", environment_name=affected.environment_name)

    assert [result.completed.returncode for result in results] == [1, 1]
    assert _selected(*results) == {
        "tests/test_runtime_helper.py::test_runtime_helper_owner",
        "tests/test_runtime_helper.py::test_runtime_helper_serial_owner",
    }
