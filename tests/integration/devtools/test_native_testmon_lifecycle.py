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

from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
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
    pytest.mark.timeout(90),
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
  "tui: serial native-testmon lane",
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
    semantic_marker = "not load_sensitive and not tui" if lane == "parallel" else "load_sensitive or tui"
    marker = semantic_marker if base_marker is None else f"({base_marker}) and ({semantic_marker})"
    selection = "--testmon-forceselect" if mode == "affected" else "--testmon-noselect"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--testmon",
        f"--testmon-env={environment_name}",
        selection,
        "-m",
        marker,
        "-p",
        "devtools.pytest_progress_plugin",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={artifact_dir / PYTEST_CANONICAL_REPORT_NAME}",
        "-n",
        str(workers),
    ]
    completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
    selection_payload = json.loads((artifact_dir / "selection.json").read_text(encoding="utf-8"))
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


def _run_production_verify(repo: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
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
import sys
from pathlib import Path
from types import SimpleNamespace

import devtools.verify as verify

root = Path(sys.argv[1]).resolve()
real_build = verify.build_verify_steps

def native_steps_only(**kwargs):
    return [step for step in real_build(**kwargs) if step[0].startswith("pytest native")]

verify.ROOT = root
verify.build_verify_steps = native_steps_only
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
    try:
        completed = subprocess.run(
            [sys.executable, "-c", driver, str(repo), *args, "--json"],
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
    payload = json.loads(completed.stdout)
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["invocation_id"] == invocation_id
    assert persisted["pytest_aggregate"] == payload["pytest_aggregate"]
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
    assert started.exists(), process.communicate(timeout=1)
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


def test_environment_and_plugin_identity_changes_start_real_fresh_bootstraps(
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

    monkeypatch.setenv("PYTEST_ADDOPTS", "-ra")
    environment_changed = prepare_native_testmon_environment(repo)
    assert environment_changed.selection_mode == "bootstrap"
    assert environment_changed.environment_name != initial.environment_name
    assert [
        result.completed.returncode
        for result in _run_plain_verify_corpus(
            repo, mode="bootstrap", environment_name=environment_changed.environment_name
        )
    ] == [0, 0]

    plugin = repo / "local_plugin.py"
    plugin.write_text(
        "import pytest\n\n@pytest.fixture\ndef native_identity():\n    return 'v1'\n",
        encoding="utf-8",
    )
    test_file.write_text(
        "def test_initial():\n    assert True\n\ndef test_plugin(native_identity):\n    assert native_identity == 'v1'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_PLUGINS", "local_plugin")
    plugin_changed = prepare_native_testmon_environment(repo)
    assert plugin_changed.selection_mode == "bootstrap"
    assert plugin_changed.environment_name != environment_changed.environment_name
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
