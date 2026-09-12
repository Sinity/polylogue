"""The datafile decides whether a run selects, seeds, or stops.

Anti-vacuity: the usable case is a datafile the installed pytest-testmon
itself wrote. A check that invents its own table list passes against nothing
real: reporting every genuine datafile as unusable discards the seed on every
provision, and reporting a foreign one as usable lets testmon silently replace
it and re-execute everything under the name of selection.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest
from testmon.db import DATA_VERSION, DB

from devtools import testmon_provision
from devtools.agent_env import HARNESS_RUN_ENV
from devtools.pytest_invocation import MANAGED_PLUGIN_ARGS
from devtools.run_tests import ROOT, build_pytest_cmd, focused_pytest_env
from devtools.testmon_provision import (
    TESTMON_COVERAGE_CORE,
    TESTMON_ENVIRONMENT,
    TestmonGraphStatus,
    current_environment_key,
    discard_testmon_graph,
    inspect_testmon_graph,
)
from devtools.toolchain import venv_python
from devtools.verify_runs import VerifyRun


def _seed_with_testmon(root: Path, *, packages: str | None = None, python_version: str | None = None) -> Path:
    path = testmon_provision.testmon_datafile(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    database = DB(str(path))
    current_packages, current_version = current_environment_key()
    database.initiate_execution(
        TESTMON_ENVIRONMENT,
        packages if packages is not None else current_packages,
        python_version or current_version,
        {},
    )
    database.con.close()
    return path


def _set_user_version(path: Path, version: int) -> None:
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()


def test_absent_datafile_is_a_seed_not_a_failure(tmp_path: Path) -> None:
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.ABSENT
    assert not state.usable


def test_a_datafile_the_installed_testmon_wrote_is_usable(tmp_path: Path) -> None:
    _seed_with_testmon(tmp_path)
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.USABLE
    assert state.full_rerun_cause is None
    assert state.recorded_tests == 0
    assert state.source_dependencies == 0


def test_a_datafile_from_another_testmon_data_version_is_unusable(tmp_path: Path) -> None:
    path = _seed_with_testmon(tmp_path)
    _set_user_version(path, DATA_VERSION + 1)
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.UNUSABLE
    assert f"version {DATA_VERSION + 1}" in state.reason


def test_a_datafile_missing_testmon_tables_is_unusable(tmp_path: Path) -> None:
    path = testmon_provision.testmon_datafile(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY)")
    connection.execute(f"PRAGMA user_version = {DATA_VERSION}")
    connection.commit()
    connection.close()
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.UNUSABLE
    assert "incompatible testmon version" in state.reason


def test_a_package_change_is_reported_as_the_cause_of_a_full_rerun(tmp_path: Path) -> None:
    _seed_with_testmon(tmp_path, packages="somepkg 1.0")
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.USABLE
    assert state.full_rerun_cause == "the installed packages changed"


def test_an_interpreter_change_is_reported_as_the_cause_of_a_full_rerun(tmp_path: Path) -> None:
    _seed_with_testmon(tmp_path, python_version="2.7.18")
    state = inspect_testmon_graph(tmp_path)
    assert state.usable
    assert state.full_rerun_cause is not None
    assert "interpreter changed" in state.full_rerun_cause


def test_a_graph_with_no_source_dependencies_is_unusable(tmp_path: Path) -> None:
    """Traced without dynamic contexts, every test depends only on its own file."""
    path = _seed_with_testmon(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO test_execution (environment_id, test_name, duration, failed, forced) VALUES (1, 'tests/test_x.py::test_a', 0.1, 0, 0)"
    )
    connection.execute(
        "INSERT INTO file_fp (filename, method_checksums, mtime, fsha) VALUES ('tests/test_x.py', X'00', 0, 'abc')"
    )
    connection.execute("INSERT INTO test_execution_file_fp VALUES (1, 1)")
    connection.commit()
    connection.close()
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.UNUSABLE
    assert "no dependency on any source file" in state.reason
    assert state.recorded_tests == 1
    assert state.source_dependencies == 0

    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO file_fp (filename, method_checksums, mtime, fsha) VALUES ('polylogue/x.py', X'00', 0, 'def')"
    )
    connection.execute("INSERT INTO test_execution_file_fp VALUES (1, 2)")
    connection.commit()
    connection.close()
    assert inspect_testmon_graph(tmp_path).usable


def test_a_corrupt_datafile_is_unusable(tmp_path: Path) -> None:
    path = testmon_provision.testmon_datafile(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a database at all")
    assert inspect_testmon_graph(tmp_path).status is TestmonGraphStatus.UNUSABLE


def test_provision_discards_only_an_unusable_datafile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _seed_with_testmon(tmp_path)
    _set_user_version(path, DATA_VERSION + 1)
    monkeypatch.chdir(tmp_path)

    assert testmon_provision.main([]) == 0
    assert not path.exists()

    _seed_with_testmon(tmp_path)
    assert testmon_provision.main([]) == 0
    assert path.exists()


def test_provision_json_includes_inspected_graph_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_with_testmon(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert testmon_provision.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["recorded_tests"] == 0
    assert payload["source_dependencies"] == 0


def test_seeding_snapshots_a_database_with_an_uncommitted_writer(tmp_path: Path) -> None:
    """Anti-vacuity: a byte copy in place of the backup API makes this red.

    The source is left mid-transaction with its committed tail in the WAL, so
    a copy of the main file alone either loses the committed row or carries
    the uncommitted one. The backup API reads under SQLite's locking and
    writes one consistent, sidecar-free file.
    """
    source = tmp_path / "primary" / ".cache" / "testmon" / "testmondata"
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as setup:
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("CREATE TABLE fingerprint (value TEXT)")
        setup.execute("INSERT INTO fingerprint VALUES ('committed')")

    writer = sqlite3.connect(source)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO fingerprint VALUES ('uncommitted')")

        destination = testmon_provision.testmon_datafile(tmp_path / "worktree")
        assert testmon_provision.snapshot_testmon_graph(source, destination) is True
    finally:
        writer.rollback()
        writer.close()

    assert not destination.with_name(destination.name + "-wal").exists()
    with sqlite3.connect(destination) as check:
        assert [row[0] for row in check.execute("SELECT value FROM fingerprint")] == ["committed"]


def test_seeding_an_absent_or_unreadable_source_leaves_no_datafile(tmp_path: Path) -> None:
    """Anti-vacuity: writing the destination before the source is validated, or
    keeping a partial destination on failure, makes this red — the provision
    check would then accept a file no testmon wrote.
    """
    destination = testmon_provision.testmon_datafile(tmp_path / "worktree")
    assert testmon_provision.snapshot_testmon_graph(tmp_path / "missing", destination) is False
    assert not destination.exists()

    junk = tmp_path / "junk"
    junk.write_bytes(b"not a database at all")
    assert testmon_provision.snapshot_testmon_graph(junk, destination) is False
    assert not destination.exists()


def test_discard_removes_the_datafile_and_its_sidecars(tmp_path: Path) -> None:
    path = _seed_with_testmon(tmp_path)
    path.with_name(path.name + "-wal").write_bytes(b"")

    discard_testmon_graph(tmp_path)

    assert not path.exists()
    assert not path.with_name(path.name + "-wal").exists()


def test_seeding_keeps_a_usable_local_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout's own current graph is never replaced by the seed.

    Anti-vacuity: seeding unconditionally (the previous behaviour) would replace
    the local file, and the marker row written below would disappear.
    """
    seed_root = tmp_path / "seed"
    seed = _seed_with_testmon(seed_root)
    local_root = tmp_path / "local"
    local = _seed_with_testmon(local_root)
    connection = sqlite3.connect(local)
    connection.execute("CREATE TABLE local_marker (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    monkeypatch.chdir(local_root)

    assert testmon_provision.main(["--seed", str(seed)]) == 0

    connection = sqlite3.connect(local)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert "local_marker" in tables


def test_seeding_replaces_a_local_graph_that_would_rerun_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_root = tmp_path / "seed"
    seed = _seed_with_testmon(seed_root)
    local_root = tmp_path / "local"
    local = _seed_with_testmon(local_root, packages="stale 0.1")
    monkeypatch.chdir(local_root)

    assert testmon_provision.main(["--seed", str(seed)]) == 0

    assert inspect_testmon_graph(local_root).full_rerun_cause is None
    assert local.exists()


def test_seeding_keeps_a_local_graph_when_the_seed_is_no_fresher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_root = tmp_path / "seed"
    seed = _seed_with_testmon(seed_root, packages="stale-seed 0.1")
    local_root = tmp_path / "local"
    local = _seed_with_testmon(local_root, packages="stale-local 0.1")
    connection = sqlite3.connect(local)
    connection.execute("CREATE TABLE local_marker (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    monkeypatch.chdir(local_root)

    assert testmon_provision.main(["--seed", str(seed)]) == 0

    connection = sqlite3.connect(local)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert "local_marker" in tables


def test_a_newer_primary_graph_does_not_replace_a_working_local_graph(tmp_path: Path) -> None:
    """A checkout that runs repeatedly keeps its own graph; age is not the test.

    Anti-vacuity: comparing modification times instead (refresh whenever the
    primary is newer) drops the marker table below. That is what emptied the
    CI runner's accumulated graph on every run and imported the primary's
    package set with it.
    """
    primary_root = tmp_path / "primary"
    primary = _seed_with_testmon(primary_root)
    local_root = tmp_path / "runner"
    local = _seed_with_testmon(local_root)
    connection = sqlite3.connect(local)
    connection.execute("CREATE TABLE local_marker (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    os.utime(primary, ns=(local.stat().st_mtime_ns + 10**9, local.stat().st_mtime_ns + 10**9))

    assert testmon_provision.sync_testmon_graph(local_root, source=primary) is False

    connection = sqlite3.connect(local)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert "local_marker" in tables


def test_an_absent_local_graph_is_taken_from_the_primary(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    primary = _seed_with_testmon(primary_root)
    local_root = tmp_path / "worktree"
    local_root.mkdir()

    assert testmon_provision.sync_testmon_graph(local_root, source=primary) is True
    assert inspect_testmon_graph(local_root).usable


def test_a_local_graph_that_would_rerun_everything_takes_the_primary(tmp_path: Path) -> None:
    """The one case where the primary wins: its environment matches, the local
    one does not, so keeping the local graph costs a full re-execution."""
    primary_root = tmp_path / "primary"
    primary = _seed_with_testmon(primary_root)
    local_root = tmp_path / "runner"
    _seed_with_testmon(local_root, packages="stale 0.1")

    assert testmon_provision.sync_testmon_graph(local_root, source=primary) is True
    assert inspect_testmon_graph(local_root).full_rerun_cause is None


def _synthetic_corpus(root: Path) -> None:
    """A corpus whose every test runs one shared module, so editing that module
    leaves no recorded test stable -- the state in which testmon prunes."""
    (root / "tests").mkdir(parents=True)
    (root / "hub.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
    for index in range(5):
        (root / f"leaf{index}.py").write_text(f"def value():\n    return {index}\n", encoding="utf-8")
        lines = [f"import leaf{index}", "import hub", ""]
        for test_index in range(4):
            lines += [
                f"def test_{test_index}():",
                f"    assert leaf{index}.value() + hub.shared() == {index} + 1",
                "",
            ]
        (root / "tests" / f"test_leaf{index}.py").write_text("\n".join(lines), encoding="utf-8")


def _recorded_tests(datafile: Path) -> int:
    connection = sqlite3.connect(f"file:{datafile}?mode=ro", uri=True)
    with contextlib.closing(connection):
        return int(connection.execute("SELECT count(*) FROM test_execution").fetchone()[0])


def _managed_pytest_environment(root: Path, datafile: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(ROOT)))
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["COVERAGE_CORE"] = TESTMON_COVERAGE_CORE
    environment["TESTMON_DATAFILE"] = str(datafile)
    for leaked in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"):
        environment.pop(leaked, None)
    return environment


def test_a_focused_run_does_not_configure_a_testmon_graph(tmp_path: Path) -> None:
    """Focused runs do not load testmon or configure a datafile.

    Anti-vacuity: returning an environment with ``TESTMON_DATAFILE`` would
    make a future plugin or ambient option mutate a graph for a non-corpus
    selection. Only an affected/full verification run owns the corpus graph.
    """
    run = VerifyRun(tier="focused-test", argv=["tests/unit/devtools"], git_head=None, root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["python", "-m", "pytest"])

    environment = focused_pytest_env(run=run, artifacts=artifacts)

    assert "TESTMON_DATAFILE" not in environment


def test_the_focused_command_leaves_a_corpus_graph_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production focused command never opens the corpus graph.

    Anti-vacuity: enabling testmon in the focused command would prune the
    twenty-test corpus graph down to the four tests named by the selection.
    """
    _synthetic_corpus(tmp_path)
    corpus_graph = tmp_path / "corpus-testmondata"

    corpus = subprocess.run(
        [
            str(venv_python(root=ROOT)),
            "-m",
            "pytest",
            "-q",
            *MANAGED_PLUGIN_ARGS,
            "--testmon",
            f"--testmon-env={TESTMON_ENVIRONMENT}",
            "--testmon-noselect",
            "tests",
        ],
        cwd=tmp_path,
        env=_managed_pytest_environment(tmp_path, corpus_graph),
        capture_output=True,
        text=True,
        check=False,
    )
    assert corpus.returncode == 0, corpus.stdout + corpus.stderr
    assert _recorded_tests(corpus_graph) == 20

    # Every recorded test now depends on a module that changed, so none of them
    # is stable and none of them would survive a prune.
    (tmp_path / "hub.py").write_text("def shared():\n    return 1\n\n\ndef extra():\n    return 2\n", encoding="utf-8")

    # Exercise a top-level focused command. The outer managed test advertises
    # that it already holds the pytest slot, which correctly disables testmon
    # for genuinely nested commands.
    monkeypatch.delenv(HARNESS_RUN_ENV, raising=False)
    focused = subprocess.run(
        build_pytest_cmd(["tests/test_leaf0.py"]),
        cwd=tmp_path,
        env=_managed_pytest_environment(tmp_path, tmp_path / "scratch-testmondata"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert focused.returncode == 0, focused.stdout + focused.stderr
    assert _recorded_tests(corpus_graph) == 20
    assert not (tmp_path / "scratch-testmondata").exists()
