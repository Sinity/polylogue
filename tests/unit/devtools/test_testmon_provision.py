"""The datafile decides whether a run selects, seeds, or stops.

Anti-vacuity: the usable case is a datafile the installed pytest-testmon
itself wrote. A check that invents its own table list passes against nothing
real: reporting every genuine datafile as unusable discards the seed on every
provision, and reporting a foreign one as usable lets testmon silently replace
it and re-execute everything under the name of selection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from testmon.db import DATA_VERSION, DB

from devtools import testmon_provision
from devtools.testmon_provision import (
    TESTMON_ENVIRONMENT,
    TestmonGraphStatus,
    current_environment_key,
    discard_testmon_graph,
    inspect_testmon_graph,
)


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


def test_discard_removes_the_datafile_and_its_sidecars(tmp_path: Path) -> None:
    path = _seed_with_testmon(tmp_path)
    path.with_name(path.name + "-wal").write_bytes(b"")

    discard_testmon_graph(tmp_path)

    assert not path.exists()
    assert not path.with_name(path.name + "-wal").exists()
