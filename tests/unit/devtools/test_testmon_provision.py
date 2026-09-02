"""The datafile decides whether a run selects, seeds, or stops.

Anti-vacuity: reporting a corrupt or foreign-format datafile as usable lets
`devtools verify` select against a graph that cannot answer, which silently
skips tests. Reporting an absent one as unusable stops a lane that should just
seed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from devtools import testmon_provision
from devtools.testmon_provision import (
    TestmonGraphStatus,
    discard_testmon_graph,
    inspect_testmon_graph,
)


def _seed(root: Path, *, tables: tuple[str, ...]) -> Path:
    path = testmon_provision.testmon_datafile(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for table in tables:
        connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    return path


def test_absent_datafile_is_a_seed_not_a_failure(tmp_path: Path) -> None:
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.ABSENT
    assert not state.usable


def test_a_complete_datafile_is_usable(tmp_path: Path) -> None:
    _seed(tmp_path, tables=("environment", "node"))
    assert inspect_testmon_graph(tmp_path).status is TestmonGraphStatus.USABLE


def test_a_datafile_from_another_testmon_version_is_unusable(tmp_path: Path) -> None:
    _seed(tmp_path, tables=("environment",))
    state = inspect_testmon_graph(tmp_path)
    assert state.status is TestmonGraphStatus.UNUSABLE
    assert "incompatible testmon version" in state.reason


def test_a_corrupt_datafile_is_unusable(tmp_path: Path) -> None:
    path = testmon_provision.testmon_datafile(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a database at all")
    assert inspect_testmon_graph(tmp_path).status is TestmonGraphStatus.UNUSABLE


def test_provision_discards_only_an_unusable_datafile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _seed(tmp_path, tables=("environment",))
    monkeypatch.chdir(tmp_path)

    assert testmon_provision.main([]) == 0
    assert not path.exists()

    _seed(tmp_path, tables=("environment", "node"))
    assert testmon_provision.main([]) == 0
    assert path.exists()


def test_discard_removes_the_datafile_and_its_sidecars(tmp_path: Path) -> None:
    path = _seed(tmp_path, tables=("environment", "node"))
    path.with_name(path.name + "-wal").write_bytes(b"")

    discard_testmon_graph(tmp_path)

    assert not path.exists()
    assert not path.with_name(path.name + "-wal").exists()
