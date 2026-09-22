"""A shortened node ID stays usable as a selection.

``tests/conftest.py`` shortens any node ID over 100 characters so xdist's
controller map and every worker's node metadata stay bounded when a
parameter's repr is a whole JSON payload. That rewrite runs *after* pytest has
matched the command line against the original ids, so until the map below
existed the shortened id was the only id anyone ever saw and none of them
could be run: ``devtools verify``'s failure rerun feeds reported ids straight
back to pytest and got ``ERROR: not found`` with no retry, and a human
reproducing one failure through ``devtools test <id>`` got the same.

Only FAILING ids are recorded. Mapping every shortened id instead measured
16,371 of 24,052 collected ids and a 4.3 MB file that every worker would read
on every collection, to answer a question only a failing id ever asks.

Anti-vacuity:
- drop the ``_record_long_nodeids`` call from ``pytest_runtest_logreport``
  and ``test_shortening_is_recorded`` goes red, because nothing knows what the
  digest stood for;
- drop the ``pytest_collection`` wrapper's translation and
  ``test_a_shortened_id_is_translated`` goes red, because the selection still
  names an id no collector will produce;
- make the map best-effort in the wrong direction -- translating an id that
  was never shortened -- and ``test_an_unrelated_argument_is_untouched`` goes
  red.

The end-to-end half was verified against a live run: with the map present,
``devtools test '<shortened id>'`` selects and runs the original test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.conftest as harness_conftest

_ORIGINAL = (
    "tests/unit/storage/test_audit_continuity.py::"
    "test_current_schema_missing_a_continuity_table_is_damage[continuity_attempts-and-a-long-parameter]"
)
_SHORTENED = "tests/unit/storage/test_audit_continuity.py::test_current_schema_missing[param-0011223344556677]"


@pytest.fixture
def _map_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cache" / "pytest-long-nodeids.json"
    monkeypatch.setattr(harness_conftest, "LONG_NODEID_MAP_PATH", path)
    return path


def test_shortening_is_recorded(_map_path: Path) -> None:
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})

    assert json.loads(_map_path.read_text(encoding="utf-8")) == {_SHORTENED: _ORIGINAL}


def test_a_shortened_id_is_translated(_map_path: Path) -> None:
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})

    assert harness_conftest._restore_long_nodeid_arguments([_SHORTENED]) == [_ORIGINAL]


def test_an_unrelated_argument_is_untouched(_map_path: Path) -> None:
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})
    selection = ["tests/unit/storage", "-k", "damage", "tests/x.py::test_y[param-ok]"]

    assert harness_conftest._restore_long_nodeid_arguments(selection) == selection


def test_an_absent_map_refuses_nothing(_map_path: Path) -> None:
    """A missing map leaves the selection exactly as the caller wrote it.

    The translation is a convenience over a cache; it must never turn an
    unreadable cache into a collection error.
    """
    assert not _map_path.exists()

    assert harness_conftest._restore_long_nodeid_arguments([_SHORTENED]) == [_SHORTENED]
    _map_path.parent.mkdir(parents=True)
    _map_path.write_text("{ not json", encoding="utf-8")
    assert harness_conftest._restore_long_nodeid_arguments([_SHORTENED]) == [_SHORTENED]


def test_recording_merges_rather_than_replaces(_map_path: Path) -> None:
    """Two xdist workers report into one file; neither may erase the other."""
    other = "tests/unit/storage/test_other.py::test_case[param-8877665544332211]"
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})
    harness_conftest._record_long_nodeids({other: "tests/unit/storage/test_other.py::test_case[long]"})

    stored = json.loads(_map_path.read_text(encoding="utf-8"))
    assert set(stored) == {_SHORTENED, other}
