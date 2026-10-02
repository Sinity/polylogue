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

import fcntl
import json
import threading
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.uses_real_clock
@pytest.mark.parametrize("keep_lock", [True, False])
def test_concurrent_failure_hooks_preserve_both_original_selectors(
    _map_path: Path, monkeypatch: pytest.MonkeyPatch, keep_lock: bool
) -> None:
    """Removing read/merge/publish exclusion loses a deliberately interleaved failure.

    The first reporter pauses after reading. The second reporter reaches either
    the real lock or, under the unlocked mutant, its own completed stale read
    before the first is released. Both reports exercise the loaded failure hook.
    """
    prior = "tests/prior.py::test_prior[param-0000000000000000]"
    second = "tests/second.py::test_second[param-1111111111111111]"
    second_original = "tests/second.py::test_second[a-long-original-parameter]"
    harness_conftest._record_long_nodeids({prior: "tests/prior.py::test_prior[original]"})
    monkeypatch.setitem(harness_conftest._SHORTENED_NODEIDS, _SHORTENED, _ORIGINAL)
    monkeypatch.setitem(harness_conftest._SHORTENED_NODEIDS, second, second_original)
    first_read = threading.Event()
    second_attempted = threading.Event()
    release_first = threading.Event()
    original_load = harness_conftest._load_long_nodeid_map
    original_flock = fcntl.flock

    def load() -> dict[str, str]:
        loaded = original_load()
        if threading.current_thread().name == "nodeid-first":
            first_read.set()
            assert release_first.wait(10), "first reporter was never released"
        elif threading.current_thread().name == "nodeid-second":
            second_attempted.set()
        return loaded

    def flock(fd: int, operation: int) -> None:
        if keep_lock and threading.current_thread().name == "nodeid-second" and operation == fcntl.LOCK_EX:
            second_attempted.set()
        if keep_lock:
            original_flock(fd, operation)

    def report(nodeid: str, name: str) -> None:
        threading.current_thread().name = name
        harness_conftest.pytest_runtest_logreport(
            pytest.TestReport(
                nodeid=nodeid,
                location=("synthetic.py", 1, "test_case"),
                keywords={},
                outcome="failed",
                longrepr="synthetic failure",
                when="call",
            )
        )

    monkeypatch.setattr(harness_conftest, "_load_long_nodeid_map", load)
    monkeypatch.setattr(fcntl, "flock", flock)
    with ThreadPoolExecutor(max_workers=2) as reporters:
        first = reporters.submit(report, _SHORTENED, "nodeid-first")
        try:
            assert first_read.wait(10), "first reporter never read the original map"
            other = reporters.submit(report, second, "nodeid-second")
            assert second_attempted.wait(10), "second reporter never attempted the shared map"
        finally:
            release_first.set()
        first.result(timeout=10)
        other.result(timeout=10)
    expected = {prior: "tests/prior.py::test_prior[original]", _SHORTENED: _ORIGINAL, second: second_original}
    if keep_lock:
        assert original_load() == expected
        assert harness_conftest._restore_long_nodeid_arguments([prior, _SHORTENED, second]) == list(expected.values())
    else:
        # The same real hooks and forced interleaving expose the lost update
        # when the existing kernel exclusion is removed.
        assert original_load() != expected
        assert prior in original_load()
        assert len(original_load()) == 2


@pytest.mark.parametrize("contents", ["{not-json", "[]"])
def test_recording_replaces_an_invalid_map(_map_path: Path, contents: str) -> None:
    """Invalid cache contents cannot prevent a later failed selector being recorded."""
    _map_path.parent.mkdir(parents=True)
    _map_path.write_text(contents, encoding="utf-8")
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})
    assert harness_conftest._load_long_nodeid_map() == {_SHORTENED: _ORIGINAL}


def test_recording_preserves_existing_duplicate_key_behavior(_map_path: Path) -> None:
    """Locking keeps duplicate-only no-op and mixed-batch update semantics."""
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})
    harness_conftest._record_long_nodeids({_SHORTENED: "other-original"})
    assert harness_conftest._load_long_nodeid_map() == {_SHORTENED: _ORIGINAL}
    other = "tests/other.py::test_other[param-2222222222222222]"
    harness_conftest._record_long_nodeids({_SHORTENED: "other-original", other: "new-original"})
    assert harness_conftest._load_long_nodeid_map() == {_SHORTENED: "other-original", other: "new-original"}


def test_unwritable_map_does_not_replace_published_selectors(_map_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed atomic publication preserves the last readable cache without failing the run."""
    harness_conftest._record_long_nodeids({_SHORTENED: _ORIGINAL})

    def refuse(_path: Path, _content: str) -> None:
        raise PermissionError("synthetic read-only cache")

    monkeypatch.setattr(harness_conftest, "write_if_changed", refuse)
    harness_conftest._record_long_nodeids({"new-shortened": "new-original"})
    assert harness_conftest._load_long_nodeid_map() == {_SHORTENED: _ORIGINAL}
