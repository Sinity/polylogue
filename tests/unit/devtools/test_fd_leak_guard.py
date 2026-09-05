"""Descriptor exhaustion is reported against the tests that caused it.

The 2026-09-05 cascade retained about three descriptors per test across
hundreds of tests, so no single-test bound would have caught it; eleven tests
then died with ``OSError: [Errno 24]`` for reasons that were not their own.
The guard therefore has two parts: an outright per-test leak fails that test,
and reaching exhaustion names the largest cumulative retainers.

Anti-vacuity:
- delete the ``retained > FD_LEAK_ALLOWANCE`` branch and
  ``test_an_outright_leak_fails_that_test`` goes red;
- delete the ``FD_EXHAUSTION_FRACTION`` branch and
  ``test_exhaustion_names_the_accumulating_retainers`` goes red;
- drop ``_retainer_report`` from the exhaustion message and that same case goes
  red, because the report is the only thing that points past the victim.

The attribution half was verified against a live pytest session: a test opening
30 unclosed connections is reported as ``ERROR ...::test_leaks`` and the test
that runs after it passes.
"""

from __future__ import annotations

import pytest

import tests.conftest as harness_conftest


class _Item:
    def __init__(self, nodeid: str, before: int | None) -> None:
        self.nodeid = nodeid
        self.stash: dict[object, int] = {}
        if before is not None:
            self.stash[harness_conftest._FD_BEFORE] = before


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retainer ledger is process-global; never let a case see another's."""
    monkeypatch.setattr(harness_conftest, "FD_RETAINED", {})


def _check(
    before: int | None,
    after: int | None,
    *,
    limit: int,
    monkeypatch: pytest.MonkeyPatch,
    nodeid: str = "tests/unit/example.py::test_thing",
) -> None:
    monkeypatch.setattr(harness_conftest, "_open_fd_count", lambda: after)
    monkeypatch.setattr(harness_conftest, "_fd_soft_limit", lambda: limit)
    # The real stash is a typed mapping; a plain dict answers the same lookup.
    harness_conftest.check_descriptor_balance(_Item(nodeid, before))  # type: ignore[arg-type]


def test_a_test_within_its_working_set_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _check(100, 100 + harness_conftest.FD_LEAK_ALLOWANCE, limit=1024, monkeypatch=monkeypatch)


def test_an_outright_leak_fails_that_test(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError) as caught:
        _check(100, 400, limit=1024, monkeypatch=monkeypatch)
    assert "leaked 300 file descriptors" in str(caught.value)
    assert "tests/unit/example.py::test_thing" in str(caught.value)


def test_a_small_per_test_retention_is_recorded_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three descriptors is under any workable bound, and is the real leak shape."""
    _check(100, 103, limit=1024, monkeypatch=monkeypatch, nodeid="a::test_one")
    assert harness_conftest.FD_RETAINED == {"a::test_one": 3}


def test_exhaustion_names_the_accumulating_retainers(monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(3):
        _check(100, 103, limit=1024, monkeypatch=monkeypatch, nodeid=f"a::test_{index}")
    with pytest.raises(AssertionError) as caught:
        _check(900, 950, limit=1024, monkeypatch=monkeypatch, nodeid="victim::test_unlucky")
    message = str(caught.value)
    assert "file-descriptor table is 950/1024 full" in message
    # The victim is named as the point of failure, and the accumulators as the cause.
    assert "victim::test_unlucky" in message
    assert "largest cumulative retainers" in message
    assert "a::test_0" in message and "a::test_2" in message


def test_the_guard_is_inert_where_proc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _check(100, None, limit=1024, monkeypatch=monkeypatch)
    _check(None, 900, limit=1024, monkeypatch=monkeypatch)
    assert harness_conftest.FD_RETAINED == {}
