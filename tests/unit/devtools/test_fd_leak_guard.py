"""The harness fails the test that leaks descriptors, not its successor.

Anti-vacuity: delete either ``pytest.fail`` in ``pytest_runtest_protocol``, or
raise ``FD_LEAK_ALLOWANCE`` above the leak these cases simulate, and the
corresponding case goes green while a real leak again lands on a later victim.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import tests.conftest as harness_conftest


class _Item:
    nodeid = "tests/unit/example.py::test_thing"


def _drive(counts: list[int], *, limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the hook wrapper end to end with a scripted descriptor count."""
    remaining = list(counts)
    monkeypatch.setattr(harness_conftest, "_open_fd_count", lambda: remaining.pop(0))
    monkeypatch.setattr(harness_conftest, "_fd_soft_limit", lambda: limit)
    wrapper: Iterator[Any] = harness_conftest.pytest_runtest_protocol(_Item(), None)
    next(wrapper)
    with pytest.raises(StopIteration):
        next(wrapper)


def test_a_test_within_its_working_set_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _drive([100, 100 + harness_conftest.FD_LEAK_ALLOWANCE], limit=1024, monkeypatch=monkeypatch)


def test_a_leaking_test_fails_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(Exception) as caught:
        _drive([100, 400], limit=1024, monkeypatch=monkeypatch)
    assert "leaked 300 file descriptors" in str(caught.value)
    assert _Item.nodeid in str(caught.value)


def test_a_nearly_exhausted_table_fails_before_the_next_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaustion is attributed to a running test, never to its successor."""
    with pytest.raises(Exception) as caught:
        _drive([900, 950], limit=1024, monkeypatch=monkeypatch)
    assert "file-descriptor table is 950/1024 full" in str(caught.value)


def test_the_guard_is_inert_where_proc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness_conftest, "_open_fd_count", lambda: None)
    wrapper: Iterator[Any] = harness_conftest.pytest_runtest_protocol(_Item(), None)
    next(wrapper)
    with pytest.raises(StopIteration):
        next(wrapper)
