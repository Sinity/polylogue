"""The harness fails the test that leaks descriptors, not its successor.

Anti-vacuity: delete either ``raise AssertionError`` in
``check_descriptor_balance``, or raise ``FD_LEAK_ALLOWANCE`` above the leak
these cases simulate, and the corresponding case goes green while a real leak
again lands on a later victim.

The attribution half was verified against a live pytest session: a test opening
30 unclosed connections is reported as ``ERROR ...::test_leaks`` and the test
that runs after it passes.
"""

from __future__ import annotations

import pytest

import tests.conftest as harness_conftest


class _Item:
    nodeid = "tests/unit/example.py::test_thing"

    def __init__(self, before: int | None) -> None:
        self.stash: dict[object, int] = {}
        if before is not None:
            self.stash[harness_conftest._FD_BEFORE] = before


def _check(before: int | None, after: int | None, *, limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness_conftest, "_open_fd_count", lambda: after)
    monkeypatch.setattr(harness_conftest, "_fd_soft_limit", lambda: limit)
    # The real stash is a typed mapping; a plain dict answers the same lookup.
    harness_conftest.check_descriptor_balance(_Item(before))  # type: ignore[arg-type]


def test_a_test_within_its_working_set_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _check(100, 100 + harness_conftest.FD_LEAK_ALLOWANCE, limit=1024, monkeypatch=monkeypatch)


def test_a_leaking_test_fails_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError) as caught:
        _check(100, 400, limit=1024, monkeypatch=monkeypatch)
    assert "leaked 300 file descriptors" in str(caught.value)
    assert _Item.nodeid in str(caught.value)


def test_a_nearly_exhausted_table_fails_before_the_next_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaustion is attributed to a running test, never to its successor."""
    with pytest.raises(AssertionError) as caught:
        _check(900, 950, limit=1024, monkeypatch=monkeypatch)
    assert "file-descriptor table is 950/1024 full" in str(caught.value)


def test_the_guard_is_inert_where_proc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _check(100, None, limit=1024, monkeypatch=monkeypatch)
    _check(None, 900, limit=1024, monkeypatch=monkeypatch)
