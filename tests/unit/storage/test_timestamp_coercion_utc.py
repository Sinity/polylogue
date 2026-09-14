"""Naive ISO timestamps coerce as UTC, not as the host's local zone.

Anti-vacuity: each test runs under a forced non-UTC local zone (POSIX ``TZ``,
so no tzdata lookup is needed) and asserts the naive spelling of an instant
lands on the same epoch-ms as its explicitly-UTC spelling. Delete either
``tzinfo is None -> replace(tzinfo=UTC)`` guard and both tests go red by
exactly the forced offset (5 h = 18_000_000 ms), on any host including a
UTC-configured CI runner.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest

from polylogue.sources.hook_producer import timestamp_ms
from polylogue.storage.sqlite.queries.artifacts import _iso_to_ms

# 2026-01-01T12:00:00Z and 2026-07-01T12:00:00Z, so a host zone with DST cannot
# make one spelling pass while the other fails.
_WINTER_UTC_MS = 1767268800000
_SUMMER_UTC_MS = 1782907200000


@pytest.fixture
def local_zone_utc_plus_5() -> Iterator[None]:
    """Force local time to UTC+5 for the duration of one test."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "XXX-5"
    time.tzset()
    try:
        assert time.timezone == -18000, "POSIX TZ did not take effect"
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.parametrize(
    ("naive", "expected_ms"),
    [
        ("2026-01-01T12:00:00", _WINTER_UTC_MS),
        ("2026-07-01T12:00:00", _SUMMER_UTC_MS),
    ],
)
@pytest.mark.parametrize("coerce", [_iso_to_ms, timestamp_ms], ids=["artifacts", "hook_producer"])
def test_naive_iso_is_utc_not_host_local(
    local_zone_utc_plus_5: None,
    coerce: object,
    naive: str,
    expected_ms: int,
) -> None:
    assert coerce(naive) == expected_ms  # type: ignore[operator]


@pytest.mark.parametrize("coerce", [_iso_to_ms, timestamp_ms], ids=["artifacts", "hook_producer"])
def test_naive_and_aware_spellings_agree(local_zone_utc_plus_5: None, coerce: object) -> None:
    naive = coerce("2026-01-01T12:00:00")  # type: ignore[operator]
    for aware in ("2026-01-01T12:00:00Z", "2026-01-01T12:00:00+00:00"):
        assert coerce(aware) == naive  # type: ignore[operator]


def test_all_digit_input_is_still_passed_through_as_milliseconds() -> None:
    """The digit branch is unchanged by the UTC fix (its unit policy is polylogue-z3sv)."""
    assert _iso_to_ms("1767268800000") == _WINTER_UTC_MS
