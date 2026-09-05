"""Anti-vacuity: without this helper the pin silently degrades.

Deleting the `TZDIR` derivation in `pinned_local_timezone` turns
`TZ=America/Los_Angeles` into a zero-offset POSIX zone on a host with no
`/usr/share/zoneinfo`, and `test_pin_holds_without_an_ambient_zoneinfo_directory`
goes red.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.infra.local_timezone import pinned_local_timezone


def _rendered(monkeypatch: pytest.MonkeyPatch) -> str:
    with pinned_local_timezone(monkeypatch, "America/Los_Angeles"):
        return datetime(2026, 7, 1, 19, 48, tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def test_pin_holds_without_an_ambient_zoneinfo_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TZDIR", raising=False)

    assert _rendered(monkeypatch) == "2026-07-01 12:48 PDT"


def test_pin_restores_the_previous_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "UTC")

    _rendered(monkeypatch)

    assert datetime(2026, 7, 1, 19, 48, tzinfo=timezone.utc).astimezone().utcoffset() == timedelta(0)
