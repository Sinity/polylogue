"""Pin a named local timezone for tests that render host-local times.

`datetime.astimezone()` resolves the local zone through libc, which finds a
named zone only when a zoneinfo database is on disk at a location it knows.
This host ships no `/usr/share/zoneinfo`, so libc resolves names only via
`TZDIR`, and an environment that does not forward `TZDIR` silently degrades
`TZ=America/Los_Angeles` to a POSIX zero-offset zone named `America`. Deriving
`TZDIR` from `zoneinfo.TZPATH` keeps the pin independent of ambient state.
"""

from __future__ import annotations

import os
import time
import zoneinfo
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _zoneinfo_directory(zone: str) -> str | None:
    for candidate in zoneinfo.TZPATH:
        if (Path(candidate) / zone).exists():
            return candidate
    return None


@contextmanager
def pinned_local_timezone(monkeypatch: pytest.MonkeyPatch, zone: str) -> Iterator[None]:
    """Make libc-local time render `zone`, restoring the prior zone after."""
    previous_tz = os.environ.get("TZ")
    previous_tzdir = os.environ.get("TZDIR")
    directory = _zoneinfo_directory(zone)
    if directory is not None:
        monkeypatch.setenv("TZDIR", directory)
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    probe = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    expected = probe.astimezone(zoneinfo.ZoneInfo(zone)).utcoffset()
    if probe.astimezone().utcoffset() != expected:
        raise AssertionError(
            f"could not pin local timezone to {zone!r}: libc reported {time.tzname!r}; "
            "no zoneinfo database was reachable"
        )
    try:
        yield
    finally:
        for key, value in (("TZ", previous_tz), ("TZDIR", previous_tzdir)):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        time.tzset()


__all__ = ["pinned_local_timezone"]
