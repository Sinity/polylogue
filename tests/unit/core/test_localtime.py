from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from polylogue.core.localtime import format_local_datetime


@pytest.fixture
def fixed_local_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", previous)
        time.tzset()


def test_format_local_datetime_converts_utc_and_marks_zone(fixed_local_timezone: None) -> None:
    rendered = format_local_datetime(datetime(2026, 7, 1, 19, 48, tzinfo=timezone.utc))

    assert rendered == "2026-07-01 12:48 PDT"


def test_format_local_datetime_supports_localized_date_boundaries(fixed_local_timezone: None) -> None:
    rendered = format_local_datetime(datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc), "%Y-%m-%d")

    assert rendered == "2026-06-30"
