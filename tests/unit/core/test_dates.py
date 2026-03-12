from __future__ import annotations

from datetime import timezone

from polylogue.lib.dates import parse_date


def test_parse_date_returns_utc_aware_datetime_for_iso_dates() -> None:
    result = parse_date("2024-01-15")
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset() == timezone.utc.utcoffset(result)


def test_parse_date_accepts_relative_dates() -> None:
    result = parse_date("yesterday")
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset() == timezone.utc.utcoffset(result)


def test_parse_date_returns_none_for_invalid_input() -> None:
    assert parse_date("not a date at all!!!!") is None
