from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib.dates import parse_date
from polylogue.lib.timestamps import format_timestamp, parse_timestamp


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


@given(st.floats(min_value=0, max_value=2**31 - 1, allow_nan=False, allow_infinity=False))
def test_parse_timestamp_accepts_valid_epoch_numbers(epoch: float) -> None:
    assert parse_timestamp(epoch) is not None


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_timestamp_format_parse_roundtrip(epoch: int) -> None:
    formatted = format_timestamp(epoch)
    parsed = parse_timestamp(formatted)
    assert parsed is not None
    assert abs(parsed.timestamp() - epoch) < 2


@given(st.text())
def test_parse_timestamp_never_crashes(text: str) -> None:
    result = parse_timestamp(text)
    assert result is None or isinstance(result, datetime)
