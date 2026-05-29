"""Tests for payload_coercion — especially timezone-aware datetime handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from polylogue.core.payload_coercion import optional_datetime


class TestOptionalDatetime:
    def test_none_returns_none(self) -> None:
        assert optional_datetime(None) is None

    def test_aware_datetime_converted_to_utc(self) -> None:
        cet = timezone(timedelta(hours=2))
        dt = datetime(2026, 5, 28, 14, 0, 0, tzinfo=cet)
        result = optional_datetime(dt)
        assert result is not None
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)
        assert result.hour == 12  # 14 CET → 12 UTC

    def test_naive_datetime_assumed_utc(self) -> None:
        dt = datetime(2026, 5, 28, 12, 0, 0)
        result = optional_datetime(dt)
        assert result is not None
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)
        assert result.hour == 12  # unchanged — assumed already UTC

    def test_iso_string_with_z(self) -> None:
        result = optional_datetime("2026-05-28T12:00:00Z")
        assert result is not None
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)
        assert result.hour == 12

    def test_iso_string_with_offset(self) -> None:
        result = optional_datetime("2026-05-28T14:00:00+02:00")
        assert result is not None
        assert result.tzinfo is not None
        assert result.hour == 12  # converted to UTC

    def test_iso_string_without_tz_assumed_utc(self) -> None:
        result = optional_datetime("2026-05-28T12:00:00")
        assert result is not None
        assert result.tzinfo is not None
        assert result.hour == 12  # assumed UTC

    def test_timestamp_consistent_regardless_of_input_tz(self) -> None:
        """A datetime with timezone should produce the same .timestamp() as UTC."""
        cet = timezone(timedelta(hours=2))
        aware_dt = datetime(2026, 5, 28, 14, 0, 0, tzinfo=cet)
        naive_dt = datetime(2026, 5, 28, 12, 0, 0)
        iso_str = "2026-05-28T12:00:00Z"

        a = optional_datetime(aware_dt)
        b = optional_datetime(naive_dt)
        c = optional_datetime(iso_str)

        assert a is not None and b is not None and c is not None
        assert a.timestamp() == pytest.approx(b.timestamp())
        assert a.timestamp() == pytest.approx(c.timestamp())

    def test_already_utc_datetime_preserved(self) -> None:
        dt = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        result = optional_datetime(dt)
        assert result is not None
        assert result == dt

    def test_date_only_iso_string(self) -> None:
        result = optional_datetime("2026-05-28")
        assert result is not None
        assert result.tzinfo is not None
        assert result.hour == 0
