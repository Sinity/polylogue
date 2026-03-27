"""Property-based and regression tests for timestamp parsing and schema enum sanitization.

Consolidates:
- test_timestamp_guards.py (original regression tests)
- Property tests for parse_timestamp robustness

Covers bugs from:
- f33ef29: year strings misclassified as epoch seconds
- f0a5e4c: naive datetime leakage (should always return UTC-aware)
- d770e29: _is_safe_enum_value PII prevention
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from polylogue.lib.dates import parse_date
from polylogue.lib.timestamps import format_timestamp, parse_timestamp
from polylogue.schemas.schema_inference import _is_safe_enum_value
from tests.infra.tables import FORMAT_TIMESTAMP_TABLE, PARSE_TIMESTAMP_FORMAT_TABLE

# =============================================================================
# Property: parse_timestamp never crashes on any input
# =============================================================================


@given(st.text())
@example("")
@example("99999999999999999")
@example("<script>alert(1)</script>")
def test_parse_timestamp_never_crashes_on_text(value: str):
    """parse_timestamp never raises on any text input."""
    result = parse_timestamp(value)
    assert result is None or isinstance(result, datetime)


@given(st.one_of(st.none(), st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)))
@example(0)
@example(-1000)
def test_parse_timestamp_never_crashes_on_mixed_types(value):
    """parse_timestamp never raises on any type of input."""
    result = parse_timestamp(value)
    assert result is None or isinstance(result, datetime)


# =============================================================================
# Property: Result is always UTC-aware or None
# =============================================================================


@given(st.one_of(
    st.integers(min_value=86400, max_value=2_000_000_000),
    st.floats(min_value=86400, max_value=2_000_000_000, allow_nan=False),
    st.from_regex(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fullmatch=True),
))
def test_parse_timestamp_result_always_utc_aware(value):
    """When parse_timestamp returns a datetime, it is always UTC-aware."""
    result = parse_timestamp(value)
    if result is not None:
        assert result.tzinfo is not None


# =============================================================================
# Regression: epoch vs year string disambiguation (f33ef29)
# =============================================================================


class TestTimestampYearGuard:
    """Year-like strings must NOT be treated as epoch seconds."""

    @pytest.mark.parametrize("value", ["2024", "2025", "2026", "1999", "3000"])
    def test_year_strings_not_treated_as_epoch(self, value):
        result = parse_timestamp(value)
        assert result is None

    def test_boundary_86399_not_treated_as_epoch(self):
        result = parse_timestamp("86399")
        assert result is None

    def test_boundary_86400_is_valid_epoch(self):
        result = parse_timestamp("86400")
        assert result is not None
        assert result.year == 1970
        assert result.month == 1
        assert result.day == 2

    def test_typical_epoch_string(self):
        result = parse_timestamp("1700000000")
        assert result is not None
        assert result.year == 2023

    def test_epoch_with_decimal(self):
        result = parse_timestamp("1700000000.123")
        assert result is not None


# =============================================================================
# Regression: UTC awareness (f0a5e4c)
# =============================================================================


class TestTimestampUTCAwareness:
    """All returned datetimes must be UTC-aware (tzinfo != None)."""

    def test_int_epoch_is_utc_aware(self):
        result = parse_timestamp(1700000000)
        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_float_epoch_is_utc_aware(self):
        result = parse_timestamp(1700000000.5)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_string_epoch_is_utc_aware(self):
        result = parse_timestamp("1700000000")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_iso_with_z_is_utc_aware(self):
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_naive_iso_gets_utc(self):
        result = parse_timestamp("2024-01-15T10:30:00")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_iso_with_offset_preserves_tz(self):
        result = parse_timestamp("2024-01-15T10:30:00+05:00")
        assert result is not None
        assert result.tzinfo is not None


# =============================================================================
# Regression: edge cases and None handling
# =============================================================================


class TestTimestampEdgeCases:
    def test_none_returns_none(self):
        assert parse_timestamp(None) is None

    def test_empty_string_returns_none(self):
        assert parse_timestamp("") is None

    def test_garbage_string_returns_none(self):
        assert parse_timestamp("not-a-timestamp") is None

    def test_negative_epoch_returns_none(self):
        assert parse_timestamp("-1000") is None

    def test_extremely_large_epoch_does_not_crash(self):
        result = parse_timestamp("99999999999999999")
        assert result is None or isinstance(result, datetime)

    def test_zero_as_int(self):
        result = parse_timestamp(0)
        assert result is not None
        assert result.year == 1970

    def test_zero_as_string_not_epoch(self):
        result = parse_timestamp("0")
        assert result is None


# =============================================================================
# Canonical format table (parametrized regression documentation)
# =============================================================================


class TestParseTimestampFormats:
    """Comprehensive format coverage using canonical table."""

    @pytest.mark.parametrize(
        "input_val,exp_year,exp_month,exp_day,exp_micro,desc",
        PARSE_TIMESTAMP_FORMAT_TABLE,
    )
    def test_parse_timestamp_table(self, input_val, exp_year, exp_month, exp_day, exp_micro, desc):
        result = parse_timestamp(input_val)
        if exp_year is None:
            assert result is None, f"Expected None for {desc!r}, got {result}"
        else:
            assert result is not None, f"Failed to parse {desc!r}: {input_val!r}"
            assert result.year == exp_year, f"Wrong year for {desc!r}"
            assert result.month == exp_month, f"Wrong month for {desc!r}"
            assert result.day == exp_day, f"Wrong day for {desc!r}"
            if exp_micro is not None:
                assert result.microsecond == exp_micro, f"Wrong microseconds for {desc!r}"


# =============================================================================
# format_timestamp: round-trip consistency
# =============================================================================


class TestFormatTimestamp:
    @pytest.mark.parametrize("input_val,expected_prefix,desc", FORMAT_TIMESTAMP_TABLE)
    def test_format_timestamp_table(self, input_val, expected_prefix, desc):
        result = format_timestamp(input_val)
        assert result is not None
        assert expected_prefix in result, f"Expected {expected_prefix!r} in result for {desc!r}"

    def test_epoch_roundtrip(self):
        formatted = format_timestamp(1700000000)
        assert "+00:00" in formatted
        assert "2023" in formatted

    def test_datetime_roundtrip(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        formatted = format_timestamp(dt)
        assert "2024-06-15" in formatted
        assert "+00:00" in formatted

    def test_naive_datetime_treated_as_utc(self):
        dt = datetime(2024, 6, 15, 12, 0, 0)
        formatted = format_timestamp(dt)
        assert "+00:00" in formatted


# =============================================================================
# _is_safe_enum_value: PII prevention (d770e29)
# =============================================================================


SAFE_ENUM_ALLOWED = [
    ("user", "role"),
    ("assistant", "role"),
    ("system", "role"),
    ("tool", "role"),
    ("text", "content_type"),
    ("code", "content_type"),
    ("STOP", "stop_reason"),
    ("MAX_TOKENS", "stop_reason"),
    ("finished_successfully", "stop_reason"),
    ("application/json", "mime_type"),
]

SAFE_ENUM_BLOCKED = [
    ("john.doe@gmail.com", "email"),
    ("user@company.org", "email"),
    ("https://example.com/path", "url"),
    ("http://internal.server:8080", "url"),
    ("/home/user/documents/file.txt", "absolute_path"),
    ("/etc/passwd", "absolute_path"),
    ("photo.png", "file_extension"),
    ("document.json", "file_extension"),
    ("archive.zip", "file_extension"),
    ("image.jpg", "file_extension"),
    ("script.py", "file_extension"),
    ("2024-01-15T10:30:00", "timestamp"),
    ("2024-01-15T10:30:00Z", "timestamp"),
    ("", "empty"),
    ("caf\u00e9", "non_ascii"),
    ("hello world", "spaces"),
    ("example.com", "domain"),
    ("+1234567890", "plus_prefix"),
    ("+base64data", "plus_prefix"),
]


@pytest.mark.parametrize("value,category", SAFE_ENUM_ALLOWED)
def test_safe_enum_value_allowed(value, category):
    assert _is_safe_enum_value(value) is True, f"Should allow {category}: {value!r}"


@pytest.mark.parametrize("value,category", SAFE_ENUM_BLOCKED)
def test_safe_enum_value_blocked(value, category):
    assert _is_safe_enum_value(value) is False, f"Should block {category}: {value!r}"


# =============================================================================
# Merged from test_dates.py (2024-03-15)
# =============================================================================


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
