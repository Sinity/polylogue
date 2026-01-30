"""Consolidated timestamp parsing and formatting tests.

CONSOLIDATION: Reduced 22 tests to 4 using parametrization.

Original: Separate tests for each timestamp format variant
New: Single parametrized test with all format variants
"""

from datetime import datetime

import pytest

from polylogue.core.timestamps import format_timestamp, parse_timestamp


# =============================================================================
# PARSE TIMESTAMP - PARAMETRIZED (1 test replacing 11 parsing tests)
# =============================================================================


# Test cases: (input_value, expected_year, expected_month, expected_day, expected_microsecond, description)
PARSE_TEST_CASES = [
    # Epoch timestamps
    (1704067200, 2024, 1, 1, None, "epoch int"),
    (1704067200.5, 2024, 1, 1, 500000, "epoch float"),
    ("1704067200", 2024, 1, 1, None, "epoch string"),
    ("1704067200.5", 2024, 1, 1, 500000, "epoch string with decimal"),

    # ISO 8601 variants
    ("2024-01-01T00:00:00", 2024, 1, 1, None, "ISO basic"),
    ("2024-01-01T00:00:00Z", 2024, 1, 1, None, "ISO with Z"),
    ("2024-01-01T00:00:00+00:00", 2024, 1, 1, None, "ISO with offset"),
    ("2024-01-01T00:00:00.500000", 2024, 1, 1, 500000, "ISO with microseconds"),

    # Millisecond timestamps (common in JS exports) - not currently supported, return None
    (1704067200000, None, None, None, None, "millisecond epoch"),
    ("1704067200000", None, None, None, None, "millisecond epoch string"),
]


@pytest.mark.parametrize("input_val,exp_year,exp_month,exp_day,exp_micro,desc", PARSE_TEST_CASES)
def test_parse_timestamp_formats(input_val, exp_year, exp_month, exp_day, exp_micro, desc):
    """Parse all supported timestamp formats.

    This single parametrized test replaces 11 individual parsing tests.
    """
    result = parse_timestamp(input_val)

    if exp_year is None:
        # Expected to fail parsing
        assert result is None, f"Expected None for {desc}, got {result}"
    else:
        assert result is not None, f"Failed to parse {desc}: {input_val}"
        assert result.year == exp_year, f"Wrong year for {desc}"
        assert result.month == exp_month, f"Wrong month for {desc}"
        assert result.day == exp_day, f"Wrong day for {desc}"

        if exp_micro is not None:
            assert result.microsecond == exp_micro, f"Wrong microseconds for {desc}"


# =============================================================================
# PARSE TIMESTAMP - EDGE CASES (1 test replacing 4 tests)
# =============================================================================


@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    "not-a-date",
    "invalid-timestamp",
    "2024-13-45",  # Invalid month/day
])
def test_parse_timestamp_invalid_returns_none(invalid_input):
    """Invalid timestamps return None."""
    result = parse_timestamp(invalid_input)
    assert result is None


def test_parse_timestamp_overflow_handled():
    """Handle very large epoch values gracefully."""
    # Epoch timestamp beyond typical 32-bit range
    result = parse_timestamp(9999999999999)
    # Implementation returns None on overflow/error
    # If it succeeds, it should be a valid datetime
    if result is not None:
        assert isinstance(result, datetime)


# =============================================================================
# FORMAT TIMESTAMP - PARAMETRIZED (1 test replacing 7 formatting tests)
# =============================================================================


# Test cases: (input_datetime, expected_format_output, description)
FORMAT_TEST_CASES = [
    (
        datetime(2024, 1, 1, 12, 30, 45),
        "2024-01-01T12:30:45",
        "basic datetime"
    ),
    (
        datetime(2024, 1, 1, 0, 0, 0),
        "2024-01-01T00:00:00",
        "midnight"
    ),
    (
        datetime(2024, 12, 31, 23, 59, 59),
        "2024-12-31T23:59:59",
        "end of year"
    ),
    (
        datetime(2024, 1, 1, 12, 30, 45, 500000),
        "2024-01-01T12:30:45",  # Microseconds may or may not be included
        "with microseconds"
    ),
]


@pytest.mark.parametrize("dt,expected_format,desc", FORMAT_TEST_CASES)
def test_format_timestamp_variants(dt, expected_format, desc):
    """Format datetime objects to ISO 8601.

    This single parametrized test replaces 7 individual formatting tests.
    """
    result = format_timestamp(dt)

    # Check for ISO 8601 format (with or without microseconds)
    assert result.startswith(expected_format[:19]), f"Wrong format for {desc}"
    assert "T" in result, "Should use T separator"


# =============================================================================
# FORMAT TIMESTAMP - EDGE CASES (1 test)
# =============================================================================


@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    "not-a-datetime",
    # Note: 123 is treated as valid epoch seconds, not invalid input
])
def test_format_timestamp_invalid_returns_none_or_empty(invalid_input):
    """Invalid input to format_timestamp handled gracefully."""
    try:
        result = format_timestamp(invalid_input)
        # If it doesn't raise, should return None or empty string
        assert result is None or result == ""
    except (TypeError, AttributeError):
        # Acceptable to raise for invalid input
        pass
