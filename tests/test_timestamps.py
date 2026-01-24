"""Tests for polylogue.core.timestamps module."""

from datetime import datetime

import pytest

from polylogue.core.timestamps import format_timestamp, parse_timestamp


class TestParseTimestamp:
    """Test parse_timestamp function."""

    def test_parse_timestamp_epoch_int(self):
        """Parse integer epoch timestamp."""
        # 2024-01-01 00:00:00 UTC
        result = parse_timestamp(1704067200)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_timestamp_epoch_float(self):
        """Parse float epoch timestamp."""
        # 2024-01-01 00:00:00.5 UTC
        result = parse_timestamp(1704067200.5)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.microsecond == 500000

    def test_parse_timestamp_epoch_string(self):
        """Parse epoch timestamp as string."""
        result = parse_timestamp("1704067200")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_timestamp_epoch_string_with_decimal(self):
        """Parse epoch timestamp as string with decimal."""
        result = parse_timestamp("1704067200.5")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.microsecond == 500000

    def test_parse_timestamp_iso8601_basic(self):
        """Parse basic ISO 8601 timestamp."""
        result = parse_timestamp("2024-01-01T00:00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_parse_timestamp_iso8601_with_z(self):
        """Parse ISO 8601 timestamp with Z suffix."""
        result = parse_timestamp("2024-01-01T00:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_parse_timestamp_iso8601_with_offset(self):
        """Parse ISO 8601 timestamp with timezone offset."""
        result = parse_timestamp("2024-01-01T00:00:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_timestamp_none_input(self):
        """Return None for None input."""
        result = parse_timestamp(None)
        assert result is None

    def test_parse_timestamp_empty_string(self):
        """Return None for empty string."""
        result = parse_timestamp("")
        assert result is None

    def test_parse_timestamp_invalid_string(self):
        """Return None for invalid string."""
        result = parse_timestamp("not-a-date")
        assert result is None

    def test_parse_timestamp_overflow(self):
        """Handle very large epoch values gracefully."""
        # Epoch timestamp beyond typical 32-bit range
        # Some systems may reject this with OSError/OverflowError
        result = parse_timestamp(9999999999999)
        # Implementation returns None on overflow/error
        # If it succeeds, it should be a valid datetime
        if result is not None:
            assert isinstance(result, datetime)

    def test_parse_timestamp_negative_epoch(self):
        """Handle pre-1970 timestamps."""
        # -1 second from epoch - result depends on local timezone
        result = parse_timestamp(-1)
        # Some systems support negative epochs, others don't
        if result is not None:
            assert isinstance(result, datetime)
            # Could be 1969 (UTC or west of UTC) or 1970 (east of UTC)
            assert result.year in (1969, 1970)

    def test_parse_timestamp_zero(self):
        """Handle epoch 0 (1970-01-01 00:00:00 UTC)."""
        result = parse_timestamp(0)
        assert result is not None
        assert result.year == 1970
        assert result.month == 1
        assert result.day == 1

    def test_parse_timestamp_string_with_letters(self):
        """Return None for string with letters that isn't ISO format."""
        result = parse_timestamp("12abc34")
        assert result is None

    def test_parse_timestamp_iso8601_with_microseconds(self):
        """Parse ISO 8601 timestamp with microseconds."""
        result = parse_timestamp("2024-01-01T12:30:45.123456")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        assert result.microsecond == 123456


class TestFormatTimestamp:
    """Test format_timestamp function."""

    def test_format_timestamp_datetime(self):
        """Format datetime object to ISO string with UTC timezone."""
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = format_timestamp(dt)
        # Naive datetimes are treated as UTC
        assert result == "2024-01-01T12:30:45+00:00"

    def test_format_timestamp_epoch_int(self):
        """Format integer epoch to ISO string."""
        # 2024-01-01 00:00:00 UTC (local time may vary)
        result = format_timestamp(1704067200)
        assert isinstance(result, str)
        assert "T" in result
        # Check it contains valid year
        assert "202" in result

    def test_format_timestamp_epoch_float(self):
        """Format float epoch to ISO string."""
        result = format_timestamp(1704067200.5)
        assert isinstance(result, str)
        assert "T" in result
        assert "202" in result

    def test_format_timestamp_includes_seconds(self):
        """Verify timespec='seconds' is used (no microseconds)."""
        dt = datetime(2024, 1, 1, 12, 30, 45, 123456)
        result = format_timestamp(dt)
        # Should not include microseconds
        assert ".123456" not in result
        assert result == "2024-01-01T12:30:45+00:00"

    def test_format_timestamp_datetime_with_timezone(self):
        """Format datetime with timezone info."""
        # Create timezone-aware datetime
        from datetime import timezone

        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        # Should include timezone offset
        assert "+" in result or "00:00" in result or result.endswith("Z")

    def test_format_timestamp_epoch_zero(self):
        """Format epoch 0."""
        result = format_timestamp(0)
        assert isinstance(result, str)
        assert "1970" in result


class TestRoundtrip:
    """Test round-trip conversions."""

    def test_roundtrip_epoch_to_string(self):
        """Parse and format epoch preserves value."""
        epoch = 1704067200
        parsed = parse_timestamp(epoch)
        formatted = format_timestamp(parsed)
        reparsed = parse_timestamp(formatted)

        # Compare timestamps (may differ slightly due to precision)
        assert parsed is not None
        assert reparsed is not None
        # Should be within a second of each other
        diff = abs((parsed - reparsed).total_seconds())
        assert diff < 1.0

    def test_roundtrip_iso_string(self):
        """Parse and format ISO string preserves value."""
        iso_string = "2024-01-01T12:30:45"
        parsed = parse_timestamp(iso_string)
        formatted = format_timestamp(parsed)

        assert parsed is not None
        # Output now includes UTC timezone suffix
        assert formatted == "2024-01-01T12:30:45+00:00"
        # Verify the datetime value is preserved
        reparsed = parse_timestamp(formatted)
        assert reparsed is not None
        assert abs((parsed - reparsed).total_seconds()) < 1.0

    def test_roundtrip_datetime(self):
        """Format and parse datetime preserves value."""
        dt = datetime(2024, 6, 15, 14, 30, 0)
        formatted = format_timestamp(dt)
        parsed = parse_timestamp(formatted)

        assert parsed is not None
        assert parsed.year == dt.year
        assert parsed.month == dt.month
        assert parsed.day == dt.day
        assert parsed.hour == dt.hour
        assert parsed.minute == dt.minute
        assert parsed.second == dt.second
