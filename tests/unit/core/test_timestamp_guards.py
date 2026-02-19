"""Regression tests for timestamp parsing and schema enum sanitization.

Covers bugs from:
- f33ef29: year strings misclassified as epoch seconds
- f0a5e4c: naive datetime leakage (should always return UTC-aware)
- d770e29: _is_safe_enum_value PII prevention
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.lib.timestamps import format_timestamp, parse_timestamp
from polylogue.schemas.schema_inference import _is_safe_enum_value


# =============================================================================
# parse_timestamp: epoch vs year string disambiguation (f33ef29)
# =============================================================================

class TestTimestampYearGuard:
    """Year-like strings must NOT be treated as epoch seconds."""

    @pytest.mark.parametrize("value", ["2024", "2025", "2026", "1999", "3000"])
    def test_year_strings_not_treated_as_epoch(self, value):
        """Values below 86400 should not be parsed as epoch (would give 1970 dates)."""
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
# parse_timestamp: UTC awareness (f0a5e4c)
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
        """Naive ISO strings should be assumed UTC, not left as naive."""
        result = parse_timestamp("2024-01-15T10:30:00")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_iso_with_offset_preserves_tz(self):
        result = parse_timestamp("2024-01-15T10:30:00+05:00")
        assert result is not None
        assert result.tzinfo is not None


# =============================================================================
# parse_timestamp: edge cases and None handling
# =============================================================================

class TestTimestampEdgeCases:
    def test_none_returns_none(self):
        assert parse_timestamp(None) is None

    def test_empty_string_returns_none(self):
        assert parse_timestamp("") is None

    def test_garbage_string_returns_none(self):
        assert parse_timestamp("not-a-timestamp") is None

    def test_negative_epoch_returns_none(self):
        # Negative epochs are technically valid but not in our digit check
        assert parse_timestamp("-1000") is None

    def test_extremely_large_epoch_does_not_crash(self):
        """OverflowError for huge values should be caught."""
        result = parse_timestamp("99999999999999999")
        # Should return None or a datetime, but NOT raise
        assert result is None or isinstance(result, datetime)

    def test_zero_as_int(self):
        result = parse_timestamp(0)
        assert result is not None
        assert result.year == 1970

    def test_zero_as_string_not_epoch(self):
        """String "0" is below 86400, should not be parsed as epoch."""
        result = parse_timestamp("0")
        assert result is None


# =============================================================================
# format_timestamp: round-trip consistency
# =============================================================================

class TestFormatTimestamp:
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

class TestSafeEnumValue:
    """Schema annotations must not leak personal data."""

    @pytest.mark.parametrize("value", [
        "user", "assistant", "system", "tool",
        "text", "code", "STOP", "MAX_TOKENS",
        "finished_successfully",
        "application/json",
    ])
    def test_technical_identifiers_allowed(self, value):
        assert _is_safe_enum_value(value) is True

    @pytest.mark.parametrize("value", [
        "john.doe@gmail.com",
        "user@company.org",
    ])
    def test_emails_blocked(self, value):
        assert _is_safe_enum_value(value) is False

    @pytest.mark.parametrize("value", [
        "https://example.com/path",
        "http://internal.server:8080",
    ])
    def test_urls_blocked(self, value):
        assert _is_safe_enum_value(value) is False

    @pytest.mark.parametrize("value", [
        "/home/user/documents/file.txt",
        "/etc/passwd",
    ])
    def test_absolute_paths_blocked(self, value):
        assert _is_safe_enum_value(value) is False

    @pytest.mark.parametrize("value", [
        "photo.png", "document.json", "archive.zip",
        "image.jpg", "script.py",
    ])
    def test_file_extensions_blocked(self, value):
        assert _is_safe_enum_value(value) is False

    @pytest.mark.parametrize("value", [
        "2024-01-15T10:30:00",
        "2024-01-15T10:30:00Z",
    ])
    def test_timestamps_blocked(self, value):
        assert _is_safe_enum_value(value) is False

    def test_empty_string_blocked(self):
        assert _is_safe_enum_value("") is False

    def test_non_ascii_blocked(self):
        assert _is_safe_enum_value("café") is False

    def test_spaces_blocked(self):
        assert _is_safe_enum_value("hello world") is False

    def test_domains_blocked(self):
        assert _is_safe_enum_value("example.com") is False

    @pytest.mark.parametrize("value", [
        "+1234567890",
        "+base64data",
    ])
    def test_plus_prefix_blocked(self, value):
        assert _is_safe_enum_value(value) is False
