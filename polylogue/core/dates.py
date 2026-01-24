"""Date parsing utilities with natural language support."""

from __future__ import annotations

from datetime import datetime, timezone

import dateparser


def parse_date(date_str: str) -> datetime | None:
    """Parse a date string in various formats.

    Supports:
    - ISO format: "2024-01-15", "2024-01-15T10:30:00"
    - Natural language: "last week", "2 days ago", "yesterday"
    - Relative: "last month", "this year", "3 weeks ago"
    - Specific: "January 2024", "Jun 15", "2024-Q1"

    Args:
        date_str: Date string to parse

    Returns:
        datetime object (UTC-aware) or None if parsing fails

    Examples:
        >>> parse_date("2024-01-15")
        datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        >>> parse_date("last week")  # doctest: +SKIP
        datetime(2026, 1, 16, ..., tzinfo=timezone.utc)
        >>> parse_date("yesterday")  # doctest: +SKIP
        datetime(2026, 1, 22, ..., tzinfo=timezone.utc)
    """
    settings = {
        "PREFER_DATES_FROM": "past",  # Default to past dates for "last week"
        "RETURN_AS_TIMEZONE_AWARE": True,  # Return UTC-aware for comparison with timestamps
        "TIMEZONE": "UTC",  # Use UTC as default timezone
        "RELATIVE_BASE": datetime.now(tz=timezone.utc),  # Base for relative dates
    }

    parsed = dateparser.parse(date_str, settings=settings)
    if parsed is not None and parsed.tzinfo is None:
        # Ensure result is always UTC-aware
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_date_iso(dt: datetime) -> str:
    """Format datetime as ISO string compatible with storage layer.

    Args:
        dt: datetime to format

    Returns:
        ISO 8601 formatted string (YYYY-MM-DD HH:MM:SS)
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")
