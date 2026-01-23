"""Date parsing utilities with natural language support."""

from __future__ import annotations

from datetime import datetime

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
        datetime object or None if parsing fails

    Examples:
        >>> parse_date("2024-01-15")
        datetime(2024, 1, 15, 0, 0)
        >>> parse_date("last week")  # doctest: +SKIP
        datetime(2026, 1, 16, ...)
        >>> parse_date("yesterday")  # doctest: +SKIP
        datetime(2026, 1, 22, ...)
    """
    settings = {
        "PREFER_DATES_FROM": "past",  # Default to past dates for "last week"
        "RETURN_AS_TIMEZONE_AWARE": False,  # Match existing timestamp handling
        "RELATIVE_BASE": datetime.now(),  # Base for relative dates
    }

    return dateparser.parse(date_str, settings=settings)


def format_date_iso(dt: datetime) -> str:
    """Format datetime as ISO string compatible with storage layer.

    Args:
        dt: datetime to format

    Returns:
        ISO 8601 formatted string (YYYY-MM-DD HH:MM:SS)
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")
