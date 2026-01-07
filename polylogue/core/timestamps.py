"""Unified timestamp parsing utilities for Polylogue.

Handles the various timestamp formats stored in the database:
- Unix epoch as int/float
- Unix epoch as string
- ISO 8601 strings
"""

from __future__ import annotations

from datetime import datetime


def parse_timestamp(value: str | int | float | None) -> datetime | None:
    """Parse a timestamp from various formats to datetime.

    Args:
        value: Timestamp as epoch (int/float/str) or ISO string, or None

    Returns:
        datetime object or None if parsing fails
    """
    if value is None:
        return None

    try:
        # Handle int/float epoch directly
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value))

        # Handle string
        if isinstance(value, str):
            # Check if it looks like an epoch (all digits, possibly with decimal)
            if value.replace(".", "").isdigit():
                return datetime.fromtimestamp(float(value))

            # Try ISO 8601 parsing
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
    except (ValueError, OSError, OverflowError):
        # OSError can occur for out-of-range timestamps
        # OverflowError for extremely large values
        pass

    return None


def format_timestamp(ts: int | float | datetime) -> str:
    """Format a timestamp as ISO 8601 string.

    Args:
        ts: Unix epoch (int/float) or datetime object

    Returns:
        ISO 8601 formatted string
    """
    if isinstance(ts, datetime):
        return ts.isoformat(timespec="seconds")
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


__all__ = ["parse_timestamp", "format_timestamp"]
