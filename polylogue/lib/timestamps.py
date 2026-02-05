"""Unified timestamp parsing utilities for Polylogue.

Handles the various timestamp formats stored in the database:
- Unix epoch as int/float
- Unix epoch as string
- ISO 8601 strings

All operations use UTC to avoid DST ambiguity issues.
"""

from __future__ import annotations

from datetime import datetime, timezone


def parse_timestamp(value: str | int | float | None) -> datetime | None:
    """Parse a timestamp from various formats to datetime.

    Args:
        value: Timestamp as epoch (int/float/str) or ISO string, or None

    Returns:
        datetime object (UTC-aware for epoch inputs) or None if parsing fails
    """
    if value is None:
        return None

    try:
        # Handle int/float epoch directly - use UTC to avoid DST issues
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        # Handle string
        if isinstance(value, str):
            # Check if it looks like an epoch (all digits, possibly with decimal)
            if value.replace(".", "").isdigit():
                return datetime.fromtimestamp(float(value), tz=timezone.utc)

            # Try ISO 8601 parsing
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                # If naive datetime, assume UTC
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                pass
    except (ValueError, OSError, OverflowError):
        # OSError can occur for out-of-range timestamps
        # OverflowError for extremely large values
        pass

    return None


def format_timestamp(ts: int | float | datetime) -> str:
    """Format a timestamp as ISO 8601 string with UTC timezone.

    Args:
        ts: Unix epoch (int/float) or datetime object

    Returns:
        ISO 8601 formatted string with +00:00 timezone suffix
    """
    if isinstance(ts, datetime):
        # If aware, convert to UTC; if naive, assume UTC
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        else:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat(timespec="seconds")
    # Convert epoch to UTC datetime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


__all__ = ["parse_timestamp", "format_timestamp"]
