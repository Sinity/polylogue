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

    # Reject Python repr strings (e.g. from str(non_string_value))
    if isinstance(value, str) and value and value[0] in ("{", "[", "("):
        return None

    try:
        # Handle int/float epoch directly - use UTC to avoid DST issues
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        # Handle string
        if isinstance(value, str):
            # Check if it looks like an epoch (all digits, possibly with decimal).
            # Realistic epoch range: 1970-01-02 through ~2100 (86400 to 4102444800).
            # Values below 86400 (like "2025") would be misinterpreted as year strings.
            if value.replace(".", "").isdigit():
                epoch_f = float(value)
                if epoch_f >= 86400:
                    return datetime.fromtimestamp(epoch_f, tz=timezone.utc)

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


def parse_archive_datetime(value: str | None) -> datetime | None:
    """Parse an archive-stored ISO 8601 datetime string, always UTC-aware.

    Consolidates six identical/near-identical ``_parse_archive_datetime``
    private copies (context/selection.py, mcp/archive_support.py,
    cli/read_views/standard.py, api/archive.py,
    archive/query/archive_execution.py, storage/insights/session/rebuild.py)
    that had silently diverged: five returned a naive ``datetime`` for a
    naive-looking input string (no ``Z``/offset suffix), while the rebuild.py
    copy forced UTC on naive results. The same stored string could therefore
    parse to offset-naive or offset-aware depending on which surface read it
    — a latent ``TypeError: can't compare offset-naive and offset-aware
    datetimes`` wherever a value from one surface met a value from another
    (polylogue-a7xr.6).

    Unlike :func:`parse_timestamp`, this deliberately does NOT swallow a
    malformed non-empty string into ``None`` — archive-stored timestamp
    columns are expected to already be well-formed ISO 8601, so a parse
    failure here indicates real data corruption worth surfacing via
    ``ValueError``, not silently discarding.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def format_timestamp(ts: int | float | datetime) -> str:
    """Format a timestamp as ISO 8601 string with UTC timezone.

    Args:
        ts: Unix epoch (int/float) or datetime object

    Returns:
        ISO 8601 formatted string with +00:00 timezone suffix
    """
    if isinstance(ts, datetime):
        # If aware, convert to UTC; if naive, assume UTC
        ts = ts.astimezone(timezone.utc) if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        return ts.isoformat(timespec="seconds")
    # Convert epoch to UTC datetime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def canonical_timestamp_text(value: str | int | float | datetime | None) -> str | None:
    """Return canonical UTC ISO-8601 text for a supported timestamp.

    Unlike :func:`parse_timestamp_pair`, this does not preserve epoch strings
    verbatim. It is intended for archive storage columns where SQL date
    functions need one timestamp representation.
    """
    parsed = value if isinstance(value, datetime) else parse_timestamp(value)
    if parsed is None:
        return None
    parsed = parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def parse_timestamp_pair(
    value: str | int | float | None,
) -> tuple[datetime, str] | None:
    """Parse a timestamp and return both the datetime and an ISO-style string.

    If ``value`` is already a string, the original string is preserved verbatim
    (so any on-disk representation round-trips unchanged). For numeric inputs,
    the canonical ISO 8601 string from :func:`format_timestamp` is returned.

    Returns ``None`` when the value cannot be parsed.
    """
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if isinstance(value, str):
        return (parsed, value)
    return (parsed, format_timestamp(parsed))


def _timestamp_sort_key(ts: str | None) -> float | None:
    """Convert a timestamp string to a numeric sort key.

    Handles epoch strings (with millisecond normalisation) and ISO-8601.
    """
    if ts is None:
        return None
    try:
        value = float(ts)
        if value > 32503680000:
            value = value / 1000
        return value
    except (ValueError, TypeError):
        pass
    parsed = parse_timestamp(ts)
    if parsed is not None:
        return parsed.timestamp()
    return None


__all__ = [
    "canonical_timestamp_text",
    "format_timestamp",
    "parse_timestamp",
    "parse_timestamp_pair",
    "_timestamp_sort_key",
]
