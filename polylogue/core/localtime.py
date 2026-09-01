"""Local-time formatting for human-facing render boundaries."""

from __future__ import annotations

from datetime import date, datetime, timezone


def format_local_datetime(
    value: datetime | date | None,
    date_format: str = "%Y-%m-%d %H:%M %Z",
) -> str:
    """Format an archive time in the host's local timezone.

    Archive timestamps are expected to be timezone-aware. Naive values are
    interpreted as UTC to keep legacy records readable without changing their
    stored or wire representation.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone().strftime(date_format)
    return value.strftime(date_format)


__all__ = ["format_local_datetime"]
