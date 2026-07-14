"""Provenance vocabulary helpers for session attributes.

@owner archive-session
"""

from __future__ import annotations


def range_timing_provenance(start_time: str | None, end_time: str | None) -> str:
    """Determine the provenance category for session time range.

    Returns one of:
    - "timestamped_range": both start and end times are present
    - "start_timestamp_only": only start time is present
    - "end_timestamp_only": only end time is present
    - "untimestamped": neither start nor end time is present
    """
    if start_time is not None and end_time is not None:
        return "timestamped_range"
    if start_time is not None:
        return "start_timestamp_only"
    if end_time is not None:
        return "end_timestamp_only"
    return "untimestamped"


def date_provenance(
    canonical_session_date: str | None,
    start_time: str | None,
    end_time: str | None,
) -> str:
    """Determine the provenance category for session date.

    Returns one of:
    - "none": no canonical date available
    - "event_timestamp": date derived from event timestamps
    - "date_only": date set but no timestamps available
    """
    if canonical_session_date is None:
        return "none"
    if start_time is not None or end_time is not None:
        return "event_timestamp"
    return "date_only"


__all__ = ["date_provenance", "range_timing_provenance"]
