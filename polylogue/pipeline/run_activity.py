"""Helpers for honest session-activity accounting in run results."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.storage.run_state import RunCounts, RunCountsPayload, RunDrift, RunDriftPayload

RunCountLike = Mapping[str, object] | RunCountsPayload | RunCounts
RunDriftLike = Mapping[str, object] | RunDriftPayload | RunDrift


def _int_value(mapping: RunCountLike, key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) else 0


def _drift_value(
    drift: RunDriftLike | None,
    bucket: str,
    key: str,
) -> int:
    if drift is None:
        return 0
    bucket_mapping = drift.get(bucket)
    if not hasattr(bucket_mapping, "get"):
        return 0
    value = bucket_mapping.get(key)
    return value if isinstance(value, int) else 0


def session_activity_counts(
    counts: RunCountLike,
    drift: RunDriftLike | None = None,
) -> tuple[int, int, int]:
    """Return total, new, and changed session counts for one run."""

    total = _int_value(counts, "sessions")
    new = _drift_value(drift, "new", "sessions") or _int_value(counts, "new_sessions")
    changed = _drift_value(drift, "changed", "sessions") or _int_value(counts, "changed_sessions")

    if new == 0 and changed == 0 and total:
        new = total
    if total < new + changed:
        total = new + changed
    return total, new, changed


__all__ = ["session_activity_counts"]
