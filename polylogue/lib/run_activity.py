"""Helpers for honest conversation-activity accounting in run results."""

from __future__ import annotations

from collections.abc import Mapping


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) else 0


def _drift_value(
    drift: Mapping[str, Mapping[str, object]] | None,
    bucket: str,
    key: str,
) -> int:
    if drift is None:
        return 0
    bucket_mapping = drift.get(bucket)
    if not isinstance(bucket_mapping, Mapping):
        return 0
    value = bucket_mapping.get(key)
    return value if isinstance(value, int) else 0


def conversation_activity_counts(
    counts: Mapping[str, object],
    drift: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[int, int, int]:
    """Return total, new, and changed conversation counts for one run."""

    total = _int_value(counts, "conversations")
    new = _drift_value(drift, "new", "conversations") or _int_value(counts, "new_conversations")
    changed = _drift_value(drift, "changed", "conversations") or _int_value(counts, "changed_conversations")

    if new == 0 and changed == 0 and total:
        new = total
    if total < new + changed:
        total = new + changed
    return total, new, changed


__all__ = ["conversation_activity_counts"]
