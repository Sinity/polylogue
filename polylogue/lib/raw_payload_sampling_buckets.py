"""Bucketing and quota helpers for raw payload sampling."""

from __future__ import annotations

from collections import Counter
from typing import Any

RECORD_CANDIDATE_KEYS = frozenset({
    "type",
    "record_type",
    "role",
    "content",
    "message",
    "uuid",
    "id",
    "timestamp",
    "parentUuid",
    "sessionId",
    "payload",
})


def is_record_candidate(item: dict[str, Any]) -> bool:
    """Return whether a decoded object looks like a record-oriented payload item."""
    if any(key in item for key in RECORD_CANDIDATE_KEYS):
        return True
    payload = item.get("payload")
    return isinstance(payload, dict)


def record_bucket_key(sample: dict[str, Any], record_type_key: str | None = None) -> str:
    """Return a stratification key for record-oriented payload samples."""
    if record_type_key:
        value = sample.get(record_type_key)
        if isinstance(value, str) and value:
            return f"{record_type_key}:{value}"
    for key in ("type", "record_type"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    payload = sample.get("payload")
    if isinstance(payload, dict):
        value = payload.get("type")
        if isinstance(value, str) and value:
            return f"payload.type:{value}"
    return "unknown"


def bucket_target_counts(bucket_counts: Counter[str], limit: int) -> Counter[str]:
    """Compute per-bucket quotas matching the existing round-robin sampler."""
    remaining = Counter(bucket_counts)
    selected = Counter()
    ordered_keys = sorted(bucket_counts, key=lambda key: bucket_counts[key], reverse=True)
    selected_total = 0

    for key in ordered_keys:
        if selected_total >= limit or remaining[key] <= 0:
            continue
        selected[key] += 1
        remaining[key] -= 1
        selected_total += 1

    while selected_total < limit:
        progressed = False
        for key in ordered_keys:
            if selected_total >= limit:
                break
            if remaining[key] <= 0:
                continue
            selected[key] += 1
            remaining[key] -= 1
            selected_total += 1
            progressed = True
        if not progressed:
            break

    return selected


def take_bucketed_samples(
    buckets: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Take a stratified sample across pre-bucketed payload records."""
    ordered_keys = sorted(buckets, key=lambda key: len(buckets[key]), reverse=True)
    selected: list[dict[str, Any]] = []

    for key in ordered_keys:
        if len(selected) >= limit:
            break
        selected.append(buckets[key].pop(0))

    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


__all__ = [
    "RECORD_CANDIDATE_KEYS",
    "bucket_target_counts",
    "is_record_candidate",
    "record_bucket_key",
    "take_bucketed_samples",
]
