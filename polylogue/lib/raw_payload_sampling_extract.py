"""Streaming and bounded sampling for decoded raw payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import orjson

from polylogue.lib.raw_payload_sampling_buckets import (
    bucket_target_counts,
    is_record_candidate,
    record_bucket_key,
    take_bucketed_samples,
)


def limit_samples(
    samples: list[dict[str, Any]],
    *,
    limit: int,
    stratify: bool,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Limit extracted samples while preserving representative record variants."""
    if limit <= 0 or len(samples) <= limit:
        return samples
    if not stratify:
        return samples[:limit]

    buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        buckets.setdefault(record_bucket_key(sample, record_type_key), []).append(sample)
    return take_bucketed_samples(buckets, limit)


def collect_limited_samples(
    sample_factory: Callable[[], Iterable[dict[str, Any]]],
    *,
    limit: int,
    stratify: bool,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Collect bounded samples from a re-iterable source without full materialization."""
    if limit <= 0:
        return []
    if not stratify:
        selected: list[dict[str, Any]] = []
        for sample in sample_factory():
            selected.append(sample)
            if len(selected) >= limit:
                break
        return selected

    bucket_counts = Counter(
        record_bucket_key(sample, record_type_key)
        for sample in sample_factory()
    )
    if not bucket_counts:
        return []

    target_counts = bucket_target_counts(bucket_counts, limit)
    collected_counts = Counter()
    selected: list[dict[str, Any]] = []

    for sample in sample_factory():
        bucket = record_bucket_key(sample, record_type_key)
        target = target_counts.get(bucket, 0)
        if target == 0 or collected_counts[bucket] >= target:
            continue
        selected.append(sample)
        collected_counts[bucket] += 1
        if len(selected) >= limit:
            break

    return selected


def extract_payload_samples(
    payload: Any,
    *,
    sample_granularity: Literal["document", "record"],
    max_samples: int | None = None,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Extract representative validation/schema samples from a decoded payload."""
    if not isinstance(payload, (dict, list)):
        return []

    if sample_granularity == "document":
        documents = [payload] if isinstance(payload, dict) else [item for item in payload if isinstance(item, dict)]
        if max_samples is None:
            return documents
        return documents[:max_samples] if max_samples > 0 else []

    if isinstance(payload, dict):
        return [payload] if is_record_candidate(payload) else []

    if max_samples is None:
        return [item for item in payload if isinstance(item, dict) and is_record_candidate(item)]

    if max_samples <= 0:
        return []

    scan_cap: int = max(1024, max_samples * 64)
    per_bucket_cap: int = 8
    buckets: dict[str, list[dict[str, Any]]] = {}
    head_items: list[dict[str, Any]] = []
    dict_count = 0
    truncated = False

    for item in payload:
        if not isinstance(item, dict) or not is_record_candidate(item):
            continue
        dict_count += 1
        if dict_count <= max_samples:
            head_items.append(item)
        bucket = buckets.setdefault(record_bucket_key(item, record_type_key), [])
        if len(bucket) < per_bucket_cap:
            bucket.append(item)
        if dict_count >= scan_cap:
            truncated = True
            break

    if dict_count == 0:
        return []
    if not truncated and dict_count <= max_samples:
        return head_items
    return take_bucketed_samples(buckets, max_samples)


def extract_record_samples_from_raw_content(
    raw_content: Path | bytes | str | Any,
    *,
    max_samples: int | None,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Stream-record sample extraction for large JSONL payloads.

    When *raw_content* is a :class:`~pathlib.Path`, lines are streamed
    directly from the file handle — enabling constant-memory sampling
    of multi-GB files.

    When *max_samples* is ``None`` (full-corpus mode), all records are
    returned without bucketing or scan caps.
    """
    if max_samples is not None and max_samples <= 0:
        return []

    if isinstance(raw_content, Path):
        stream = open(raw_content, "rb")  # noqa: SIM115 — caller-managed context
    else:
        raw = raw_content if isinstance(raw_content, (bytes, str)) else str(raw_content)
        stream = BytesIO(raw) if isinstance(raw, bytes) else StringIO(raw)

    # Full-corpus mode: collect everything without caps.
    if max_samples is None:
        all_records: list[dict[str, Any]] = []
        first_line = True
        try:
            for raw_line in stream:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if first_line:
                    line = line.lstrip("\ufeff")
                    first_line = False
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = orjson.loads(line)
                except (ValueError, orjson.JSONDecodeError):
                    continue
                if isinstance(parsed, dict) and is_record_candidate(parsed):
                    all_records.append(parsed)
        finally:
            if isinstance(raw_content, Path):
                stream.close()
        return all_records

    lines: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    dict_count = 0
    scan_cap = max(1024, max_samples * 64)
    per_bucket_cap = 8
    first_line = True

    try:
        for raw_line in stream:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if first_line:
                line = line.lstrip("\ufeff")
                first_line = False
            line = line.strip()
            if not line:
                continue
            try:
                parsed = orjson.loads(line)
            except (ValueError, orjson.JSONDecodeError):
                continue
            if not isinstance(parsed, dict) or not is_record_candidate(parsed):
                continue

            dict_count += 1
            if dict_count <= max_samples:
                lines.append(parsed)
            bucket = buckets.setdefault(record_bucket_key(parsed, record_type_key), [])
            if len(bucket) < per_bucket_cap:
                bucket.append(parsed)
            if dict_count >= scan_cap:
                break
    finally:
        if isinstance(raw_content, Path):
            stream.close()

    if dict_count == 0:
        return []
    if dict_count <= max_samples:
        return lines
    return take_bucketed_samples(buckets, max_samples)


__all__ = [
    "collect_limited_samples",
    "extract_payload_samples",
    "extract_record_samples_from_raw_content",
    "limit_samples",
]
