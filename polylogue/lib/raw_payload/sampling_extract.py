"""Streaming and bounded sampling for decoded raw payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from polylogue.lib.json import JSONDocument, JSONValue, json_document, loads
from polylogue.lib.raw_payload.sampling_buckets import (
    bucket_target_counts,
    is_record_candidate,
    record_bucket_key,
    take_bucketed_samples,
)
from polylogue.lib.raw_payload.streams import raw_line_stream


def limit_samples(
    samples: list[JSONDocument],
    *,
    limit: int,
    stratify: bool,
    record_type_key: str | None = None,
) -> list[JSONDocument]:
    """Limit extracted samples while preserving representative record variants."""
    if limit <= 0 or len(samples) <= limit:
        return samples
    if not stratify:
        return samples[:limit]

    buckets: dict[str, list[JSONDocument]] = {}
    for sample in samples:
        buckets.setdefault(record_bucket_key(sample, record_type_key), []).append(sample)
    return take_bucketed_samples(buckets, limit)


def collect_limited_samples(
    sample_factory: Callable[[], Iterable[JSONDocument]],
    *,
    limit: int,
    stratify: bool,
    record_type_key: str | None = None,
) -> list[JSONDocument]:
    """Collect bounded samples from a re-iterable source without full materialization."""
    if limit <= 0:
        return []
    if not stratify:
        selected: list[JSONDocument] = []
        for sample in sample_factory():
            selected.append(sample)
            if len(selected) >= limit:
                break
        return selected

    bucket_counts = Counter(record_bucket_key(sample, record_type_key) for sample in sample_factory())
    if not bucket_counts:
        return []

    target_counts = bucket_target_counts(bucket_counts, limit)
    collected_counts: Counter[str] = Counter()
    stratified_selected: list[JSONDocument] = []

    for sample in sample_factory():
        bucket = record_bucket_key(sample, record_type_key)
        target = target_counts.get(bucket, 0)
        if target == 0 or collected_counts[bucket] >= target:
            continue
        stratified_selected.append(sample)
        collected_counts[bucket] += 1
        if len(stratified_selected) >= limit:
            break

    return stratified_selected


def extract_payload_samples(
    payload: JSONValue,
    *,
    sample_granularity: Literal["document", "record"],
    max_samples: int | None = None,
    record_type_key: str | None = None,
) -> list[JSONDocument]:
    """Extract representative validation/schema samples from a decoded payload."""
    if not isinstance(payload, (dict, list)):
        return []

    if sample_granularity == "document":
        documents = [json_document(payload)] if isinstance(payload, dict) else [json_document(item) for item in payload]
        documents = [document for document in documents if document]
        if max_samples is None:
            return documents
        return documents[:max_samples] if max_samples > 0 else []

    if isinstance(payload, dict):
        payload_record = json_document(payload)
        return [payload_record] if payload_record and is_record_candidate(payload_record) else []

    if max_samples is None:
        return [record for item in payload if (record := json_document(item)) and is_record_candidate(record)]

    if max_samples <= 0:
        return []

    scan_cap: int = max(1024, max_samples * 64)
    per_bucket_cap: int = 8
    buckets: dict[str, list[JSONDocument]] = {}
    head_items: list[JSONDocument] = []
    dict_count = 0
    truncated = False

    for item in payload:
        record = json_document(item)
        if not record or not is_record_candidate(record):
            continue
        dict_count += 1
        if dict_count <= max_samples:
            head_items.append(record)
        bucket = buckets.setdefault(record_bucket_key(record, record_type_key), [])
        if len(bucket) < per_bucket_cap:
            bucket.append(record)
        if dict_count >= scan_cap:
            truncated = True
            break

    if dict_count == 0:
        return []
    if not truncated and dict_count <= max_samples:
        return head_items
    return take_bucketed_samples(buckets, max_samples)


def extract_record_samples_from_raw_content(
    raw_content: Path | bytes | str | JSONValue,
    *,
    max_samples: int | None,
    record_type_key: str | None = None,
) -> list[JSONDocument]:
    """Stream-record sample extraction for large JSONL payloads.

    When *raw_content* is a :class:`~pathlib.Path`, lines are streamed
    directly from the file handle. Bounded sampling stays memory-light,
    while ``max_samples=None`` still materializes the full result set.

    When *max_samples* is ``None`` (full-corpus mode), all records are
    returned without bucketing or scan caps.
    """
    if max_samples is not None and max_samples <= 0:
        return []

    if not isinstance(raw_content, (Path, bytes, str)):
        return extract_payload_samples(
            raw_content,
            sample_granularity="record",
            max_samples=max_samples,
            record_type_key=record_type_key,
        )

    with raw_line_stream(raw_content) as stream:
        # Full-corpus mode: collect everything without caps.
        if max_samples is None:
            all_records: list[JSONDocument] = []
            first_line = True
            for raw_line in stream:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if first_line:
                    line = line.lstrip("\ufeff")
                    first_line = False
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = loads(line)
                except ValueError:
                    continue
                record = json_document(parsed)
                if record and is_record_candidate(record):
                    all_records.append(record)
            return all_records

        lines: list[JSONDocument] = []
        buckets: dict[str, list[JSONDocument]] = {}
        dict_count = 0
        scan_cap = max(1024, max_samples * 64)
        per_bucket_cap = 8
        first_line = True

        for raw_line in stream:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if first_line:
                line = line.lstrip("\ufeff")
                first_line = False
            line = line.strip()
            if not line:
                continue
            try:
                parsed = loads(line)
            except ValueError:
                continue
            record = json_document(parsed)
            if not record or not is_record_candidate(record):
                continue

            dict_count += 1
            if dict_count <= max_samples:
                lines.append(record)
            bucket = buckets.setdefault(record_bucket_key(record, record_type_key), [])
            if len(bucket) < per_bucket_cap:
                bucket.append(record)
            if dict_count >= scan_cap:
                break

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
