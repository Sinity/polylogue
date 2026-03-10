"""Shared raw payload decoding, inference, and sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import orjson

from polylogue.lib.provider_identity import canonical_runtime_provider
from polylogue.sources.source import detect_provider

WireFormat = Literal["json", "jsonl"]

_RECORD_CANDIDATE_KEYS = frozenset({
    "type", "record_type", "role", "content", "message", "uuid", "id",
    "timestamp", "parentUuid", "sessionId", "payload",
})


@dataclass(frozen=True)
class RawPayloadEnvelope:
    """Canonical decoded raw payload with inferred runtime semantics."""

    payload: Any
    provider: str
    wire_format: WireFormat
    malformed_jsonl_lines: int = 0


def _decode_raw_payload(
    raw_content: bytes | str | Any,
    *,
    jsonl_dict_only: bool = False,
    prefer_jsonl: bool = False,
) -> tuple[Any, WireFormat, int]:
    """Decode JSON payload bytes, with JSONL fallback support.

    Args:
        raw_content: Raw bytes/string from storage.
        jsonl_dict_only: When true, JSONL fallback keeps only dict records.
            Used by verification workflows that operate on object records only.
        prefer_jsonl: Prefer streaming JSONL decode before attempting whole-file
            JSON parsing. Used for known record-oriented providers and ``.jsonl``
            payload paths so validation does not waste time on a doomed document
            parse attempt first.
    """
    raw = raw_content if isinstance(raw_content, (bytes, str)) else str(raw_content)
    if prefer_jsonl:
        try:
            payload, malformed_lines = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
            return payload, "jsonl", malformed_lines
        except (UnicodeDecodeError, ValueError):
            pass
    try:
        return orjson.loads(raw), "json", 0
    except (orjson.JSONDecodeError, ValueError) as exc:
        try:
            payload, malformed_lines = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
        except ValueError:
            raise exc from None
        return payload, "jsonl", malformed_lines


def _decode_jsonl_payload(
    raw: bytes | str,
    *,
    jsonl_dict_only: bool = False,
) -> tuple[list[Any], int]:
    """Decode JSONL incrementally to avoid full-file line splitting."""
    lines: list[Any] = []
    malformed_lines = 0
    first_line = True
    stream = BytesIO(raw) if isinstance(raw, bytes) else StringIO(raw)

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
        except (orjson.JSONDecodeError, ValueError):
            malformed_lines += 1
            continue
        if jsonl_dict_only and not isinstance(parsed, dict):
            continue
        lines.append(parsed)

    if not lines:
        raise ValueError("No valid JSONL records found")
    return lines, malformed_lines


def _infer_payload_provider(
    payload: Any,
    *,
    source_path: str | Path | None,
    fallback_provider: str,
    payload_provider: str | None = None,
) -> str:
    """Infer canonical provider from payload/path, with fallback."""
    if payload_provider:
        return canonical_runtime_provider(
            payload_provider,
            preserve_unknown=True,
            default=payload_provider,
        )
    source_hint = str(source_path or "")
    if source_hint:
        inferred = detect_provider(payload, Path(source_hint))
        if inferred:
            return canonical_runtime_provider(
                inferred,
                preserve_unknown=True,
                default=inferred,
            )
    return canonical_runtime_provider(
        fallback_provider,
        preserve_unknown=True,
        default=fallback_provider,
    )


def build_raw_payload_envelope(
    raw_content: bytes | str | Any,
    *,
    source_path: str | Path | None,
    fallback_provider: str,
    payload_provider: str | None = None,
    jsonl_dict_only: bool = False,
) -> RawPayloadEnvelope:
    """Decode raw payload and attach canonical provider/wire-format identity."""
    normalized_path = str(source_path or "").lower()
    prefer_jsonl = normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson"))
    preferred_provider = payload_provider or fallback_provider
    if not prefer_jsonl:
        prefer_jsonl = canonical_runtime_provider(
            preferred_provider,
            preserve_unknown=True,
            default=preferred_provider,
        ) in {"claude-code", "codex"}
    payload, wire_format, malformed_jsonl_lines = _decode_raw_payload(
        raw_content,
        jsonl_dict_only=jsonl_dict_only,
        prefer_jsonl=prefer_jsonl,
    )
    provider = _infer_payload_provider(
        payload,
        source_path=source_path,
        fallback_provider=fallback_provider,
        payload_provider=payload_provider,
    )
    return RawPayloadEnvelope(
        payload=payload,
        provider=provider,
        wire_format=wire_format,
        malformed_jsonl_lines=malformed_jsonl_lines,
    )


def is_record_candidate(item: dict[str, Any]) -> bool:
    """Return whether a decoded object looks like a record-oriented payload item."""
    if any(key in item for key in _RECORD_CANDIDATE_KEYS):
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
    return _take_bucketed_samples(buckets, limit)


def _take_bucketed_samples(
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
        return [
            item for item in payload
            if isinstance(item, dict) and is_record_candidate(item)
        ]

    if max_samples <= 0:
        return []

    scan_cap = max(1024, max_samples * 64)
    per_bucket_cap = 8
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
        key = record_bucket_key(item, record_type_key)
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_bucket_cap:
            bucket.append(item)
        if dict_count >= scan_cap:
            truncated = True
            break

    if dict_count == 0:
        return []
    if not truncated and dict_count <= max_samples:
        return head_items
    return _take_bucketed_samples(buckets, max_samples)


__all__ = [
    "RawPayloadEnvelope",
    "WireFormat",
    "build_raw_payload_envelope",
    "extract_payload_samples",
    "is_record_candidate",
    "limit_samples",
    "record_bucket_key",
]
