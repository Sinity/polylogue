"""Shared raw payload decoding and provider inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from polylogue.sources.source import detect_provider


@dataclass(frozen=True)
class DecodedPayload:
    """Decoded payload plus JSONL parse diagnostics."""

    payload: Any
    malformed_jsonl_lines: int = 0


def decode_raw_payload(
    raw_content: bytes | str | Any,
    *,
    jsonl_dict_only: bool = False,
) -> DecodedPayload:
    """Decode JSON payload bytes, with JSONL fallback support.

    Args:
        raw_content: Raw bytes/string from storage.
        jsonl_dict_only: When true, JSONL fallback keeps only dict records.
            Used by verification workflows that operate on object records only.
    """
    raw = raw_content if isinstance(raw_content, (bytes, str)) else str(raw_content)
    try:
        return DecodedPayload(payload=orjson.loads(raw), malformed_jsonl_lines=0)
    except (orjson.JSONDecodeError, ValueError):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        lines: list[Any] = []
        malformed_lines = 0
        for line in text.split("\n"):
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
            raise
        return DecodedPayload(payload=lines, malformed_jsonl_lines=malformed_lines)


def infer_payload_provider(
    payload: Any,
    *,
    source_path: str | Path | None,
    fallback_provider: str,
) -> str:
    """Infer canonical provider from payload/path, with fallback."""
    normalized = str(source_path or "")
    if ".zip:" in normalized:
        normalized = normalized.split(":", 1)[0]
    if normalized:
        inferred = detect_provider(payload, Path(normalized))
        if inferred:
            return inferred
    return fallback_provider


__all__ = ["DecodedPayload", "decode_raw_payload", "infer_payload_provider"]
