"""Text and timestamp helpers for Gemini/Drive parsing."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from polylogue.lib.timestamps import parse_timestamp


def extract_text_from_chunk(chunk: object) -> str | None:
    if not isinstance(chunk, dict):
        return None
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str) and part:
                texts.append(part)
            elif isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    texts.append(part_text)
        return "\n".join(texts) or None
    return None


def chunk_timestamp(chunk: Mapping[str, object], default_timestamp: str | None) -> str | None:
    for key in ("createTime", "timestamp", "updateTime"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    return default_timestamp


def select_timestamp(values: list[str | None], *, latest: bool) -> str | None:
    candidates: list[tuple[datetime, str]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        parsed = parse_timestamp(value)
        if parsed is None:
            continue
        seen.add(value)
        candidates.append((parsed, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1] if latest else candidates[0][1]


__all__ = ["chunk_timestamp", "extract_text_from_chunk", "select_timestamp"]
