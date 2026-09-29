"""Text and timestamp helpers for Gemini/Drive parsing."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from polylogue.core.timestamps import parse_timestamp


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


class TimestampBounds:
    """The earliest and latest parseable chunk timestamps, in one pass.

    Equal instants keep the first spelling for the earliest bound and the
    last *distinct* spelling for the latest, as a stable sort of the
    deduplicated spellings would: ``Z``, ``+00:00``, ``Z`` keeps ``+00:00``.
    Only spellings at the current latest instant are remembered, so memory
    stays bounded by that instant's spellings.
    """

    def __init__(self) -> None:
        self.earliest: tuple[datetime, str] | None = None
        self.latest: tuple[datetime, str] | None = None
        self._latest_spellings: set[str] = set()

    def observe(self, value: str | None) -> None:
        if not isinstance(value, str) or not value:
            return
        parsed = parse_timestamp(value)
        if parsed is None:
            return
        if self.earliest is None or parsed < self.earliest[0]:
            self.earliest = (parsed, value)
        if self.latest is None or parsed > self.latest[0]:
            self.latest = (parsed, value)
            self._latest_spellings = {value}
        elif parsed == self.latest[0] and value not in self._latest_spellings:
            self.latest = (parsed, value)
            self._latest_spellings.add(value)


__all__ = ["TimestampBounds", "chunk_timestamp", "extract_text_from_chunk"]
