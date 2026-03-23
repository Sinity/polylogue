"""Typed search result models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchHit:
    conversation_id: str
    provider_name: str
    source_name: str | None
    message_id: str
    title: str | None
    timestamp: str | None
    snippet: str
    conversation_path: Path


@dataclass
class SearchResult:
    hits: list[SearchHit]


__all__ = ["SearchHit", "SearchResult"]
