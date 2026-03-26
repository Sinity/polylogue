"""Typed embedding-statistics models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingStatsSnapshot:
    embedded_conversations: int = 0
    embedded_messages: int = 0
    pending_conversations: int = 0
    stale_messages: int = 0
    messages_missing_provenance: int = 0
    oldest_embedded_at: str | None = None
    newest_embedded_at: str | None = None
    model_counts: dict[str, int] = field(default_factory=dict)
    dimension_counts: dict[int, int] = field(default_factory=dict)
    retrieval_bands: dict[str, dict[str, object]] = field(default_factory=dict)


__all__ = ["EmbeddingStatsSnapshot"]
