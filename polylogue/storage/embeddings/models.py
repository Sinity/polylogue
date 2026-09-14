"""Typed embedding-statistics models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingStatsSnapshot:
    # ``None`` means *could not measure*, never zero. See
    # ``EmbeddingCoverageUnmeasurableError``: embeddings.db is re-purchased from a
    # paid provider, so an unmeasurable tier must not render as un-embedded.
    embedded_sessions: int | None = 0
    embedded_messages: int | None = 0
    pending_sessions: int | None = 0
    pending_messages: int = 0
    candidate_prose_messages: int | None = None
    candidate_prose_messages_exact: bool = False
    stale_messages: int = 0
    messages_missing_provenance: int = 0
    oldest_embedded_at: str | None = None
    newest_embedded_at: str | None = None
    model_counts: dict[str, int] = field(default_factory=dict)
    dimension_counts: dict[int, int] = field(default_factory=dict)
    retrieval_bands: dict[str, dict[str, object]] = field(default_factory=dict)
    failure_count: int = 0
    total_estimated_cost_usd: float | None = 0.0
    coverage_unmeasurable_reason: str | None = None

    @property
    def coverage_measurable(self) -> bool:
        return self.coverage_unmeasurable_reason is None


__all__ = ["EmbeddingStatsSnapshot"]
