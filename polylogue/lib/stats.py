"""Archive statistics types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchiveStats:
    """Comprehensive archive statistics.

    Provides a snapshot of the archive state including conversation counts,
    message counts, provider breakdown, and embedding coverage.
    """

    total_conversations: int
    total_messages: int
    total_attachments: int = 0
    providers: dict[str, int] = field(default_factory=dict)
    embedded_conversations: int = 0
    embedded_messages: int = 0
    pending_embedding_conversations: int = 0
    stale_embedding_messages: int = 0
    messages_missing_embedding_provenance: int = 0
    embedding_oldest_at: str | None = None
    embedding_newest_at: str | None = None
    embedding_models: dict[str, int] = field(default_factory=dict)
    embedding_dimensions: dict[int, int] = field(default_factory=dict)
    db_size_bytes: int = 0

    @property
    def provider_count(self) -> int:
        """Number of unique providers."""
        return len(self.providers)

    @property
    def avg_messages_per_conversation(self) -> float:
        """Average messages per conversation."""
        if self.total_conversations == 0:
            return 0.0
        return self.total_messages / self.total_conversations

    @property
    def embedding_coverage(self) -> float:
        """Percentage of conversations with embeddings."""
        if self.total_conversations == 0:
            return 0.0
        return (self.embedded_conversations / self.total_conversations) * 100

    @property
    def retrieval_ready(self) -> bool:
        fresh_messages = max(self.embedded_messages - self.stale_embedding_messages, 0)
        return fresh_messages > 0

    @property
    def embedding_readiness_status(self) -> str:
        if self.embedded_messages <= 0:
            return "none"
        if self.stale_embedding_messages > 0 or self.messages_missing_embedding_provenance > 0:
            return "stale"
        if self.pending_embedding_conversations > 0:
            return "partial"
        return "fresh"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "total_attachments": self.total_attachments,
            "provider_count": self.provider_count,
            "providers": self.providers,
            "embedded_conversations": self.embedded_conversations,
            "embedded_messages": self.embedded_messages,
            "pending_embedding_conversations": self.pending_embedding_conversations,
            "stale_embedding_messages": self.stale_embedding_messages,
            "messages_missing_embedding_provenance": self.messages_missing_embedding_provenance,
            "embedding_oldest_at": self.embedding_oldest_at,
            "embedding_newest_at": self.embedding_newest_at,
            "embedding_models": self.embedding_models,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_coverage_percent": round(self.embedding_coverage, 1),
            "embedding_readiness_status": self.embedding_readiness_status,
            "retrieval_ready": self.retrieval_ready,
            "avg_messages_per_conversation": round(self.avg_messages_per_conversation, 1),
            "db_size_bytes": self.db_size_bytes,
        }


__all__ = ["ArchiveStats"]
