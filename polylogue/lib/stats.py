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
    providers: dict[str, int] = field(default_factory=dict)
    embedded_conversations: int = 0
    embedded_messages: int = 0
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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "provider_count": self.provider_count,
            "providers": self.providers,
            "embedded_conversations": self.embedded_conversations,
            "embedded_messages": self.embedded_messages,
            "embedding_coverage_percent": round(self.embedding_coverage, 1),
            "avg_messages_per_conversation": round(self.avg_messages_per_conversation, 1),
            "db_size_bytes": self.db_size_bytes,
        }


__all__ = ["ArchiveStats"]
