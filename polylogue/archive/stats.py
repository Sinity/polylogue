"""Archive statistics types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchiveStats:
    """Comprehensive archive statistics.

    Provides a snapshot of the archive state including session counts,
    message counts, origin breakdown, and embedding coverage.
    """

    total_sessions: int
    total_messages: int
    total_attachments: int = 0
    origins: dict[str, int] = field(default_factory=dict)
    embedded_sessions: int = 0
    embedded_messages: int = 0
    pending_embedding_sessions: int = 0
    stale_embedding_messages: int = 0
    messages_missing_embedding_provenance: int = 0
    embedding_oldest_at: str | None = None
    embedding_newest_at: str | None = None
    embedding_models: dict[str, int] = field(default_factory=dict)
    embedding_dimensions: dict[int, int] = field(default_factory=dict)
    db_size_bytes: int = 0

    @property
    def origin_count(self) -> int:
        """Number of unique origins."""
        return len(self.origins)

    @property
    def avg_messages_per_session(self) -> float:
        """Average messages per session."""
        if self.total_sessions == 0:
            return 0.0
        return self.total_messages / self.total_sessions

    @property
    def embedding_coverage(self) -> float:
        """Percentage of sessions with embeddings."""
        if self.total_sessions == 0:
            return 0.0
        return (self.embedded_sessions / self.total_sessions) * 100

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
        if self.pending_embedding_sessions > 0:
            return "partial"
        return "fresh"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "total_attachments": self.total_attachments,
            "origin_count": self.origin_count,
            "origins": self.origins,
            "embedded_sessions": self.embedded_sessions,
            "embedded_messages": self.embedded_messages,
            "pending_embedding_sessions": self.pending_embedding_sessions,
            "stale_embedding_messages": self.stale_embedding_messages,
            "messages_missing_embedding_provenance": self.messages_missing_embedding_provenance,
            "embedding_oldest_at": self.embedding_oldest_at,
            "embedding_newest_at": self.embedding_newest_at,
            "embedding_models": self.embedding_models,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_coverage_percent": round(self.embedding_coverage, 1),
            "embedding_readiness_status": self.embedding_readiness_status,
            "retrieval_ready": self.retrieval_ready,
            "avg_messages_per_session": round(self.avg_messages_per_session, 1),
            "db_size_bytes": self.db_size_bytes,
        }


__all__ = ["ArchiveStats"]
