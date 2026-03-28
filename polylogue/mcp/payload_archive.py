"""Archive and mutation MCP payloads."""

from __future__ import annotations

from typing import Any

from polylogue.mcp.payload_base import MCPPayload, MCPRootPayload


class MCPArchiveStatsPayload(MCPPayload):
    total_conversations: int
    total_messages: int
    providers: dict[str, int]
    embedded_conversations: int | None = None
    embedded_messages: int | None = None
    pending_embedding_conversations: int | None = None
    embedding_coverage_percent: float | None = None
    stale_embedding_messages: int | None = None
    messages_missing_embedding_provenance: int | None = None
    embedding_health_status: str | None = None
    embedding_models: dict[str, int] | None = None
    embedding_dimensions: dict[int, int] | None = None
    embedding_oldest_at: str | None = None
    embedding_newest_at: str | None = None
    db_size_mb: float | int | None = None

    @classmethod
    def from_archive_stats(
        cls,
        archive_stats: Any,
        *,
        include_embedded: bool,
        include_db_size: bool,
    ) -> MCPArchiveStatsPayload:
        return cls(
            total_conversations=archive_stats.total_conversations,
            total_messages=archive_stats.total_messages,
            providers=archive_stats.providers,
            embedded_conversations=archive_stats.embedded_conversations if include_embedded else None,
            embedded_messages=archive_stats.embedded_messages if include_embedded else None,
            pending_embedding_conversations=(
                archive_stats.pending_embedding_conversations if include_embedded else None
            ),
            embedding_coverage_percent=(
                round(float(archive_stats.embedding_coverage), 1) if include_embedded else None
            ),
            stale_embedding_messages=archive_stats.stale_embedding_messages if include_embedded else None,
            messages_missing_embedding_provenance=(
                archive_stats.messages_missing_embedding_provenance if include_embedded else None
            ),
            embedding_health_status=archive_stats.embedding_health_status if include_embedded else None,
            embedding_models=archive_stats.embedding_models if include_embedded else None,
            embedding_dimensions=archive_stats.embedding_dimensions if include_embedded else None,
            embedding_oldest_at=archive_stats.embedding_oldest_at if include_embedded else None,
            embedding_newest_at=archive_stats.embedding_newest_at if include_embedded else None,
            db_size_mb=(
                round(archive_stats.db_size_bytes / 1_048_576, 1)
                if include_db_size and archive_stats.db_size_bytes
                else 0 if include_db_size else None
            ),
        )


class MCPMutationStatusPayload(MCPPayload):
    status: str
    conversation_id: str | None = None
    tag: str | None = None
    key: str | None = None
    index_exists: bool | None = None
    indexed_messages: int | None = None
    conversation_count: int | None = None


class MCPTagCountsPayload(MCPRootPayload):
    root: dict[str, int]


class MCPMetadataPayload(MCPRootPayload):
    root: dict[str, Any]


class MCPStatsByPayload(MCPRootPayload):
    root: dict[str, int]


__all__ = [
    "MCPArchiveStatsPayload",
    "MCPMetadataPayload",
    "MCPMutationStatusPayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
]
