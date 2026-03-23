"""Archive summary/statistics support."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class ArchiveStats:
    """Statistics about the archive for the public facade surface."""

    def __init__(
        self,
        conversation_count: int,
        message_count: int,
        word_count: int,
        providers: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
        recent,
    ):
        self.conversation_count = conversation_count
        self.message_count = message_count
        self.word_count = word_count
        self.providers = providers
        self.tags = tags
        self.last_sync = last_sync
        self.recent = recent

    def __repr__(self) -> str:
        return (
            f"ArchiveStats(conversations={self.conversation_count}, "
            f"messages={self.message_count}, providers={list(self.providers.keys())})"
        )


class ArchiveStatsMixin:
    """Archive summary, status, and provider-count helpers."""

    async def storage_stats(self):
        return await self.repository.get_archive_stats()

    async def summary_stats(self) -> ArchiveStats:
        storage_stats = await self.storage_stats()
        aggregate_stats = await self.repository.queries.aggregate_message_stats()
        tags = await self.repository.list_tags()
        recent = await self.list_conversations(limit=5)

        last_sync = None
        try:
            last_sync = await self.backend.queries.get_last_sync_timestamp()
        except Exception as exc:  # pragma: no cover - defensive debug path
            logger.debug("failed to query last sync timestamp", error=str(exc))

        return ArchiveStats(
            conversation_count=storage_stats.total_conversations,
            message_count=storage_stats.total_messages,
            word_count=int(aggregate_stats.get("words_approx", 0)),
            providers=storage_stats.providers,
            tags=tags,
            last_sync=last_sync,
            recent=recent,
        )

    async def provider_counts(self) -> list[tuple[str, int]]:
        rows = await self.backend.queries.get_provider_conversation_counts()
        return [(row["provider_name"] or "unknown", row["conversation_count"]) for row in rows]

    async def get_session_product_status(self) -> dict[str, int | bool]:
        return await self.repository.get_session_product_status()


__all__ = ["ArchiveStats", "ArchiveStatsMixin"]
