"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.insights.archive import (
    SessionEnrichmentInsight,
    SessionEnrichmentInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation, ConversationSummary
    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.archive.filter.filters import ConversationFilter
    from polylogue.archive.message.models import Message
    from polylogue.config import Config
    from polylogue.insights.export_bundles import InsightExportBundleRequest, InsightExportBundleResult
    from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
    from polylogue.insights.resume import ResumeBrief
    from polylogue.operations import ArchiveStats
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search.models import SearchResult

    class _ArchiveOperationsSurface(Protocol):
        async def get_conversation(
            self,
            conversation_id: str,
            *,
            content_projection: ContentProjectionSpec | None = None,
        ) -> Conversation | None: ...

        async def get_conversation_summary(self, conversation_id: str) -> ConversationSummary | None: ...

        async def get_messages_paginated(
            self,
            conversation_id: str,
            *,
            message_role: MessageRoleFilter = (),
            message_type: MessageTypeName | None = None,
            limit: int = 50,
            offset: int = 0,
            content_projection: ContentProjectionSpec | None = None,
        ) -> tuple[list[Message], int]: ...

        async def get_conversations(
            self,
            conversation_ids: list[str],
            *,
            content_projection: ContentProjectionSpec | None = None,
        ) -> list[Conversation]: ...

        async def list_conversations(
            self,
            *,
            provider: str | None = None,
            limit: int | None = None,
            content_projection: ContentProjectionSpec | None = None,
        ) -> list[Conversation]: ...

        async def search(
            self,
            query: str,
            *,
            limit: int = 100,
            source: str | None = None,
            since: str | None = None,
        ) -> SearchResult: ...

        async def get_session_insight_status(self) -> SessionInsightStatusSnapshot: ...

        async def get_session_profile_insight(
            self,
            conversation_id: str,
            *,
            tier: str = "merged",
        ) -> SessionProfileInsight | None: ...

        async def list_session_profile_insights(
            self,
            query: SessionProfileInsightQuery | None = None,
        ) -> list[SessionProfileInsight]: ...

        async def get_session_enrichment_insight(
            self,
            conversation_id: str,
        ) -> SessionEnrichmentInsight | None: ...

        async def list_session_enrichment_insights(
            self,
            query: SessionEnrichmentInsightQuery | None = None,
        ) -> list[SessionEnrichmentInsight]: ...

        async def summary_stats(self) -> ArchiveStats: ...

        async def rebuild_session_insights(
            self,
            conversation_ids: Sequence[str] | None = None,
        ) -> SessionInsightCounts: ...

        async def build_resume_brief(
            self,
            session_id: str,
            *,
            related_limit: int = 6,
        ) -> ResumeBrief | None: ...

        async def get_insight_readiness_report(
            self,
            query: InsightReadinessQuery | None = None,
        ) -> InsightReadinessReport: ...

        async def export_insight_bundle(
            self,
            request: InsightExportBundleRequest,
        ) -> InsightExportBundleResult: ...

        async def neighbor_candidates(
            self,
            *,
            conversation_id: str | None = None,
            query: str | None = None,
            provider: str | None = None,
            limit: int = 10,
            window_hours: int = 24,
        ) -> list[ConversationNeighborCandidate]: ...

        async def get_session_tree(self, conversation_id: str) -> list[Conversation]: ...

        async def list_tags(self, *, provider: str | None = None) -> dict[str, int]: ...

        async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]: ...


class ConversationNotFoundError(ValueError):
    """Raised when a requested conversation does not exist in the archive."""


class PolylogueArchiveMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def operations(self) -> _ArchiveOperationsSurface: ...

        @property
        def repository(self) -> ConversationRepository: ...

    async def get_conversation(
        self,
        conversation_id: str,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> Conversation | None:
        return await self.operations.get_conversation(conversation_id, content_projection=content_projection)

    async def get_conversations(
        self,
        conversation_ids: list[str],
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Conversation]:
        return await self.operations.get_conversations(conversation_ids, content_projection=content_projection)

    async def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Conversation]:
        return await self.operations.list_conversations(
            provider=provider,
            limit=limit,
            content_projection=content_projection,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        return await self.operations.search(
            query,
            limit=limit,
            source=source,
            since=since,
        )

    async def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        return await self.operations.get_session_insight_status()

    async def get_session_profile_insight(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        return await self.operations.get_session_profile_insight(conversation_id, tier=tier)

    async def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        return await self.operations.list_session_profile_insights(query)

    async def get_session_enrichment_insight(
        self,
        conversation_id: str,
    ) -> SessionEnrichmentInsight | None:
        return await self.operations.get_session_enrichment_insight(conversation_id)

    async def list_session_enrichment_insights(
        self,
        query: SessionEnrichmentInsightQuery | None = None,
    ) -> list[SessionEnrichmentInsight]:
        return await self.operations.list_session_enrichment_insights(query)

    def filter(self) -> ConversationFilter:
        from polylogue.archive.filter.filters import ConversationFilter
        from polylogue.storage.search_providers import create_vector_provider

        vector_provider = None
        with suppress(ValueError, ImportError):
            vector_provider = create_vector_provider(self.config)

        return ConversationFilter(self.repository, vector_provider=vector_provider)

    async def stats(self) -> ArchiveStats:
        return await self.operations.summary_stats()

    async def get_archive_stats(self) -> ArchiveStats:
        """Return archive summary statistics with an explicit API name."""
        return await self.stats()

    async def health_check(self) -> ReadinessReport:
        """Return the canonical archive readiness report."""
        from polylogue.readiness import get_readiness

        return get_readiness(self.config)

    async def rebuild_insights(
        self,
        conversation_ids: Sequence[str] | None = None,
    ) -> SessionInsightCounts:
        """Rebuild durable session-insight read models."""
        return await self.operations.rebuild_session_insights(conversation_ids=conversation_ids)

    async def resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact handoff brief for an archived session."""
        return await self.operations.build_resume_brief(session_id, related_limit=related_limit)

    async def insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        """Return insight materialization readiness for downstream consumers."""
        return await self.operations.get_insight_readiness_report(query)

    async def get_messages_paginated(
        self,
        conversation_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int]:
        """Return paginated ``Message`` objects for a conversation.

        Raises ``ConversationNotFoundError`` if the conversation does not exist.
        """
        summary = await self.operations.get_conversation_summary(conversation_id)
        if summary is None:
            raise ConversationNotFoundError(conversation_id)
        full_id = str(summary.id)

        return await self.operations.get_messages_paginated(
            full_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
            content_projection=content_projection,
        )

    async def get_raw_artifacts_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return paginated raw archive artifact rows for a conversation."""
        from polylogue.storage.sqlite.queries.raw import get_raw_records_for_conversation as _raw_query

        summary = await self.operations.get_conversation_summary(conversation_id)
        if summary is None:
            return [], 0
        full_id = str(summary.id)

        async with self.repository._backend.connection() as conn:
            records, total = await _raw_query(conn, full_id, limit=limit, offset=offset)
            result = []
            for r in records:
                result.append(
                    {
                        "raw_id": getattr(r, "raw_id", ""),
                        "provider_name": getattr(r, "provider_name", ""),
                        "source_path": getattr(r, "source_path", ""),
                        "source_name": getattr(r, "source_name", None),
                        "blob_size": getattr(r, "blob_size", 0),
                        "acquired_at": getattr(r, "acquired_at", None),
                        "parsed_at": getattr(r, "parsed_at", None),
                        "validation_status": getattr(r, "validation_status", None),
                    }
                )
            return result, total

    async def query_conversations(
        self,
        *,
        provider: str | None = None,
        tag: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        **kwargs: object,
    ) -> builtins.list[dict[str, object]]:
        """Query conversations with full filter support.

        Returns lightweight dicts suitable for the web reader and daemon API.
        For full ``Conversation`` objects use ``list_conversations``.
        """
        from polylogue.archive.query.spec import ConversationQuerySpec

        spec = ConversationQuerySpec.from_params(
            {
                "provider": provider,
                "tag": tag,
                "since": since,
                "until": until,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "filter_has_tool_use": has_tool_use,
                "filter_has_thinking": has_thinking,
                "filter_has_paste": has_paste,
                "typed_only": typed_only,
                "min_messages": min_messages,
                "max_messages": max_messages,
                "min_words": min_words,
                **kwargs,
            },
            strict=True,
        )
        filter_obj = spec.build_filter(self.repository)
        summaries = await filter_obj.list_summaries()
        return [
            {
                "id": str(s.id),
                "title": s.title,
                "provider": str(s.provider),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "message_count": getattr(s, "message_count", 0),
                "word_count": getattr(s, "word_count", 0),
            }
            for s in summaries
        ]

    async def count_conversations(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: object,
    ) -> int:
        """Count conversations matching the given filters."""
        from polylogue.archive.query.spec import ConversationQuerySpec

        spec = ConversationQuerySpec.from_params(
            {"provider": provider, "since": since, "until": until, **kwargs},
            strict=True,
        )
        return await spec.count(self.repository)

    async def export_insight_bundle(
        self,
        request: InsightExportBundleRequest,
    ) -> InsightExportBundleResult:
        """Write a versioned archive-insight export bundle."""
        return await self.operations.export_insight_bundle(request)

    async def get_conversation_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Return a summary record for a single conversation, or ``None`` if not found."""
        return await self.operations.get_conversation_summary(conversation_id)

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Return message-count and word-count stats for a single conversation."""
        return await self.operations.get_conversation_stats(conversation_id)

    async def neighbor_candidates(
        self,
        *,
        conversation_id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[ConversationNeighborCandidate]:
        """Discover explainable neighboring or near-duplicate candidates.

        At least one of ``conversation_id`` or ``query`` must be provided.
        """
        return await self.operations.neighbor_candidates(
            conversation_id=conversation_id,
            query=query,
            provider=provider,
            limit=limit,
            window_hours=window_hours,
        )

    async def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        """Return the full session tree (parent + children) for a conversation."""
        return await self.operations.get_session_tree(conversation_id)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with conversation counts, optionally filtered by provider."""
        return await self.operations.list_tags(provider=provider)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Permanently delete a conversation and all associated data.

        Returns ``True`` if something was deleted, ``False`` if the conversation
        was not found.
        """
        deleted_count = await self.filter().id(conversation_id).delete()
        return deleted_count > 0

    async def add_tag(self, conversation_id: str, tag: str) -> bool:
        """Add a tag to a conversation. Returns ``True`` if the tag was newly added."""
        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = self.repository  # type: ignore[assignment]
        existing = await store.list_tags()
        await store.add_tag(str(resolved), tag)
        after = await store.list_tags()
        return after.get(tag, 0) > existing.get(tag, 0)

    async def remove_tag(self, conversation_id: str, tag: str) -> bool:
        """Remove a tag from a conversation. Returns ``True`` if the tag was removed."""
        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = self.repository  # type: ignore[assignment]
        existing = await store.list_tags()
        await store.remove_tag(str(resolved), tag)
        after = await store.list_tags()
        return after.get(tag, 0) < existing.get(tag, 0)

    async def get_metadata(self, conversation_id: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a conversation."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = self.repository  # type: ignore[assignment]
        result: dict[str, str] = {}
        doc = await store.get_metadata(conversation_id)
        for k, v in doc.items():
            result[str(k)] = str(v) if not isinstance(v, str) else v
        return result

    async def update_metadata(self, conversation_id: str, key: str, value: str) -> None:
        """Set a metadata key on a conversation. Creates or updates the key."""
        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = self.repository  # type: ignore[assignment]
        await store.update_metadata(str(resolved), key, value)
