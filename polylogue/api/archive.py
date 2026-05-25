"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.errors import PolylogueError
from polylogue.insights.archive import (
    SessionEnrichmentInsight,
    SessionEnrichmentInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.insights.feedback import LearningCorrection
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation, ConversationSummary
    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.archive.filter.filters import ConversationFilter
    from polylogue.archive.message.models import Message
    from polylogue.archive.query.facets import FacetBuckets
    from polylogue.archive.query.spec import ConversationQuerySpec
    from polylogue.config import Config
    from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport
    from polylogue.insights.export_bundles import InsightExportBundleRequest, InsightExportBundleResult
    from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
    from polylogue.insights.resume import ResumeBrief, ResumeCandidate
    from polylogue.operations import ArchiveOperations, ArchiveStats
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search.models import SearchResult
    from polylogue.surfaces.payloads import (
        BulkTagMutationResult,
        DeleteConversationResult,
        FacetsResponse,
        MetadataMutationResult,
        SearchEnvelope,
        TagMutationResult,
    )


class ConversationNotFoundError(PolylogueError):
    """Raised when a requested conversation does not exist in the archive."""

    http_status_code = 404


class PolylogueArchiveMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def operations(self) -> ArchiveOperations: ...

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

    async def search_envelope(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        retrieval_lane: str = "auto",
        sort: str | None = None,
        cursor: str | None = None,
    ) -> SearchEnvelope:
        """Return the canonical :class:`SearchEnvelope` for a query (#1266).

        Pass ``cursor`` (an opaque token previously returned as
        :attr:`SearchEnvelope.next_cursor`) to fetch the next page
        without losing or duplicating hits even when the archive grew
        between requests (#1268).
        """
        from polylogue.api.search_envelope_builder import build_archive_search_envelope

        return await build_archive_search_envelope(
            self.operations,
            self.repository,
            query=query,
            limit=limit,
            offset=offset,
            provider=provider,
            since=since,
            until=until,
            retrieval_lane=retrieval_lane,
            sort=sort,
            cursor=cursor,
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

    async def facets(
        self,
        spec: ConversationQuerySpec | None = None,
        *,
        include_idf: bool = True,
    ) -> FacetsResponse:
        """Compute scoped + global facet aggregates over the archive.

        When ``spec`` carries any active filter, the scoped buckets are
        rolled from that filter's summary list and ``scoped_to_query``
        becomes true.  The global buckets always reflect the
        unfiltered archive.  Surfaces (daemon HTTP, MCP, CLI) call into
        this method so the scope vocabulary stays in one place
        (#1269 / slice D of #873).
        """
        from polylogue.archive.query.facets import (
            FacetBuckets as _FacetBuckets,
        )
        from polylogue.archive.query.facets import (
            compute_facets,
            compute_idf,
        )
        from polylogue.surfaces.payloads import (
            FacetBucketsPayload,
            FacetsResponse,
        )

        scoped_to_query = spec is not None and spec.has_filters()
        global_buckets = await self._compute_global_facets()
        if scoped_to_query:
            assert spec is not None
            filter_obj = spec.build_filter(self.repository)
            scoped_summaries = await filter_obj.list_summaries()
            scoped_buckets = compute_facets(scoped_summaries)
        else:
            scoped_buckets = global_buckets

        idf_map = compute_idf(global_buckets) if include_idf else {}
        active = scoped_buckets if scoped_to_query else global_buckets

        def _payload(b: _FacetBuckets) -> FacetBucketsPayload:
            return FacetBucketsPayload(
                providers=dict(b.providers),
                tags=dict(b.tags),
                total_conversations=b.total_conversations,
                total_messages=b.total_messages,
            )

        return FacetsResponse.model_validate(
            {
                "scoped_to_query": scoped_to_query,
                "providers": dict(active.providers),
                "tags": dict(active.tags),
                "total_conversations": active.total_conversations,
                "total_messages": active.total_messages,
                "scoped": _payload(scoped_buckets),
                "global": _payload(global_buckets),
                "idf": idf_map,
            }
        )

    async def _compute_global_facets(self) -> FacetBuckets:
        """Compute global facet buckets from the unfiltered archive.

        Uses the same fluent filter machinery as the scoped path with
        no active predicates so both buckets are derived from one code
        path.
        """
        from polylogue.archive.query.facets import compute_facets
        from polylogue.archive.query.spec import ConversationQuerySpec

        empty_spec = ConversationQuerySpec()
        filter_obj = empty_spec.build_filter(self.repository)
        summaries = await filter_obj.list_summaries()
        return compute_facets(summaries)

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

    async def find_resume_candidates(
        self, *, repo_path: str, cwd: str | None = None, recent_files: Sequence[str] = (), limit: int = 10
    ) -> tuple[ResumeCandidate, ...]:
        return await self.operations.find_resume_candidates(
            repo_path=repo_path, cwd=cwd, recent_files=recent_files, limit=limit
        )

    async def insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        """Return insight materialization readiness for downstream consumers."""
        return await self.operations.get_insight_readiness_report(query)

    async def insight_rigor_audit(
        self,
        query: InsightRigorAuditQuery | None = None,
    ) -> InsightRigorAuditReport:
        """Per-product rigor profile across materialized insights (#1275)."""
        return await self.operations.audit_insight_rigor(query)

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

    async def bulk_get_messages(
        self,
        conversation_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        """Return messages for many conversations using one archive batch read."""
        return await self.operations.bulk_get_messages(
            conversation_ids,
            since=since,
            until=until,
            message_role=message_role,
            content_projection=content_projection,
        )

    async def get_raw_artifacts_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return paginated raw archive artifact rows for a conversation.

        Delegates to
        ``ArchiveOperations.get_raw_artifacts_for_conversation()``
        rather than accessing the private ``_backend`` connection directly.
        """
        records, total = await self.operations.get_raw_artifacts_for_conversation(
            conversation_id,
            limit=limit,
            offset=offset,
        )
        result: list[dict[str, object]] = []
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
        was not found. Routes through :meth:`ArchiveMutationsMixin
        .delete_conversation_safe` so resolution and idempotency stay
        centralized (#862).
        """
        result = await self.operations.delete_conversation_safe(conversation_id)
        return result.outcome == "deleted"

    async def delete_conversation_safe(self, conversation_id: str) -> DeleteConversationResult:
        """Typed delete that returns ``outcome="deleted"`` or ``"not_found"``."""
        return await self.operations.delete_conversation_safe(conversation_id)

    async def add_tag(self, conversation_id: str, tag: str) -> TagMutationResult:
        """Add a tag to a conversation.

        Returns a ``TagMutationResult`` with:
        - ``outcome="added"`` if the tag was newly added
        - ``outcome="no_op"`` if the tag was already present
        """
        from polylogue.surfaces.payloads import TagMutationResult

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        was_added = await self.operations.add_tag(str(resolved), tag)
        return TagMutationResult(
            outcome="added" if was_added else "no_op",
            detail=None if was_added else "already_present",
        )

    async def remove_tag(self, conversation_id: str, tag: str) -> TagMutationResult:
        """Remove a tag from a conversation.

        Returns a ``TagMutationResult`` with:
        - ``outcome="removed"`` if the tag was removed
        - ``outcome="not_present"`` if the tag was not present
        """
        from polylogue.surfaces.payloads import TagMutationResult

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        was_removed = await self.operations.remove_tag(str(resolved), tag)
        return TagMutationResult(
            outcome="removed" if was_removed else "not_present",
            detail=None if was_removed else "tag_not_present",
        )

    async def get_metadata(self, conversation_id: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a conversation."""
        result: dict[str, str] = {}
        doc = await self.operations.get_metadata(conversation_id)
        for k, v in doc.items():
            result[str(k)] = str(v) if not isinstance(v, str) else v
        return result

    async def update_metadata(self, conversation_id: str, key: str, value: str) -> bool:
        """Set a metadata key on a conversation.

        Returns ``True`` if the value was changed, ``False`` if it was already set
        to the same value. Routes through :meth:`ArchiveMutationsMixin
        .set_metadata_validated` so key validation and conversation
        resolution stay centralized (#862).
        """
        result = await self.operations.set_metadata_validated(conversation_id, key, value)
        return result.outcome == "set"

    async def set_metadata(self, conversation_id: str, key: str, value: object) -> MetadataMutationResult:
        """Typed metadata-set returning ``outcome="set"`` or ``"unchanged"``."""
        return await self.operations.set_metadata_validated(conversation_id, key, value)

    async def delete_metadata(self, conversation_id: str, key: str) -> MetadataMutationResult:
        """Typed metadata-delete returning ``outcome="deleted"`` or ``"not_found"``."""
        return await self.operations.delete_metadata_validated(conversation_id, key)

    async def bulk_tag_conversations(self, conversation_ids: list[str], tags: list[str]) -> BulkTagMutationResult:
        """Apply a bulk-tag operation across many conversations (#862).

        Validation (empty inputs and size limits) is enforced inside the
        :class:`ArchiveMutationsMixin` so every surface sees the same
        behavior.
        """
        return await self.operations.bulk_tag_conversations(conversation_ids, tags)

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    async def _resolve_user_state_target(
        self,
        conversation_id: str,
        *,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, str | None]:
        from polylogue.api.user_state_resolver import resolve_insight_target
        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        resolved_conversation_id = str(resolved)
        if target_type == "conversation":
            return {
                "target_type": "conversation",
                "target_id": resolved_conversation_id,
                "conversation_id": resolved_conversation_id,
                "message_id": None,
            }
        if target_type == "message":
            resolved_message_id = message_id or target_id
            if not resolved_message_id:
                raise ValueError("message target requires message_id or target_id")
            messages = await self.repository.get_messages(resolved_conversation_id)
            if not any(str(message.message_id) == resolved_message_id for message in messages):
                raise ValueError(f"message {resolved_message_id!r} is not in conversation {resolved_conversation_id!r}")
            return {
                "target_type": "message",
                "target_id": resolved_message_id,
                "conversation_id": resolved_conversation_id,
                "message_id": resolved_message_id,
            }
        if target_type not in TARGET_KIND_NAMES:
            raise ValueError(f"target_type must be one of: {', '.join(TARGET_KIND_NAMES)}")
        resolved_target = await resolve_insight_target(
            self.repository,
            target_type=target_type,
            target_id=target_id,
            conversation_id=resolved_conversation_id,
            message_id=message_id,
        )
        # Strip the identity_key — the storage layer doesn't carry it,
        # the recall-pack/workspace resolver re-derives it.
        return {
            "target_type": resolved_target["target_type"],
            "target_id": resolved_target["target_id"],
            "conversation_id": resolved_target["conversation_id"],
            "message_id": resolved_target.get("message_id"),
        }

    async def add_mark(
        self,
        conversation_id: str,
        mark_type: str,
        *,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Add a mark (star/pin/archive) to a conversation or message.

        Returns ``True`` if the mark was newly added, ``False`` if it already
        existed.
        """
        target = await self._resolve_user_state_target(
            conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        return await self.operations.add_mark(
            str(target["conversation_id"]),
            mark_type,
            target_type=str(target["target_type"]),
            target_id=str(target["target_id"]),
            message_id=target["message_id"],
        )

    async def remove_mark(
        self,
        conversation_id: str,
        mark_type: str,
        *,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Remove a mark from a conversation or message. Returns ``True`` if removed."""
        target = await self._resolve_user_state_target(
            conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        return await self.operations.remove_mark(str(target["target_type"]), str(target["target_id"]), mark_type)

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks, optionally filtered by type, target, conversation, or message."""
        return await self.operations.list_marks(
            mark_type=mark_type,
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    async def save_annotation(
        self,
        annotation_id: str,
        conversation_id: str,
        note_text: str,
        *,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Create or update an annotation. Returns ``True`` if newly created."""
        if not annotation_id.strip():
            raise ValueError("annotation_id must not be empty")
        if not note_text.strip():
            raise ValueError("note_text must not be empty")
        target = await self._resolve_user_state_target(
            conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        return await self.operations.save_annotation(
            annotation_id=annotation_id,
            target_type=str(target["target_type"]),
            target_id=str(target["target_id"]),
            conversation_id=str(target["conversation_id"]),
            message_id=target["message_id"],
            note_text=note_text,
        )

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Get an annotation by ID."""
        return await self.operations.get_annotation(annotation_id)

    async def list_annotations(
        self,
        *,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations, optionally filtered by target, conversation, or message."""
        return await self.operations.list_annotations(
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    async def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns ``True`` if deleted."""
        return await self.operations.delete_annotation(annotation_id)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Save a named query view. Returns ``True`` if newly created."""
        return await self.operations.save_view(view_id, name, query_json)

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        return await self.operations.get_view(view_id)

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get a saved view by name."""
        return await self.operations.get_view_by_name(name)

    async def list_views(self) -> list[dict[str, str]]:
        """List all saved views."""
        return await self.operations.list_views()

    async def delete_view(self, view_id: str) -> bool:
        """Delete a saved view. Returns ``True`` if deleted."""
        return await self.operations.delete_view(view_id)

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def _resolve_recall_pack_item(self, item: dict[str, object]) -> dict[str, object]:
        item_type = str(item.get("target_type") or item.get("type") or "conversation")
        if item_type == "conversation":
            conversation_id = str(item.get("conversation_id") or item.get("target_id") or item.get("id") or "")
            resolved = await self.repository.resolve_id(conversation_id, strict=True) if conversation_id else None
            if resolved is None:
                return {
                    "target_type": "conversation",
                    "target_id": conversation_id,
                    "conversation_id": conversation_id or None,
                    "status": "missing",
                    "disabled_reason": "conversation_not_found",
                }
            resolved_id = str(resolved)
            return {
                "target_type": "conversation",
                "target_id": resolved_id,
                "conversation_id": resolved_id,
                "status": "resolved",
                "identity_key": f"conversation:{resolved_id}",
            }

        if item_type == "message":
            conversation_id = str(item.get("conversation_id") or "")
            message_id = str(item.get("message_id") or item.get("target_id") or item.get("id") or "")
            try:
                target = await self._resolve_user_state_target(
                    conversation_id,
                    target_type="message",
                    message_id=message_id,
                )
            except (ConversationNotFoundError, ValueError) as exc:
                return {
                    "target_type": "message",
                    "target_id": message_id,
                    "conversation_id": conversation_id or None,
                    "message_id": message_id or None,
                    "status": "missing",
                    "disabled_reason": str(exc) or "message_not_found",
                }
            conversation_target_id = str(target["conversation_id"])
            resolved_message_id = str(target["message_id"])
            return {
                "target_type": "message",
                "target_id": resolved_message_id,
                "conversation_id": conversation_target_id,
                "message_id": resolved_message_id,
                "status": "resolved",
                "identity_key": f"message:{conversation_target_id}:{resolved_message_id}",
            }

        if item_type == "annotation":
            annotation_id = str(item.get("annotation_id") or item.get("target_id") or item.get("id") or "")
            row = await self.get_annotation(annotation_id) if annotation_id else None
            if row is None:
                return {
                    "target_type": "annotation",
                    "target_id": annotation_id,
                    "annotation_id": annotation_id or None,
                    "status": "missing",
                    "disabled_reason": "annotation_not_found",
                }
            return {
                "target_type": "annotation",
                "target_id": row["annotation_id"],
                "annotation_id": row["annotation_id"],
                "conversation_id": row["conversation_id"],
                "message_id": row["message_id"] or None,
                "annotated_target_type": row["target_type"],
                "annotated_target_id": row["target_id"],
                "note_text": row["note_text"],
                "status": "resolved",
                "identity_key": f"annotation:{row['annotation_id']}",
            }

        if item_type == "mark":
            mark_type = str(item.get("mark_type") or "")
            mark_target_type = str(item.get("mark_target_type") or item.get("target_ref_type") or "conversation")
            mark_target_id = str(item.get("mark_target_id") or item.get("target_id") or item.get("id") or "")
            conversation_id = str(item.get("conversation_id") or "")
            mark_message_id: str | None = str(item.get("message_id") or "") or None
            if not mark_type:
                return {
                    "target_type": "mark",
                    "target_id": mark_target_id,
                    "conversation_id": conversation_id or None,
                    "message_id": mark_message_id,
                    "status": "missing",
                    "disabled_reason": "mark_type_missing",
                }
            rows = await self.list_marks(
                mark_type=mark_type,
                conversation_id=conversation_id or None,
                target_type=mark_target_type,
                target_id=mark_target_id or None,
                message_id=mark_message_id,
            )
            if not rows:
                return {
                    "target_type": "mark",
                    "target_id": f"{mark_target_type}:{mark_target_id}:{mark_type}",
                    "conversation_id": conversation_id or None,
                    "message_id": mark_message_id,
                    "mark_type": mark_type,
                    "mark_target_type": mark_target_type,
                    "mark_target_id": mark_target_id,
                    "status": "missing",
                    "disabled_reason": "mark_not_found",
                }
            row = rows[0]
            return {
                "target_type": "mark",
                "target_id": f"{row['target_type']}:{row['target_id']}:{row['mark_type']}",
                "conversation_id": row["conversation_id"],
                "message_id": row["message_id"] or None,
                "mark_type": row["mark_type"],
                "mark_target_type": row["target_type"],
                "mark_target_id": row["target_id"],
                "status": "resolved",
                "identity_key": f"mark:{row['target_type']}:{row['target_id']}:{row['mark_type']}",
            }

        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        if item_type in TARGET_KIND_NAMES:
            return await self._resolve_recall_pack_insight_item(item, item_type)

        return {
            "target_type": item_type,
            "target_id": str(item.get("target_id") or item.get("id") or ""),
            "status": "unsupported",
            "disabled_reason": "unsupported_target_type",
        }

    async def _resolve_recall_pack_insight_item(
        self,
        item: dict[str, object],
        item_type: str,
    ) -> dict[str, object]:
        """Resolve a recall-pack item for a non-conversation/message kind (#1113)."""

        conversation_id = str(item.get("conversation_id") or "")
        target_id = str(item.get("target_id") or item.get("id") or "")
        message_id_raw = item.get("message_id")
        message_id: str | None = str(message_id_raw) if message_id_raw else None

        # session targets default target_id to the conversation_id when omitted.
        if item_type == "session" and not target_id and conversation_id:
            target_id = conversation_id

        if not conversation_id:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "conversation_id": None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": "conversation_id_required",
            }
        try:
            resolved = await self._resolve_user_state_target(
                conversation_id,
                target_type=item_type,
                target_id=target_id or None,
                message_id=message_id,
            )
        except (ConversationNotFoundError, ValueError) as exc:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "conversation_id": conversation_id or None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": str(exc) or f"{item_type}_not_found",
            }
        from polylogue.core.user_state_targets import identity_key

        resolved_target_id = str(resolved["target_id"])
        resolved_conversation_id = str(resolved["conversation_id"])
        resolved_message_id_raw = resolved.get("message_id")
        resolved_message_id: str | None = str(resolved_message_id_raw) if resolved_message_id_raw else None
        return {
            "target_type": item_type,
            "target_id": resolved_target_id,
            "conversation_id": resolved_conversation_id,
            "message_id": resolved_message_id,
            "status": "resolved",
            "identity_key": identity_key(
                item_type,
                conversation_id=resolved_conversation_id,
                target_id=resolved_target_id,
                message_id=resolved_message_id,
            ),
        }

    async def _build_recall_pack_payload(
        self,
        *,
        label: str,
        payload: dict[str, object],
    ) -> tuple[list[str], str]:
        explicit_items = payload.get("items")
        if not isinstance(explicit_items, list) or not all(isinstance(item, dict) for item in explicit_items):
            raise ValueError("recall pack payload must include an items list of objects")
        raw_items = list(explicit_items)

        items = [await self._resolve_recall_pack_item(item) for item in raw_items]
        resolved_conversation_ids: list[str] = []
        for item in items:
            conversation_id = item.get("conversation_id")
            if (
                item.get("status") == "resolved"
                and isinstance(conversation_id, str)
                and conversation_id not in resolved_conversation_ids
            ):
                resolved_conversation_ids.append(conversation_id)

        normalized_payload = {
            "schema_version": 1,
            "label": label,
            "summary": payload.get("summary") or payload.get("reason") or "",
            "items": items,
            "resolved_count": sum(1 for item in items if item.get("status") == "resolved"),
            "degraded_count": sum(1 for item in items if item.get("status") != "resolved"),
        }
        for key, value in payload.items():
            if key not in {"items", "summary", "reason"}:
                normalized_payload[key] = value
        import json

        return resolved_conversation_ids, json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))

    async def create_recall_pack(self, pack_id: str, label: str, payload_json: str) -> bool:
        """Save a recall pack. Returns ``True`` if newly created."""
        import json

        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("recall pack payload must be a JSON object")
        resolved_conversation_ids, normalized_payload_json = await self._build_recall_pack_payload(
            label=label,
            payload=payload,
        )
        conversation_ids_json = json.dumps(resolved_conversation_ids, sort_keys=True)
        return await self.operations.save_recall_pack(pack_id, label, conversation_ids_json, normalized_payload_json)

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get a recall pack by ID."""
        return await self.operations.get_recall_pack(pack_id)

    async def list_recall_packs(self) -> list[dict[str, str]]:
        """List all recall packs."""
        return await self.operations.list_recall_packs()

    async def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete a recall pack. Returns ``True`` if deleted."""
        return await self.operations.delete_recall_pack(pack_id)

    # ------------------------------------------------------------------
    # Reader workspaces
    # ------------------------------------------------------------------

    async def _build_workspace_targets(
        self, open_targets: Sequence[dict[str, object]]
    ) -> tuple[list[dict[str, object]], str]:
        import json

        items = [await self._resolve_recall_pack_item(item) for item in open_targets]
        return items, json.dumps(items, sort_keys=True, separators=(",", ":"))

    async def _build_workspace_active_target(self, active_target: dict[str, object]) -> str:
        import json

        if not active_target:
            return "{}"
        return json.dumps(await self._resolve_recall_pack_item(active_target), sort_keys=True, separators=(",", ":"))

    async def save_workspace(
        self,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str = "{}",
    ) -> bool:
        """Create or update a durable reader workspace."""
        import json

        workspace_id = workspace_id.strip()
        name = name.strip()
        mode = mode.strip()
        if not workspace_id:
            raise ValueError("workspace_id must not be empty")
        if not name:
            raise ValueError("name must not be empty")
        if mode not in {"tabs", "stack", "compare", "timeline"}:
            raise ValueError("mode must be one of: tabs, stack, compare, timeline")

        open_targets = json.loads(open_targets_json)
        if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
            raise ValueError("open_targets_json must encode a list of objects")
        _, normalized_targets_json = await self._build_workspace_targets(open_targets)

        layout = json.loads(layout_json)
        if not isinstance(layout, dict):
            raise ValueError("layout_json must encode an object")
        normalized_layout_json = json.dumps(layout, sort_keys=True, separators=(",", ":"))

        active_target = json.loads(active_target_json)
        if not isinstance(active_target, dict):
            raise ValueError("active_target_json must encode an object")
        normalized_active_json = await self._build_workspace_active_target(active_target)

        return await self.operations.save_workspace(
            workspace_id=workspace_id,
            name=name,
            mode=mode,
            open_targets_json=normalized_targets_json,
            layout_json=normalized_layout_json,
            active_target_json=normalized_active_json,
        )

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get a durable reader workspace by ID."""
        return await self.operations.get_workspace(workspace_id)

    async def list_workspaces(self) -> list[dict[str, str]]:
        """List durable reader workspaces."""
        return await self.operations.list_workspaces()

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a durable reader workspace. Returns ``True`` if deleted."""
        return await self.operations.delete_workspace(workspace_id)

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # User-recorded overrides that the insight materialization paths
    # consult after computing their base suggestion. Lives outside the
    # content-hash boundary by construction; see
    # :mod:`polylogue.insights.feedback` and
    # :mod:`polylogue.storage.insights.feedback`.
    # ------------------------------------------------------------------

    async def record_correction(
        self,
        conversation_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        """Record a typed user correction for a session.

        Resolves the conversation ID first (short IDs are accepted) so
        the durable row is keyed by the canonical ID. Raises
        :class:`ConversationNotFoundError` when the target session does
        not exist and
        :class:`~polylogue.insights.feedback.UnknownCorrectionKindError`
        when ``kind`` is not a recognized
        :class:`~polylogue.insights.feedback.CorrectionKind`.
        """

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        return await self.operations.record_correction(str(resolved), kind, payload, note=note)

    async def list_corrections(
        self,
        *,
        conversation_id: str | None = None,
        kind: str | None = None,
    ) -> list[LearningCorrection]:
        """List stored corrections, optionally filtered by session/kind."""

        resolved: str | None = None
        if conversation_id is not None:
            looked_up = await self.repository.resolve_id(conversation_id, strict=True)
            if looked_up is None:
                raise ConversationNotFoundError(conversation_id)
            resolved = str(looked_up)
        return await self.operations.list_corrections(conversation_id=resolved, kind=kind)

    async def delete_correction(self, conversation_id: str, kind: str) -> bool:
        """Delete one correction. Returns ``True`` when a row was removed."""

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        return await self.operations.delete_correction(str(resolved), kind)

    async def clear_corrections(self, conversation_id: str) -> int:
        """Delete every correction for a session. Returns the count."""

        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        return await self.operations.clear_corrections(str(resolved))
