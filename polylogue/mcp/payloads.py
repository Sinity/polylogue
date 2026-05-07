"""Typed MCP payload models shared by server tools and resources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import RootModel
from typing_extensions import TypedDict

from polylogue.core.json import JSONDocument
from polylogue.mcp.context_pack import (
    ContextPackActionSummary as MCPContextPackActionSummary,
)
from polylogue.mcp.context_pack import (
    ContextPackConversation as MCPContextPackConversation,
)
from polylogue.mcp.context_pack import (
    ContextPackDateRange as MCPContextPackDateRange,
)
from polylogue.mcp.context_pack import (
    ContextPackMessage as MCPContextPackMessage,
)
from polylogue.mcp.context_pack import (
    ContextPackPayload as MCPContextPackPayload,
)
from polylogue.mcp.context_pack import (
    ContextPackProject as MCPContextPackProject,
)
from polylogue.mcp.context_pack import (
    ContextPackProvenance as MCPContextPackProvenance,
)
from polylogue.mcp.context_pack import (
    ContextPackQueryContext as MCPContextPackQueryContext,
)
from polylogue.mcp.context_pack import (
    ContextPackUnresolvedWork as MCPContextPackUnresolvedWork,
)
from polylogue.surfaces.payloads import (
    ConversationDetailPayload as MCPConversationDetailPayload,
)
from polylogue.surfaces.payloads import (
    ConversationMessagePayload as MCPMessagePayload,
)
from polylogue.surfaces.payloads import (
    ConversationNeighborCandidatePayload as MCPConversationNeighborCandidatePayload,
)
from polylogue.surfaces.payloads import (
    ConversationSearchHitPayload as MCPConversationSearchHitPayload,
)
from polylogue.surfaces.payloads import (
    ConversationSummaryPayload as MCPConversationSummaryPayload,
)
from polylogue.surfaces.payloads import (
    SurfacePayloadModel,
    model_json_document,
    normalize_role,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.archive.models import Conversation
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics, QueryMissReason
    from polylogue.archive.query.search_hits import ConversationSearchHit
    from polylogue.archive.stats import ArchiveStats
    from polylogue.readiness import ReadinessCheck, ReadinessReport
    from polylogue.storage.runtime import RawConversationRecord

TRoot = TypeVar("TRoot")


class MCPRootPayload(RootModel[TRoot], Generic[TRoot]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPErrorPayload(SurfacePayloadModel):
    error: str
    code: int | str | None = None
    detail: str | None = None
    tool: str | None = None
    conversation_id: str | None = None
    is_error: Literal[True] = True


class MCPFencedCodeBlock(TypedDict):
    language: str
    code: str


class MCPConversationSummaryListPayload(MCPRootPayload[list[MCPConversationSummaryPayload]]):
    root: list[MCPConversationSummaryPayload]


class MCPConversationSearchHitListPayload(MCPRootPayload[list[MCPConversationSearchHitPayload]]):
    root: list[MCPConversationSearchHitPayload]


class MCPConversationNeighborCandidateListPayload(MCPRootPayload[list[MCPConversationNeighborCandidatePayload]]):
    root: list[MCPConversationNeighborCandidatePayload]


class MCPQueryMissReasonPayload(SurfacePayloadModel):
    code: str
    severity: str
    summary: str
    detail: str | None = None
    count: int | None = None

    @classmethod
    def from_reason(cls, reason: QueryMissReason) -> MCPQueryMissReasonPayload:
        return cls(
            code=reason.code,
            severity=reason.severity,
            summary=reason.summary,
            detail=reason.detail,
            count=reason.count,
        )


class MCPQueryMissDiagnosticsPayload(SurfacePayloadModel):
    message: str
    filters: tuple[str, ...]
    reasons: tuple[MCPQueryMissReasonPayload, ...]
    archive_conversation_count: int | None = None
    raw_conversation_count: int | None = None

    @classmethod
    def from_diagnostics(cls, diagnostics: QueryMissDiagnostics) -> MCPQueryMissDiagnosticsPayload:
        return cls(
            message=diagnostics.message,
            filters=diagnostics.filters,
            reasons=tuple(MCPQueryMissReasonPayload.from_reason(reason) for reason in diagnostics.reasons),
            archive_conversation_count=diagnostics.archive_conversation_count,
            raw_conversation_count=diagnostics.raw_conversation_count,
        )


class MCPConversationQueryNoResultsPayload(SurfacePayloadModel):
    results: tuple[MCPConversationSummaryPayload, ...] = ()
    diagnostics: MCPQueryMissDiagnosticsPayload


class MCPConversationSearchNoResultsPayload(SurfacePayloadModel):
    results: tuple[MCPConversationSearchHitPayload, ...] = ()
    diagnostics: MCPQueryMissDiagnosticsPayload


class MCPPaginatedQueryResultPayload(SurfacePayloadModel):
    """Paginated query result envelope for list_conversations."""

    items: tuple[MCPConversationSummaryPayload, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None
    diagnostics: MCPQueryMissDiagnosticsPayload | None = None


class MCPPaginatedSearchResultPayload(SurfacePayloadModel):
    """Paginated search result envelope for search."""

    hits: tuple[MCPConversationSearchHitPayload, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None
    diagnostics: MCPQueryMissDiagnosticsPayload | None = None


def conversation_summary_list_payload(
    conversations: Sequence[Conversation],
) -> MCPConversationSummaryListPayload:
    return MCPConversationSummaryListPayload(
        root=[MCPConversationSummaryPayload.from_conversation(conv) for conv in conversations]
    )


def conversation_query_result_payload(
    conversations: Sequence[Conversation],
    *,
    total: int,
    limit: int,
    offset: int,
    diagnostics: QueryMissDiagnostics | None = None,
) -> MCPPaginatedQueryResultPayload:
    next_offset = offset + len(conversations) if len(conversations) == limit and offset + limit < total else None
    return MCPPaginatedQueryResultPayload(
        items=tuple(MCPConversationSummaryPayload.from_conversation(conv) for conv in conversations),
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
        diagnostics=(MCPQueryMissDiagnosticsPayload.from_diagnostics(diagnostics) if diagnostics else None),
    )


def conversation_search_hit_list_payload(
    hits: Sequence[ConversationSearchHit],
) -> MCPConversationSearchHitListPayload:
    return MCPConversationSearchHitListPayload(
        root=[
            MCPConversationSearchHitPayload.from_search_hit(
                hit,
                message_count=hit.summary.message_count,
            )
            for hit in hits
        ]
    )


def conversation_neighbor_candidate_list_payload(
    candidates: Sequence[ConversationNeighborCandidate],
) -> MCPConversationNeighborCandidateListPayload:
    return MCPConversationNeighborCandidateListPayload(
        root=[MCPConversationNeighborCandidatePayload.from_candidate(candidate) for candidate in candidates]
    )


def conversation_search_result_payload(
    hits: Sequence[ConversationSearchHit],
    *,
    total: int,
    limit: int,
    offset: int,
    diagnostics: QueryMissDiagnostics | None = None,
) -> MCPPaginatedSearchResultPayload:
    next_offset = offset + len(hits) if len(hits) == limit and offset + limit < total else None
    return MCPPaginatedSearchResultPayload(
        hits=tuple(
            MCPConversationSearchHitPayload.from_search_hit(
                hit,
                message_count=hit.summary.message_count,
            )
            for hit in hits
        ),
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
        diagnostics=(MCPQueryMissDiagnosticsPayload.from_diagnostics(diagnostics) if diagnostics else None),
    )


class MCPArchiveStatsPayload(SurfacePayloadModel):
    total_conversations: int
    total_messages: int
    providers: dict[str, int]
    embedded_conversations: int | None = None
    embedded_messages: int | None = None
    pending_embedding_conversations: int | None = None
    embedding_coverage_percent: float | None = None
    stale_embedding_messages: int | None = None
    messages_missing_embedding_provenance: int | None = None
    embedding_readiness_status: str | None = None
    embedding_models: dict[str, int] | None = None
    embedding_dimensions: dict[int, int] | None = None
    embedding_oldest_at: str | None = None
    embedding_newest_at: str | None = None
    db_size_mb: float | int | None = None

    @classmethod
    def from_archive_stats(
        cls,
        archive_stats: ArchiveStats,
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
            embedding_readiness_status=archive_stats.embedding_readiness_status if include_embedded else None,
            embedding_models=archive_stats.embedding_models if include_embedded else None,
            embedding_dimensions=archive_stats.embedding_dimensions if include_embedded else None,
            embedding_oldest_at=archive_stats.embedding_oldest_at if include_embedded else None,
            embedding_newest_at=archive_stats.embedding_newest_at if include_embedded else None,
            db_size_mb=(
                round(archive_stats.db_size_bytes / 1_048_576, 1)
                if include_db_size and archive_stats.db_size_bytes
                else 0
                if include_db_size
                else None
            ),
        )


class MCPMutationStatusPayload(SurfacePayloadModel):
    status: str
    conversation_id: str | None = None
    tag: str | None = None
    key: str | None = None
    index_exists: bool | None = None
    indexed_messages: int | None = None
    conversation_count: int | None = None


class MutationResultPayload(SurfacePayloadModel):
    """Shared result envelope for tag, metadata, and delete mutations.

    Carries idempotent status codes, context fields, and bulk-operation
    counts so that CLI, MCP, API, and daemon surfaces all expose the
    same mutation contract shape.
    """

    status: str
    """``ok``, ``deleted``, ``not_found``, ``unchanged``, or ``partial``."""

    conversation_id: str | None = None
    detail: str | None = None
    """Machine-readable detail: ``already_present``, ``tag_not_present``,
    ``key_not_found``, ``value_unchanged``, ``conversation_not_found``."""

    affected_count: int | None = None
    skipped_count: int | None = None
    tag: str | None = None
    key: str | None = None
    conversation_count: int | None = None
    tag_count: int | None = None
    applied_count: int | None = None


class MCPTagCountsPayload(MCPRootPayload[dict[str, int]]):
    root: dict[str, int]


class MCPMetadataPayload(SurfacePayloadModel):
    root: dict[str, object]

    @classmethod
    def from_document(cls, document: JSONDocument) -> MCPMetadataPayload:
        return cls(root=dict(document))

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.root, indent=2)


class MCPStatsByPayload(MCPRootPayload[dict[str, int]]):
    root: dict[str, int]


class MCPMessagesListPayload(SurfacePayloadModel):
    """Paginated message list response for get_messages tool."""

    conversation_id: str
    messages: tuple[MCPMessagePayload, ...]
    total: int
    limit: int
    offset: int


class MCPRawArtifactPayload(SurfacePayloadModel):
    """One raw archive artifact for the raw_artifacts tool."""

    raw_id: str
    provider_name: str
    source_name: str | None = None
    source_path: str
    blob_size: int
    acquired_at: str
    parsed_at: str | None = None
    parse_error: str | None = None
    validated_at: str | None = None
    validation_status: str | None = None
    validation_error: str | None = None

    @classmethod
    def from_record(cls, record: RawConversationRecord) -> MCPRawArtifactPayload:
        return cls(
            raw_id=record.raw_id,
            provider_name=record.provider_name,
            source_name=record.source_name,
            source_path=record.source_path,
            blob_size=record.blob_size,
            acquired_at=record.acquired_at,
            parsed_at=record.parsed_at,
            parse_error=record.parse_error,
            validated_at=record.validated_at,
            validation_status=str(record.validation_status) if record.validation_status else None,
            validation_error=record.validation_error,
        )


class MCPRawArtifactsListPayload(SurfacePayloadModel):
    """Paginated raw archive artifact response for the raw_artifacts tool."""

    conversation_id: str
    raw_artifacts: tuple[MCPRawArtifactPayload, ...]
    total: int
    limit: int
    offset: int


class MCPReadinessCheckPayload(SurfacePayloadModel):
    name: str
    status: str
    count: int | None = None
    detail: str | None = None

    @classmethod
    def from_check(
        cls,
        check: ReadinessCheck,
        *,
        include_counts: bool,
        include_detail: bool,
    ) -> MCPReadinessCheckPayload:
        return cls(
            name=check.name,
            status=check.status.value,
            count=check.count if include_counts else None,
            detail=check.detail if include_detail else None,
        )


def _extract_readiness_source(report: ReadinessReport) -> str | None:
    provenance = report.provenance
    if provenance is None or provenance.source is None:
        return None
    return provenance.source


class MCPReadinessReportPayload(SurfacePayloadModel):
    checks: list[MCPReadinessCheckPayload]
    summary: str | dict[str, int]
    source: str | None = None

    @classmethod
    def from_report(
        cls,
        report: ReadinessReport,
        *,
        include_counts: bool,
        include_detail: bool,
        include_cached: bool,
    ) -> MCPReadinessReportPayload:
        return cls(
            checks=[
                MCPReadinessCheckPayload.from_check(
                    check,
                    include_counts=include_counts,
                    include_detail=include_detail,
                )
                for check in report.checks
            ],
            summary=report.summary,
            source=_extract_readiness_source(report) if include_cached else None,
        )


__all__ = [
    "MCPArchiveStatsPayload",
    "MCPContextPackActionSummary",
    "MCPContextPackConversation",
    "MCPContextPackDateRange",
    "MCPContextPackMessage",
    "MCPContextPackPayload",
    "MCPContextPackProject",
    "MCPContextPackProvenance",
    "MCPContextPackQueryContext",
    "MCPContextPackUnresolvedWork",
    "MCPConversationDetailPayload",
    "MCPConversationNeighborCandidateListPayload",
    "MCPConversationNeighborCandidatePayload",
    "MCPConversationQueryNoResultsPayload",
    "MCPConversationSearchHitListPayload",
    "MCPConversationSearchHitPayload",
    "MCPConversationSearchNoResultsPayload",
    "MCPConversationSummaryListPayload",
    "MCPConversationSummaryPayload",
    "MCPErrorPayload",
    "MCPFencedCodeBlock",
    "MCPMessagePayload",
    "MCPMessagesListPayload",
    "MCPMetadataPayload",
    "MCPMutationStatusPayload",
    "MutationResultPayload",
    "MCPQueryMissDiagnosticsPayload",
    "MCPQueryMissReasonPayload",
    "MCPRawArtifactPayload",
    "MCPRawArtifactsListPayload",
    "MCPReadinessCheckPayload",
    "MCPReadinessReportPayload",
    "MCPRootPayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
    "conversation_neighbor_candidate_list_payload",
    "conversation_query_result_payload",
    "conversation_search_hit_list_payload",
    "conversation_search_result_payload",
    "conversation_summary_list_payload",
    "model_json_document",
    "normalize_role",
]
