"""Typed MCP payload models shared by server tools and resources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import RootModel
from typing_extensions import TypedDict

from polylogue.schemas.json_types import JSONDocument
from polylogue.surface_payloads import (
    ConversationDetailPayload as MCPConversationDetailPayload,
)
from polylogue.surface_payloads import (
    ConversationMessagePayload as MCPMessagePayload,
)
from polylogue.surface_payloads import (
    ConversationSummaryPayload as MCPConversationSummaryPayload,
)
from polylogue.surface_payloads import (
    SurfacePayloadModel,
    model_json_document,
    normalize_role,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.lib.models import Conversation
    from polylogue.lib.stats import ArchiveStats
    from polylogue.readiness import ReadinessCheck, ReadinessReport

TRoot = TypeVar("TRoot")


class MCPRootPayload(RootModel[TRoot], Generic[TRoot]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPErrorPayload(SurfacePayloadModel):
    error: str
    tool: str | None = None
    conversation_id: str | None = None


class MCPFencedCodeBlock(TypedDict):
    language: str
    code: str


class MCPConversationSummaryListPayload(MCPRootPayload[list[MCPConversationSummaryPayload]]):
    root: list[MCPConversationSummaryPayload]


def conversation_summary_list_payload(
    conversations: Sequence[Conversation],
) -> MCPConversationSummaryListPayload:
    return MCPConversationSummaryListPayload(
        root=[MCPConversationSummaryPayload.from_conversation(conv) for conv in conversations]
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
    "MCPConversationDetailPayload",
    "MCPConversationSummaryListPayload",
    "MCPConversationSummaryPayload",
    "MCPErrorPayload",
    "MCPFencedCodeBlock",
    "MCPReadinessCheckPayload",
    "MCPReadinessReportPayload",
    "MCPMessagePayload",
    "MCPMetadataPayload",
    "MCPRootPayload",
    "MCPMutationStatusPayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
    "conversation_summary_list_payload",
    "model_json_document",
    "normalize_role",
]
