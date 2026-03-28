"""Typed MCP payload models shared by server tools and resources."""

from __future__ import annotations

<<<<<<< HEAD
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel

from polylogue.lib.models import Conversation, ConversationSummary


def _normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if hasattr(role, "value"):
        role = role.value
    return str(role)


class MCPPayload(BaseModel):
    """Base model for JSON payloads returned by MCP surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPRootPayload(RootModel[Any]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPErrorPayload(MCPPayload):
    error: str
    tool: str | None = None
    conversation_id: str | None = None


class MCPMessagePayload(MCPPayload):
    id: str
    role: str
    text: str
    timestamp: datetime | None = None

    @classmethod
    def from_message(cls, message: Any) -> MCPMessagePayload:
        return cls(
            id=str(message.id),
            role=_normalize_role(message.role),
            text=message.text or "",
            timestamp=message.timestamp,
        )


class MCPConversationSummaryPayload(MCPPayload):
    id: str
    provider: str
    title: str
    message_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationSummaryPayload:
        return cls(
            id=str(conversation.id),
            provider=conversation.provider,
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int | None = None,
    ) -> MCPConversationSummaryPayload:
        return cls(
            id=str(summary.id),
            provider=summary.provider,
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class MCPConversationDetailPayload(MCPConversationSummaryPayload):
    messages: list[MCPMessagePayload]

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationDetailPayload:
        summary = MCPConversationSummaryPayload.from_conversation(conversation)
        return cls(
            **summary.model_dump(),
            messages=[MCPMessagePayload.from_message(msg) for msg in conversation.messages],
        )


class MCPConversationSummaryListPayload(MCPRootPayload):
    root: list[MCPConversationSummaryPayload]


class MCPArchiveStatsPayload(MCPPayload):
    total_conversations: int
    total_messages: int
    providers: dict[str, int]
    embedded_conversations: int | None = None
    embedded_messages: int | None = None
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


class MCPHealthCheckPayload(MCPPayload):
    name: str
    status: str
    count: int | None = None
    detail: str | None = None

    @classmethod
    def from_check(cls, check: Any, *, include_counts: bool, include_detail: bool) -> MCPHealthCheckPayload:
        return cls(
            name=check.name,
            status=check.status.value if hasattr(check.status, "value") else str(check.status),
            count=check.count if include_counts else None,
            detail=check.detail if include_detail else None,
        )


class MCPHealthReportPayload(MCPPayload):
    checks: list[MCPHealthCheckPayload]
    summary: str
    cached: bool | None = None

    @classmethod
    def from_report(
        cls,
        report: Any,
        *,
        include_counts: bool,
        include_detail: bool,
        include_cached: bool,
    ) -> MCPHealthReportPayload:
        return cls(
            checks=[
                MCPHealthCheckPayload.from_check(
                    check,
                    include_counts=include_counts,
                    include_detail=include_detail,
                )
                for check in report.checks
            ],
            summary=report.summary,
            cached=getattr(report, "cached", False) if include_cached else None,
        )

||||||| parent of f5cb862b (refactor: close codebase-wide cleanup hotspots)
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel

from polylogue.lib.models import Conversation, ConversationSummary


def _normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if hasattr(role, "value"):
        role = role.value
    return str(role)


class MCPPayload(BaseModel):
    """Base model for JSON payloads returned by MCP surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPRootPayload(RootModel[Any]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPErrorPayload(MCPPayload):
    error: str
    tool: str | None = None
    conversation_id: str | None = None


class MCPMessagePayload(MCPPayload):
    id: str
    role: str
    text: str
    timestamp: datetime | None = None

    @classmethod
    def from_message(cls, message: Any) -> MCPMessagePayload:
        return cls(
            id=str(message.id),
            role=_normalize_role(message.role),
            text=message.text or "",
            timestamp=message.timestamp,
        )


class MCPConversationSummaryPayload(MCPPayload):
    id: str
    provider: str
    title: str
    message_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationSummaryPayload:
        return cls(
            id=str(conversation.id),
            provider=conversation.provider,
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int | None = None,
    ) -> MCPConversationSummaryPayload:
        return cls(
            id=str(summary.id),
            provider=summary.provider,
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class MCPConversationDetailPayload(MCPConversationSummaryPayload):
    messages: list[MCPMessagePayload]

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationDetailPayload:
        summary = MCPConversationSummaryPayload.from_conversation(conversation)
        return cls(
            **summary.model_dump(),
            messages=[MCPMessagePayload.from_message(msg) for msg in conversation.messages],
        )


class MCPConversationSummaryListPayload(MCPRootPayload):
    root: list[MCPConversationSummaryPayload]


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
            stale_embedding_messages=(
                archive_stats.stale_embedding_messages if include_embedded else None
            ),
            messages_missing_embedding_provenance=(
                archive_stats.messages_missing_embedding_provenance if include_embedded else None
            ),
            embedding_health_status=(
                archive_stats.embedding_health_status if include_embedded else None
            ),
            embedding_models=(
                archive_stats.embedding_models if include_embedded else None
            ),
            embedding_dimensions=(
                archive_stats.embedding_dimensions if include_embedded else None
            ),
            embedding_oldest_at=(
                archive_stats.embedding_oldest_at if include_embedded else None
            ),
            embedding_newest_at=(
                archive_stats.embedding_newest_at if include_embedded else None
            ),
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


class MCPHealthCheckPayload(MCPPayload):
    name: str
    status: str
    count: int | None = None
    detail: str | None = None

    @classmethod
    def from_check(cls, check: Any, *, include_counts: bool, include_detail: bool) -> MCPHealthCheckPayload:
        return cls(
            name=check.name,
            status=check.status.value if hasattr(check.status, "value") else str(check.status),
            count=check.count if include_counts else None,
            detail=check.detail if include_detail else None,
        )


class MCPHealthReportPayload(MCPPayload):
    checks: list[MCPHealthCheckPayload]
    summary: str
    source: str | None = None
    cache_age_seconds: int | None = None
    cache_ttl_seconds: int | None = None

    @classmethod
    def from_report(
        cls,
        report: Any,
        *,
        include_counts: bool,
        include_detail: bool,
        include_cached: bool,
    ) -> MCPHealthReportPayload:
        return cls(
            checks=[
                MCPHealthCheckPayload.from_check(
                    check,
                    include_counts=include_counts,
                    include_detail=include_detail,
                )
                for check in report.checks
            ],
            summary=report.summary,
            source=(
                getattr(getattr(report, "provenance", None), "source", None).value
                if include_cached and getattr(report, "provenance", None) is not None
                else None
            ),
            cache_age_seconds=(
                getattr(getattr(report, "provenance", None), "cache_age_seconds", None)
                if include_cached
                else None
            ),
            cache_ttl_seconds=(
                getattr(getattr(report, "provenance", None), "cache_ttl_seconds", None)
                if include_cached
                else None
            ),
        )

=======
from polylogue.mcp.payload_archive import (
    MCPArchiveStatsPayload,
    MCPMetadataPayload,
    MCPMutationStatusPayload,
    MCPStatsByPayload,
    MCPTagCountsPayload,
)
from polylogue.mcp.payload_base import MCPPayload, MCPRootPayload
from polylogue.mcp.payload_conversations import (
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPErrorPayload,
    MCPMessagePayload,
)
from polylogue.mcp.payload_health import MCPHealthCheckPayload, MCPHealthReportPayload
>>>>>>> f5cb862b (refactor: close codebase-wide cleanup hotspots)

__all__ = [
    "MCPArchiveStatsPayload",
    "MCPConversationDetailPayload",
    "MCPConversationSummaryListPayload",
    "MCPConversationSummaryPayload",
    "MCPErrorPayload",
    "MCPHealthCheckPayload",
    "MCPHealthReportPayload",
    "MCPMessagePayload",
    "MCPMetadataPayload",
    "MCPRootPayload",
    "MCPPayload",
    "MCPMutationStatusPayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
]
