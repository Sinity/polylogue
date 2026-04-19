"""Typed models for static-site generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.paths import safe_path_component
from polylogue.types import SearchProvider

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import ConversationSummary

logger = get_logger(__name__)


def format_summary_date(value: object, fmt: str, summary_id: str) -> str | None:
    """Format a summary timestamp for stable site display."""
    if value is None:
        return None
    try:
        strftime = getattr(value, "strftime", None)
        if not callable(strftime):
            raise AttributeError("value has no strftime()")
        formatted = strftime(fmt)
        return formatted if isinstance(formatted, str) else str(formatted)
    except (AttributeError, ValueError) as exc:
        logger.debug("Timestamp format error for %s: %s", summary_id, exc)
        return str(value)[:10] if fmt == "%Y-%m-%d" else str(value)


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Configuration for static site generation."""

    title: str = "Polylogue Archive"
    description: str = "AI conversation archive"
    enable_search: bool = True
    search_provider: SearchProvider = SearchProvider.PAGEFIND
    conversations_per_page: int = 100
    include_dashboard: bool = True

    @property
    def search_provider_name(self) -> str | None:
        """Return the configured search backend name when search is enabled."""
        return str(self.search_provider) if self.enable_search else None

    @property
    def uses_pagefind(self) -> bool:
        """Whether this site build should produce Pagefind assets."""
        return self.enable_search and self.search_provider is SearchProvider.PAGEFIND

    def to_payload(self) -> dict[str, object]:
        """Return a stable manifest-friendly config payload."""
        return {
            "title": self.title,
            "description": self.description,
            "enable_search": self.enable_search,
            "search_provider": str(self.search_provider),
            "conversations_per_page": self.conversations_per_page,
            "include_dashboard": self.include_dashboard,
        }


@dataclass(frozen=True, slots=True)
class ConversationIndex:
    """Indexed conversation for site generation."""

    id: str
    title: str
    provider: str
    created_at: str | None
    updated_at: str | None
    message_count: int
    preview: str
    path: str

    @classmethod
    def from_summary(cls, summary: ConversationSummary, message_count: int) -> ConversationIndex:
        sid = str(summary.id)
        provider = getattr(summary.provider, "value", str(summary.provider))
        return cls(
            id=sid,
            title=summary.display_title or sid[:12],
            provider=provider,
            created_at=format_summary_date(summary.created_at, "%Y-%m-%d", sid),
            updated_at=format_summary_date(summary.updated_at, "%Y-%m-%d %H:%M", sid),
            message_count=message_count,
            preview=summary.summary or "",
            path=f"{safe_path_component(provider, fallback='provider')}/{sid[:12]}/conversation.html",
        )


@dataclass(frozen=True, slots=True)
class SearchDocument:
    """Compact search payload for one indexed conversation."""

    id: str
    title: str
    provider: str
    preview: str
    path: str

    @classmethod
    def from_conversation(cls, conversation: ConversationIndex) -> SearchDocument:
        return cls(
            id=conversation.id,
            title=conversation.title,
            provider=conversation.provider,
            preview=conversation.preview,
            path=conversation.path,
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "provider": self.provider,
            "preview": self.preview,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class ProviderIndex:
    """Provider-scoped aggregate row for index and dashboard pages."""

    name: str
    conversation_count: int
    message_count: int
    path: str

    @classmethod
    def from_counts(
        cls,
        provider: str,
        *,
        conversation_count: int,
        message_count: int,
    ) -> ProviderIndex:
        return cls(
            name=provider,
            conversation_count=conversation_count,
            message_count=message_count,
            path=f"{safe_path_component(provider, fallback='provider')}/index.html",
        )


@dataclass
class ArchiveIndexStats:
    """Streaming archive aggregates used by site-generation surfaces."""

    root_page_conversations: list[ConversationIndex] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=dict)
    provider_messages: dict[str, int] = field(default_factory=dict)
    provider_order: list[str] = field(default_factory=list)
    total_conversations: int = 0
    total_messages: int = 0

    def record(self, conversation: ConversationIndex, *, root_page_size: int) -> None:
        """Accumulate counters and the root index first page in scan order."""
        if len(self.root_page_conversations) < root_page_size:
            self.root_page_conversations.append(conversation)

        self.total_conversations += 1
        self.total_messages += conversation.message_count
        if conversation.provider not in self.provider_counts:
            self.provider_order.append(conversation.provider)
        self.provider_counts[conversation.provider] = self.provider_counts.get(conversation.provider, 0) + 1
        self.provider_messages[conversation.provider] = (
            self.provider_messages.get(conversation.provider, 0) + conversation.message_count
        )

    def provider_indexes(self) -> tuple[ProviderIndex, ...]:
        """Return provider aggregates in stable archive order."""
        return tuple(
            ProviderIndex.from_counts(
                provider,
                conversation_count=self.provider_counts[provider],
                message_count=self.provider_messages.get(provider, 0),
            )
            for provider in self.provider_order
        )


@dataclass(slots=True)
class ConversationPageBuildStats:
    """Conversation-page materialization counts for one site build."""

    total: int = 0
    rendered: int = 0
    reused: int = 0
    failed: int = 0

    def record(self, status: str) -> None:
        self.total += 1
        if status == "rendered":
            self.rendered += 1
        elif status == "reused":
            self.reused += 1
        elif status == "failed":
            self.failed += 1

    def to_payload(self) -> dict[str, int]:
        return {
            "total": self.total,
            "rendered": self.rendered,
            "reused": self.reused,
            "failed": self.failed,
        }
