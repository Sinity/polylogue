"""Typed models for static-site generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.logging import get_logger
from polylogue.paths import safe_path_component
from polylogue.types import SearchProvider

logger = get_logger(__name__)


def format_summary_date(value: object, fmt: str, summary_id: str) -> str | None:
    """Format a summary timestamp for stable site display."""
    if value is None:
        return None
    try:
        return value.strftime(fmt)
    except (AttributeError, ValueError) as exc:
        logger.debug("Timestamp format error for %s: %s", summary_id, exc)
        return str(value)[:10] if fmt == "%Y-%m-%d" else str(value)


@dataclass
class SiteConfig:
    """Configuration for static site generation."""

    title: str = "Polylogue Archive"
    description: str = "AI conversation archive"
    enable_search: bool = True
    search_provider: SearchProvider = SearchProvider.PAGEFIND
    conversations_per_page: int = 100
    include_dashboard: bool = True


@dataclass
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
    def from_summary(cls, summary: object, message_count: int) -> ConversationIndex:
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


@dataclass
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
