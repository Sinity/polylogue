"""Archive operations value types and re-exports shared across surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.insights.archive import ArchiveDebtInsight
from polylogue.insights.tool_usage import build_tool_usage_insight
from polylogue.operations.completion_aggregates import CompletionAggregate

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session


class ArchiveStats:
    """Statistics about the archive for the public facade surface."""

    def __init__(
        self,
        session_count: int,
        message_count: int,
        word_count: int,
        origins: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
        recent: list[Session],
    ) -> None:
        self.session_count = session_count
        self.message_count = message_count
        self.word_count = word_count
        self.origins = origins
        self.tags = tags
        self.last_sync = last_sync
        self.recent = recent

    def __repr__(self) -> str:
        return (
            f"ArchiveStats(sessions={self.session_count}, "
            f"messages={self.message_count}, origins={list(self.origins.keys())})"
        )


__all__ = [
    "ArchiveDebtInsight",
    "ArchiveStats",
    "CompletionAggregate",
    "build_tool_usage_insight",
]
