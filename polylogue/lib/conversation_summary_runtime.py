"""Runtime behavior mixin for conversation summaries."""

from __future__ import annotations

from polylogue.lib.branch_type import BranchType


class ConversationSummaryRuntimeMixin:
    @property
    def display_date(self):
        return self.updated_at or self.created_at

    @property
    def display_title(self) -> str:
        user_title = self.metadata.get("title")
        if user_title:
            return str(user_title)
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def tags(self) -> list[str]:
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        return []

    @property
    def summary(self) -> str | None:
        summary = self.metadata.get("summary")
        return str(summary) if summary is not None else None

    @property
    def is_continuation(self) -> bool:
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


__all__ = ["ConversationSummaryRuntimeMixin"]
