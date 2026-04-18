"""Runtime behavior helpers for ``ConversationSummary`` models."""

from __future__ import annotations

from datetime import datetime

from polylogue.lib.branch_type import BranchType
from polylogue.types import ConversationId


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _metadata_tags(metadata: dict[str, object]) -> list[str]:
    raw_tags = metadata.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags]


class ConversationSummaryRuntimeMixin:
    id: ConversationId
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None
    metadata: dict[str, object]
    parent_id: ConversationId | None
    branch_type: BranchType | None

    @property
    def display_date(self) -> datetime | None:
        return self.updated_at or self.created_at

    @property
    def display_title(self) -> str:
        user_title = _metadata_string(self.metadata, "title")
        if user_title:
            return user_title
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def tags(self) -> list[str]:
        return _metadata_tags(self.metadata)

    @property
    def summary(self) -> str | None:
        return _metadata_string(self.metadata, "summary")

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
