"""Runtime behavior helpers for ``ConversationSummary`` models."""

from __future__ import annotations

from datetime import datetime

from polylogue.lib.conversation.branch_type import BranchType
from polylogue.lib.tail_overlay import TailOverlayInfo, tail_overlay_from_provider_meta
from polylogue.types import ConversationId


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _provider_meta_string(provider_meta: dict[str, object] | None, key: str) -> str | None:
    if provider_meta is None:
        return None
    value = provider_meta.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    provider_meta: dict[str, object] | None
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
        provider_label = _provider_meta_string(self.provider_meta, "display_label")
        if provider_label:
            return provider_label
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def tags(self) -> list[str]:
        return _metadata_tags(self.metadata)

    @property
    def tail_overlay(self) -> TailOverlayInfo | None:
        return tail_overlay_from_provider_meta(self.provider_meta)

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
