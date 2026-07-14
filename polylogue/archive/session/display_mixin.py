"""Shared display/title/tags mixin for Session and SessionSummary models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.archive.session.branch_type import BranchType
from polylogue.types import SessionId

if TYPE_CHECKING:
    pass


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    """Extract a string value from metadata by key."""
    value = metadata.get(key)
    return str(value) if value is not None else None


def _metadata_tags(metadata: dict[str, object]) -> list[str]:
    """Extract tags from metadata, defaulting to empty list."""
    raw_tags = metadata.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags]


class DisplayTitleTagsMixin:
    """Shared mixin for display_title, tags, and summary properties.

    Classes using this mixin must provide:
    - id: SessionId
    - title: str | None
    - created_at: datetime | None
    - updated_at: datetime | None
    - metadata: dict[str, object]
    - parent_id: SessionId | None
    - branch_type: BranchType | None
    """

    id: SessionId
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None
    metadata: dict[str, object]
    parent_id: SessionId | None
    branch_type: BranchType | None

    @property
    def display_date(self) -> datetime | None:
        """Return the session's display date (updated or created)."""
        return self.updated_at or self.created_at

    @property
    def user_title(self) -> str | None:
        """Return the user-provided title from metadata."""
        return _metadata_string(self.metadata, "title")

    @property
    def display_title(self) -> str:
        """Return the display title with precedence: user_title > title > id[:8]."""
        user_title = self.user_title
        if user_title:
            return user_title
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def summary(self) -> str | None:
        """Return the summary from metadata."""
        return _metadata_string(self.metadata, "summary")

    @property
    def tags(self) -> list[str]:
        """Return tags with precedence: M2M-hydrated tags > metadata tags.

        See #1240: M2M-sourced tags are authoritative once hydrated.
        """
        m2m = getattr(self, "tags_m2m", None)
        if m2m:
            return list(m2m)
        return _metadata_tags(self.metadata)

    @property
    def is_continuation(self) -> bool:
        """Return whether this session is a continuation."""
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        """Return whether this session is a sidechain."""
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        """Return whether this session is a root (has no parent)."""
        return self.parent_id is None


__all__ = ["DisplayTitleTagsMixin", "_metadata_string", "_metadata_tags"]
