"""Shared display/title/tags mixin for Session and SessionSummary models.

@owner archive-session
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import DisplayLabelSource, TitleSource
from polylogue.core.types import SessionId

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
    - display_name: str | None
    - display_label: str | None
    - display_label_source: DisplayLabelSource | None
    - title_source: TitleSource | None
    """

    id: SessionId
    title: str | None
    display_label: str | None
    display_label_source: DisplayLabelSource | None
    title_source: TitleSource | None
    created_at: datetime | None
    updated_at: datetime | None
    metadata: dict[str, object]
    parent_id: SessionId | None
    branch_type: BranchType | None
    display_name: str | None

    @property
    def display_date(self) -> datetime | None:
        """Return the session's display date (updated or created)."""
        return self.updated_at or self.created_at

    @property
    def user_title(self) -> str | None:
        """Return the user-provided title from metadata."""
        return _metadata_string(self.metadata, "title")

    @property
    def explicit_display_title(self) -> str | None:
        """Return title-worthy evidence, or ``None`` when the session has none.

        Callers that render a bounded column use this and supply their own
        identity fallback: a fallback derived here would see one session and
        so could not stay distinct from its siblings.
        """
        user_title = self.user_title
        if user_title:
            return user_title
        if self.display_label:
            return self.display_label
        # polylogue-4p1.6: a HEURISTIC title is a prompt echo the parser
        # already recognized as one. It is stored evidence, not title-worthy
        # evidence -- rendering it republishes the user's own opening message
        # as if the provider had named the session. Reads that can compose a
        # structural label supply it as ``display_label`` above; reads that
        # cannot fall through to the identity fallback rather than to the echo.
        if self.title and self.title_source is not TitleSource.HEURISTIC:
            return self.title
        # polylogue-cgfy: provider-assigned display name (e.g. Claude Code's
        # slug, "greedy-squishing-hamming") is title-worthy evidence.
        return self.display_name or None

    @property
    def display_title(self) -> str:
        """Return the read-time display label, preserving provider titles."""
        return self.explicit_display_title or str(self.id)

    @property
    def display_title_is_synthesized(self) -> bool:
        """Whether ``display_title`` was composed here rather than asserted.

        A synthesized label must never read as a stored title. ``True`` means
        no user, provider title or provider-assigned name produced the string;
        the archive composed it from structural evidence -- which is the case
        for a session whose stored title is a recognized prompt echo
        (``TitleSource.HEURISTIC``), where the echo is deliberately not
        display authority.

        The session-id fallback also counts as synthesized: it is likewise a
        string no provider asserted as a name.
        """
        if self.user_title:
            return False
        if self.display_label:
            return self.display_label_source is DisplayLabelSource.SYNTHESIZED
        if self.title and self.title_source is not TitleSource.HEURISTIC:
            return False
        return not self.display_name

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
