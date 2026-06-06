"""Reusable semantic query cases for archive-scenario tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArchiveQueryCase:
    """Expected session-id projection for one archive query.

    A case may carry one primary projection axis (``provider`` or
    ``search_text``) plus optional additional filter axes
    (``since``/``until``/``limit``/``offset`` and stats-join filters
    such as ``has_tool_use``).  Adapter surfaces translate the case into
    the equivalent call shape for their entrypoint so cross-surface
    parity is asserted on the same filter chain rather than near-misses.
    """

    name: str
    expected_ids: tuple[str, ...]
    provider: str | None = None
    search_text: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = None
    offset: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None

    @property
    def expected_id_set(self) -> set[str]:
        return set(self.expected_ids)

    @property
    def has_extended_filters(self) -> bool:
        """Whether the case carries filter axes beyond the primary projection."""
        return (
            self.since is not None
            or self.until is not None
            or self.limit is not None
            or self.offset != 0
            or self.has_tool_use
            or self.has_thinking
            or self.min_messages is not None
            or self.max_messages is not None
            or self.min_words is not None
        )


__all__ = ["ArchiveQueryCase"]
