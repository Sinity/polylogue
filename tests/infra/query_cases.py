"""Reusable semantic query cases for archive-scenario tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArchiveQueryCase:
    """Expected conversation-id projection for one archive query."""

    name: str
    expected_ids: tuple[str, ...]
    provider: str | None = None
    search_text: str | None = None

    @property
    def expected_id_set(self) -> set[str]:
        return set(self.expected_ids)


__all__ = ["ArchiveQueryCase"]
