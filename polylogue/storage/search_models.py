"""Typed search result models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchHit:
    conversation_id: str
    provider_name: str
    source_name: str | None
    message_id: str
    title: str | None
    timestamp: str | None
    snippet: str
    conversation_path: Path


@dataclass
class SearchResult:
    hits: list[SearchHit]


@dataclass(frozen=True)
class ConversationSearchHit:
    conversation_id: str
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class ConversationSearchEvidenceHit:
    conversation_id: str
    rank: int
    score: float | None = None
    message_id: str | None = None
    snippet: str | None = None
    match_surface: str = "message"
    retrieval_lane: str = "dialogue"


@dataclass(frozen=True)
class ConversationSearchResult:
    hits: list[ConversationSearchHit]

    @classmethod
    def from_ids(cls, conversation_ids: Sequence[str]) -> ConversationSearchResult:
        return cls(
            hits=[
                ConversationSearchHit(conversation_id=conversation_id, rank=rank)
                for rank, conversation_id in enumerate(conversation_ids, start=1)
            ]
        )

    def conversation_ids(self) -> list[str]:
        return [hit.conversation_id for hit in self.hits]


__all__ = [
    "ConversationSearchEvidenceHit",
    "ConversationSearchHit",
    "ConversationSearchResult",
    "SearchHit",
    "SearchResult",
]
