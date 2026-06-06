"""Typed search result models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SearchHit:
    session_id: str
    source_name: str | None
    message_id: str
    title: str | None
    timestamp: str | None
    snippet: str
    session_url: str


@dataclass
class SearchResult:
    hits: list[SearchHit]


@dataclass(frozen=True)
class SessionSearchIdHit:
    session_id: str
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class SessionSearchEvidenceRow:
    """Storage-level ranked evidence row before archive summary hydration."""

    session_id: str
    rank: int
    score: float | None = None
    message_id: str | None = None
    snippet: str | None = None
    match_surface: str = "message"
    retrieval_lane: str = "dialogue"
    matched_terms: tuple[str, ...] = ()
    score_components: dict[str, float] | None = None
    lane_rank: int | None = None
    lane_contribution: float | None = None
    raw_score: float | None = None
    # Identifies the meaning of ``score``: ``"bm25"`` (lower is better),
    # ``"rrf"`` (higher is better), ``"vector_distance"`` (lower is closer),
    # or ``None`` when no ranking applied.
    score_kind: str | None = None


@dataclass(frozen=True)
class SessionSearchResult:
    hits: list[SessionSearchIdHit]

    @classmethod
    def from_ids(cls, session_ids: Sequence[str]) -> SessionSearchResult:
        return cls(
            hits=[
                SessionSearchIdHit(session_id=session_id, rank=rank)
                for rank, session_id in enumerate(session_ids, start=1)
            ]
        )

    def session_ids(self) -> list[str]:
        return [hit.session_id for hit in self.hits]


__all__ = [
    "SessionSearchEvidenceRow",
    "SessionSearchIdHit",
    "SessionSearchResult",
    "SearchHit",
    "SearchResult",
]
