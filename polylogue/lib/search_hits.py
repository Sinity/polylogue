"""Evidence-bearing search-hit contracts shared by CLI, MCP, and archive ops."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polylogue.lib.query.support import conversation_to_summary

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary


@dataclass(frozen=True, slots=True)
class ConversationSearchHit:
    """A conversation summary plus evidence explaining why it matched."""

    summary: ConversationSummary
    rank: int
    retrieval_lane: str
    match_surface: str
    message_id: str | None = None
    snippet: str | None = None
    score: float | None = None

    @property
    def conversation_id(self) -> str:
        return str(self.summary.id)

    def with_message_count(self, message_count: int | None) -> ConversationSearchHit:
        return replace(self, summary=self.summary.model_copy(update={"message_count": message_count}))


def search_query_text(query_terms: tuple[str, ...]) -> str:
    return " ".join(term.strip() for term in query_terms if term.strip()).strip()


def search_terms(query_terms: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    for query_term in query_terms:
        terms.extend(term.lower() for term in query_term.split() if term.strip())
    return tuple(terms)


def build_search_snippet(text: str, query_terms: tuple[str, ...]) -> str:
    """Create a deterministic snippet around the earliest query-term match."""
    if not text:
        return ""

    lowered = text.lower()
    positions = [lowered.find(term) for term in search_terms(query_terms) if lowered.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - 60)
    end = min(len(text), anchor + 140)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def search_hit_surface(retrieval_lane: str) -> str:
    if retrieval_lane == "actions":
        return "action"
    if retrieval_lane == "hybrid":
        return "hybrid"
    if retrieval_lane == "semantic":
        return "semantic"
    return "message"


def conversation_search_hit_from_conversation(
    conversation: Conversation,
    *,
    query_terms: tuple[str, ...],
    rank: int,
    retrieval_lane: str,
    match_surface: str | None = None,
    score: float | None = None,
) -> ConversationSearchHit:
    terms = search_terms(query_terms)
    matching_message = next(
        (
            message
            for message in conversation.messages
            if message.text and any(term in message.text.lower() for term in terms)
        ),
        next((message for message in conversation.messages if message.text), None),
    )
    snippet = build_search_snippet(matching_message.text or "", query_terms) if matching_message else None
    return ConversationSearchHit(
        summary=conversation_to_summary(conversation),
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface=match_surface or search_hit_surface(retrieval_lane),
        message_id=str(matching_message.id) if matching_message else None,
        snippet=snippet,
        score=score,
    )


def conversation_search_hit_from_summary(
    summary: ConversationSummary,
    *,
    rank: int,
    retrieval_lane: str,
    match_surface: str,
    message_id: str | None,
    snippet: str | None,
    score: float | None = None,
) -> ConversationSearchHit:
    return ConversationSearchHit(
        summary=summary,
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface=match_surface,
        message_id=message_id,
        snippet=snippet,
        score=score,
    )


__all__ = [
    "ConversationSearchHit",
    "build_search_snippet",
    "conversation_search_hit_from_conversation",
    "conversation_search_hit_from_summary",
    "search_hit_surface",
    "search_query_text",
    "search_terms",
]
