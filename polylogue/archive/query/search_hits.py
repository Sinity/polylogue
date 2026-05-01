"""Evidence-bearing search-hit contracts and execution over query plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polylogue.archive.query.fields import plan_has_fields_matching
from polylogue.archive.query.retrieval import search_limit
from polylogue.archive.query.retrieval_search import search_query_text as plan_search_query_text
from polylogue.archive.query.support import conversation_to_summary, provider_values

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation, ConversationSummary
    from polylogue.archive.query.plan import ConversationQueryPlan
    from polylogue.protocols import ConversationQueryRuntimeStore


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


def plan_has_search_hit_evidence(plan: ConversationQueryPlan) -> bool:
    return bool(plan.fts_terms or plan.similar_text)


def _simple_message_hit_plan(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.fts_terms
        and plan.retrieval_lane in {"auto", "dialogue"}
        and not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_simple_message_hit)
    )


def _resolved_retrieval_lane(plan: ConversationQueryPlan) -> str:
    if plan.similar_text:
        return "semantic"
    if plan.retrieval_lane == "auto":
        return "dialogue"
    return plan.retrieval_lane


async def search_hits_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> list[ConversationSearchHit]:
    """Return evidence-bearing hits for search-like query plans."""
    if not plan_has_search_hit_evidence(plan):
        return []

    query_text = plan.similar_text or plan_search_query_text(plan)
    if not query_text:
        return []

    if _simple_message_hit_plan(plan):
        provider_names = list(provider_values(plan.providers)) or None
        limit = plan.limit or search_limit(plan)
        return await repository.search_summary_hits(
            query_text,
            limit=limit,
            providers=provider_names,
            since=plan.since.isoformat() if plan.since else None,
        )

    conversations = await plan.list(repository)
    retrieval_lane = _resolved_retrieval_lane(plan)
    query_terms = (query_text,)
    return [
        conversation_search_hit_from_conversation(
            conversation,
            query_terms=query_terms,
            rank=rank,
            retrieval_lane=retrieval_lane,
        )
        for rank, conversation in enumerate(conversations, start=1)
    ]


__all__ = [
    "ConversationSearchHit",
    "build_search_snippet",
    "conversation_search_hit_from_conversation",
    "conversation_search_hit_from_summary",
    "plan_has_search_hit_evidence",
    "search_hit_surface",
    "search_hits_for_plan",
    "search_query_text",
    "search_terms",
]
