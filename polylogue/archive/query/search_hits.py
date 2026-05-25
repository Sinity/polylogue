"""Evidence-bearing search-hit contracts and execution over query plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    """A conversation summary plus evidence explaining why it matched.

    ``score_kind`` declares how to interpret ``score``:

    - ``"bm25"`` — SQLite FTS5 BM25 (lower magnitude is a better match;
      values are typically negative; never compare across queries).
    - ``"rrf"`` — Reciprocal Rank Fusion (higher is better; bounded by
      ``sum(1/(k+1))`` across lanes).
    - ``"vector_distance"`` — vector cosine distance (lower is closer).
    - ``None`` — no rank-derived score, e.g. action or attachment lanes
      that surface evidence without a numeric score.
    """

    summary: ConversationSummary
    rank: int
    retrieval_lane: str
    match_surface: str
    message_id: str | None = None
    snippet: str | None = None
    score: float | None = None
    matched_terms: tuple[str, ...] = ()
    score_components: dict[str, float] = field(default_factory=dict)
    score_kind: str | None = None
    lane_rank: int | None = None
    lane_contribution: float | None = None
    raw_score: float | None = None

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
    matched_terms: tuple[str, ...] = (),
    score_components: dict[str, float] | None = None,
    score_kind: str | None = None,
    lane_rank: int | None = None,
    lane_contribution: float | None = None,
    raw_score: float | None = None,
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
        matched_terms=matched_terms,
        score_components=score_components or {},
        score_kind=score_kind or default_score_kind(retrieval_lane),
        lane_rank=lane_rank,
        lane_contribution=lane_contribution,
        raw_score=raw_score,
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
    matched_terms: tuple[str, ...] = (),
    score_components: dict[str, float] | None = None,
    score_kind: str | None = None,
    lane_rank: int | None = None,
    lane_contribution: float | None = None,
    raw_score: float | None = None,
) -> ConversationSearchHit:
    return ConversationSearchHit(
        summary=summary,
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface=match_surface,
        message_id=message_id,
        snippet=snippet,
        score=score,
        matched_terms=matched_terms,
        score_components=score_components or {},
        score_kind=score_kind or default_score_kind(retrieval_lane),
        lane_rank=lane_rank,
        lane_contribution=lane_contribution,
        raw_score=raw_score,
    )


_HYBRID_RRF_K = 60


def _hybrid_score_components(
    lane_info: dict[str, int | None],
) -> tuple[dict[str, float], float | None]:
    """Expand a per-lane rank map into RRF-explanation components.

    Returns ``(score_components, fused_score)`` where ``score_components``
    contains ``<lane>_rank`` and ``<lane>_rrf`` entries for every lane that
    contributed, and ``fused_score`` is the sum of those RRF contributions
    (``None`` when no lane contributed). The constant ``k=60`` matches
    :func:`polylogue.storage.search_providers.hybrid.reciprocal_rank_fusion`.
    """
    components: dict[str, float] = {}
    fused = 0.0
    any_lane = False
    for lane_name, lane_rank_val in lane_info.items():
        if lane_rank_val is None:
            continue
        any_lane = True
        lane_rank = int(lane_rank_val)
        components[f"{lane_name}_rank"] = float(lane_rank)
        contribution = 1.0 / (_HYBRID_RRF_K + lane_rank)
        components[f"{lane_name}_rrf"] = contribution
        fused += contribution
    return components, (fused if any_lane else None)


def primary_lane_evidence(score_components: dict[str, float]) -> tuple[int | None, float | None]:
    """Return the strongest lane rank/contribution from RRF components."""
    best_rank: int | None = None
    best_contribution: float | None = None
    for key, value in score_components.items():
        if not key.endswith("_rank"):
            continue
        lane = key.removesuffix("_rank")
        contribution = score_components.get(f"{lane}_rrf")
        rank = int(value)
        if contribution is None:
            continue
        if best_contribution is None or contribution > best_contribution:
            best_rank = rank
            best_contribution = contribution
    return best_rank, best_contribution


def default_score_kind(retrieval_lane: str) -> str | None:
    """Map a retrieval lane to the natural ``score`` interpretation.

    Lanes that do not produce a single numeric score (``actions``,
    ``attachment``) return ``None`` so consumers know not to render or
    compare a numeric score for those evidence types.
    """
    if retrieval_lane in {"dialogue", "auto"}:
        return "bm25"
    if retrieval_lane == "hybrid":
        return "rrf"
    if retrieval_lane == "semantic":
        return "vector_distance"
    return None


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
        source_names = list(provider_values(plan.providers)) or None
        limit = plan.limit or search_limit(plan)
        return await repository.search_summary_hits(
            query_text,
            limit=limit,
            providers=source_names,
            since=plan.since.isoformat() if plan.since else None,
        )

    # Hybrid lane: use direct search to preserve per-lane rank contributions
    if plan.retrieval_lane == "hybrid":
        from polylogue.archive.query.retrieval_search import search_hybrid_results as _search_hybrid

        limit = plan.limit or search_limit(plan)
        conversations, lane_ranks = await _search_hybrid(plan, repository, limit=limit)
        query_terms = (query_text,)
        terms = search_terms(query_terms)
        hits: list[ConversationSearchHit] = []
        for rank, conversation in enumerate(conversations, start=1):
            conv_id = str(conversation.id)
            lane_info = lane_ranks.get(conv_id, {})
            score_components, fused_score = _hybrid_score_components(lane_info)
            lane_rank, lane_contribution = primary_lane_evidence(score_components)
            hits.append(
                conversation_search_hit_from_conversation(
                    conversation,
                    query_terms=query_terms,
                    rank=rank,
                    retrieval_lane="hybrid",
                    score=fused_score,
                    score_kind="rrf" if fused_score is not None else None,
                    score_components=score_components,
                    matched_terms=terms,
                    lane_rank=lane_rank,
                    lane_contribution=lane_contribution,
                    raw_score=fused_score,
                )
            )
        return hits

    conversations = await plan.list(repository)
    retrieval_lane = _resolved_retrieval_lane(plan)
    query_terms = (query_text,)
    terms = search_terms(query_terms)
    return [
        conversation_search_hit_from_conversation(
            conversation,
            query_terms=query_terms,
            rank=rank,
            retrieval_lane=retrieval_lane,
            matched_terms=terms,
        )
        for rank, conversation in enumerate(conversations, start=1)
    ]


__all__ = [
    "ConversationSearchHit",
    "build_search_snippet",
    "conversation_search_hit_from_conversation",
    "conversation_search_hit_from_summary",
    "default_score_kind",
    "plan_has_search_hit_evidence",
    "primary_lane_evidence",
    "search_hit_surface",
    "search_hits_for_plan",
    "search_query_text",
    "search_terms",
]
