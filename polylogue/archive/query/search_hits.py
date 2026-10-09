"""Evidence-bearing search-hit contracts and execution over query plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from polylogue.archive.query.retrieval import search_limit
from polylogue.archive.query.retrieval import search_query_text as plan_search_query_text
from polylogue.archive.query.search_contract import SearchExecution
from polylogue.archive.query.search_cursor import SearchPosition
from polylogue.archive.query.support import session_to_summary

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.search_contract import ArchiveSearchResult
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.config import Config
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary

DEFAULT_SEARCH_SNIPPET_MAX_CHARS = 320
DEFAULT_TITLE_MAX_CHARS = 96


@dataclass(frozen=True, slots=True)
class SessionSearchHit:
    """A session summary plus evidence explaining why it matched.

    ``score_kind`` declares how to interpret ``score``:

    - ``"bm25"`` — SQLite FTS5 BM25 (lower magnitude is a better match;
      values are typically negative; never compare across queries).
    - ``"rrf"`` — Reciprocal Rank Fusion (higher is better; bounded by
      ``sum(1/(k+1))`` across lanes).
    - ``"vector_distance"`` — vector L2 distance (lower is closer).
    - ``None`` — no rank-derived score, e.g. action or attachment lanes
      that surface evidence without a numeric score.
    """

    summary: SessionSummary
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
    block_id: str | None = None
    position: SearchPosition | None = None

    @property
    def session_id(self) -> str:
        return str(self.summary.id)

    def with_message_count(self, message_count: int | None) -> SessionSearchHit:
        return replace(self, summary=self.summary.model_copy(update={"message_count": message_count}))


class SearchHitResults(list[SessionSearchHit]):
    """List-compatible hits carrying canonical lane execution evidence."""

    def __init__(self, hits: list[SessionSearchHit], execution: SearchExecution) -> None:
        super().__init__(hits)
        self.execution = execution


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


def _bound_normalized_text(normalized: str, *, max_chars: int) -> str:
    if max_chars <= 3:
        return normalized[:max_chars]
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def bound_search_snippet(
    snippet: str | None,
    *,
    max_chars: int = DEFAULT_SEARCH_SNIPPET_MAX_CHARS,
) -> str | None:
    """Return a display-safe snippet, never a full transcript payload."""
    if snippet is None:
        return None
    normalized = " ".join(snippet.split())
    return _bound_normalized_text(normalized, max_chars=max_chars)


def bound_display_text(value: object, *, max_chars: int = DEFAULT_TITLE_MAX_CHARS) -> str:
    """Single-line-normalize and truncate arbitrary display text.

    Shared row-projection budget primitive (polylogue-x7d): titles, short
    text previews, and non-search-hit snippets across CLI/API/MCP row
    surfaces all bound through this one truncation rule instead of each
    surface carrying its own copy that can silently drift out of sync
    (the original bug: ``format_summary_list``'s JSON title output was
    completely unbounded while its plain-text sibling truncated to 50
    chars — a giant title exploded machine-readable output even though
    the interactive table stayed legible).
    """
    normalized = " ".join(str(value).split()) if value is not None else ""
    return _bound_normalized_text(normalized, max_chars=max_chars)


def bound_display_title(value: object, fallback: object = "", *, max_chars: int = DEFAULT_TITLE_MAX_CHARS) -> str:
    """Bound a display title, falling back to ``fallback`` when empty."""
    return bound_display_text(value or fallback, max_chars=max_chars)


def search_hit_surface(retrieval_lane: str) -> str:
    if retrieval_lane == "actions":
        return "action"
    if retrieval_lane == "hybrid":
        return "hybrid"
    if retrieval_lane == "semantic":
        return "semantic"
    return "message"


def session_search_hit_from_session(
    session: Session,
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
    block_id: str | None = None,
    position: SearchPosition | None = None,
) -> SessionSearchHit:
    terms = search_terms(query_terms)
    matching_message = next(
        (
            message
            for message in session.messages
            if message.text and any(term in message.text.lower() for term in terms)
        ),
        next((message for message in session.messages if message.text), None),
    )
    snippet = build_search_snippet(matching_message.text or "", query_terms) if matching_message else None
    return SessionSearchHit(
        summary=session_to_summary(session),
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface=match_surface or search_hit_surface(retrieval_lane),
        message_id=str(matching_message.id) if matching_message else None,
        snippet=bound_search_snippet(snippet),
        score=score,
        matched_terms=matched_terms,
        score_components=score_components or {},
        score_kind=(score_kind or default_score_kind(retrieval_lane)) if match_surface != "session" else None,
        lane_rank=lane_rank,
        lane_contribution=lane_contribution,
        raw_score=raw_score,
        block_id=block_id,
        position=position,
    )


def session_search_hit_from_summary(
    summary: SessionSummary,
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
    block_id: str | None = None,
    position: SearchPosition | None = None,
) -> SessionSearchHit:
    return SessionSearchHit(
        summary=summary,
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface=match_surface,
        message_id=message_id,
        snippet=bound_search_snippet(snippet),
        score=score,
        matched_terms=matched_terms,
        score_components=score_components or {},
        score_kind=(score_kind or default_score_kind(retrieval_lane)) if match_surface != "session" else None,
        lane_rank=lane_rank,
        lane_contribution=lane_contribution,
        raw_score=raw_score,
        block_id=block_id,
        position=position,
    )


def _hybrid_score_components(
    lane_info: dict[str, int | None],
) -> tuple[dict[str, float], float | None]:
    """Expand a per-lane rank map into RRF-explanation components.

    Returns ``(score_components, fused_score)`` where ``score_components``
    contains ``<lane>_rank`` and ``<lane>_rrf`` entries for every lane that
    contributed, and ``fused_score`` is the sum of those RRF contributions
    (``None`` when no lane contributed). The constant ``k=60`` matches
    the archive SQL lane-rank settlement owner.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import HYBRID_RRF_K

    components: dict[str, float] = {}
    fused = 0.0
    any_lane = False
    for lane_name, lane_rank_val in lane_info.items():
        if lane_rank_val is None:
            continue
        any_lane = True
        lane_rank = int(lane_rank_val)
        components[f"{lane_name}_rank"] = float(lane_rank)
        contribution = 1.0 / (HYBRID_RRF_K + lane_rank)
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


def plan_has_search_hit_evidence(plan: SessionQueryPlan) -> bool:
    return bool(plan.fts_terms or plan.similar_text or plan.similar_session_id)


async def search_hits_for_plan(
    plan: SessionQueryPlan,
    config: Config,
) -> SearchHitResults:
    """Return the native ranked read and its lane evidence through one projection."""
    from polylogue.archive.query.archive_execution import archive_search_hits
    from polylogue.archive.query.transaction import run_archive_read
    from polylogue.storage.archive_identity import archive_file_set_root

    archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
    result = await run_archive_read(
        archive_root,
        operation="archive.query.search-hits-for-plan",
        arguments={"plan": plan, "default_limit": plan.limit or search_limit(plan)},
        work=lambda archive: archive_search_hits(
            plan,
            archive_root=archive_root,
            config=config,
            default_limit=plan.limit or search_limit(plan),
            archive=archive,
        ),
        page_size=plan.limit or search_limit(plan),
        offset=plan.offset,
        projection="search-hits",
        workload_class="scan" if plan.limit is None or plan.limit > 1000 else "interactive",
    )
    return project_search_hits(plan, result)


def project_search_hits(
    plan: SessionQueryPlan,
    result: ArchiveSearchResult,
) -> SearchHitResults:
    """Hydrate hits without reconstructing or discarding the native lane outcome."""
    query_text = plan.similar_text or plan_search_query_text(plan)
    query_terms = (query_text,) if query_text else ()
    terms = search_terms(query_terms)
    hits: list[SessionSearchHit] = []
    for rank, (native_hit, summary) in enumerate(result.hits, start=1):
        fused_score: float | None
        primary_contribution: float | None
        if result.retrieval_lane == "hybrid":
            components, _component_score = _hybrid_score_components(native_hit.lane_ranks or {})
            fused_score = native_hit.score
            primary_rank, primary_contribution = primary_lane_evidence(components)
        else:
            components = {
                f"{lane}_rank": float(rank_value)
                for lane, rank_value in (native_hit.lane_ranks or {}).items()
                if rank_value is not None
            }
            fused_score = native_hit.score
            primary_contribution = None
            # A single-lane hit has no RRF contribution and no lane_ranks
            # mapping: its native rank is the lane rank.
            native_rank = getattr(native_hit, "rank", None)
            primary_rank = (
                int(native_rank)
                if isinstance(native_rank, (int, float)) and not isinstance(native_rank, bool)
                else min((int(value) for value in components.values()), default=None)
            )
        hits.append(
            session_search_hit_from_summary(
                _archive_summary_to_domain(summary),
                rank=native_hit.rank or rank,
                retrieval_lane=result.retrieval_lane,
                match_surface="session" if not native_hit.message_id else search_hit_surface(result.retrieval_lane),
                message_id=native_hit.message_id or None,
                block_id=native_hit.block_id,
                position=native_hit.position,
                snippet=native_hit.snippet,
                matched_terms=terms,
                score=fused_score if native_hit.message_id else None,
                score_kind="bm25" if result.retrieval_lane == "actions" else default_score_kind(result.retrieval_lane),
                score_components=components,
                raw_score=fused_score if native_hit.message_id else None,
                lane_rank=primary_rank if native_hit.message_id else None,
                lane_contribution=primary_contribution,
            )
        )
    return SearchHitResults(hits, result.execution)


def _archive_summary_to_domain(summary: object) -> SessionSummary:
    """Accept an already-hydrated summary or route a raw row through hydration."""
    from polylogue.archive.hydration import archive_summary_to_domain
    from polylogue.archive.session.domain_models import SessionSummary as _SessionSummary

    if isinstance(summary, _SessionSummary):
        return summary
    return archive_summary_to_domain(cast("ArchiveSessionSummary", summary))


__all__ = [
    "DEFAULT_SEARCH_SNIPPET_MAX_CHARS",
    "DEFAULT_TITLE_MAX_CHARS",
    "SessionSearchHit",
    "SearchHitResults",
    "bound_display_text",
    "bound_display_title",
    "bound_search_snippet",
    "build_search_snippet",
    "session_search_hit_from_session",
    "session_search_hit_from_summary",
    "default_score_kind",
    "plan_has_search_hit_evidence",
    "primary_lane_evidence",
    "search_hit_surface",
    "search_hits_for_plan",
    "project_search_hits",
    "search_query_text",
    "search_terms",
]
