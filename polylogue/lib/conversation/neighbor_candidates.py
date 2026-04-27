"""Explainable neighboring-conversation candidate discovery."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.conversation.models import Conversation, ConversationSummary
    from polylogue.lib.search_hits import ConversationSearchHit
    from polylogue.protocols import ConversationQueryRuntimeStore


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)
_ATTACHMENT_IDENTITY_KEYS = frozenset(("provider_id", "id", "fileId", "driveId"))
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "but",
        "can",
        "could",
        "for",
        "from",
        "have",
        "how",
        "into",
        "not",
        "our",
        "please",
        "that",
        "the",
        "this",
        "was",
        "what",
        "when",
        "with",
        "would",
        "you",
        "your",
    }
)


class NeighborDiscoveryError(ValueError):
    """Raised when a neighboring-candidate request cannot be executed."""


@dataclass(frozen=True, slots=True)
class NeighborReason:
    """Evidence explaining why a conversation was suggested as a neighbor."""

    kind: str
    detail: str
    evidence: str | None = None
    weight: float = 0.0


@dataclass(frozen=True, slots=True)
class ConversationNeighborCandidate:
    """Ranked neighboring-conversation candidate with explainable evidence."""

    summary: ConversationSummary
    rank: int
    score: float
    reasons: tuple[NeighborReason, ...]
    source_conversation_id: str | None = None
    query: str | None = None

    @property
    def conversation_id(self) -> str:
        return str(self.summary.id)

    def with_rank(self, rank: int) -> ConversationNeighborCandidate:
        return replace(self, rank=rank)

    def with_message_count(self, message_count: int | None) -> ConversationNeighborCandidate:
        if message_count is None:
            return self
        summary = self.summary.model_copy(update={"message_count": message_count})
        return replace(self, summary=summary)


@dataclass(frozen=True, slots=True)
class NeighborDiscoveryRequest:
    """Request model for neighboring-conversation candidate discovery."""

    conversation_id: str | None = None
    query: str | None = None
    provider: str | None = None
    limit: int = 10
    window_hours: int = 24
    candidate_pool_limit: int = 100
    similarity_threshold: float = 0.12


@dataclass(slots=True)
class _CandidateDraft:
    summary: ConversationSummary
    reasons: list[NeighborReason] = field(default_factory=list)
    score: float = 0.0


def _normalized_title(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.casefold().split())


def _canonical_provider(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Provider.from_string(value))


def _provider_list(value: str | None) -> list[str] | None:
    provider = _canonical_provider(value)
    return [provider] if provider else None


def _source_timestamp(conversation: Conversation) -> datetime | None:
    return conversation.updated_at or conversation.created_at


def _summary_timestamp(summary: ConversationSummary) -> datetime | None:
    return summary.updated_at or summary.created_at


def _epoch(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return value.timestamp()


def _format_hours(delta: timedelta) -> str:
    hours = abs(delta.total_seconds()) / 3600
    if hours < 1:
        return f"{round(hours * 60)}m"
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _date_window(anchor: datetime, window_hours: int) -> tuple[str, str]:
    window = timedelta(hours=max(0, window_hours))
    return (anchor - window).isoformat(), (anchor + window).isoformat()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text) if token.casefold() not in _STOPWORDS}


def _ordered_tokens(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(text):
        token = raw_token.casefold()
        if token in seen or token in _STOPWORDS:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _conversation_text(conversation: Conversation) -> str:
    return "\n".join(message.text for message in conversation.messages if message.text)


def _source_search_seed(conversation: Conversation) -> str | None:
    tokens = _ordered_tokens(_conversation_text(conversation))
    if not tokens:
        return None
    selected = sorted(tokens, key=lambda token: (-len(token), token))[:2]
    return " ".join(selected)


def _summary_text(summary: ConversationSummary) -> str:
    parts = [summary.display_title]
    if summary.summary:
        parts.append(summary.summary)
    return "\n".join(parts)


def _content_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / math.sqrt(len(left) * len(right))


def _shared_terms(left: set[str], right: set[str], *, limit: int = 6) -> tuple[str, ...]:
    return tuple(sorted(left & right)[:limit])


def _attachment_identities(conversation: Conversation) -> tuple[str, ...]:
    identities: set[str] = set()
    for message in conversation.messages:
        for attachment in message.attachments:
            if attachment.id:
                identities.add(attachment.id)
            for key, value in (attachment.provider_meta or {}).items():
                if key not in _ATTACHMENT_IDENTITY_KEYS or value is None:
                    continue
                text = str(value).strip()
                if text:
                    identities.add(text)
    return tuple(sorted(identities))


def _add_candidate_reason(
    drafts: dict[str, _CandidateDraft],
    summary: ConversationSummary,
    reason: NeighborReason,
) -> None:
    conversation_id = str(summary.id)
    draft = drafts.get(conversation_id)
    if draft is None:
        draft = _CandidateDraft(summary=summary)
        drafts[conversation_id] = draft
    reason_key = (reason.kind, reason.detail, reason.evidence)
    if not any((existing.kind, existing.detail, existing.evidence) == reason_key for existing in draft.reasons):
        draft.reasons.append(reason)
        draft.score += reason.weight


def _non_source_summaries(
    summaries: Iterable[ConversationSummary],
    *,
    source_id: str | None,
) -> Iterable[ConversationSummary]:
    for summary in summaries:
        if source_id is not None and str(summary.id) == source_id:
            continue
        yield summary


def _query_hits_reason(hit: ConversationSearchHit, query: str) -> NeighborReason:
    surface = hit.match_surface.replace("_", " ")
    detail = f"matches query via {surface}"
    return NeighborReason(
        kind="query_match",
        detail=detail,
        evidence=hit.snippet,
        weight=2.0 if hit.score is None else max(0.5, min(3.0, float(hit.score))),
    )


async def _load_target(
    store: ConversationQueryRuntimeStore,
    conversation_id: str | None,
) -> Conversation | None:
    if conversation_id is None:
        return None
    resolved = await store.resolve_id(conversation_id)
    target_id = str(resolved) if resolved is not None else conversation_id
    target = await store.get(target_id)
    if target is None:
        raise NeighborDiscoveryError(f"Conversation not found: {conversation_id}")
    return target


async def _add_same_title_candidates(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    target: Conversation,
    provider: str | None,
    source_id: str,
    pool_limit: int,
) -> None:
    title = _normalized_title(target.display_title)
    if not title:
        return
    summaries = await store.list_summaries_by_query(
        ConversationRecordQuery(
            provider=provider,
            title_contains=target.display_title,
            limit=pool_limit,
        )
    )
    for summary in _non_source_summaries(summaries, source_id=source_id):
        if _normalized_title(summary.display_title) != title:
            continue
        _add_candidate_reason(
            drafts,
            summary,
            NeighborReason(
                kind="same_title",
                detail=f"same normalized title: {summary.display_title}",
                weight=3.0,
            ),
        )


async def _add_nearby_candidates(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    target: Conversation,
    provider: str | None,
    source_id: str,
    window_hours: int,
    pool_limit: int,
) -> None:
    anchor = _source_timestamp(target)
    if anchor is None:
        return
    since, until = _date_window(anchor, window_hours)
    summaries = await store.list_summaries_by_query(
        ConversationRecordQuery(
            provider=provider,
            since=since,
            until=until,
            limit=pool_limit,
        )
    )
    window_seconds = max(1.0, float(max(1, window_hours) * 3600))
    for summary in _non_source_summaries(summaries, source_id=source_id):
        candidate_time = _summary_timestamp(summary)
        if candidate_time is None:
            continue
        delta = abs((candidate_time - anchor).total_seconds())
        closeness = max(0.1, 1.0 - (delta / window_seconds))
        _add_candidate_reason(
            drafts,
            summary,
            NeighborReason(
                kind="nearby_time",
                detail=f"within {_format_hours(candidate_time - anchor)} of source conversation",
                evidence=f"source={anchor.isoformat()} candidate={candidate_time.isoformat()}",
                weight=1.5 * closeness,
            ),
        )


async def _add_query_candidates(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    query: str | None,
    providers: list[str] | None,
    source_id: str | None,
    pool_limit: int,
) -> None:
    if not query or not query.strip():
        return
    hits = await store.search_summary_hits(query.strip(), limit=pool_limit, providers=providers)
    for hit in hits:
        if source_id is not None and hit.conversation_id == source_id:
            continue
        _add_candidate_reason(drafts, hit.summary, _query_hits_reason(hit, query.strip()))


async def _add_shared_attachment_candidates(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    target: Conversation,
    providers: list[str] | None,
    source_id: str,
    pool_limit: int,
) -> None:
    for identity in _attachment_identities(target):
        hits = await store.search_summary_hits(identity, limit=pool_limit, providers=providers)
        for hit in hits:
            if hit.conversation_id == source_id:
                continue
            _add_candidate_reason(
                drafts,
                hit.summary,
                NeighborReason(
                    kind="shared_attachment_identity",
                    detail=f"shares attachment identity: {identity}",
                    evidence=hit.snippet,
                    weight=4.0,
                ),
            )


async def _add_source_content_search_candidates(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    target: Conversation,
    providers: list[str] | None,
    source_id: str,
    pool_limit: int,
) -> None:
    seed = _source_search_seed(target)
    if seed is None:
        return
    hits = await store.search_summary_hits(seed, limit=pool_limit, providers=providers)
    for hit in hits:
        if hit.conversation_id == source_id:
            continue
        _add_candidate_reason(
            drafts,
            hit.summary,
            NeighborReason(
                kind="content_search",
                detail=f"matches source content seed: {seed}",
                evidence=hit.snippet,
                weight=1.75,
            ),
        )


async def _add_content_similarity_reasons(
    drafts: dict[str, _CandidateDraft],
    store: ConversationQueryRuntimeStore,
    *,
    target: Conversation | None,
    query: str | None,
    threshold: float,
) -> None:
    if not drafts:
        return
    target_tokens = _tokens(_conversation_text(target)) if target is not None else _tokens(query or "")
    if not target_tokens:
        return

    for draft in drafts.values():
        candidate = await store.get(str(draft.summary.id))
        candidate_tokens = (
            _tokens(_conversation_text(candidate)) if candidate is not None else _tokens(_summary_text(draft.summary))
        )
        similarity = _content_similarity(target_tokens, candidate_tokens)
        if similarity < threshold:
            continue
        terms = _shared_terms(target_tokens, candidate_tokens)
        detail = "shared content terms"
        if terms:
            detail = f"shared content terms: {', '.join(terms)}"
        reason = NeighborReason(
            kind="content_similarity",
            detail=detail,
            evidence=f"token_overlap={similarity:.3f}",
            weight=3.0 * similarity,
        )
        _add_candidate_reason(drafts, draft.summary, reason)


def _rank_candidates(
    drafts: dict[str, _CandidateDraft],
    *,
    source_conversation_id: str | None,
    query: str | None,
    limit: int,
) -> list[ConversationNeighborCandidate]:
    candidates = [
        ConversationNeighborCandidate(
            summary=draft.summary,
            rank=0,
            score=round(draft.score, 6),
            reasons=tuple(draft.reasons),
            source_conversation_id=source_conversation_id,
            query=query,
        )
        for draft in drafts.values()
        if draft.reasons
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -_epoch(_summary_timestamp(candidate.summary)),
            candidate.summary.display_title.casefold(),
            candidate.conversation_id,
        )
    )
    return [candidate.with_rank(rank) for rank, candidate in enumerate(candidates[:limit], start=1)]


async def discover_neighbor_candidates(
    store: ConversationQueryRuntimeStore,
    request: NeighborDiscoveryRequest,
) -> list[ConversationNeighborCandidate]:
    """Return ranked, explainable neighboring-conversation candidates."""
    if request.limit <= 0:
        return []
    if not request.conversation_id and not (request.query and request.query.strip()):
        raise NeighborDiscoveryError("Neighbor discovery requires a conversation id or query.")

    target = await _load_target(store, request.conversation_id)
    source_id = str(target.id) if target is not None else None
    provider = _canonical_provider(request.provider) or (str(target.provider) if target is not None else None)
    providers = _provider_list(provider)
    pool_limit = max(request.candidate_pool_limit, request.limit)
    drafts: dict[str, _CandidateDraft] = {}

    if target is not None and source_id is not None:
        await _add_same_title_candidates(
            drafts,
            store,
            target=target,
            provider=provider,
            source_id=source_id,
            pool_limit=pool_limit,
        )
        await _add_nearby_candidates(
            drafts,
            store,
            target=target,
            provider=provider,
            source_id=source_id,
            window_hours=request.window_hours,
            pool_limit=pool_limit,
        )
        await _add_shared_attachment_candidates(
            drafts,
            store,
            target=target,
            providers=providers,
            source_id=source_id,
            pool_limit=pool_limit,
        )
        await _add_source_content_search_candidates(
            drafts,
            store,
            target=target,
            providers=providers,
            source_id=source_id,
            pool_limit=pool_limit,
        )

    await _add_query_candidates(
        drafts,
        store,
        query=request.query,
        providers=providers,
        source_id=source_id,
        pool_limit=pool_limit,
    )
    await _add_content_similarity_reasons(
        drafts,
        store,
        target=target,
        query=request.query,
        threshold=request.similarity_threshold,
    )
    return _rank_candidates(
        drafts,
        source_conversation_id=source_id,
        query=request.query.strip() if request.query else None,
        limit=request.limit,
    )


__all__ = [
    "ConversationNeighborCandidate",
    "NeighborDiscoveryError",
    "NeighborDiscoveryRequest",
    "NeighborReason",
    "discover_neighbor_candidates",
]
