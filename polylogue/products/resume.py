"""Typed resume-brief assembly over archived conversations and products."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from polylogue.archive.action_event.action_events import build_tool_calls_from_content_blocks
from polylogue.lib.conversation.models import Conversation
from polylogue.products.archive import (
    ArchiveProductUnavailableError,
    SessionEnrichmentProduct,
    SessionPhaseProduct,
    SessionProfileProduct,
    SessionWorkEventProduct,
    WorkThreadProduct,
    WorkThreadProductQuery,
)
from polylogue.products.archive_models import ArchiveProductModel
from polylogue.storage.search.query_support import normalize_fts5_query

if TYPE_CHECKING:
    from polylogue.lib.message.models import Message


class ResumeLastMessage(ArchiveProductModel):
    role: str
    timestamp: str | None = None
    preview: str = ""


class ResumeFacts(ArchiveProductModel):
    conversation_id: str
    provider_name: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    parent_id: str | None = None
    branch_type: str | None = None
    message_count: int = 0
    tags: tuple[str, ...] = ()
    repo_paths: tuple[str, ...] = ()
    cwd_paths: tuple[str, ...] = ()
    branch_names: tuple[str, ...] = ()
    file_paths_touched: tuple[str, ...] = ()
    tool_categories: dict[str, int] = Field(default_factory=dict)
    last_message: ResumeLastMessage | None = None


class ResumeWorkEvent(ArchiveProductModel):
    kind: str
    summary: str
    confidence: float
    support_level: str


class ResumePhase(ArchiveProductModel):
    phase_index: int
    message_range: tuple[int, int]
    confidence: float
    support_level: str


class ResumeWorkThread(ArchiveProductModel):
    thread_id: str
    root_id: str
    dominant_repo: str | None = None
    session_count: int = 0
    depth: int = 0
    branch_count: int = 0
    session_ids: tuple[str, ...] = ()


class ResumeInferences(ArchiveProductModel):
    intent_summary: str | None = None
    outcome_summary: str | None = None
    blockers: tuple[str, ...] = ()
    confidence: float = 0.0
    support_level: str = "unknown"
    repo_names: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    work_events: tuple[ResumeWorkEvent, ...] = ()
    phases: tuple[ResumePhase, ...] = ()
    work_thread: ResumeWorkThread | None = None


class ResumeRelatedSession(ArchiveProductModel):
    conversation_id: str
    relation: str
    provider_name: str
    title: str | None = None
    updated_at: str | None = None
    message_count: int = 0


class ResumeUncertainty(ArchiveProductModel):
    source: str
    detail: str


class ResumeBrief(ArchiveProductModel):
    session_id: str
    facts: ResumeFacts
    inferences: ResumeInferences
    related_sessions: tuple[ResumeRelatedSession, ...] = ()
    uncertainties: tuple[ResumeUncertainty, ...] = ()
    next_steps: tuple[str, ...] = ()


class ResumeOperations(Protocol):
    async def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    async def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]: ...

    async def get_session_tree(self, conversation_id: str) -> list[Conversation]: ...

    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None: ...

    async def get_session_enrichment_product(self, conversation_id: str) -> SessionEnrichmentProduct | None: ...

    async def get_session_work_event_products(self, conversation_id: str) -> list[SessionWorkEventProduct]: ...

    async def get_session_phase_products(self, conversation_id: str) -> list[SessionPhaseProduct]: ...

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]: ...


def _iso(value: object) -> str | None:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value) if value is not None else None


def _preview(text: str | None, *, limit: int = 180) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _metadata_tags(metadata: Mapping[str, object] | None) -> tuple[str, ...]:
    raw = metadata.get("tags") if metadata else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str) and item)


def _last_message(messages: Sequence[Message]) -> ResumeLastMessage | None:
    if not messages:
        return None
    message = messages[-1]
    return ResumeLastMessage(
        role=str(message.role),
        timestamp=_iso(message.timestamp),
        preview=_preview(message.text),
    )


def _tool_facts(conversation: Conversation) -> tuple[dict[str, int], tuple[str, ...]]:
    categories: Counter[str] = Counter()
    paths: list[str] = []
    for message in conversation.messages:
        tool_calls = build_tool_calls_from_content_blocks(
            provider=conversation.provider,
            content_blocks=message.content_blocks,
        )
        for tool_call in tool_calls:
            category = getattr(tool_call.category, "value", str(tool_call.category))
            categories[str(category)] += 1
            paths.extend(tool_call.affected_paths)
    return dict(sorted(categories.items())), tuple(dict.fromkeys(paths))


def _facts_from_conversation(
    conversation: Conversation,
    profile: SessionProfileProduct | None,
) -> ResumeFacts:
    messages = conversation.messages.to_list()
    tool_categories, file_paths = _tool_facts(conversation)
    evidence = profile.evidence if profile is not None else None
    return ResumeFacts(
        conversation_id=str(conversation.id),
        provider_name=str(conversation.provider),
        title=conversation.title,
        created_at=_iso(conversation.created_at),
        updated_at=_iso(conversation.updated_at),
        parent_id=str(conversation.parent_id) if conversation.parent_id is not None else None,
        branch_type=str(conversation.branch_type) if conversation.branch_type is not None else None,
        message_count=len(messages),
        tags=evidence.tags if evidence is not None else _metadata_tags(conversation.metadata),
        repo_paths=evidence.repo_paths if evidence is not None else (),
        cwd_paths=evidence.cwd_paths if evidence is not None else (),
        branch_names=evidence.branch_names if evidence is not None else (),
        file_paths_touched=evidence.file_paths_touched if evidence is not None else file_paths,
        tool_categories=evidence.tool_categories if evidence is not None else tool_categories,
        last_message=_last_message(messages),
    )


def _event_summary(events: Sequence[SessionWorkEventProduct]) -> tuple[ResumeWorkEvent, ...]:
    return tuple(
        ResumeWorkEvent(
            kind=event.inference.kind,
            summary=event.inference.summary,
            confidence=event.inference.confidence,
            support_level=event.inference.support_level,
        )
        for event in events[:5]
    )


def _phase_summary(phases: Sequence[SessionPhaseProduct]) -> tuple[ResumePhase, ...]:
    return tuple(
        ResumePhase(
            phase_index=phase.phase_index,
            message_range=phase.evidence.message_range,
            confidence=phase.inference.confidence,
            support_level=phase.inference.support_level,
        )
        for phase in phases[:5]
    )


def _work_thread_summary(thread: WorkThreadProduct | None) -> ResumeWorkThread | None:
    if thread is None:
        return None
    return ResumeWorkThread(
        thread_id=thread.thread_id,
        root_id=thread.root_id,
        dominant_repo=thread.dominant_repo,
        session_count=thread.thread.session_count,
        depth=thread.thread.depth,
        branch_count=thread.thread.branch_count,
        session_ids=thread.thread.session_ids,
    )


def _inferences(
    *,
    profile: SessionProfileProduct | None,
    enrichment: SessionEnrichmentProduct | None,
    events: Sequence[SessionWorkEventProduct],
    phases: Sequence[SessionPhaseProduct],
    work_thread: WorkThreadProduct | None,
) -> ResumeInferences:
    profile_inference = profile.inference if profile is not None else None
    enrichment_payload = enrichment.enrichment if enrichment is not None else None
    return ResumeInferences(
        intent_summary=enrichment_payload.intent_summary if enrichment_payload is not None else None,
        outcome_summary=enrichment_payload.outcome_summary if enrichment_payload is not None else None,
        blockers=enrichment_payload.blockers if enrichment_payload is not None else (),
        confidence=enrichment_payload.confidence if enrichment_payload is not None else 0.0,
        support_level=enrichment_payload.support_level if enrichment_payload is not None else "unknown",
        repo_names=profile_inference.repo_names if profile_inference is not None else (),
        auto_tags=profile_inference.auto_tags if profile_inference is not None else (),
        work_events=_event_summary(events),
        phases=_phase_summary(phases),
        work_thread=_work_thread_summary(work_thread),
    )


def _relation(target: Conversation, candidate: Conversation, work_thread_ids: set[str]) -> str:
    candidate_id = str(candidate.id)
    if target.parent_id is not None and candidate_id == str(target.parent_id):
        return "parent"
    if candidate.parent_id is not None and str(candidate.parent_id) == str(target.id):
        return str(candidate.branch_type or "child")
    if candidate_id in work_thread_ids:
        return "work_thread"
    return "session_tree"


def _related_session(
    target: Conversation,
    candidate: Conversation,
    work_thread_ids: set[str],
) -> ResumeRelatedSession:
    return ResumeRelatedSession(
        conversation_id=str(candidate.id),
        relation=_relation(target, candidate, work_thread_ids),
        provider_name=str(candidate.provider),
        title=candidate.title,
        updated_at=_iso(candidate.updated_at or candidate.created_at),
        message_count=len(candidate.messages),
    )


async def _related_sessions(
    operations: ResumeOperations,
    target: Conversation,
    work_thread: WorkThreadProduct | None,
    *,
    related_limit: int,
) -> tuple[ResumeRelatedSession, ...]:
    work_thread_ids = set(work_thread.thread.session_ids if work_thread is not None else ())
    conversations_by_id: dict[str, Conversation] = {}

    for conversation in await operations.get_session_tree(str(target.id)):
        conversations_by_id[str(conversation.id)] = conversation

    missing_work_thread_ids = [session_id for session_id in work_thread_ids if session_id not in conversations_by_id]
    if missing_work_thread_ids:
        for conversation in await operations.get_conversations(missing_work_thread_ids):
            conversations_by_id[str(conversation.id)] = conversation

    conversations_by_id.pop(str(target.id), None)
    related = [_related_session(target, conversation, work_thread_ids) for conversation in conversations_by_id.values()]
    related.sort(key=lambda item: item.updated_at or "", reverse=True)
    return tuple(related[:related_limit])


async def _find_work_thread(
    operations: ResumeOperations,
    conversation_id: str,
    uncertainties: list[ResumeUncertainty],
) -> WorkThreadProduct | None:
    fts_query = normalize_fts5_query(conversation_id)
    if fts_query is not None:
        try:
            for candidate in await operations.list_work_thread_products(
                WorkThreadProductQuery(query=fts_query, limit=10)
            ):
                if conversation_id in candidate.thread.session_ids:
                    return candidate
        except ArchiveProductUnavailableError:
            pass

    try:
        candidates = await operations.list_work_thread_products(WorkThreadProductQuery(limit=None))
    except ArchiveProductUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="work_thread", detail=str(exc)))
        return None

    for candidate in candidates:
        if conversation_id in candidate.thread.session_ids:
            return candidate
    return None


def _next_steps(
    conversation: Conversation,
    inferences: ResumeInferences,
) -> tuple[str, ...]:
    steps: list[str] = []
    if inferences.blockers:
        steps.append(f"Resolve blocker: {inferences.blockers[0]}")

    last = inferences.work_events[-1] if inferences.work_events else None
    if last is not None:
        steps.append(f"Continue after latest work event: {last.summary}")

    last_message = _last_message(conversation.messages.to_list())
    if last_message is not None and last_message.role == "user" and last_message.preview:
        steps.append(f"Respond to latest user request: {last_message.preview}")

    if not steps and inferences.intent_summary:
        steps.append(f"Continue intent: {inferences.intent_summary}")
    if not steps and inferences.outcome_summary:
        steps.append(f"Verify or close out outcome: {inferences.outcome_summary}")
    if not steps:
        steps.append("Review the latest archived message and continue from that state.")

    return tuple(dict.fromkeys(steps[:3]))


async def build_resume_brief(
    operations: ResumeOperations,
    session_id: str,
    *,
    related_limit: int = 6,
) -> ResumeBrief | None:
    """Build a compact handoff brief for one archived session."""
    conversation = await operations.get_conversation(session_id)
    if conversation is None:
        return None

    conversation_id = str(conversation.id)
    uncertainties: list[ResumeUncertainty] = []

    profile: SessionProfileProduct | None = None
    try:
        profile = await operations.get_session_profile_product(conversation_id)
    except ArchiveProductUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="session_profile", detail=str(exc)))

    enrichment: SessionEnrichmentProduct | None = None
    try:
        enrichment = await operations.get_session_enrichment_product(conversation_id)
    except ArchiveProductUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="session_enrichment", detail=str(exc)))

    events: list[SessionWorkEventProduct] = []
    try:
        events = await operations.get_session_work_event_products(conversation_id)
    except ArchiveProductUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="work_events", detail=str(exc)))

    phases: list[SessionPhaseProduct] = []
    try:
        phases = await operations.get_session_phase_products(conversation_id)
    except ArchiveProductUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="phases", detail=str(exc)))

    work_thread = await _find_work_thread(operations, conversation_id, uncertainties)
    inferences = _inferences(
        profile=profile,
        enrichment=enrichment,
        events=events,
        phases=phases,
        work_thread=work_thread,
    )

    return ResumeBrief(
        session_id=conversation_id,
        facts=_facts_from_conversation(conversation, profile),
        inferences=inferences,
        related_sessions=await _related_sessions(
            operations,
            conversation,
            work_thread,
            related_limit=related_limit,
        ),
        uncertainties=tuple(uncertainties),
        next_steps=_next_steps(conversation, inferences),
    )


__all__ = [
    "ResumeBrief",
    "ResumeFacts",
    "ResumeInferences",
    "ResumeLastMessage",
    "ResumeOperations",
    "ResumePhase",
    "ResumeRelatedSession",
    "ResumeUncertainty",
    "ResumeWorkEvent",
    "ResumeWorkThread",
    "build_resume_brief",
]
