"""Typed resume-brief assembly over archived sessions and insights."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import PurePath
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, field_validator

from polylogue.archive.actions.actions import build_tool_calls_from_content_blocks
from polylogue.archive.session.domain_models import Session
from polylogue.core.sources import provider_from_origin
from polylogue.insights.archive import (
    ArchiveInsightUnavailableError,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionWorkEventInsight,
    ThreadInsight,
    ThreadInsightQuery,
)
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.storage.search.query_support import normalize_fts5_query

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message


RESUME_BRIEF_MATERIALIZER_VERSION = 1
"""Bumped whenever the resume-brief composition contract changes shape.

Owned by ``polylogue/insights/resume.py``. The brief is composed on read
from already-materialized session insights (profile with folded enrichment,
work events, phases, work thread); the version bumps when fields or
composition semantics change so consumers can invalidate cached
renderings.
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResumeLastMessage(ArchiveInsightModel):
    role: str
    timestamp: str | None = None
    preview: str = ""


class ResumeFacts(ArchiveInsightModel):
    session_id: str
    source_name: str
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


class ResumeWorkEvent(ArchiveInsightModel):
    heuristic_label: str
    summary: str
    confidence: float
    support_level: str


class ResumePhase(ArchiveInsightModel):
    phase_index: int
    message_range: tuple[int, int]
    confidence: float
    support_level: str


class ResumeThread(ArchiveInsightModel):
    thread_id: str
    root_id: str
    dominant_repo: str | None = None
    session_count: int = 0
    depth: int = 0
    branch_count: int = 0
    session_ids: tuple[str, ...] = ()


class ResumeInferences(ArchiveInsightModel):
    inferred_topic: str | None = None
    intent_summary: str | None = None
    outcome_summary: str | None = None
    blockers: tuple[str, ...] = ()
    confidence: float = 0.0
    support_level: str = "unknown"
    repo_names: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    work_events: tuple[ResumeWorkEvent, ...] = ()
    phases: tuple[ResumePhase, ...] = ()
    thread: ResumeThread | None = None


class ResumeRelatedSession(ArchiveInsightModel):
    session_id: str
    relation: str
    source_name: str
    title: str | None = None
    updated_at: str | None = None
    message_count: int = 0


class ResumeUncertainty(ArchiveInsightModel):
    source: str
    detail: str


class ResumeProvenance(ArchiveInsightModel):
    """Cites the substrate rows that contributed to a resume brief.

    Every brief surface (CLI, MCP, reader) must be able to point back at
    the specific session, message, work-event, phase, and related-session
    IDs it composed from — no opaque prose.
    """

    materializer_version: int
    computed_at: str
    cited_session_ids: tuple[str, ...] = ()
    cited_message_ids: tuple[str, ...] = ()
    cited_work_event_ids: tuple[str, ...] = ()
    cited_phase_ids: tuple[str, ...] = ()
    cited_thread_id: str | None = None


class ResumeBrief(ArchiveInsightModel):
    session_id: str
    facts: ResumeFacts
    inferences: ResumeInferences
    related_sessions: tuple[ResumeRelatedSession, ...] = ()
    uncertainties: tuple[ResumeUncertainty, ...] = ()
    next_steps: tuple[str, ...] = ()
    provenance: ResumeProvenance = Field(
        default_factory=lambda: ResumeProvenance(
            materializer_version=RESUME_BRIEF_MATERIALIZER_VERSION,
            computed_at=_utc_now_iso(),
        )
    )


class ResumeCandidate(ArchiveInsightModel):
    logical_session_id: str
    canonical_session_date: str | None = None
    last_message_at: str | None = None
    title: str
    terminal_state: str = "unknown"
    workflow_shape: str = "unknown"
    file_overlap: tuple[str, ...] = ()
    score: float
    score_breakdown: dict[str, float]
    brief_url: str

    @field_validator("logical_session_id", "title", "brief_url")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field cannot be empty")
        return value


class ResumeOperations(Protocol):
    async def get_session(self, session_id: str) -> Session | None: ...

    async def get_sessions(self, session_ids: list[str]) -> list[Session]: ...

    async def get_session_tree(self, session_id: str) -> list[Session]: ...

    async def get_session_profile_insight(
        self,
        session_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None: ...

    async def get_session_work_event_insights(self, session_id: str) -> list[SessionWorkEventInsight]: ...

    async def get_session_phase_insights(self, session_id: str) -> list[SessionPhaseInsight]: ...

    async def list_thread_insights(
        self,
        query: ThreadInsightQuery | None = None,
    ) -> list[ThreadInsight]: ...

    async def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]: ...


def _iso(value: object) -> str | None:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value) if value is not None else None


def _normalize_path(value: str) -> str:
    text = str(PurePath(value.strip()).as_posix()) if value and value.strip() else ""
    if text.endswith("/") and len(text) > 1:
        text = text.rstrip("/")
    return text


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if not path or not prefix:
        return False
    return path == prefix or path.startswith(f"{prefix}/")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _profile_last_message_at(profile: SessionProfileInsight) -> str | None:
    evidence = profile.evidence
    provenance = profile.provenance
    if evidence is None:
        return provenance.source_updated_at
    return (
        evidence.last_message_at
        or evidence.session_timestamp
        or evidence.updated_at
        or evidence.first_message_at
        or provenance.source_updated_at
    )


def _candidate_title(profile: SessionProfileInsight) -> str:
    inferred = profile.inference.inferred_topic if profile.inference is not None else None
    return inferred or profile.title or str(profile.session_id)


def _terminal_weight(state: str) -> float:
    return {
        "tool_left": 1.0,
        "error_left": 0.95,
        "question_left": 0.85,
        "agent_hanging": 0.8,
        "unknown": 0.25,
        "clean_finish": 0.0,
    }.get(state, 0.25)


def _workflow_weight(shape: str) -> float:
    return {
        "agentic_loop": 1.0,
        "subagent_dispatch": 0.9,
        "debugging": 0.75,
        "implementation": 0.7,
        "planning": 0.55,
        "chat": 0.15,
        "unknown": 0.2,
    }.get(shape, 0.2)


def _profile_terminal_state(profile: SessionProfileInsight) -> str:
    if profile.inference is not None and profile.inference.terminal_state != "unknown":
        return profile.inference.terminal_state
    if profile.evidence is not None:
        return profile.evidence.terminal_state
    return "unknown"


def _profile_workflow_shape(profile: SessionProfileInsight) -> str:
    if profile.inference is not None and profile.inference.workflow_shape != "unknown":
        return profile.inference.workflow_shape
    if profile.evidence is not None:
        return profile.evidence.workflow_shape
    return "unknown"


def _strongest_terminal_state(profiles: Sequence[SessionProfileInsight]) -> str:
    return max((_profile_terminal_state(profile) for profile in profiles), key=_terminal_weight, default="unknown")


def _strongest_workflow_shape(profiles: Sequence[SessionProfileInsight]) -> str:
    return max((_profile_workflow_shape(profile) for profile in profiles), key=_workflow_weight, default="unknown")


def _profile_paths(profile: SessionProfileInsight) -> set[str]:
    evidence = profile.evidence
    if evidence is None:
        return set()
    paths = (
        *evidence.file_paths_touched,
        *evidence.repo_paths,
        *evidence.cwd_paths,
    )
    return {_normalize_path(path) for path in paths if _normalize_path(path)}


def _profile_cwds(profile: SessionProfileInsight) -> set[str]:
    evidence = profile.evidence
    if evidence is None:
        return set()
    return {_normalize_path(path) for path in evidence.cwd_paths if _normalize_path(path)}


def _profile_repo_matches(profile: SessionProfileInsight, repo_path: str) -> bool:
    evidence = profile.evidence
    if evidence is None:
        return False
    repo = _normalize_path(repo_path)
    if not repo:
        return True
    candidates = (*evidence.repo_paths, *evidence.cwd_paths, *evidence.file_paths_touched)
    return any(_path_matches_prefix(_normalize_path(path), repo) for path in candidates)


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


def _tool_facts(session: Session) -> tuple[dict[str, int], tuple[str, ...]]:
    categories: Counter[str] = Counter()
    paths: list[str] = []
    for message in session.messages:
        tool_calls = build_tool_calls_from_content_blocks(
            provider=provider_from_origin(session.origin),
            content_blocks=message.blocks,
        )
        for tool_call in tool_calls:
            category = getattr(tool_call.category, "value", str(tool_call.category))
            categories[str(category)] += 1
            paths.extend(tool_call.affected_paths)
    return dict(sorted(categories.items())), tuple(dict.fromkeys(paths))


def _facts_from_session(
    session: Session,
    profile: SessionProfileInsight | None,
) -> ResumeFacts:
    messages = session.messages.to_list()
    tool_categories, file_paths = _tool_facts(session)
    evidence = profile.evidence if profile is not None else None
    return ResumeFacts(
        session_id=str(session.id),
        source_name=provider_from_origin(session.origin).value,
        title=session.title,
        created_at=_iso(session.created_at),
        updated_at=_iso(session.updated_at),
        parent_id=str(session.parent_id) if session.parent_id is not None else None,
        branch_type=str(session.branch_type) if session.branch_type is not None else None,
        message_count=len(messages),
        tags=evidence.tags if evidence is not None else _metadata_tags(session.metadata),
        repo_paths=evidence.repo_paths if evidence is not None else (),
        cwd_paths=evidence.cwd_paths if evidence is not None else (),
        branch_names=evidence.branch_names if evidence is not None else (),
        file_paths_touched=evidence.file_paths_touched if evidence is not None else file_paths,
        tool_categories=evidence.tool_categories if evidence is not None else tool_categories,
        last_message=_last_message(messages),
    )


def _event_summary(events: Sequence[SessionWorkEventInsight]) -> tuple[ResumeWorkEvent, ...]:
    return tuple(
        ResumeWorkEvent(
            heuristic_label=event.inference.heuristic_label,
            summary=event.inference.summary,
            confidence=event.inference.confidence,
            support_level=event.inference.support_level,
        )
        for event in events[:5]
    )


def _phase_summary(phases: Sequence[SessionPhaseInsight]) -> tuple[ResumePhase, ...]:
    return tuple(
        ResumePhase(
            phase_index=phase.phase_index,
            message_range=phase.evidence.message_range,
            confidence=phase.inference.confidence,
            support_level=phase.inference.support_level,
        )
        for phase in phases[:5]
    )


def _thread_summary(thread: ThreadInsight | None) -> ResumeThread | None:
    if thread is None:
        return None
    return ResumeThread(
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
    profile: SessionProfileInsight | None,
    events: Sequence[SessionWorkEventInsight],
    phases: Sequence[SessionPhaseInsight],
    thread: ThreadInsight | None,
) -> ResumeInferences:
    profile_inference = profile.inference if profile is not None else None
    enrichment_payload = profile.enrichment if profile is not None else None
    return ResumeInferences(
        inferred_topic=profile_inference.inferred_topic if profile_inference is not None else None,
        intent_summary=enrichment_payload.intent_summary if enrichment_payload is not None else None,
        outcome_summary=enrichment_payload.outcome_summary if enrichment_payload is not None else None,
        blockers=enrichment_payload.blockers if enrichment_payload is not None else (),
        confidence=enrichment_payload.confidence if enrichment_payload is not None else 0.0,
        support_level=enrichment_payload.support_level if enrichment_payload is not None else "unknown",
        repo_names=profile_inference.repo_names if profile_inference is not None else (),
        auto_tags=profile_inference.auto_tags if profile_inference is not None else (),
        work_events=_event_summary(events),
        phases=_phase_summary(phases),
        thread=_thread_summary(thread),
    )


def _relation(target: Session, candidate: Session, thread_ids: set[str]) -> str:
    candidate_id = str(candidate.id)
    if target.parent_id is not None and candidate_id == str(target.parent_id):
        return "parent"
    if candidate.parent_id is not None and str(candidate.parent_id) == str(target.id):
        return str(candidate.branch_type or "child")
    if candidate_id in thread_ids:
        return "thread"
    return "session_tree"


def _related_session(
    target: Session,
    candidate: Session,
    thread_ids: set[str],
) -> ResumeRelatedSession:
    return ResumeRelatedSession(
        session_id=str(candidate.id),
        relation=_relation(target, candidate, thread_ids),
        source_name=provider_from_origin(candidate.origin).value,
        title=candidate.title,
        updated_at=_iso(candidate.updated_at or candidate.created_at),
        message_count=len(candidate.messages),
    )


async def _related_sessions(
    operations: ResumeOperations,
    target: Session,
    thread: ThreadInsight | None,
    *,
    related_limit: int,
) -> tuple[ResumeRelatedSession, ...]:
    thread_ids = set(thread.thread.session_ids if thread is not None else ())
    sessions_by_id: dict[str, Session] = {}

    for session in await operations.get_session_tree(str(target.id)):
        sessions_by_id[str(session.id)] = session

    missing_thread_ids = [session_id for session_id in thread_ids if session_id not in sessions_by_id]
    if missing_thread_ids:
        for session in await operations.get_sessions(missing_thread_ids):
            sessions_by_id[str(session.id)] = session

    sessions_by_id.pop(str(target.id), None)
    related = [_related_session(target, session, thread_ids) for session in sessions_by_id.values()]
    related.sort(key=lambda item: item.updated_at or "", reverse=True)
    return tuple(related[:related_limit])


async def _find_thread(
    operations: ResumeOperations,
    session_id: str,
    uncertainties: list[ResumeUncertainty],
) -> ThreadInsight | None:
    fts_query = normalize_fts5_query(session_id)
    if fts_query is not None:
        try:
            for candidate in await operations.list_thread_insights(ThreadInsightQuery(query=fts_query, limit=10)):
                if session_id in candidate.thread.session_ids:
                    return candidate
        except ArchiveInsightUnavailableError:
            pass

    try:
        candidates = await operations.list_thread_insights(ThreadInsightQuery(limit=None))
    except ArchiveInsightUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="thread", detail=str(exc)))
        return None

    for candidate in candidates:
        if session_id in candidate.thread.session_ids:
            return candidate
    return None


def _next_steps(
    session: Session,
    inferences: ResumeInferences,
) -> tuple[str, ...]:
    steps: list[str] = []
    if inferences.blockers:
        steps.append(f"Resolve blocker: {inferences.blockers[0]}")

    last = inferences.work_events[-1] if inferences.work_events else None
    if last is not None:
        steps.append(f"Continue after latest work event: {last.summary}")

    last_message = _last_message(session.messages.to_list())
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
    session = await operations.get_session(session_id)
    if session is None:
        return None

    session_id = str(session.id)
    uncertainties: list[ResumeUncertainty] = []

    profile: SessionProfileInsight | None = None
    try:
        profile = await operations.get_session_profile_insight(session_id)
    except ArchiveInsightUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="session_profile", detail=str(exc)))
    if profile is None and not any(u.source == "session_profile" for u in uncertainties):
        uncertainties.append(
            ResumeUncertainty(
                source="session_profile",
                detail="session_insights not materialized for this session; run rebuild_insights",
            )
        )

    events: list[SessionWorkEventInsight] = []
    try:
        events = await operations.get_session_work_event_insights(session_id)
    except ArchiveInsightUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="work_events", detail=str(exc)))

    phases: list[SessionPhaseInsight] = []
    try:
        phases = await operations.get_session_phase_insights(session_id)
    except ArchiveInsightUnavailableError as exc:
        uncertainties.append(ResumeUncertainty(source="phases", detail=str(exc)))

    thread = await _find_thread(operations, session_id, uncertainties)
    inferences = _inferences(
        profile=profile,
        events=events,
        phases=phases,
        thread=thread,
    )
    related_sessions = await _related_sessions(
        operations,
        session,
        thread,
        related_limit=related_limit,
    )

    cited_session_ids: tuple[str, ...] = (session_id,) + tuple(related.session_id for related in related_sessions)
    cited_message_ids: tuple[str, ...] = tuple(str(message.id) for message in session.messages)
    cited_work_event_ids: tuple[str, ...] = tuple(event.event_id for event in events[:5])
    cited_phase_ids: tuple[str, ...] = tuple(phase.phase_id for phase in phases[:5])
    provenance = ResumeProvenance(
        materializer_version=RESUME_BRIEF_MATERIALIZER_VERSION,
        computed_at=_utc_now_iso(),
        cited_session_ids=cited_session_ids,
        cited_message_ids=cited_message_ids,
        cited_work_event_ids=cited_work_event_ids,
        cited_phase_ids=cited_phase_ids,
        cited_thread_id=thread.thread_id if thread is not None else None,
    )

    return ResumeBrief(
        session_id=session_id,
        facts=_facts_from_session(session, profile),
        inferences=inferences,
        related_sessions=related_sessions,
        uncertainties=tuple(uncertainties),
        next_steps=_next_steps(session, inferences),
        provenance=provenance,
    )


async def find_resume_candidates(
    operations: ResumeOperations,
    *,
    repo_path: str,
    cwd: str | None = None,
    recent_files: Sequence[str] = (),
    limit: int = 10,
) -> tuple[ResumeCandidate, ...]:
    """Rank logical sessions likely to match the operator's current context."""

    normalized_repo = _normalize_path(repo_path)
    normalized_cwd = _normalize_path(cwd or "")
    normalized_recent = {_normalize_path(path) for path in recent_files if _normalize_path(path)}
    profiles = await operations.list_session_profile_insights(
        SessionProfileInsightQuery(
            sort="last-message",
            tier="merged",
            limit=None,
        )
    )
    grouped: dict[str, list[SessionProfileInsight]] = {}
    for profile in profiles:
        logical_id = str(profile.logical_session_id or profile.session_id)
        grouped.setdefault(logical_id, []).append(profile)

    if normalized_repo:
        repo_grouped = {
            logical_id: members
            for logical_id, members in grouped.items()
            if any(_profile_repo_matches(member, normalized_repo) for member in members)
        }
        if repo_grouped:
            grouped = repo_grouped

    latest_times = [
        parsed
        for members in grouped.values()
        for parsed in (_parse_timestamp(_profile_last_message_at(member)) for member in members)
        if parsed is not None
    ]
    newest = max(latest_times) if latest_times else None

    candidates: list[ResumeCandidate] = []
    for logical_id, members in grouped.items():
        representative = max(
            members,
            key=lambda profile: (
                _parse_timestamp(_profile_last_message_at(profile)) or datetime.min.replace(tzinfo=timezone.utc),
                str(profile.session_id),
            ),
        )
        evidence = representative.evidence
        last_message_at = _profile_last_message_at(representative)
        last_dt = _parse_timestamp(last_message_at)
        all_paths = set().union(*(_profile_paths(member) for member in members))
        all_cwds = set().union(*(_profile_cwds(member) for member in members))
        file_overlap = tuple(sorted(normalized_recent & all_paths))
        file_overlap_score = (
            len(file_overlap) / len(normalized_recent | all_paths)
            if normalized_recent and (normalized_recent | all_paths)
            else 0.0
        )
        cwd_score = (
            1.0
            if normalized_cwd
            and any(
                _path_matches_prefix(normalized_cwd, item) or _path_matches_prefix(item, normalized_cwd)
                for item in all_cwds
            )
            else 0.0
        )
        if newest is not None and last_dt is not None:
            age_hours = max((newest - last_dt).total_seconds() / 3600.0, 0.0)
            recency_score = 1.0 / (1.0 + (age_hours / 72.0))
        else:
            recency_score = 0.0
        terminal_state = _strongest_terminal_state(members)
        workflow_shape = _strongest_workflow_shape(members)
        breakdown = {
            "recency": round(recency_score, 4),
            "file_overlap": round(file_overlap_score, 4),
            "cwd_match": round(cwd_score, 4),
            "terminal_state": round(_terminal_weight(terminal_state), 4),
            "workflow_shape": round(_workflow_weight(workflow_shape), 4),
        }
        score = round(
            (0.35 * breakdown["recency"])
            + (0.25 * breakdown["file_overlap"])
            + (0.15 * breakdown["cwd_match"])
            + (0.15 * breakdown["terminal_state"])
            + (0.10 * breakdown["workflow_shape"]),
            6,
        )
        candidates.append(
            ResumeCandidate(
                logical_session_id=logical_id,
                canonical_session_date=(evidence.canonical_session_date if evidence is not None else None),
                last_message_at=last_message_at,
                title=_candidate_title(representative),
                terminal_state=terminal_state,
                workflow_shape=workflow_shape,
                file_overlap=file_overlap,
                score=score,
                score_breakdown=breakdown,
                brief_url=f"polylogue://resume/{logical_id}",
            )
        )

    if not normalized_recent and not normalized_cwd:
        candidates = [candidate for candidate in candidates if candidate.terminal_state not in {"clean_finish"}]
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.last_message_at or "",
            candidate.logical_session_id,
        )
    )
    return tuple(candidates[: max(0, int(limit))])


__all__ = [
    "RESUME_BRIEF_MATERIALIZER_VERSION",
    "ResumeBrief",
    "ResumeCandidate",
    "ResumeFacts",
    "ResumeInferences",
    "ResumeLastMessage",
    "ResumeOperations",
    "ResumePhase",
    "ResumeProvenance",
    "ResumeRelatedSession",
    "ResumeUncertainty",
    "ResumeWorkEvent",
    "ResumeThread",
    "build_resume_brief",
    "find_resume_candidates",
]
