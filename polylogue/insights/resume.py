"""Typed resume-brief assembly over archived sessions and insights."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import Field, field_validator

from polylogue.archive.actions.actions import build_tool_calls_from_content_blocks
from polylogue.archive.session.domain_models import Session
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


RESUME_BRIEF_MATERIALIZER_VERSION = 2
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
    origin: str
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
    origin: str
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


class ResumePathOverlap(ArchiveInsightModel):
    candidate_path: str
    recent_file: str


class ResumeOverlapBasis(ArchiveInsightModel):
    exact: tuple[ResumePathOverlap, ...] = ()
    dir: tuple[ResumePathOverlap, ...] = ()
    dead_excluded: tuple[str, ...] = ()


class ResumeBrief(ArchiveInsightModel):
    session_id: str
    facts: ResumeFacts
    inferences: ResumeInferences
    related_sessions: tuple[ResumeRelatedSession, ...] = ()
    uncertainties: tuple[ResumeUncertainty, ...] = ()
    next_steps: tuple[str, ...] = ()
    overlap_basis: ResumeOverlapBasis | None = None
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
    overlap_basis: ResumeOverlapBasis = Field(default_factory=ResumeOverlapBasis)
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


@dataclass(frozen=True)
class _PathEvidence:
    display: str
    local_path: Path | None


@dataclass(frozen=True)
class _ResolvedCandidatePath:
    evidence: _PathEvidence
    local_path: Path


@dataclass(frozen=True)
class _FileOverlapScore:
    score: float
    file_overlap: tuple[str, ...]
    basis: ResumeOverlapBasis
    resolvable_paths: tuple[str, ...] = ()
    dead_paths: tuple[str, ...] = ()
    resolution_available: bool = False


@dataclass
class _PathResolutionContext:
    repo_root: Path | None
    exists_cache: dict[Path, bool] = field(default_factory=dict)

    @classmethod
    def from_repo_path(cls, repo_path: str) -> _PathResolutionContext:
        return cls(repo_root=_resume_repo_root(repo_path))

    def exists(self, path: Path) -> bool:
        cached = self.exists_cache.get(path)
        if cached is not None:
            return cached
        try:
            exists = os.path.exists(path)
        except OSError:
            exists = False
        self.exists_cache[path] = exists
        return exists


_ResumeOverlapMode = Literal["legacy", "refactor-aware"]


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if not path or not prefix:
        return False
    return path == prefix or path.startswith(f"{prefix}/")


def _resume_repo_root(repo_path: str) -> Path | None:
    normalized = _normalize_path(repo_path)
    if not normalized:
        return None
    try:
        root = Path(normalized).expanduser().resolve(strict=False)
        return root if os.path.isdir(root) else None
    except OSError:
        return None


def _absolute_root(value: str) -> Path | None:
    normalized = _normalize_path(value)
    if not normalized:
        return None
    path = Path(normalized).expanduser()
    if not path.is_absolute():
        return None
    try:
        return path.resolve(strict=False)
    except OSError:
        return None


def _inferred_candidate_repo_roots(
    candidate_paths: set[str],
    recent_files: set[str],
) -> tuple[str, ...]:
    """Infer captured checkout roots from an exact repo-relative suffix.

    Some older profiles have absolute file touches but no ``repo_paths`` evidence.
    A current relative path is sufficient to recover the old checkout root only when
    the complete relative path is an exact suffix of the captured absolute path.
    """
    recent_parts = {
        tuple(path.parts)
        for display in recent_files
        if (path := Path(display)).parts and not path.is_absolute() and ".." not in path.parts
    }
    roots: set[str] = set()
    for display in candidate_paths:
        absolute = _absolute_root(display)
        if absolute is None:
            continue
        for parts in recent_parts:
            if len(absolute.parts) <= len(parts) or tuple(absolute.parts[-len(parts) :]) != parts:
                continue
            root = absolute
            for _ in parts:
                root = root.parent
            roots.add(root.as_posix())
    return tuple(sorted(roots))


def _repo_local_path(
    repo_root: Path,
    display: str,
    *,
    candidate_repo_roots: Sequence[str] = (),
) -> Path | None:
    path = Path(display).expanduser()
    try:
        if not path.is_absolute():
            local = (repo_root / path).resolve(strict=False)
            local.relative_to(repo_root)
            return local

        absolute = path.resolve(strict=False)
        try:
            absolute.relative_to(repo_root)
        except ValueError:
            historical_roots = sorted(
                (root for value in candidate_repo_roots if (root := _absolute_root(value)) is not None),
                key=lambda root: len(root.parts),
                reverse=True,
            )
            for historical_root in historical_roots:
                try:
                    relative = absolute.relative_to(historical_root)
                except ValueError:
                    continue
                local = (repo_root / relative).resolve(strict=False)
                local.relative_to(repo_root)
                return local
            return None
        return absolute
    except (OSError, ValueError):
        return None


def _partition_candidate_paths(
    context: _PathResolutionContext,
    candidate_paths: set[str],
    *,
    candidate_repo_roots: Sequence[str],
) -> tuple[tuple[_ResolvedCandidatePath, ...], tuple[_PathEvidence, ...]]:
    repo_root = context.repo_root
    if repo_root is None:
        return (), ()

    resolved: list[_ResolvedCandidatePath] = []
    dead: list[_PathEvidence] = []
    for display in sorted(candidate_paths):
        evidence = _PathEvidence(
            display=display,
            local_path=_repo_local_path(
                repo_root,
                display,
                candidate_repo_roots=candidate_repo_roots,
            ),
        )
        local_path = evidence.local_path
        if local_path is None:
            dead.append(evidence)
            continue
        if context.exists(local_path):
            resolved.append(_ResolvedCandidatePath(evidence=evidence, local_path=local_path))
            continue
        if local_path.suffix == ".py" and local_path.name != "__init__.py":
            package_init = local_path.with_suffix("") / "__init__.py"
            if context.exists(package_init):
                resolved.append(_ResolvedCandidatePath(evidence=evidence, local_path=package_init))
                continue
        dead.append(evidence)
    return tuple(resolved), tuple(dead)


def _path_identity(evidence: _PathEvidence) -> str:
    if evidence.local_path is None:
        return f"raw:{evidence.display}"
    return evidence.local_path.as_posix()


def _is_path_prefix(prefix: Path, path: Path) -> bool:
    return prefix == path or prefix in path.parents


def _directory_overlap_pairs(
    repo_root: Path,
    dead_paths: Sequence[_PathEvidence],
    recent_paths: Sequence[_PathEvidence],
) -> tuple[ResumePathOverlap, ...]:
    scored_pairs: list[tuple[int, int, str, str]] = []
    for dead in dead_paths:
        if dead.local_path is None:
            continue
        dead_parent = dead.local_path.parent
        if dead_parent == repo_root:
            continue
        for recent in recent_paths:
            if recent.local_path is None:
                continue
            recent_parent = recent.local_path.parent
            try:
                dead_parent.relative_to(repo_root)
                recent_parent.relative_to(repo_root)
            except ValueError:
                continue
            if not (_is_path_prefix(dead_parent, recent_parent) or _is_path_prefix(recent_parent, dead_parent)):
                continue
            shorter = dead_parent if len(dead_parent.parts) <= len(recent_parent.parts) else recent_parent
            if shorter == repo_root:
                continue
            common_depth = len(shorter.relative_to(repo_root).parts)
            distance = abs(len(dead_parent.parts) - len(recent_parent.parts))
            scored_pairs.append((-common_depth, distance, dead.display, recent.display))

    matched_dead: set[str] = set()
    matched_recent: set[str] = set()
    matches: list[ResumePathOverlap] = []
    for _, _, dead_display, recent_display in sorted(scored_pairs):
        if dead_display in matched_dead or recent_display in matched_recent:
            continue
        matched_dead.add(dead_display)
        matched_recent.add(recent_display)
        matches.append(
            ResumePathOverlap(
                candidate_path=dead_display,
                recent_file=recent_display,
            )
        )
    return tuple(matches)


def _legacy_file_overlap(*, recent_files: set[str], candidate_paths: set[str]) -> _FileOverlapScore:
    exact_paths = tuple(sorted(recent_files & candidate_paths))
    union = recent_files | candidate_paths
    score = len(exact_paths) / len(union) if recent_files and union else 0.0
    return _FileOverlapScore(
        score=score,
        file_overlap=exact_paths,
        basis=ResumeOverlapBasis(
            exact=tuple(ResumePathOverlap(candidate_path=path, recent_file=path) for path in exact_paths)
        ),
    )


def _score_file_overlap(
    *,
    context: _PathResolutionContext,
    recent_files: set[str],
    candidate_paths: set[str],
    candidate_repo_roots: Sequence[str] = (),
    mode: _ResumeOverlapMode = "refactor-aware",
) -> _FileOverlapScore:
    legacy = _legacy_file_overlap(recent_files=recent_files, candidate_paths=candidate_paths)
    if mode == "legacy" or context.repo_root is None:
        return legacy

    repo_root = context.repo_root
    recent_evidence = tuple(
        _PathEvidence(
            display=display,
            local_path=_repo_local_path(repo_root, display),
        )
        for display in sorted(recent_files)
    )
    recent_by_identity = {_path_identity(evidence): evidence for evidence in recent_evidence}
    effective_repo_roots = tuple(
        sorted(
            {
                *candidate_repo_roots,
                *_inferred_candidate_repo_roots(candidate_paths, recent_files),
            }
        )
    )
    resolved, dead = _partition_candidate_paths(
        context,
        candidate_paths,
        candidate_repo_roots=effective_repo_roots,
    )

    resolved_by_identity: dict[str, _ResolvedCandidatePath] = {}
    for resolved_candidate in resolved:
        resolved_by_identity.setdefault(resolved_candidate.local_path.as_posix(), resolved_candidate)

    exact_matches: list[ResumePathOverlap] = []
    exact_recent_identities: set[str] = set()
    for identity, recent in sorted(recent_by_identity.items()):
        matched_candidate = resolved_by_identity.get(identity)
        if matched_candidate is None:
            continue
        exact_recent_identities.add(identity)
        exact_matches.append(
            ResumePathOverlap(
                candidate_path=matched_candidate.evidence.display,
                recent_file=recent.display,
            )
        )

    available_recent = tuple(
        evidence for identity, evidence in sorted(recent_by_identity.items()) if identity not in exact_recent_identities
    )
    directory_matches = _directory_overlap_pairs(repo_root, dead, available_recent)
    recent_identity_by_display = {evidence.display: identity for identity, evidence in recent_by_identity.items()}
    directory_recent_identities = {recent_identity_by_display[match.recent_file] for match in directory_matches}
    matched_dead = {match.candidate_path for match in directory_matches}

    candidate_identities = set(resolved_by_identity) | directory_recent_identities
    overlap_identities = exact_recent_identities | directory_recent_identities
    union = set(recent_by_identity) | candidate_identities
    score = len(overlap_identities) / len(union) if recent_by_identity and union else 0.0
    file_overlap = tuple(sorted(recent_by_identity[identity].display for identity in overlap_identities))
    basis = ResumeOverlapBasis(
        exact=tuple(exact_matches),
        dir=directory_matches,
        dead_excluded=tuple(sorted(path.display for path in dead if path.display not in matched_dead)),
    )
    return _FileOverlapScore(
        score=score,
        file_overlap=file_overlap,
        basis=basis,
        resolvable_paths=tuple(sorted(candidate.evidence.display for candidate in resolved)),
        dead_paths=tuple(sorted(path.display for path in dead)),
        resolution_available=True,
    )


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
    return "unknown"


def _profile_workflow_shape(profile: SessionProfileInsight) -> str:
    if profile.inference is not None and profile.inference.workflow_shape != "unknown":
        return profile.inference.workflow_shape
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


def _profile_repo_roots(profile: SessionProfileInsight) -> set[str]:
    evidence = profile.evidence
    if evidence is None:
        return set()
    return {_normalize_path(path) for path in evidence.repo_paths if _normalize_path(path)}


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
            origin=session.origin,
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
        origin=session.origin.value,
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
            confidence=phase.inference.confidence if phase.inference is not None else 0.0,
            support_level=phase.inference.support_level if phase.inference is not None else "weak",
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
        origin=candidate.origin.value,
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

    if not steps and last_message is not None and last_message.role == "assistant" and last_message.preview:
        steps.append(f"Continue from latest assistant state: {last_message.preview}")

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
    repo_path: str | None = None,
    recent_files: Sequence[str] = (),
) -> ResumeBrief | None:
    """Build a compact handoff brief for one archived session."""
    if repo_path is None and recent_files:
        raise ValueError("repo_path is required when recent_files are supplied")

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
    elif profile is not None:
        missing_profile_parts = tuple(
            name
            for name, value in (
                ("evidence", profile.evidence),
                ("inference", profile.inference),
                ("enrichment", profile.enrichment),
            )
            if value is None
        )
        if missing_profile_parts:
            uncertainties.append(
                ResumeUncertainty(
                    source="session_profile",
                    detail=f"merged session profile is missing: {', '.join(missing_profile_parts)}",
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

    overlap_basis: ResumeOverlapBasis | None = None
    if repo_path is not None:
        try:
            ranking_profiles = await operations.list_session_profile_insights(
                SessionProfileInsightQuery(
                    sort="last-message",
                    tier="merged",
                    limit=None,
                )
            )
            logical_session_id = (
                str(profile.logical_session_id or profile.session_id) if profile is not None else session_id
            )
            members = [
                candidate
                for candidate in ranking_profiles
                if str(candidate.logical_session_id or candidate.session_id) == logical_session_id
            ]
            if members:
                normalized_recent = {_normalize_path(path) for path in recent_files if _normalize_path(path)}
                all_paths = set().union(*(_profile_paths(member) for member in members))
                candidate_repo_roots = set().union(*(_profile_repo_roots(member) for member in members))
                overlap_basis = _score_file_overlap(
                    context=_PathResolutionContext.from_repo_path(repo_path),
                    recent_files=normalized_recent,
                    candidate_paths=all_paths,
                    candidate_repo_roots=tuple(sorted(candidate_repo_roots)),
                ).basis
            else:
                uncertainties.append(
                    ResumeUncertainty(
                        source="resume_overlap",
                        detail="no merged session profile was available for overlap explanation",
                    )
                )
        except ArchiveInsightUnavailableError as exc:
            uncertainties.append(ResumeUncertainty(source="resume_overlap", detail=str(exc)))

    return ResumeBrief(
        session_id=session_id,
        facts=_facts_from_session(session, profile),
        inferences=inferences,
        related_sessions=related_sessions,
        uncertainties=tuple(uncertainties),
        next_steps=_next_steps(session, inferences),
        overlap_basis=overlap_basis,
        provenance=provenance,
    )


def _rank_resume_profiles(
    profiles: Sequence[SessionProfileInsight],
    *,
    repo_path: str,
    cwd: str | None = None,
    recent_files: Sequence[str] = (),
    limit: int = 10,
    overlap_mode: _ResumeOverlapMode = "refactor-aware",
) -> tuple[ResumeCandidate, ...]:
    normalized_repo = _normalize_path(repo_path)
    normalized_cwd = _normalize_path(cwd or "")
    normalized_recent = {_normalize_path(path) for path in recent_files if _normalize_path(path)}
    path_context = _PathResolutionContext.from_repo_path(normalized_repo)
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
        candidate_repo_roots = set().union(*(_profile_repo_roots(member) for member in members))
        overlap_score = _score_file_overlap(
            context=path_context,
            recent_files=normalized_recent,
            candidate_paths=all_paths,
            candidate_repo_roots=tuple(sorted(candidate_repo_roots)),
            mode=overlap_mode,
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
            "file_overlap": round(overlap_score.score, 4),
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
                file_overlap=overlap_score.file_overlap,
                overlap_basis=overlap_score.basis,
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


async def find_resume_candidates(
    operations: ResumeOperations,
    *,
    repo_path: str,
    cwd: str | None = None,
    recent_files: Sequence[str] = (),
    limit: int = 10,
) -> tuple[ResumeCandidate, ...]:
    """Rank logical sessions likely to match the operator's current context."""

    profiles = await operations.list_session_profile_insights(
        SessionProfileInsightQuery(
            sort="last-message",
            tier="merged",
            limit=None,
        )
    )
    return _rank_resume_profiles(
        profiles,
        repo_path=repo_path,
        cwd=cwd,
        recent_files=recent_files,
        limit=limit,
    )


__all__ = [
    "RESUME_BRIEF_MATERIALIZER_VERSION",
    "ResumeBrief",
    "ResumeCandidate",
    "ResumeFacts",
    "ResumeInferences",
    "ResumeLastMessage",
    "ResumeOperations",
    "ResumeOverlapBasis",
    "ResumePathOverlap",
    "ResumePhase",
    "ResumeProvenance",
    "ResumeRelatedSession",
    "ResumeUncertainty",
    "ResumeWorkEvent",
    "ResumeThread",
    "build_resume_brief",
    "find_resume_candidates",
]
