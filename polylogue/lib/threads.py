"""Thread-level semantic products derived from session profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias, cast

from polylogue.lib.payload_coercion import (
    coerce_float,
    coerce_int,
    mapping_sequence,
    optional_datetime,
    optional_string,
    string_int_mapping,
    string_sequence,
)
from polylogue.lib.repo_identity import normalize_repo_names
from polylogue.lib.session_payload_documents import WorkThreadDocument, WorkThreadMemberEvidenceDocument
from polylogue.lib.session_profile import SessionProfile

WorkThreadPayload: TypeAlias = WorkThreadDocument


def _work_thread_payload(thread: WorkThread) -> WorkThreadPayload:
    return {
        "thread_id": thread.thread_id,
        "root_id": thread.root_id,
        "session_ids": list(thread.session_ids),
        "session_count": len(thread.session_ids),
        "depth": thread.depth,
        "branch_count": thread.branch_count,
        "start_time": thread.start_time.isoformat() if thread.start_time else None,
        "end_time": thread.end_time.isoformat() if thread.end_time else None,
        "wall_duration_ms": thread.wall_duration_ms,
        "total_messages": thread.total_messages,
        "total_cost_usd": thread.total_cost_usd,
        "dominant_repo": thread.dominant_repo,
        "provider_breakdown": dict(thread.provider_breakdown),
        "work_event_breakdown": dict(thread.work_event_breakdown),
        "confidence": thread.confidence,
        "support_level": thread.support_level,
        "support_signals": list(thread.support_signals),
        "member_evidence": [
            cast(WorkThreadMemberEvidenceDocument, member.to_dict()) for member in thread.member_evidence
        ],
    }


def _work_thread_from_mapping(payload: Mapping[str, object]) -> WorkThread:
    return WorkThread(
        thread_id=str(payload["thread_id"]),
        root_id=str(payload["root_id"]),
        session_ids=string_sequence(payload.get("session_ids")),
        depth=coerce_int(payload.get("depth"), 0),
        branch_count=coerce_int(payload.get("branch_count"), 0),
        start_time=optional_datetime(payload.get("start_time")),
        end_time=optional_datetime(payload.get("end_time")),
        wall_duration_ms=coerce_int(payload.get("wall_duration_ms"), 0),
        total_messages=coerce_int(payload.get("total_messages"), 0),
        total_cost_usd=coerce_float(payload.get("total_cost_usd"), 0.0),
        dominant_repo=optional_string(payload.get("dominant_repo")),
        provider_breakdown=string_int_mapping(payload.get("provider_breakdown")),
        work_event_breakdown=string_int_mapping(payload.get("work_event_breakdown")),
        confidence=coerce_float(payload.get("confidence"), 0.0),
        support_level=optional_string(payload.get("support_level")) or "weak",
        support_signals=string_sequence(payload.get("support_signals")),
        member_evidence=tuple(
            WorkThreadMemberEvidence.from_dict(item) for item in mapping_sequence(payload.get("member_evidence"))
        ),
    )


@dataclass(frozen=True)
class WorkThreadMemberEvidence:
    conversation_id: str
    parent_id: str | None
    role: str
    depth: int
    confidence: float
    support_signals: tuple[str, ...]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "parent_id": self.parent_id,
            "role": self.role,
            "depth": self.depth,
            "confidence": self.confidence,
            "support_signals": list(self.support_signals),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WorkThreadMemberEvidence:
        return cls(
            conversation_id=str(payload["conversation_id"]),
            parent_id=optional_string(payload.get("parent_id")),
            role=optional_string(payload.get("role")) or "member",
            depth=coerce_int(payload.get("depth"), 0),
            confidence=coerce_float(payload.get("confidence"), 0.0),
            support_signals=string_sequence(payload.get("support_signals")),
            evidence=string_sequence(payload.get("evidence")),
        )


@dataclass(frozen=True)
class WorkThread:
    thread_id: str
    root_id: str
    session_ids: tuple[str, ...]
    depth: int
    branch_count: int
    start_time: datetime | None
    end_time: datetime | None
    wall_duration_ms: int
    total_messages: int
    total_cost_usd: float
    dominant_repo: str | None
    provider_breakdown: dict[str, int]
    work_event_breakdown: dict[str, int]
    confidence: float
    support_level: str
    support_signals: tuple[str, ...]
    member_evidence: tuple[WorkThreadMemberEvidence, ...]

    def to_dict(self) -> WorkThreadPayload:
        return _work_thread_payload(self)

    @classmethod
    def from_payload(cls, payload: WorkThreadPayload) -> WorkThread:
        return _work_thread_from_mapping(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WorkThread:
        return _work_thread_from_mapping(payload)


def _bfs_depth(adjacency: dict[str, list[str]], root: str) -> int:
    visited = {root}
    frontier = [root]
    depth = 0
    while frontier:
        next_frontier: list[str] = []
        for node in frontier:
            for child in adjacency.get(node, []):
                if child not in visited:
                    visited.add(child)
                    next_frontier.append(child)
        if not next_frontier:
            break
        depth += 1
        frontier = next_frontier
    return depth


def _member_depths(children: dict[str, list[str]], root: str) -> dict[str, int]:
    depths = {root: 0}
    frontier = [root]
    while frontier:
        next_frontier: list[str] = []
        for parent_id in frontier:
            for child_id in children.get(parent_id, []):
                if child_id in depths:
                    continue
                depths[child_id] = depths[parent_id] + 1
                next_frontier.append(child_id)
        frontier = next_frontier
    return depths


def _thread_member_evidence(
    thread_profiles: Iterable[SessionProfile],
    *,
    root_id: str,
    depths: Mapping[str, int],
) -> tuple[WorkThreadMemberEvidence, ...]:
    members: list[WorkThreadMemberEvidence] = []
    for profile in thread_profiles:
        conversation_id = profile.conversation_id
        parent_id = profile.parent_id
        if conversation_id == root_id:
            role = "root"
            signals: tuple[str, ...] = ("root_conversation", "explicit_lineage")
            evidence: tuple[str, ...] = ("conversation has no archived parent inside this thread",)
        else:
            role = "parent_continuation"
            signals = ("parent_conversation_id", "explicit_lineage")
            evidence = (f"parent_id={parent_id}", f"root_id={root_id}")
        members.append(
            WorkThreadMemberEvidence(
                conversation_id=conversation_id,
                parent_id=parent_id,
                role=role,
                depth=depths.get(conversation_id, 0),
                confidence=1.0,
                support_signals=signals,
                evidence=evidence,
            )
        )
    return tuple(members)


def _thread_support_signals(
    *,
    session_count: int,
    branch_count: int,
    dominant_repo: str | None,
    start_time: datetime | None,
    end_time: datetime | None,
) -> tuple[str, ...]:
    signals = ["explicit_lineage"]
    if session_count > 1:
        signals.append("multi_session_thread")
    if branch_count > 1:
        signals.append("branching_continuations")
    if dominant_repo is not None:
        signals.append("dominant_repo")
    if start_time is not None and end_time is not None:
        signals.append("wallclock_bounds")
    return tuple(signals)


def _thread_support_level(*, session_count: int) -> str:
    return "strong" if session_count > 1 else "moderate"


def _thread_confidence(*, session_count: int) -> float:
    return 1.0 if session_count > 1 else 0.85


def build_session_threads(profiles: Iterable[SessionProfile]) -> list[WorkThread]:
    all_profiles = list(profiles)
    by_id = {profile.conversation_id: profile for profile in all_profiles}
    children: dict[str, list[str]] = defaultdict(list)
    for profile in all_profiles:
        if profile.parent_id and profile.parent_id in by_id:
            children[profile.parent_id].append(profile.conversation_id)
    child_ids = {child_id for child_list in children.values() for child_id in child_list}
    roots = [profile for profile in all_profiles if profile.conversation_id not in child_ids]

    threads: list[WorkThread] = []
    for root in roots:
        thread_ids: list[str] = []
        frontier = [root.conversation_id]
        while frontier:
            next_frontier: list[str] = []
            for conversation_id in frontier:
                thread_ids.append(conversation_id)
                next_frontier.extend(children.get(conversation_id, []))
            frontier = next_frontier

        thread_profiles = [by_id[conversation_id] for conversation_id in thread_ids if conversation_id in by_id]
        timestamps_start = [profile.first_message_at for profile in thread_profiles if profile.first_message_at]
        timestamps_end = [profile.last_message_at for profile in thread_profiles if profile.last_message_at]
        start_time = min(timestamps_start) if timestamps_start else None
        end_time = max(timestamps_end) if timestamps_end else None
        wall_ms = 0
        if start_time is not None and end_time is not None:
            wall_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

        repo_counter: Counter[str] = Counter()
        provider_counter: Counter[str] = Counter()
        work_event_counter: Counter[str] = Counter()
        for profile in thread_profiles:
            repo_counter.update(profile.repo_names or normalize_repo_names(repo_paths=profile.repo_paths))
            provider_counter.update((profile.provider,))
            work_event_counter.update(
                event.kind.value if hasattr(event.kind, "value") else str(event.kind) for event in profile.work_events
            )
        dominant_repo = repo_counter.most_common(1)[0][0] if repo_counter else None
        member_depths = _member_depths(children, root.conversation_id)
        member_evidence = _thread_member_evidence(
            thread_profiles,
            root_id=root.conversation_id,
            depths=member_depths,
        )
        support_signals = _thread_support_signals(
            session_count=len(thread_ids),
            branch_count=sum(1 for conversation_id in thread_ids if not children.get(conversation_id)),
            dominant_repo=dominant_repo,
            start_time=start_time,
            end_time=end_time,
        )

        threads.append(
            WorkThread(
                thread_id=root.conversation_id,
                root_id=root.conversation_id,
                session_ids=tuple(thread_ids),
                depth=_bfs_depth(children, root.conversation_id),
                branch_count=sum(1 for conversation_id in thread_ids if not children.get(conversation_id)),
                start_time=start_time,
                end_time=end_time,
                wall_duration_ms=wall_ms,
                total_messages=sum(profile.message_count for profile in thread_profiles),
                total_cost_usd=sum(profile.total_cost_usd for profile in thread_profiles),
                dominant_repo=dominant_repo,
                provider_breakdown=dict(provider_counter),
                work_event_breakdown=dict(work_event_counter),
                confidence=_thread_confidence(session_count=len(thread_ids)),
                support_level=_thread_support_level(session_count=len(thread_ids)),
                support_signals=support_signals,
                member_evidence=member_evidence,
            )
        )
    return threads


__all__ = ["WorkThread", "WorkThreadMemberEvidence", "build_session_threads"]
