"""Thread-level semantic products derived from session profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from polylogue.lib.payload_coercion import (
    coerce_float,
    coerce_int,
    optional_datetime,
    optional_string,
    string_int_mapping,
    string_sequence,
)
from polylogue.lib.repo_identity import normalize_repo_names
from polylogue.lib.session_profile import SessionProfile


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

    def to_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "root_id": self.root_id,
            "session_ids": list(self.session_ids),
            "session_count": len(self.session_ids),
            "depth": self.depth,
            "branch_count": self.branch_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "wall_duration_ms": self.wall_duration_ms,
            "total_messages": self.total_messages,
            "total_cost_usd": self.total_cost_usd,
            "dominant_repo": self.dominant_repo,
            "provider_breakdown": self.provider_breakdown,
            "work_event_breakdown": self.work_event_breakdown,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WorkThread:
        return cls(
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
        )


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
                dominant_repo=repo_counter.most_common(1)[0][0] if repo_counter else None,
                provider_breakdown=dict(provider_counter),
                work_event_breakdown=dict(work_event_counter),
            )
        )
    return threads


__all__ = ["WorkThread", "build_session_threads"]
