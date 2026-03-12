"""Work thread model — groups continuation sessions into logical work units.

A "work thread" is a root session plus all its continuations (sessions with
parent_id pointing back to the root or to other sessions in the chain).
This provides a temporal scale between single session and multi-day episode.

Example: debugging a hard crash across 5 continuation sessions over 2 days
is one work thread, not 5 independent sessions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from polylogue.lib.session_profile import SessionProfile


@dataclass(frozen=True)
class WorkThread:
    """A logical unit of work spanning one or more continuation sessions.

    thread_id is the conversation_id of the root (non-continuation) session.
    """
    thread_id: str                      # root conversation_id
    root_id: str                        # same as thread_id
    session_ids: tuple[str, ...]        # all sessions in thread (root + continuations)
    depth: int                          # max chain length root→leaf
    branch_count: int                   # number of leaf sessions (divergent continuations)
    start_time: Optional[datetime]      # earliest first_message_at
    end_time: Optional[datetime]        # latest last_message_at
    wall_duration_ms: int               # start_time → end_time in ms
    total_messages: int
    total_cost_usd: float
    dominant_project: Optional[str]     # most common canonical project
    work_event_breakdown: dict[str, int]  # {kind: count} across all sessions


def _bfs_depth(adjacency: dict[str, list[str]], root: str) -> int:
    """BFS to find maximum chain depth from root."""
    visited = {root}
    frontier = [root]
    depth = 0
    while frontier:
        next_frontier = []
        for node in frontier:
            for child in adjacency.get(node, []):
                if child not in visited:
                    visited.add(child)
                    next_frontier.append(child)
        if next_frontier:
            depth += 1
            frontier = next_frontier
        else:
            break
    return depth


def build_session_threads(profiles: Iterable[SessionProfile]) -> list[WorkThread]:
    """Group SessionProfiles into WorkThreads by following parent_id chains.

    Sessions without parent_id (is_continuation=False, or parent_id=None)
    are thread roots. Sessions with parent_id are children. Orphans (parent_id
    points to an unknown session) are treated as roots of their own thread.
    """
    all_profiles = list(profiles)
    by_id = {p.conversation_id: p for p in all_profiles}

    # Build child adjacency: parent_id → [child_conversation_ids]
    children: dict[str, list[str]] = defaultdict(list)
    for p in all_profiles:
        if p.parent_id and p.parent_id in by_id:
            children[p.parent_id].append(p.conversation_id)

    # Find roots: sessions that are not someone else's continuation
    # (either not a continuation, or parent not in archive)
    child_ids = {cid for cids in children.values() for cid in cids}
    roots = [p for p in all_profiles if p.conversation_id not in child_ids]

    # For each root, BFS to collect all descendants
    threads: list[WorkThread] = []
    for root in roots:
        # Collect all sessions in this thread
        thread_ids: list[str] = []
        frontier = [root.conversation_id]
        while frontier:
            next_frontier = []
            for cid in frontier:
                thread_ids.append(cid)
                next_frontier.extend(children.get(cid, []))
            frontier = next_frontier

        thread_profiles = [by_id[cid] for cid in thread_ids if cid in by_id]

        # Aggregate
        total_messages = sum(p.message_count for p in thread_profiles)
        total_cost = sum(p.total_cost_usd for p in thread_profiles)

        timestamps_start = [p.first_message_at for p in thread_profiles if p.first_message_at]
        timestamps_end = [p.last_message_at for p in thread_profiles if p.last_message_at]
        start_time = min(timestamps_start) if timestamps_start else None
        end_time = max(timestamps_end) if timestamps_end else None
        wall_ms = 0
        if start_time and end_time:
            wall_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

        # Dominant project from canonical_projects across all sessions
        project_counter: Counter[str] = Counter()
        for p in thread_profiles:
            for proj in p.canonical_projects:
                project_counter[proj] += 1
        dominant_project = project_counter.most_common(1)[0][0] if project_counter else None

        # Work event breakdown
        event_counter: Counter[str] = Counter()
        for p in thread_profiles:
            for we in p.work_events:
                kind_str = we.kind.value if hasattr(we.kind, "value") else str(we.kind)
                event_counter[kind_str] += 1

        # Depth and branch count
        depth = _bfs_depth(children, root.conversation_id)
        # Leaves = sessions with no children
        branch_count = sum(1 for cid in thread_ids if not children.get(cid))

        threads.append(WorkThread(
            thread_id=root.conversation_id,
            root_id=root.conversation_id,
            session_ids=tuple(thread_ids),
            depth=depth,
            branch_count=branch_count,
            start_time=start_time,
            end_time=end_time,
            wall_duration_ms=wall_ms,
            total_messages=total_messages,
            total_cost_usd=total_cost,
            dominant_project=dominant_project,
            work_event_breakdown=dict(event_counter),
        ))

    return threads
