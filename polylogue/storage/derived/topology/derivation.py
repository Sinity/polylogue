"""Canonical session topology projection over ``session_links``.

The ``sessions`` parent/root columns are write-side accelerators. They are
not read here: every public topology edge starts as a preserved natural
``session_links`` row, then this module solely decides composability.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyEdgeKind, TopologyNode
from polylogue.archive.topology.edge import status_excludes_composition
from polylogue.core.types import SessionId
from polylogue.storage.runtime import SessionRecord


@dataclass(frozen=True)
class TopologyNodeInput:
    """The only session-node facts the graph engine reads.

    ``compose_session_topology`` never needed a whole :class:`SessionRecord`; narrowing the
    input to these three fields is what lets a caller holding raw ``sessions``
    rows (the coordination context envelope) reach the one canonical engine
    instead of growing a second graph construction.
    """

    session_id: str
    origin: str
    title: str | None = None


def node_input_from_record(record: SessionRecord) -> TopologyNodeInput:
    return TopologyNodeInput(
        session_id=str(record.session_id),
        origin=record.origin.value,
        title=record.title,
    )


class _SessionQuerySource(Protocol):
    async def get_session(self, session_id: str) -> SessionRecord | None: ...
    async def list_session_links_for_session(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, object]]: ...
    async def list_session_links_to_session(self, session_id: str, *, limit: int) -> list[dict[str, object]]: ...


def _kind(value: object) -> TopologyEdgeKind:
    try:
        return TopologyEdgeKind(str(value))
    except ValueError:
        return TopologyEdgeKind.UNKNOWN


def _evidence(value: object) -> list[object]:
    """Normalize stored JSON into the public evidence-array contract."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [{"state": "invalid-json"}]
    return parsed if isinstance(parsed, list) else [parsed]


def _draft_edge(link: Mapping[str, object]) -> TopologyEdge:
    status = link.get("status")
    parent = link.get("resolved_dst_session_id")
    resolved = isinstance(parent, str) and bool(parent)
    parent_id = parent if isinstance(parent, str) and parent else None
    excluded = status_excludes_composition(status)
    dst_native_id = str(link["dst_native_id"]) if link.get("dst_native_id") is not None else None
    return TopologyEdge(
        child_id=SessionId(str(link["src_session_id"])),
        parent_id=SessionId(parent_id) if parent_id is not None else None,
        dst_origin=str(link.get("dst_origin") or ""),
        dst_native_id=dst_native_id,
        parent_native_id=dst_native_id if not resolved else None,
        kind=_kind(link.get("link_type")),
        resolved=resolved,
        inheritance=str(link["inheritance"]) if link.get("inheritance") is not None else None,
        branch_point_message_id=str(link["branch_point_message_id"]) if link.get("branch_point_message_id") else None,
        authority_state=str(status) if status is not None else "accepted",
        resolution_state="resolved" if resolved else "unresolved",
        composable=resolved and not excluded,
        composability_reason=str(status) if excluded else (None if resolved else "unresolved-parent"),
        parent_tool_use_block_id=(
            str(link["parent_tool_use_block_id"]) if link.get("parent_tool_use_block_id") else None
        ),
        method=str(link["method"]) if link.get("method") is not None else None,
        confidence=float(str(link.get("confidence") or 0.0)),
        observed_at_ms=(int(str(link["observed_at_ms"])) if link.get("observed_at_ms") is not None else None),
        resolved_at_ms=(int(str(link["resolved_at_ms"])) if link.get("resolved_at_ms") is not None else None),
        evidence=_evidence(link.get("evidence_json")),
    )


def _exclude(edge: TopologyEdge, reason: str) -> TopologyEdge:
    return edge.model_copy(update={"composable": False, "composability_reason": reason})


def _cycle_indexes(edges: Sequence[TopologyEdge]) -> set[int]:
    """Mark every edge on a path that reaches a back edge.

    The traversal carries its own stack rather than recursing. The writer admits
    a lineage chain up to ``_CYCLE_WALK_BUDGET`` (1024) steps deep, and a
    recursive depth-first walk raises ``RecursionError`` before that -- which
    this classifier runs over every link in the archive, so one deep but
    perfectly acyclic lineage made unrelated topology reads fail outright.
    Frame-for-frame the same walk: a node enters ``visiting`` on push and moves
    to ``visited`` on pop, a child already in ``visiting`` is a back edge that
    marks the whole current ancestry, and roots are still taken in sorted order.
    """
    children: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        if edge.composable and edge.parent_id is not None:
            children[str(edge.parent_id)].append((index, str(edge.child_id)))
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[int] = set()

    for root in sorted(children):
        if root in visited:
            continue
        visiting.add(root)
        stack: list[tuple[str, list[int], Iterator[tuple[int, str]]]] = [(root, [], iter(children.get(root, ())))]
        while stack:
            node, ancestry, pending = stack[-1]
            descended = False
            for index, child in pending:
                if child in visiting:
                    cycles.update(ancestry)
                    cycles.add(index)
                elif child not in visited:
                    visiting.add(child)
                    stack.append((child, [*ancestry, index], iter(children.get(child, ()))))
                    descended = True
                    break
            if not descended:
                stack.pop()
                visiting.discard(node)
                visited.add(node)
    return cycles


def compose_session_topology(
    target_id: str,
    nodes: Sequence[TopologyNodeInput],
    links: Sequence[Mapping[str, object]],
) -> SessionTopology | None:
    """Classify one topology graph from canonical ``session_links`` rows.

    This is the single graph classification engine.  Every public topology
    route -- async paged reads, the sync adapter, and the coordination context
    envelope -- composes through here; no caller may classify an edge itself.
    """
    records_by_id = {str(record.session_id): record for record in nodes}
    if target_id not in records_by_id:
        return None
    unique_links: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for link in links:
        if str(link.get("src_session_id")) not in records_by_id:
            continue
        key = (
            str(link.get("src_session_id") or ""),
            str(link.get("dst_origin") or ""),
            str(link.get("dst_native_id") or ""),
            str(link.get("link_type") or ""),
        )
        unique_links[key] = link
    edges = [_draft_edge(link) for _, link in sorted(unique_links.items())]

    # A resolved edge can point at a session outside this bounded scope (a
    # tight page, or an ancestor the walk stopped short of). Traversing it
    # would compose a node we hold no record for. Name the gap on the edge
    # instead of crashing or silently dropping it.
    edges = [
        _exclude(edge, "parent-out-of-scope")
        if edge.composable and edge.parent_id is not None and str(edge.parent_id) not in records_by_id
        else edge
        for edge in edges
    ]

    parent_sets: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.composable and edge.parent_id is not None:
            parent_sets[str(edge.child_id)].add(str(edge.parent_id))
    conflicts = {child for child, parents in parent_sets.items() if len(parents) > 1}
    edges = [
        _exclude(edge, "conflicting-parent") if edge.composable and str(edge.child_id) in conflicts else edge
        for edge in edges
    ]
    cycles = _cycle_indexes(edges)
    edges = [_exclude(edge, "cycle") if index in cycles else edge for index, edge in enumerate(edges)]

    parent_of: dict[str, str] = {}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.composable and edge.parent_id is not None:
            child, parent = str(edge.child_id), str(edge.parent_id)
            parent_of[child] = parent
            children[parent].append(child)
    for value in children.values():
        value.sort()

    root, seen = target_id, {target_id}
    while root in parent_of and parent_of[root] not in seen:
        root = parent_of[root]
        seen.add(root)
    depths: dict[str, int] = {root: 0}
    node_ids: list[str] = []
    queue: deque[str] = deque([root])
    while queue:
        current = queue.popleft()
        node_ids.append(current)
        for child in children.get(current, ()):
            if child not in depths:
                depths[child] = depths[current] + 1
                queue.append(child)
    included = set(node_ids)
    return SessionTopology(
        target_id=SessionId(target_id),
        root_id=SessionId(root),
        nodes=tuple(
            TopologyNode(
                session_id=SessionId(node_id),
                origin=records_by_id[node_id].origin,
                title=records_by_id[node_id].title,
                depth=depths[node_id],
                is_root=node_id == root,
            )
            for node_id in node_ids
        ),
        edges=tuple(edge for edge in edges if str(edge.child_id) in included),
        cycle_detected=bool(cycles),
        conflicting_parent_detected=bool(conflicts),
    )


async def derive_session_topology_async(
    source: _SessionQuerySource,
    session_id: str,
    *,
    node_offset: int = 0,
    node_limit: int = 200,
    edge_limit: int = 500,
) -> SessionTopology | None:
    """Compose one bounded, stable page from canonical link-neighborhood reads.

    The continuation is an opaque ``node-offset`` token.  Each request
    recomputes a deterministic BFS prefix from canonical rows, then returns a
    page from that prefix.  No route scans sessions or session_links globally.
    """
    target = await source.get_session(session_id)
    if target is None:
        return None
    if node_offset < 0 or node_limit < 1 or edge_limit < 1:
        raise ValueError("topology page bounds must be positive")

    # Resolve the root by walking only this child's canonical outbound rows.
    records: dict[str, SessionRecord] = {str(target.session_id): target}
    links: list[Mapping[str, object]] = []
    current = target
    seen_ancestors = {str(current.session_id)}
    link_truncated = False
    while True:
        outbound = await source.list_session_links_for_session(str(current.session_id), limit=edge_limit + 1)
        if len(outbound) > edge_limit:
            link_truncated = True
            outbound = outbound[:edge_limit]
        links.extend(outbound)
        candidates = [edge for edge in (_draft_edge(link) for link in outbound) if edge.composable and edge.parent_id]
        parents = {str(edge.parent_id) for edge in candidates if edge.parent_id is not None}
        if len(parents) != 1:
            break
        parent_id = parents.pop()
        if parent_id in seen_ancestors:
            break
        parent = await source.get_session(parent_id)
        if parent is None:
            break
        records[parent_id] = parent
        seen_ancestors.add(parent_id)
        current = parent
    root_id = str(current.session_id)

    scan_limit = node_offset + node_limit + 1
    queue: deque[str] = deque([root_id])
    discovered: set[str] = set()
    bfs_ids: list[str] = []
    while queue and len(bfs_ids) < scan_limit:
        current_id = queue.popleft()
        if current_id in discovered:
            continue
        record = records.get(current_id) or await source.get_session(current_id)
        if record is None:
            continue
        records[current_id] = record
        discovered.add(current_id)
        bfs_ids.append(current_id)
        outbound = await source.list_session_links_for_session(current_id, limit=edge_limit + 1)
        if len(outbound) > edge_limit:
            link_truncated = True
            outbound = outbound[:edge_limit]
        links.extend(outbound)
        inbound = await source.list_session_links_to_session(current_id, limit=scan_limit + 1)
        if len(inbound) > scan_limit:
            link_truncated = True
            inbound = inbound[:scan_limit]
        links.extend(inbound)
        for edge in sorted((_draft_edge(link) for link in inbound), key=lambda item: str(item.child_id)):
            if not edge.composable or str(edge.parent_id) != current_id:
                continue
            child_id = str(edge.child_id)
            if child_id not in discovered:
                child = await source.get_session(child_id)
                if child is not None:
                    records[child_id] = child
                    queue.append(child_id)

    composed = compose_session_topology(
        str(target.session_id), [node_input_from_record(record) for record in records.values()], links
    )
    if composed is None:
        return None
    # `compose_session_topology` is the one graph classification; paging only trims its
    # already classified stable BFS output and never remaps an edge.
    all_nodes = composed.nodes
    page_nodes = all_nodes[node_offset : node_offset + node_limit]
    page_ids = {str(node.session_id) for node in page_nodes}
    page_edges = tuple(edge for edge in composed.edges if str(edge.child_id) in page_ids)[:edge_limit]
    more_nodes = len(all_nodes) > node_offset + len(page_nodes) or bool(queue)
    edges_complete = not link_truncated and len(page_edges) == len(
        tuple(edge for edge in composed.edges if str(edge.child_id) in page_ids)
    )
    return composed.model_copy(
        update={
            "nodes": page_nodes,
            "edges": page_edges,
            "nodes_complete": not more_nodes,
            "edges_complete": edges_complete,
            "continuation": None if not more_nodes else f"node-offset:{node_offset + len(page_nodes)}",
        }
    )


_SyncFetcher = Callable[[str], SessionRecord | None]
_SyncChildrenFetcher = Callable[[str], list[SessionRecord]]
_SyncLinksFetcher = Callable[[str], list[Mapping[str, object]]]


def derive_session_topology_sync(
    session_id: str,
    *,
    fetch: _SyncFetcher,
    fetch_children: _SyncChildrenFetcher,
    fetch_links: _SyncLinksFetcher | None = None,
) -> SessionTopology | None:
    """Test adapter which invokes the same canonical composition engine."""
    target = fetch(session_id)
    if target is None:
        return None
    records: dict[str, SessionRecord] = {str(target.session_id): target}
    pending: deque[SessionRecord] = deque([target])
    while pending:
        record = pending.popleft()
        for child in fetch_children(str(record.session_id)):
            if str(child.session_id) not in records:
                records[str(child.session_id)] = child
                pending.append(child)
    links: list[Mapping[str, object]] = []
    if fetch_links is not None:
        pending_ids: deque[str] = deque(records)
        queried: set[str] = set()
        while pending_ids:
            record_id = pending_ids.popleft()
            if record_id in queried:
                continue
            queried.add(record_id)
            for link in fetch_links(record_id):
                links.append(link)
                parent_id = link.get("resolved_dst_session_id")
                if isinstance(parent_id, str) and parent_id not in records:
                    parent = fetch(parent_id)
                    if parent is not None:
                        records[parent_id] = parent
                        pending_ids.append(parent_id)
    return compose_session_topology(
        str(target.session_id), [node_input_from_record(record) for record in records.values()], links
    )


__all__ = [
    "TopologyNodeInput",
    "compose_session_topology",
    "derive_session_topology_async",
    "derive_session_topology_sync",
    "node_input_from_record",
]
