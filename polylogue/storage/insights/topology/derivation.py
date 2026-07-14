"""Derive ``SessionTopology`` from the canonical archive tables.

This module is the single source of truth for how the topology read model
is built. The async derivation walks from the requested session up
to the topology root (cycle-safe), then performs a BFS over resolved
children to enumerate the rooted subtree, classifying every visited
edge and recording unresolved native-parent pointers from ``session_links``.

Cycle handling: a cycle is reported through
``SessionTopology.cycle_detected`` and the ancestry walk terminates at
the repeated node. Cycle quarantine is the caller's responsibility (see
#866).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from polylogue.core.types import SessionId
from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import SessionRecord


class _SessionQuerySource(Protocol):
    """Subset of the query-store surface needed by the derivation."""

    async def get_session(self, session_id: str) -> SessionRecord | None: ...

    async def list_sessions(self, request: SessionRecordQuery) -> list[SessionRecord]: ...

    async def list_session_links_for_session(self, session_id: str) -> list[dict[str, object]]: ...


def _edge_kind(record: SessionRecord) -> TopologyEdgeKind:
    return TopologyEdgeKind.from_branch_type(record.branch_type)


def _node_from_record(record: SessionRecord, *, depth: int, is_root: bool) -> TopologyNode:
    return TopologyNode(
        session_id=SessionId(str(record.session_id)),
        source_name=record.origin.value,
        title=record.title,
        depth=depth,
        is_root=is_root,
    )


def _resolved_edge(record: SessionRecord) -> TopologyEdge | None:
    if record.parent_session_id is None:
        return None
    return TopologyEdge(
        child_id=SessionId(str(record.session_id)),
        parent_id=SessionId(str(record.parent_session_id)),
        kind=_edge_kind(record),
        resolved=True,
    )


def _edge_kind_from_link(link_type: object) -> TopologyEdgeKind:
    if isinstance(link_type, str) and link_type:
        try:
            return TopologyEdgeKind(link_type)
        except ValueError:
            return TopologyEdgeKind.UNKNOWN
    return TopologyEdgeKind.UNKNOWN


def _unresolved_edges_from_links(record: SessionRecord, links: Sequence[Mapping[str, object]]) -> list[TopologyEdge]:
    edges: list[TopologyEdge] = []
    for link in links:
        if link.get("resolved_dst_session_id") is not None:
            continue
        if link.get("status") == "quarantined":
            continue
        dst_native_id = link.get("dst_native_id")
        if not isinstance(dst_native_id, str) or not dst_native_id.strip():
            continue
        edges.append(
            TopologyEdge(
                child_id=SessionId(str(record.session_id)),
                parent_id=None,
                parent_native_id=dst_native_id,
                kind=_edge_kind_from_link(link.get("link_type")),
                resolved=False,
            )
        )
    return edges


async def _walk_to_root(
    source: _SessionQuerySource,
    start: SessionRecord,
) -> tuple[SessionRecord, bool]:
    """Walk parent pointers from ``start`` to the topology root.

    Returns the root record plus ``cycle_detected``. A cycle is signaled
    when the walk revisits an already-seen session; in that case
    the function returns the last unique record reached.
    """

    seen: set[str] = {str(start.session_id)}
    current = start
    cycle = False
    while current.parent_session_id is not None:
        parent_id = str(current.parent_session_id)
        if parent_id in seen:
            cycle = True
            break
        parent = await source.get_session(parent_id)
        if parent is None:
            # Parent is referenced but not present — topology root for
            # the resolved subtree is ``current``. The dangling pointer
            # itself is reported on ``current``'s edge through the
            # resolved-edge classification (parent_id retained).
            break
        seen.add(parent_id)
        current = parent
    return current, cycle


async def derive_session_topology_async(
    source: _SessionQuerySource,
    session_id: str,
) -> SessionTopology | None:
    """Derive ``SessionTopology`` for ``session_id``.

    Returns ``None`` if the session does not exist.
    """

    target = await source.get_session(session_id)
    if target is None:
        return None

    root, cycle = await _walk_to_root(source, target)

    nodes: list[TopologyNode] = []
    edges: list[TopologyEdge] = []
    seen_nodes: set[str] = set()
    queue: list[tuple[SessionRecord, int]] = [(root, 0)]

    while queue:
        record, depth = queue.pop(0)
        record_id = str(record.session_id)
        if record_id in seen_nodes:
            # BFS guard — a cycle in the descendant graph is the same
            # structural failure mode as the ancestry one; surface it
            # through ``cycle_detected`` and skip the revisit.
            cycle = True
            continue
        seen_nodes.add(record_id)
        nodes.append(_node_from_record(record, depth=depth, is_root=record_id == str(root.session_id)))

        resolved = _resolved_edge(record)
        if resolved is not None:
            edges.append(resolved)
        edges.extend(_unresolved_edges_from_links(record, await source.list_session_links_for_session(record_id)))

        children = await source.list_sessions(SessionRecordQuery(parent_id=record_id))
        for child in children:
            queue.append((child, depth + 1))

    return SessionTopology(
        target_id=SessionId(str(target.session_id)),
        root_id=SessionId(str(root.session_id)),
        nodes=tuple(nodes),
        edges=tuple(edges),
        cycle_detected=cycle,
    )


# A pure-Python sync derivation is exposed for tests and offline tools
# that operate on a sequence of ``SessionRecord`` objects without a
# query-store. The async path above is the production entry point.

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
    """Sync variant of :func:`derive_session_topology_async`."""

    target = fetch(session_id)
    if target is None:
        return None

    seen_ancestors: set[str] = {str(target.session_id)}
    current = target
    cycle = False
    while current.parent_session_id is not None:
        parent_id = str(current.parent_session_id)
        if parent_id in seen_ancestors:
            cycle = True
            break
        parent = fetch(parent_id)
        if parent is None:
            break
        seen_ancestors.add(parent_id)
        current = parent

    nodes: list[TopologyNode] = []
    edges: list[TopologyEdge] = []
    seen_nodes: set[str] = set()
    queue: list[tuple[SessionRecord, int]] = [(current, 0)]
    root_id = str(current.session_id)

    while queue:
        record, depth = queue.pop(0)
        record_id = str(record.session_id)
        if record_id in seen_nodes:
            cycle = True
            continue
        seen_nodes.add(record_id)
        nodes.append(_node_from_record(record, depth=depth, is_root=record_id == root_id))

        resolved = _resolved_edge(record)
        if resolved is not None:
            edges.append(resolved)
        edges.extend(_unresolved_edges_from_links(record, [] if fetch_links is None else fetch_links(record_id)))

        for child in fetch_children(record_id):
            queue.append((child, depth + 1))

    return SessionTopology(
        target_id=SessionId(str(target.session_id)),
        root_id=SessionId(root_id),
        nodes=tuple(nodes),
        edges=tuple(edges),
        cycle_detected=cycle,
    )


__all__ = [
    "derive_session_topology_async",
    "derive_session_topology_sync",
]
