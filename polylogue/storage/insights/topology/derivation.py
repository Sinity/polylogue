"""Derive ``SessionTopology`` from the canonical archive tables.

This module is the single source of truth for how the topology read model
is built. The async derivation walks from the requested session up
to the topology root (cycle-safe), then performs a BFS over resolved
children to enumerate the rooted subtree, classifying every visited
edge and recording unresolved native-parent pointers found in
``provider_meta``.

Cycle handling: a cycle is reported through
``SessionTopology.cycle_detected`` and the ancestry walk terminates at
the repeated node. Cycle quarantine is the caller's responsibility (see
#866).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import SessionRecord
from polylogue.types import SessionId

# Field names inside ``sessions.provider_meta`` that have been
# observed to carry a provider-native parent identifier. The topology
# derivation treats a non-empty value at any of these keys as evidence
# of an unresolved native edge when the session has no resolved
# ``parent_session_id``.
UNRESOLVED_NATIVE_PARENT_KEYS: tuple[str, ...] = (
    "parent_session_id",
    "parentSessionId",
    "parent_session_id",
    "parentSessionId",
    "parent_id",
    "parentId",
)


class _SessionQuerySource(Protocol):
    """Subset of the query-store surface needed by the derivation."""

    async def get_session(self, session_id: str) -> SessionRecord | None: ...

    async def list_sessions(self, request: SessionRecordQuery) -> list[SessionRecord]: ...


def _extract_native_parent_id(provider_meta: dict[str, object] | None) -> str | None:
    if not provider_meta:
        return None
    for key in UNRESOLVED_NATIVE_PARENT_KEYS:
        value = provider_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _edge_kind(record: SessionRecord) -> TopologyEdgeKind:
    return TopologyEdgeKind.from_branch_type(record.branch_type)


def _node_from_record(record: SessionRecord, *, depth: int, is_root: bool) -> TopologyNode:
    return TopologyNode(
        session_id=SessionId(str(record.session_id)),
        source_name=record.source_name,
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


def _unresolved_edge(record: SessionRecord) -> TopologyEdge | None:
    """Return an unresolved native-pointer edge for ``record`` if any.

    We only flag an unresolved edge when the session has no resolved
    parent: a resolved parent always supersedes a native pointer in the
    canonical read model.
    """

    if record.parent_session_id is not None:
        return None
    native_id = _extract_native_parent_id(record.provider_meta)
    if native_id is None:
        return None
    # Treat the branch_type, when present, as the edge kind; otherwise
    # mark the edge as UNRESOLVED_NATIVE so consumers can distinguish
    # a typed-but-unresolved edge from one that lacks all classification.
    kind = (
        TopologyEdgeKind.from_branch_type(record.branch_type)
        if record.branch_type is not None
        else TopologyEdgeKind.UNRESOLVED_NATIVE
    )
    return TopologyEdge(
        child_id=SessionId(str(record.session_id)),
        parent_id=None,
        parent_native_id=native_id,
        kind=kind,
        resolved=False,
    )


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
        unresolved = _unresolved_edge(record)
        if unresolved is not None:
            edges.append(unresolved)

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


def derive_session_topology_sync(
    session_id: str,
    *,
    fetch: _SyncFetcher,
    fetch_children: _SyncChildrenFetcher,
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
        unresolved = _unresolved_edge(record)
        if unresolved is not None:
            edges.append(unresolved)

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
    "UNRESOLVED_NATIVE_PARENT_KEYS",
    "derive_session_topology_async",
    "derive_session_topology_sync",
]
