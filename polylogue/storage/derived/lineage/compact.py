"""Derive the compact seed-relative lineage graph (polylogue-4ts.9).

The derivation is a bounded set of SQL statements over ``sessions``,
``session_links`` and message *counts*. It never selects a message body or
touches ``blocks``: a 130-session family costs a handful of indexed reads, not
a family-wide transcript hydration.

Structure of one derivation:

1. resolve the seed, walk to the topology root, enumerate the rooted subtree;
2. classify every node and edge relative to the seed;
3. account each node's unique-versus-inherited message counts by replaying the
   *shape* of composition (segment lengths only), reporting ``unknown`` rather
   than a plausible number when stored rows cannot reproduce it;
4. page nodes and edges independently, with the seed always present.

Every statement runs on the caller's connection, so a caller holding the
shared query transaction gets one snapshot for the whole graph.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TypeVar

from polylogue.analysis.lineage_graph import (
    DEFAULT_LINEAGE_PAGE_LIMIT,
    CompactLineageEdge,
    CompactLineageGraph,
    CompactLineageNode,
    LineageAccountingStatus,
    LineageEdgeResolution,
    LineageEdgeRole,
    LineageMessageAccounting,
    LineageNodeRole,
    LineagePage,
)
from polylogue.archive.topology.edge import status_excludes_composition, topology_status_composes_sql
from polylogue.core.enums import TopologyEdgeStatus
from polylogue.core.types import MessageId, SessionId
from polylogue.storage.runtime.store_constants import LINEAGE_ITERATIVE_DEPTH_LIMIT

#: Runaway guard for the ancestry walk and the descendant sweep. A ``visited``
#: set is the real cycle guard; this only bounds a pathological archive.
_MAX_DEPTH = LINEAGE_ITERATIVE_DEPTH_LIMIT

DEFAULT_PAGE_LIMIT = DEFAULT_LINEAGE_PAGE_LIMIT

_ACCOUNTING_DANGLING = "branch point resolves to no stored message"
_ACCOUNTING_DEPTH_LIMIT = "lineage chain exceeds the composition depth limit"
_ACCOUNTING_NOT_REQUESTED = "accounting not requested for this read"


def _resolve_seed(conn: sqlite3.Connection, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? LIMIT 1",
        (session_id, session_id),
    ).fetchone()
    return None if row is None else str(row[0])


def _session_rows(conn: sqlite3.Connection, session_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"""
        SELECT session_id, origin, title, parent_session_id, branch_type
        FROM sessions
        WHERE session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    return {str(row["session_id"]): row for row in rows}


def _ancestry(conn: sqlite3.Connection, seed_id: str) -> tuple[list[str], bool]:
    """Return the seed's parent chain (nearest first) and whether it cycled."""
    chain: list[str] = []
    seen = {seed_id}
    current = seed_id
    for _ in range(_MAX_DEPTH):
        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            (current,),
        ).fetchone()
        if row is None or row[0] is None:
            return chain, False
        parent = str(row[0])
        if parent in seen:
            return chain, True
        if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (parent,)).fetchone() is None:
            # Parent referenced but absent: the resolved subtree's root is the
            # last stored session. The dangling pointer surfaces as an edge.
            return chain, False
        seen.add(parent)
        chain.append(parent)
        current = parent
    return chain, False


def _subtree(conn: sqlite3.Connection, root_id: str) -> tuple[dict[str, int], bool]:
    """Breadth-first sweep of the rooted subtree, returning depth-from-root."""
    depths: dict[str, int] = {root_id: 0}
    frontier = [root_id]
    cycle = False
    depth = 0
    while frontier and depth < _MAX_DEPTH:
        placeholders = ", ".join("?" for _ in frontier)
        rows = conn.execute(
            f"""
            SELECT session_id, parent_session_id
            FROM sessions
            WHERE parent_session_id IN ({placeholders})
            ORDER BY session_id
            """,
            tuple(frontier),
        ).fetchall()
        depth += 1
        next_frontier: list[str] = []
        for row in rows:
            child = str(row["session_id"])
            if child in depths:
                cycle = True
                continue
            depths[child] = depth
            next_frontier.append(child)
        frontier = next_frontier
    return depths, cycle


def _descendants(conn: sqlite3.Connection, seed_id: str, depths: Mapping[str, int]) -> dict[str, int]:
    """Seed-relative depth of every session in the seed's own subtree."""
    out: dict[str, int] = {seed_id: 0}
    frontier = [seed_id]
    depth = 0
    while frontier and depth < _MAX_DEPTH:
        placeholders = ", ".join("?" for _ in frontier)
        rows = conn.execute(
            f"""
            SELECT session_id FROM sessions
            WHERE parent_session_id IN ({placeholders})
            ORDER BY session_id
            """,
            tuple(frontier),
        ).fetchall()
        depth += 1
        next_frontier = [str(row["session_id"]) for row in rows if str(row["session_id"]) not in out]
        for child in next_frontier:
            out[child] = depth
        frontier = next_frontier
    return {key: value for key, value in out.items() if key in depths}


def _link_rows(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[sqlite3.Row]:
    if not session_ids:
        return []
    placeholders = ", ".join("?" for _ in session_ids)
    return conn.execute(
        f"""
        SELECT src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
               branch_point_message_id, inheritance, status, method, confidence
        FROM session_links
        WHERE src_session_id IN ({placeholders})
        ORDER BY src_session_id, link_type, dst_origin, dst_native_id
        """,
        tuple(session_ids),
    ).fetchall()


def _resolution(row: sqlite3.Row) -> LineageEdgeResolution:
    status = row["status"]
    if isinstance(status, str) and status.strip():
        if status_excludes_composition(status):
            return (
                LineageEdgeResolution.QUARANTINED
                if status.strip() == TopologyEdgeStatus.QUARANTINED.value
                else LineageEdgeResolution.AUTHORITY_CONTRADICTED
            )
        if status.strip() == TopologyEdgeStatus.REPAIRED.value:
            return LineageEdgeResolution.REPAIRED
    if row["resolved_dst_session_id"] is not None:
        return LineageEdgeResolution.RESOLVED
    return LineageEdgeResolution.UNRESOLVED


def _edge_role(seed_id: str, child_id: str, parent_id: str | None) -> LineageEdgeRole:
    if child_id == seed_id:
        return LineageEdgeRole.SEED_PARENT
    if parent_id is not None and parent_id == seed_id:
        return LineageEdgeRole.SEED_CHILD
    return LineageEdgeRole.FAMILY


def _node_role(
    session_id: str,
    *,
    seed_id: str,
    ancestors: AbstractSet[str],
    descendants: Mapping[str, int],
    seed_parent: str | None,
    parent_of: Mapping[str, str | None],
) -> LineageNodeRole:
    if session_id == seed_id:
        return LineageNodeRole.SEED
    if session_id in ancestors:
        return LineageNodeRole.ANCESTOR
    if session_id in descendants:
        return LineageNodeRole.DESCENDANT
    if seed_parent is not None and parent_of.get(session_id) == seed_parent:
        return LineageNodeRole.SIBLING
    return LineageNodeRole.FAMILY


class _CompositionShape:
    """Composed-transcript segment lengths, without any message body.

    Composition splices a child's stored tail onto its parent's composed
    prefix at the branch point. Only the *lengths* of those segments are
    needed to say how many messages a session inherited, so this replays the
    same walk over counts and ranks.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._own_counts: dict[str, int] = {}
        self._segments: dict[str, list[tuple[str, int]] | None] = {}

    def _own_count(self, session_id: str) -> int:
        if session_id not in self._own_counts:
            row = self._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
            self._own_counts[session_id] = int(row[0])
        return self._own_counts[session_id]

    def _prefix_edge(self, session_id: str) -> tuple[str, str] | None:
        row = self._conn.execute(
            f"""
            SELECT resolved_dst_session_id, branch_point_message_id
            FROM session_links
            WHERE src_session_id = ?
              AND inheritance = 'prefix-sharing'
              AND resolved_dst_session_id IS NOT NULL
              AND branch_point_message_id IS NOT NULL
              AND {topology_status_composes_sql()}
            ORDER BY link_type, dst_origin, dst_native_id
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]))

    def _rank_within_own(self, session_id: str, message_id: str) -> int | None:
        """1-based transcript rank of a message inside its session's own rows."""
        row = self._conn.execute(
            "SELECT position, variant_index FROM messages WHERE message_id = ? AND session_id = ?",
            (message_id, session_id),
        ).fetchone()
        if row is None:
            return None
        count = self._conn.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE session_id = ?
              AND (position < ? OR (position = ? AND variant_index <= ?))
            """,
            (session_id, row["position"], row["position"], row["variant_index"]),
        ).fetchone()
        return int(count[0])

    def segments(self, session_id: str, _depth: int = 0) -> list[tuple[str, int]] | None:
        """Composed transcript as ``(owning_session, length)`` runs, or ``None``.

        ``None`` means composition is not reproducible from stored rows, which
        is exactly when an accounting number would have to be invented.
        """
        if session_id in self._segments:
            return self._segments[session_id]
        if _depth >= _MAX_DEPTH:
            self._segments[session_id] = None
            return None
        edge = self._prefix_edge(session_id)
        if edge is None:
            result: list[tuple[str, int]] | None = [(session_id, self._own_count(session_id))]
            self._segments[session_id] = result
            return result
        parent_id, branch_point_message_id = edge
        # Guard the recursion against a cyclic link before descending.
        self._segments[session_id] = None
        parent_segments = self.segments(parent_id, _depth + 1)
        if parent_segments is None:
            return None
        prefix = self._truncate_at(parent_segments, branch_point_message_id)
        if prefix is None:
            return None
        result = [*prefix, (session_id, self._own_count(session_id))]
        self._segments[session_id] = result
        return result

    def _truncate_at(
        self, segments: Sequence[tuple[str, int]], branch_point_message_id: str
    ) -> list[tuple[str, int]] | None:
        """Cut a composed segment list after the branch-point message."""
        out: list[tuple[str, int]] = []
        for owner, length in segments:
            rank = self._rank_within_own(owner, branch_point_message_id)
            if rank is None or rank > length:
                out.append((owner, length))
                continue
            out.append((owner, rank))
            return out
        return None

    def accounting(self, session_id: str) -> LineageMessageAccounting:
        edge = self._prefix_edge(session_id)
        own = self._own_count(session_id)
        if edge is None:
            return LineageMessageAccounting(status=LineageAccountingStatus.KNOWN, unique=own, inherited=0)
        parent_id, branch_point_message_id = edge
        parent_segments = self.segments(parent_id)
        if parent_segments is None:
            return LineageMessageAccounting(status=LineageAccountingStatus.UNKNOWN, reason=_ACCOUNTING_DEPTH_LIMIT)
        prefix = self._truncate_at(parent_segments, branch_point_message_id)
        if prefix is None:
            return LineageMessageAccounting(status=LineageAccountingStatus.UNKNOWN, reason=_ACCOUNTING_DANGLING)
        return LineageMessageAccounting(
            status=LineageAccountingStatus.KNOWN,
            unique=own,
            inherited=sum(length for _, length in prefix),
        )


_Item = TypeVar("_Item")


def _window(items: Sequence[_Item], offset: int, limit: int | None) -> list[_Item]:
    start = max(0, offset)
    return list(items[start:] if limit is None else items[start : start + limit])


def derive_compact_lineage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    node_offset: int = 0,
    node_limit: int | None = DEFAULT_PAGE_LIMIT,
    edge_offset: int = 0,
    edge_limit: int | None = DEFAULT_PAGE_LIMIT,
    include_accounting: bool = True,
) -> CompactLineageGraph | None:
    """Return the seed-relative compact lineage graph, or ``None`` if absent.

    A ``None`` limit is an unbounded window. ``include_accounting=False`` skips
    the per-node composition-shape queries, which are the only work that scales
    with family size beyond the flat table reads; the counts then report
    ``unknown`` rather than a number nobody computed.
    """

    seed_id = _resolve_seed(conn, session_id)
    if seed_id is None:
        return None

    ancestors, cycle = _ancestry(conn, seed_id)
    root_id = ancestors[-1] if ancestors else seed_id
    depths, subtree_cycle = _subtree(conn, root_id)
    cycle = cycle or subtree_cycle
    # A cyclic parent chain can stop the ancestry walk at a node the subtree
    # sweep never reaches. The seed is the one node this read exists to return.
    depths.setdefault(seed_id, len(ancestors))
    descendants = _descendants(conn, seed_id, depths)

    family = sorted(depths)
    rows = _session_rows(conn, family)
    parent_of = {
        key: (str(row["parent_session_id"]) if row["parent_session_id"] else None) for key, row in rows.items()
    }
    ancestor_set = set(ancestors)
    seed_parent = parent_of.get(seed_id)
    shape = _CompositionShape(conn)

    ancestor_depth = {ancestor: -(index + 1) for index, ancestor in enumerate(ancestors)}
    nodes: list[CompactLineageNode] = []
    for member in family:
        row = rows.get(member)
        if row is None:
            continue
        role = _node_role(
            member,
            seed_id=seed_id,
            ancestors=ancestor_set,
            descendants=descendants,
            seed_parent=seed_parent,
            parent_of=parent_of,
        )
        nodes.append(
            CompactLineageNode(
                session_id=SessionId(member),
                origin=str(row["origin"] or ""),
                title=row["title"],
                role=role,
                depth_from_root=depths[member],
                depth_from_seed=ancestor_depth.get(member, descendants.get(member)),
                is_root=member == root_id,
                is_seed=member == seed_id,
                branch_type=row["branch_type"],
            )
        )

    # Stable order, seed first: the seed occupies one slot of every page, so a
    # caller paging by depth never loses the session it asked about.
    nodes.sort(key=lambda node: (node.depth_from_root, str(node.session_id)))
    seed_node = next(node for node in nodes if node.is_seed)
    remainder = [node for node in nodes if not node.is_seed]
    node_window = (
        seed_node,
        *_window(remainder, node_offset, None if node_limit is None else max(0, node_limit - 1)),
    )
    # Accounting is the only per-node work, so it runs for the page the caller
    # asked for, not for the whole family behind it.
    node_window = tuple(
        node.model_copy(
            update={
                "accounting": (
                    shape.accounting(str(node.session_id))
                    if include_accounting
                    else LineageMessageAccounting(reason=_ACCOUNTING_NOT_REQUESTED)
                )
            }
        )
        for node in node_window
    )

    edges = _compact_edges(conn, seed_id, family, parent_of)
    edge_window = _window(edges, edge_offset, edge_limit)

    return CompactLineageGraph(
        seed_id=SessionId(seed_id),
        root_id=SessionId(root_id),
        nodes=node_window,
        edges=tuple(edge_window),
        # The page addresses the non-seed nodes; the seed rides every page for
        # free, so paging forward by ``returned`` never skips or repeats a node.
        node_page=LineagePage(
            offset=max(0, node_offset),
            limit=node_limit,
            returned=len(node_window) - 1,
            total=len(remainder),
        ),
        edge_page=LineagePage(
            offset=max(0, edge_offset),
            limit=edge_limit,
            returned=len(edge_window),
            total=len(edges),
        ),
        cycle_detected=cycle,
    )


def _compact_edges(
    conn: sqlite3.Connection,
    seed_id: str,
    family: Sequence[str],
    parent_of: Mapping[str, str | None],
) -> list[CompactLineageEdge]:
    """Project ``session_links`` rows, then cover any structural parent they miss.

    ``sessions.parent_session_id`` is the structural edge; ``session_links`` is
    where the relationship's meaning lives. A resolved parent with no link row
    still gets an edge, marked with the column it came from, so the graph never
    silently omits a relationship the tree shows.
    """
    edges: list[CompactLineageEdge] = []
    covered: set[tuple[str, str]] = set()
    for row in _link_rows(conn, family):
        child_id = str(row["src_session_id"])
        parent_id = str(row["resolved_dst_session_id"]) if row["resolved_dst_session_id"] else None
        if parent_id is not None:
            covered.add((child_id, parent_id))
        edges.append(
            CompactLineageEdge(
                child_id=SessionId(child_id),
                parent_id=SessionId(parent_id) if parent_id else None,
                parent_native_id=str(row["dst_native_id"]) if row["dst_native_id"] else None,
                parent_origin=str(row["dst_origin"]) if row["dst_origin"] else None,
                link_type=str(row["link_type"] or "unknown"),
                inheritance=row["inheritance"],
                branch_point_message_id=(
                    MessageId(str(row["branch_point_message_id"])) if row["branch_point_message_id"] else None
                ),
                method=row["method"],
                confidence=float(row["confidence"] if row["confidence"] is not None else 1.0),
                resolution=_resolution(row),
                role=_edge_role(seed_id, child_id, parent_id),
            )
        )
    for child_id in family:
        parent_id = parent_of.get(child_id)
        if parent_id is None or (child_id, parent_id) in covered:
            continue
        edges.append(
            CompactLineageEdge(
                child_id=SessionId(child_id),
                parent_id=SessionId(parent_id),
                link_type="unknown",
                method="sessions.parent_session_id",
                resolution=LineageEdgeResolution.RESOLVED,
                role=_edge_role(seed_id, child_id, parent_id),
            )
        )
    edges.sort(
        key=lambda edge: (
            str(edge.child_id),
            edge.link_type,
            str(edge.parent_origin or ""),
            str(edge.parent_native_id or ""),
        )
    )
    return edges


__all__ = ["DEFAULT_PAGE_LIMIT", "derive_compact_lineage"]
