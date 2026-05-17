"""Per-conversation session-topology HTTP envelope (#1121).

Builds the JSON envelope shipped by
``GET /api/conversations/{id}/topology``. Kept in its own module so the
route handler in :mod:`polylogue.daemon.http` stays inside its declared
file-size budget.

The envelope is bounded by construction:

- ``node_limit`` caps the number of nodes copied into the payload. The
  default mirrors the reader's BFS list length; the hard cap protects the
  daemon from operator-requested unbounded subtrees.
- Edges referencing dropped nodes are filtered so the UI never plots a
  dangling endpoint.
- ``readiness`` summarises the lineage state as one of
  ``ok`` / ``partial`` / ``empty`` so the reader can show a chip without
  duplicating the truncation/cycle/unresolved logic.

The shape is consumed by the Lineage inspector tab (#1121 AC) and by
``tests/unit/daemon/test_topology_endpoint.py``.
"""

from __future__ import annotations

from typing import Final

from polylogue.insights.topology import SessionTopology, TopologyEdge, TopologyNode

#: Default ``node_limit`` used when the client does not pass ``?limit=``.
DEFAULT_NODE_LIMIT: Final[int] = 200

#: Hard cap rejected by :func:`coerce_node_limit` regardless of client input.
MAX_NODE_LIMIT: Final[int] = 1000

#: Readiness vocabulary mirrored into the reader's MK3 chip classes.
READINESS_OK: Final[str] = "ok"
READINESS_PARTIAL: Final[str] = "partial"
READINESS_EMPTY: Final[str] = "empty"


def coerce_node_limit(raw: str | None) -> int | None:
    """Parse the ``?limit=`` query param.

    Returns ``None`` to signal that the client supplied a value outside
    the ``[1, MAX_NODE_LIMIT]`` window; the caller turns that into a 400.
    Missing or empty input falls back to :data:`DEFAULT_NODE_LIMIT`.
    """

    if raw is None or raw == "":
        return DEFAULT_NODE_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value < 1 or value > MAX_NODE_LIMIT:
        return None
    return value


def _node_dict(node: TopologyNode) -> dict[str, object]:
    return {
        "conversation_id": str(node.conversation_id),
        "provider_name": node.provider_name,
        "title": node.title,
        "depth": node.depth,
        "is_root": node.is_root,
    }


def _edge_dict(edge: TopologyEdge) -> dict[str, object]:
    return {
        "child_id": str(edge.child_id),
        "parent_id": str(edge.parent_id) if edge.parent_id is not None else None,
        "parent_native_id": edge.parent_native_id,
        "kind": edge.kind.value,
        "resolved": edge.resolved,
    }


def _readiness(
    *,
    truncated_count: int,
    unresolved_edge_count: int,
    cycle_detected: bool,
    node_count: int,
) -> str:
    """Map structural state to the chip vocabulary.

    A lineage rooted at one isolated conversation (one node, zero edges,
    no unresolved pointers, no truncation, no cycle) is reported as
    ``empty`` so the reader can render the dedicated empty state. Any
    truncation, unresolved edge, or cycle is ``partial``. Everything else
    is ``ok``.
    """

    if node_count <= 1 and truncated_count == 0 and unresolved_edge_count == 0 and not cycle_detected:
        return READINESS_EMPTY
    if truncated_count > 0 or unresolved_edge_count > 0 or cycle_detected:
        return READINESS_PARTIAL
    return READINESS_OK


def build_topology_envelope(
    topology: SessionTopology,
    *,
    node_limit: int = DEFAULT_NODE_LIMIT,
) -> dict[str, object]:
    """Project a :class:`SessionTopology` into the public reader envelope.

    The envelope is shaped for direct JSON serialization:

    - ``nodes`` and ``edges`` are bounded by ``node_limit``;
    - ``truncated_count`` records the number of nodes dropped;
    - ``unresolved_edge_count`` and ``cycle_detected`` are surfaced so
      the reader's readiness chip can attribute partial state;
    - edges that point at a dropped node are filtered out — the UI
      never plots a dangling endpoint — but the truncation count tells
      the operator the graph is incomplete.
    """

    effective_limit = max(1, min(node_limit, MAX_NODE_LIMIT))
    full_nodes = list(topology.nodes)
    kept_nodes = full_nodes[:effective_limit]
    kept_ids = {str(node.conversation_id) for node in kept_nodes}
    truncated_count = len(full_nodes) - len(kept_nodes)

    kept_edges: list[dict[str, object]] = []
    unresolved_edge_count = 0
    for edge in topology.edges:
        child_key = str(edge.child_id)
        if child_key not in kept_ids:
            continue
        if edge.resolved:
            if edge.parent_id is None or str(edge.parent_id) not in kept_ids:
                # Resolved-but-parent-dropped: skip to avoid dangling lines.
                continue
        else:
            unresolved_edge_count += 1
        kept_edges.append(_edge_dict(edge))

    readiness = _readiness(
        truncated_count=truncated_count,
        unresolved_edge_count=unresolved_edge_count,
        cycle_detected=topology.cycle_detected,
        node_count=len(kept_nodes),
    )

    return {
        "target_id": str(topology.target_id),
        "root_id": str(topology.root_id),
        "nodes": [_node_dict(node) for node in kept_nodes],
        "edges": kept_edges,
        "node_count": len(kept_nodes),
        "total_node_count": len(full_nodes),
        "truncated_count": truncated_count,
        "unresolved_edge_count": unresolved_edge_count,
        "cycle_detected": topology.cycle_detected,
        "readiness": readiness,
        "node_limit": effective_limit,
    }


__all__ = [
    "DEFAULT_NODE_LIMIT",
    "MAX_NODE_LIMIT",
    "READINESS_EMPTY",
    "READINESS_OK",
    "READINESS_PARTIAL",
    "build_topology_envelope",
    "coerce_node_limit",
]
