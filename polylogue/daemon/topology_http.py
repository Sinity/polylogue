"""Per-session session-topology HTTP envelope (#1121).

Builds the JSON envelope shipped by
``GET /api/sessions/{id}/topology``. Kept in its own module so the
route handler in :mod:`polylogue.daemon.http` stays inside its declared
file-size budget.

The envelope is bounded by construction:

- ``node_limit`` caps the number of nodes copied into the payload. The
  default mirrors the reader's BFS list length; the hard cap protects the
  daemon from operator-requested unbounded subtrees.
- Edges are dropped only when an endpoint is not in the bounded node
  set, so the UI never plots a dangling endpoint. An excluded or
  unresolved edge between two kept nodes stays visible as evidence.
- ``readiness`` summarises the lineage state as one of
  ``ok`` / ``partial`` / ``empty`` for the reader's chip. It is not a
  terminal outcome: the canonical ``outcome`` field is authoritative.

The shape is consumed by the Lineage inspector tab (#1121 AC) and by
``tests/unit/daemon/test_topology_endpoint.py``.
"""

from __future__ import annotations

from typing import Final, cast

from polylogue.analysis.topology import SessionTopology
from polylogue.operations.topology_envelope import (
    DEFAULT_NODE_LIMIT,
    MAX_NODE_LIMIT,
    topology_public_envelope,
)

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


def coerce_node_offset(raw: str | None) -> int | None:
    """Parse the stable topology continuation's numeric offset."""

    if raw is None or raw == "":
        return 0
    token = raw.removeprefix("node-offset:")
    try:
        value = int(token)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _readiness(
    *,
    truncated_count: int,
    unresolved_edge_count: int,
    cycle_detected: bool,
    node_count: int,
) -> str:
    """Map structural state to the chip vocabulary.

    A lineage rooted at one isolated session (one node, zero edges,
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
    """Frame the canonical topology envelope for the HTTP reader.

    The envelope is the one produced by
    :func:`polylogue.operations.topology_envelope.topology_public_envelope`
    -- same nodes, same complete edge projection, same ancestor/descendant/
    sibling/thread lists, same terminal ``outcome`` -- bounded to
    ``node_limit`` by the shared bounding helper. HTTP adds only reader
    affordances (``node_count``, ``total_node_count``, ``truncated_count``,
    ``unresolved_edge_count``, ``readiness``, ``node_limit``); it does not
    drop canonical keys and does not re-decide the outcome.

    ``readiness`` is retained as the reader's chip vocabulary only. It is
    not a second terminal-outcome vocabulary: ``outcome`` is authoritative
    and is what the transport status maps from.
    """

    effective_limit = max(1, min(node_limit, MAX_NODE_LIMIT))
    bounded = topology_public_envelope(topology, node_limit=effective_limit)

    kept_nodes = cast("list[dict[str, object]]", bounded["nodes"])
    kept_edges = cast("list[dict[str, object]]", bounded["edges"])
    truncated_count = max(len(topology.nodes) - len(kept_nodes), int(not topology.nodes_complete))
    unresolved_edge_count = sum(1 for edge in kept_edges if not edge["resolved"])

    readiness = _readiness(
        truncated_count=truncated_count,
        unresolved_edge_count=unresolved_edge_count,
        cycle_detected=topology.cycle_detected,
        node_count=len(kept_nodes),
    )

    return {
        **bounded,
        "node_count": len(kept_nodes),
        "total_node_count": len(topology.nodes) if topology.nodes_complete else None,
        "truncated_count": truncated_count,
        "unresolved_edge_count": unresolved_edge_count,
        "readiness": readiness,
        "node_limit": effective_limit,
    }


def build_parent_chain_envelope(
    topology: SessionTopology,
    *,
    include_descendants: bool = True,
) -> dict[str, object]:
    """Project a :class:`SessionTopology` into a stack-ready chain envelope.

    Returns the ordered chain of session IDs from the topology root
    down to ``topology.target_id``, optionally followed by the BFS-ordered
    descendants of the target. The returned envelope is shaped to seed
    the reader's stack workspace route:

    - ``chain_ids`` is the canonical oldest-to-newest list the stack
      workspace consumes via ``/w/stack?ids=...``;
    - ``focus_id`` is the session the operator clicked from (the
      target), so the stack view auto-scrolls to it;
    - ``ancestors`` / ``descendants`` are split out so the popover can
      label the chain segments distinctly;
    - ``branch_kind`` carries the resolved edge kind incoming to the
      target (continuation / sidechain / fork / subagent / unknown), so
      the reader can pick the right chip vocabulary without re-walking
      the edges.

    When the target is the root and has no descendants, ``chain_ids`` is
    a single-element list. Isolated leaves still produce a valid
    envelope so the UI never has to special-case the empty state.
    """

    target_id = str(topology.target_id)
    ancestors = [str(node_id) for node_id in topology.ancestors(target_id)]
    chain_ids: list[str] = [*ancestors, target_id]
    descendants_ordered: list[str] = []
    if include_descendants:
        descendants_ordered = [str(node_id) for node_id in topology.descendants(target_id)]
        chain_ids.extend(descendants_ordered)

    # Resolved incoming edge kind (if any) for the target session.
    branch_kind: str | None = None
    parent_id: str | None = None
    for edge in topology.edges:
        if str(edge.child_id) != target_id or not edge.resolved:
            continue
        branch_kind = edge.kind.value
        parent_id = str(edge.parent_id) if edge.parent_id is not None else None
        break

    # Sibling lookup so a popover can render "compare with sibling N".
    siblings = [str(sid) for sid in topology.siblings(target_id)]

    return {
        "target_id": target_id,
        "root_id": str(topology.root_id),
        "parent_id": parent_id,
        "branch_kind": branch_kind,
        "chain_ids": chain_ids,
        "ancestors": ancestors,
        "descendants": descendants_ordered,
        "siblings": siblings,
        "focus_id": target_id,
        "node_count": len(topology.nodes),
        "cycle_detected": topology.cycle_detected,
    }


__all__ = [
    "DEFAULT_NODE_LIMIT",
    "MAX_NODE_LIMIT",
    "READINESS_EMPTY",
    "READINESS_OK",
    "READINESS_PARTIAL",
    "build_parent_chain_envelope",
    "build_topology_envelope",
    "coerce_node_limit",
    "coerce_node_offset",
]
