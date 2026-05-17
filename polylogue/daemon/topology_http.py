"""Lineage topology HTTP envelope helper (#1121).

Pure logic for shaping the ``SessionTopology`` read into the
``GET /api/conversations/{id}/topology`` JSON envelope the reader
consumes. Lives outside ``daemon/http.py`` so that file stays inside
its declared per-file LOC budget (``docs/plans/file-size-budgets.yaml``).

The envelope is the typed ``SessionTopology`` payload with three
operator-facing additions:

- ``readiness`` — ``ok`` / ``partial`` / ``empty``. The reader renders
  this with the MK3 data-quality chip vocabulary so an operator can
  tell at a glance whether the lineage view is complete, degraded, or
  genuinely absent (single-node session).
- ``truncated_count`` — when ``node_limit`` cut the BFS-ordered node
  list, this is the count of dropped nodes. Edges that referenced a
  dropped node are filtered so the reader never plots a dangling
  endpoint; ``truncated_count > 0`` is the operator's signal that the
  graph is incomplete.
- ``unresolved_edge_count`` — provider-native parent IDs that did not
  resolve to a stored conversation. Promoted to a top-level number so
  the reader does not have to walk the edge array to render the
  summary chip.

Issue #1121 acceptance criterion: lineage rendering does not
unbound-expand the graph in one payload. The handler that drives this
helper applies a hard cap of ``1000`` nodes; clients cannot escape it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.insights.topology import SessionTopology


def topology_envelope(topology: SessionTopology, *, node_limit: int) -> dict[str, object]:
    """Shape a ``SessionTopology`` into the reader-facing JSON envelope."""

    all_nodes = list(topology.nodes)
    truncated_count = max(0, len(all_nodes) - node_limit)
    nodes = all_nodes[:node_limit]
    kept_ids = {str(n.conversation_id) for n in nodes}
    edges = [
        edge
        for edge in topology.edges
        if str(edge.child_id) in kept_ids and (edge.parent_id is None or str(edge.parent_id) in kept_ids)
    ]
    unresolved = sum(1 for edge in topology.edges if not edge.resolved)
    if topology.cycle_detected or truncated_count > 0 or unresolved > 0:
        readiness = "partial"
    elif len(all_nodes) <= 1:
        readiness = "empty"
    else:
        readiness = "ok"
    return {
        "target_id": str(topology.target_id),
        "root_id": str(topology.root_id),
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [edge.model_dump(mode="json", exclude_none=True) for edge in edges],
        "cycle_detected": topology.cycle_detected,
        "unresolved_edge_count": unresolved,
        "node_count": len(all_nodes),
        "edge_count": len(topology.edges),
        "truncated_count": truncated_count,
        "node_limit": node_limit,
        "readiness": readiness,
    }


__all__ = ["topology_envelope"]
