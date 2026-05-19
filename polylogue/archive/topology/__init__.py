"""Topology-level abstractions (parent/child edge graph across conversations).

Distinct from per-conversation ``branch_type`` (which is a single column on
``conversations``): the topology edge surface persists every parent reference
emitted by a parser as a typed row, including references whose parent has not
yet been ingested (#1258 / #866 slice A).
"""

from __future__ import annotations

from polylogue.archive.topology.edge import (
    TopologyEdgeRecord,
    TopologyEdgeStatus,
    TopologyEdgeType,
    branch_type_to_edge_type,
)

__all__ = [
    "TopologyEdgeRecord",
    "TopologyEdgeStatus",
    "TopologyEdgeType",
    "branch_type_to_edge_type",
]
