"""Storage-side derivation of the session topology read model.

The topology graph is derived on demand from the canonical archive tables
(``sessions.parent_session_id`` + ``branch_type`` + ``session_links``).
No extra metadata bucket is consulted: unresolved parent assertions are durable
rows in ``session_links``.

Unresolved-native edges are surfaced from ``session_links`` rows whose
``resolved_dst_session_id`` is still NULL. This preserves the topology contract
defined in :mod:`polylogue.insights.topology` even when the parent row has not
yet been ingested.
"""

from polylogue.storage.insights.topology.derivation import (
    derive_session_topology_async,
    derive_session_topology_sync,
)

__all__ = [
    "derive_session_topology_async",
    "derive_session_topology_sync",
]
