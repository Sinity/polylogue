"""Storage-side derivation of the session topology read model.

The topology graph is derived on demand from the canonical archive tables
(``sessions.parent_session_id`` + ``branch_type`` +
``sessions.provider_meta``). No new DDL is required: the durable
edge data is already persisted at ingest time through the parent
resolution in ``polylogue.pipeline.prepare_enrichment`` and parsed
``BranchType`` values.

Unresolved-native edges are surfaced by inspecting ``provider_meta`` for
common parent-pointer field names. This preserves the topology contract
defined in :mod:`polylogue.insights.topology` even when the canonical
parent row has not yet been ingested.
"""

from polylogue.storage.insights.topology.derivation import (
    UNRESOLVED_NATIVE_PARENT_KEYS,
    derive_session_topology_async,
    derive_session_topology_sync,
)

__all__ = [
    "UNRESOLVED_NATIVE_PARENT_KEYS",
    "derive_session_topology_async",
    "derive_session_topology_sync",
]
