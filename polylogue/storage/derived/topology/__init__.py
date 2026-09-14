"""Storage-side derivation of the session topology read model.

Every public topology edge is a ``session_links`` row. The
``sessions.parent_session_id`` / ``root_session_id`` columns are write-side
lookup accelerators used to *discover* candidate nodes; they never supply an
edge, its type, or its provenance. (This docstring previously claimed the
graph was derived from ``parent_session_id`` + ``branch_type``, contradicting
:mod:`~polylogue.storage.derived.topology.derivation`'s own contract.)

Unresolved-native edges are surfaced from ``session_links`` rows whose
``resolved_dst_session_id`` is still NULL. This preserves the topology contract
defined in :mod:`polylogue.analysis.topology` even when the parent row has not
yet been ingested.

:func:`compose_session_topology` is the one graph classification engine. A
caller that already holds canonical rows composes through it rather than
classifying edges itself.
"""

from polylogue.storage.derived.topology.derivation import (
    TopologyNodeInput,
    compose_session_topology,
    derive_session_topology_async,
    derive_session_topology_sync,
    node_input_from_record,
)

__all__ = [
    "TopologyNodeInput",
    "compose_session_topology",
    "derive_session_topology_async",
    "derive_session_topology_sync",
    "node_input_from_record",
]
