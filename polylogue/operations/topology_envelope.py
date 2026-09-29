"""The one topology envelope every public surface serializes.

`session_links` is the edge authority and
:mod:`polylogue.storage.derived.topology.derivation` is the one graph
engine over it. This module is the single operation boundary above that
engine: it takes the domain :class:`~polylogue.analysis.topology.SessionTopology`
and produces the canonical transport-neutral envelope, including the one
terminal ``outcome`` decision.

Surfaces call :func:`topology_public_envelope` and frame the result for
their transport. They do not decide the outcome, do not re-derive the
lineage helper lists, and do not map edge fields. A surface that bounds
the envelope for its own wire budget calls :func:`bound_topology_envelope`
so the bound is reported as a named gap rather than silently narrowing a
"complete" answer.

Layering: ``operations`` is the product layer, so it may read ``analysis``
and the ``surfaces`` outcome contract. ``analysis`` itself must not import
the terminal-outcome vocabulary, which is why
:meth:`SessionTopology.degraded_gaps` returns facts and this module turns
them into a decision.
"""

from __future__ import annotations

from typing import Final, cast

from polylogue.analysis.topology import TOPOLOGY_GAP_TRUNCATED, SessionTopology
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome

#: Reason recorded when the requested session has no topology at all.
TOPOLOGY_EMPTY_REASON: Final[str] = "no_topology_in_scope"

#: Default topology page size and its hard transport ceiling.  The producer
#: below is shared by MCP and HTTP, so neither surface can retain a separate
#: unbounded topology serialization path.
DEFAULT_NODE_LIMIT: Final[int] = 200
MAX_NODE_LIMIT: Final[int] = 1000


def topology_outcome(topology: SessionTopology) -> OutcomeEnvelope:
    """Decide the one terminal outcome for a topology answer.

    ``degraded`` outranks ``empty``: a topology behind a named gap -- a
    bounded page, an unresolved parent, an edge excluded from composition,
    a cycle, a conflicting parent -- is never reported as an empty scope,
    because the gap and not the archive may be why rows are absent.
    """

    return decide_outcome(
        matched=len(topology.nodes),
        degraded=topology.degraded_gaps(),
        empty_reason=TOPOLOGY_EMPTY_REASON,
    )


def topology_public_envelope(
    topology: SessionTopology,
    *,
    session_id: str | None = None,
    node_limit: int = MAX_NODE_LIMIT,
    node_offset: int = 0,
) -> dict[str, object]:
    """Return the bounded canonical public topology envelope with its outcome.

    This is the one payload CLI, MCP, HTTP and the Python API serialize. The
    shared hard ceiling applies even when a caller supplies a larger limit;
    transport framing is the only difference permitted between them.
    """

    payload = topology.public_payload(session_id)
    payload["outcome"] = topology_outcome(topology).to_dict()
    effective_limit = max(1, min(node_limit, MAX_NODE_LIMIT))
    return bound_topology_envelope(payload, node_limit=effective_limit, source_node_offset=node_offset)


def bound_topology_envelope(
    envelope: dict[str, object],
    *,
    node_limit: int,
    source_node_offset: int = 0,
) -> dict[str, object]:
    """Narrow a canonical envelope to ``node_limit`` nodes, honestly.

    Nodes beyond the limit are dropped. An edge belongs to the page that
    holds its child, as it does in the topology derivation's own paging, so
    an edge whose parent sits on an earlier page is still served exactly
    once and concatenated pages rebuild the whole graph. Kept edges are kept
    regardless of resolution or composability -- an excluded edge stays
    visible as evidence, it is simply never traversed.

    Any narrowing is recorded as the ``topology_truncated`` gap and forces
    ``nodes_complete`` / ``edges_complete`` false, so a bounded page can
    never present itself as a complete graph, and re-decides the outcome
    from the narrowed facts.
    """

    nodes = list(cast("list[dict[str, object]]", envelope["nodes"]))
    kept_nodes = nodes[: max(1, node_limit)]
    kept_ids = {str(node["session_id"]) for node in kept_nodes}
    dropped = len(nodes) - len(kept_nodes)

    kept_edges = [
        edge for edge in cast("list[dict[str, object]]", envelope["edges"]) if str(edge["child_id"]) in kept_ids
    ]

    source_incomplete = not bool(envelope["nodes_complete"]) or not bool(envelope["edges_complete"])
    truncated = dropped > 0 or source_incomplete

    bounded = dict(envelope)
    bounded["nodes"] = kept_nodes
    bounded["edges"] = kept_edges
    for key in ("ancestors", "descendants", "siblings", "thread"):
        if key in envelope:
            bounded[key] = [
                item
                for item in cast("list[object]", envelope[key])
                if isinstance(item, dict) and str(item.get("session_id")) in kept_ids
            ]
    bounded["nodes_complete"] = not truncated
    bounded["edges_complete"] = not truncated
    # A source page already carries the requested offset; only synthesize a
    # continuation when this bound is what did the narrowing.
    if envelope.get("continuation") is None and dropped > 0:
        bounded["continuation"] = f"node-offset:{source_node_offset + len(kept_nodes)}"

    source_outcome = cast("dict[str, object]", envelope["outcome"])
    source_detail = cast("dict[str, object]", source_outcome.get("detail") or {})
    gaps = [str(reason) for reason in cast("list[object]", source_detail.get("gaps") or [])]
    if truncated and TOPOLOGY_GAP_TRUNCATED not in gaps:
        gaps.append(TOPOLOGY_GAP_TRUNCATED)
    bounded["outcome"] = decide_outcome(
        matched=len(kept_nodes),
        degraded=tuple(gaps),
        empty_reason=TOPOLOGY_EMPTY_REASON,
    ).to_dict()
    return bounded


__all__ = [
    "DEFAULT_NODE_LIMIT",
    "MAX_NODE_LIMIT",
    "TOPOLOGY_EMPTY_REASON",
    "bound_topology_envelope",
    "topology_outcome",
    "topology_public_envelope",
]
