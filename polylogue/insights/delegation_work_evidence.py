"""Project the existing ``delegations`` query surface into the generic
:mod:`polylogue.insights.work_evidence` graph.

``polylogue-1vpm.6.1`` AC6 requires that "existing delegation/correlation
surfaces become projections/adapters or retire" -- not that they disappear.
``delegation_facts`` (materialized by
:mod:`polylogue.storage.sqlite.delegation_facts`, read through the
``delegations`` structural query unit and
:meth:`~polylogue.storage.sqlite.archive_tiers.archive.ArchiveTier.query_delegations`)
is a genuinely richer, typed surface than the generic work-evidence graph for
its own purpose: honest per-dispatch cost/token/wall-clock/model columns that
the provider-neutral graph does not (and should not) carry. Retiring it would
be a real regression, not a simplification. What AC6 actually asks for is
satisfied here: delegation dispatch facts get a real, evidence-preserving
*projection* onto the shared node/edge vocabulary, so a caller that only knows
the generic graph (traversal, reconciliation, mutation-test suite) can reach
delegation-sourced work identities exactly like Claude Workflow or ordinary
Agent/Task ones -- without inventing a second, competing hierarchy (AC4) or
collapsing a dispatch attempt into its parent session (the
task=session/invocation=run shortcuts AC6 names).

Mapping, grounded entirely in existing ``ArchiveDelegationQueryRow`` fields
(never re-derived or guessed):

- One ``call`` node per dispatch instruction (the tool-use block that
  requested a child run), identity keyed on
  ``parent_session_id``/``instruction_tool_use_block_id`` -- never on the
  child session id, so a dispatch that never resolved to a child run still
  gets an honest call identity.
- One ``attempt`` node per resolved child session, ``invoked``-linked from
  the call. A dispatch with no ``child_session_id`` (an ``edge_only``/
  ``quarantined`` mapping state) gets no attempt node at all; the call node's
  own ``association_state`` carries the honest unresolved/contradicted fact
  instead of fabricating a one-to-one attempt.
- ``mapping_state`` maps onto :data:`~polylogue.insights.work_evidence.WorkEvidenceAssociationState`
  without reinterpretation: ``resolved``/``unresolved`` pass through
  directly, ``edge_only`` (an asserted link with no resolved child) becomes
  ``unresolved``, and ``quarantined`` (a cycle-break) becomes
  ``contradicted`` -- the same vocabulary ``session_links``'s own
  ``TopologyEdgeStatus`` already uses for the identical concept.
  ``WorkEvidenceAssociationState`` retains its own ``ambiguous`` value for
  other producers (e.g. Claude Workflow evidence); ``delegation_facts.
  mapping_state`` itself no longer produces it (polylogue-1vpm.7).
- A dispatch's own ``artifact_text`` (the child's self-reported result
  artifact), when present, becomes a ``claim`` node ``produced``-linked from
  the attempt -- a first-class, evidence-backed claim about outcome, never a
  mutation of ``result_status``/``result_is_error`` (those stay exactly what
  they are: the parent session's own observed tool-result structure, not this
  module's business).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.work_evidence import (
    WorkEvidenceAssociationState,
    WorkEvidenceEdge,
    WorkEvidenceEdgeKind,
    WorkEvidenceGraph,
    WorkEvidenceNode,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveDelegationQueryRow

_MAPPING_STATE_TO_ASSOCIATION: dict[str, WorkEvidenceAssociationState] = {
    "resolved": "resolved",
    "unresolved": "unresolved",
    "edge_only": "unresolved",
    "quarantined": "contradicted",
}


def _dispatch_evidence_ref(row: ArchiveDelegationQueryRow) -> EvidenceRef:
    if row.instruction_message_id is not None:
        return EvidenceRef(session_id=row.parent_session_id, message_id=row.instruction_message_id)
    return EvidenceRef(session_id=row.parent_session_id)


def _call_object_id(row: ArchiveDelegationQueryRow) -> str:
    dispatch_key = row.instruction_tool_use_block_id or row.child_session_id or "unresolved"
    return f"delegation:{row.parent_session_id}:{dispatch_key}"


def materialize_delegation_work_evidence_graph(
    *,
    graph_id: str,
    corpus_snapshot_ref: ObjectRef,
    rows: Sequence[ArchiveDelegationQueryRow],
) -> WorkEvidenceGraph:
    """Adapt real ``delegations`` query rows into one work-evidence graph.

    Pure: every node/edge is grounded in the ``rows`` argument the caller
    already loaded (via ``query_delegations`` in production, a fixture in
    tests). Nothing here re-reads storage.
    """

    nodes: dict[str, WorkEvidenceNode] = {}
    edges: dict[str, WorkEvidenceEdge] = {}

    for row in rows:
        association_state = _MAPPING_STATE_TO_ASSOCIATION[row.mapping_state]
        evidence_ref = _dispatch_evidence_ref(row)
        call_ref = ObjectRef(kind="work-call", object_id=_call_object_id(row))
        call_label = f"delegation dispatch {row.parent_session_id} -> {row.child_session_id or 'unresolved'}"
        nodes[call_ref.format()] = WorkEvidenceNode(
            ref=call_ref,
            kind="call",
            label=call_label,
            evidence_refs=(evidence_ref,),
            corpus_snapshot_ref=corpus_snapshot_ref,
            authority="provider",
            confidence=1.0 if association_state == "resolved" else 0.5,
            association_state=association_state,
        )

        if row.child_session_id is None:
            continue

        attempt_ref = ObjectRef(kind="work-attempt", object_id=f"attempt:{row.child_session_id}")
        nodes[attempt_ref.format()] = WorkEvidenceNode(
            ref=attempt_ref,
            kind="attempt",
            label=f"delegation attempt {row.child_session_id}",
            evidence_refs=(evidence_ref,),
            corpus_snapshot_ref=corpus_snapshot_ref,
            authority="provider",
            confidence=1.0 if association_state == "resolved" else 0.5,
            association_state=association_state,
        )
        edge_kind: WorkEvidenceEdgeKind = "invoked" if association_state == "resolved" else "unresolved"
        edge_ref = ObjectRef(kind="work-edge", object_id=f"{edge_kind}:{call_ref.format()}->{attempt_ref.format()}")
        edges[edge_ref.format()] = WorkEvidenceEdge(
            ref=edge_ref,
            kind=edge_kind,
            source_ref=call_ref,
            target_ref=attempt_ref,
            evidence_refs=(evidence_ref,),
            corpus_snapshot_ref=corpus_snapshot_ref,
            authority="provider",
            confidence=1.0 if association_state == "resolved" else 0.5,
            association_state=association_state,
        )

        if row.artifact_text:
            claim_ref = ObjectRef(kind="work-claim", object_id=f"claim:delegation:{row.child_session_id}")
            nodes[claim_ref.format()] = WorkEvidenceNode(
                ref=claim_ref,
                kind="claim",
                label=f"delegation self-reported result {row.child_session_id}",
                claim_text=row.artifact_text,
                evidence_refs=(evidence_ref,),
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
            )
            produced_ref = ObjectRef(
                kind="work-edge", object_id=f"produced:{attempt_ref.format()}->{claim_ref.format()}"
            )
            edges[produced_ref.format()] = WorkEvidenceEdge(
                ref=produced_ref,
                kind="produced",
                source_ref=attempt_ref,
                target_ref=claim_ref,
                evidence_refs=(evidence_ref,),
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
            )

    return WorkEvidenceGraph(
        graph_id=graph_id,
        corpus_snapshot_ref=corpus_snapshot_ref,
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
    )


__all__ = ["materialize_delegation_work_evidence_graph"]
