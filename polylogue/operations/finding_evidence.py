"""Project one finding's ancestry into the shared evidence-integrity graph.

polylogue-rxdo.4.  ``storage/sqlite/finding_provenance.py`` says in its own
docstring that it "does not compute support, cycle, staleness, frame or
privacy"; ``core/evidence_integrity.py`` owns exactly those semantics and
declares :class:`FindingEvidenceAdapter` for this consumer -- but nothing in
production ever built one, so the finding surfaces carried an advisory
staleness string instead of a verdict, and circular ancestry was undetected.

This module is that population, and only that: it maps provenance facts onto
the small node/edge protocol and hands them to the one evaluator.  No second
graph walker, no finding-only support rules.  Ancestry expansion follows
``assertion`` refs transitively -- the step that makes a citation cycle
observable at all -- bounded by the evaluator's own node budget.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from polylogue.core.enums import AssertionStatus
from polylogue.core.evidence_integrity import (
    EvidenceAuthority,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    EvidenceIntegrityVerdict,
    FindingEvidenceAdapter,
    evaluate_adapter,
)
from polylogue.core.refs import ObjectRef

if TYPE_CHECKING:
    from polylogue.storage.sqlite.finding_provenance import FindingEvidenceResolution, FindingProvenance

__all__ = [
    "FINDING_ANCESTRY_MAX_NODES",
    "build_finding_evidence_adapter",
    "evaluate_finding_evidence",
]

#: Ancestry expansion and evaluation share one budget so a pathological
#: citation graph cannot make a ref resolution unbounded. Exhausting it is
#: reported as ``UNRESOLVED`` by the evaluator, never as support.
FINDING_ANCESTRY_MAX_NODES = 512

#: What each ref kind establishes. ``assertion`` deliberately maps to
#: ``assertion`` rather than to a grounding authority: a finding whose whole
#: ancestry is other assertions is a closed loop, which is the evaluator's own
#: ``authorities <= {"agent", "assertion"}`` rule and not a policy this module
#: re-implements. An unlisted kind stays ``unknown`` -- ungrounded -- because
#: ``_resolve_evidence_ref`` cannot resolve it either.
_REF_KIND_AUTHORITY: dict[str, EvidenceAuthority] = {
    "query": "tool",
    "result-set": "tool",
    "assertion": "assertion",
    "finding": "assertion",
    "session": "raw",
    "message": "raw",
    "block": "raw",
    "action": "raw",
    "commit": "git",
    "pr": "pr",
    "agent": "agent",
}


def _ref_authority(ref: str) -> EvidenceAuthority:
    try:
        parsed = ObjectRef.parse(ref)
    except ValueError:
        return "unknown"
    return _REF_KIND_AUTHORITY.get(parsed.kind, "unknown")


def _review_state(status: str) -> str:
    """Map a *cited* assertion's lifecycle status onto the review vocabulary.

    Only ``active`` is approved evidence. A rejected, superseded, deleted or
    still-candidate row a finding cites is addressable but cannot ground it,
    which the evaluator turns into ``NOT_SUPPORTED`` through
    ``review_unapproved``.
    """
    return "approved" if status == AssertionStatus.ACTIVE.value else f"unapproved:{status}"


def _assertion_object_id(ref: str) -> str | None:
    try:
        parsed = ObjectRef.parse(ref)
    except ValueError:
        return None
    return parsed.object_id if parsed.kind in {"assertion", "finding"} else None


def _frame_and_definition(conn: sqlite3.Connection, provenance: FindingProvenance) -> tuple[str | None, str | None]:
    """Return ``(frame_hash, definition_hash)`` a finding actually declares.

    The frame is the evaluated relation's corpus epoch and the definition is
    the query identity -- both already durable. A finding that declares neither
    has no frame, and the evaluator reports ``FRAME_INCOMPLETE`` rather than
    inventing one; that is the honest state for a finding recorded without a
    result set, and AC3 forbids calling it supported.
    """
    from polylogue.storage.sqlite.query_objects import get_query, get_result_set

    frame_hash: str | None = None
    definition_hash: str | None = None
    result_set_ref = provenance.result_set_ref
    if result_set_ref is not None:
        object_id = _object_id_of_kind(result_set_ref, "result-set")
        if object_id is not None:
            manifest = get_result_set(conn, object_id)
            if manifest is not None:
                frame_hash = manifest.corpus_epoch
    query_ref_value = provenance.query_ref
    if query_ref_value is not None:
        object_id = _object_id_of_kind(query_ref_value, "query")
        if object_id is not None:
            query = get_query(conn, object_id)
            if query is not None:
                definition_hash = query.query_hash
    return frame_hash, definition_hash


def _object_id_of_kind(ref: str, kind: str) -> str | None:
    try:
        parsed = ObjectRef.parse(ref)
    except ValueError:
        return None
    return parsed.object_id if parsed.kind == kind else None


def build_finding_evidence_adapter(
    conn: sqlite3.Connection,
    provenance: FindingProvenance,
    *,
    frame_hash: str | None,
    definition_hash: str | None,
    max_nodes: int = FINDING_ANCESTRY_MAX_NODES,
) -> FindingEvidenceAdapter:
    """Project one finding and its transitive assertion ancestry into a graph.

    The root carries the finding's own review state, visibility and declared
    frame; every cited ref becomes a node whose ``ref_state`` is the resolution
    ``compute_finding_provenance`` already measured. Cited *assertions* are
    expanded in turn, which is what lets a citation cycle be witnessed instead
    of silently terminating at a leaf.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope

    root_ref = f"assertion:{provenance.assertion_id}"
    nodes: dict[str, EvidenceGraphNode] = {
        root_ref: EvidenceGraphNode(
            ref=root_ref,
            kind="finding",
            authority="assertion",
            ref_state="ok",
            frame_hash=frame_hash,
            definition_hash=definition_hash,
            # The root is the claim, not grounding evidence: its own lifecycle
            # status is a publication-review fact the payload already carries
            # separately, and folding it in here would make every unjudged
            # detector candidate report an evidence-integrity failure it does
            # not have. Only cited assertions are gated on review below.
            review_state="approved",
            public=True,
        )
    }
    edges: list[EvidenceGraphEdge] = []
    pending: list[tuple[str, tuple[FindingEvidenceResolution, ...]]] = [(root_ref, provenance.evidence)]
    expanded: set[str] = {root_ref}

    while pending:
        parent_ref, items = pending.pop()
        for item in items:
            ref = item.ref
            resolvable = item.resolvable
            edges.append(EvidenceGraphEdge(src_ref=parent_ref, dst_ref=ref, purpose="supports"))
            if ref not in nodes:
                nodes[ref] = EvidenceGraphNode(
                    ref=ref,
                    kind=_ref_kind(ref),
                    authority=_ref_authority(ref),
                    ref_state="ok" if resolvable else "missing",
                    frame_hash=frame_hash if resolvable else None,
                )
            if len(nodes) >= max_nodes:
                continue
            assertion_id = _assertion_object_id(ref)
            if assertion_id is None or ref in expanded or not resolvable:
                continue
            expanded.add(ref)
            envelope = read_assertion_envelope(conn, assertion_id)
            if envelope is None:
                continue
            nodes[ref] = EvidenceGraphNode(
                ref=ref,
                kind="assertion",
                authority="assertion",
                ref_state="ok",
                frame_hash=frame_hash,
                review_state=_review_state(str(envelope.status)),
                # Visibility is deliberately not gated here. This consumer is
                # the internal ref-resolution route, and every detector finding
                # is written PRIVATE by ``upsert_findings_as_assertions``; a
                # ``held_private`` verdict short-circuits the status ladder
                # ahead of ``cycle``, so gating on it here would hide the
                # ancestry failures this bead exists to surface. The
                # publication consumer -- ``analysis/measurement/public_claims``
                # -- is where "publicly supported" is decided, and it is the
                # half of polylogue-rxdo.4 this change does not reach.
                public=True,
            )
            pending.append(
                (
                    ref,
                    tuple(_cited(str(child)) for child in envelope.evidence_refs),
                )
            )

    return FindingEvidenceAdapter(graph_nodes=tuple(nodes.values()), graph_edges=tuple(edges))


def _cited(ref: str) -> FindingEvidenceResolution:
    """One ref discovered during ancestry expansion.

    Resolvability of a *transitively* cited ref is deliberately not re-measured
    here: this module does not re-run the storage resolution for every
    ancestor, and an ancestor that does not exist becomes a ``missing_ref``
    witness from the evaluator's own node lookup instead.
    """
    from polylogue.storage.sqlite.finding_provenance import FindingEvidenceResolution

    return FindingEvidenceResolution(ref=ref, resolvable=True)


def _ref_kind(ref: str) -> str:
    try:
        return ObjectRef.parse(ref).kind
    except ValueError:
        return "unknown"


def evaluate_finding_evidence(
    conn: sqlite3.Connection,
    provenance: FindingProvenance,
    *,
    max_nodes: int = FINDING_ANCESTRY_MAX_NODES,
) -> EvidenceIntegrityVerdict:
    """Return the shared evaluator's verdict for one finding's ancestry."""
    frame_hash, definition_hash = _frame_and_definition(conn, provenance)
    adapter = build_finding_evidence_adapter(
        conn,
        provenance,
        frame_hash=frame_hash,
        definition_hash=definition_hash,
        max_nodes=max_nodes,
    )
    detector_ref = provenance.detector_ref
    return evaluate_adapter(
        f"assertion:{provenance.assertion_id}",
        adapter,
        definition_hash=definition_hash,
        frame_hash=frame_hash,
        detector_output_refs=frozenset({detector_ref}) if detector_ref else frozenset(),
        max_nodes=max_nodes,
    )
