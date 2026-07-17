"""Provider-neutral work topology and claim graph.

The graph represents what a runtime reported or what archive evidence can
support about work execution.  It intentionally stops before repository
effects and acceptance evaluation: a ``claim`` is a first-class node, never a
state update on an external object.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, model_validator

from polylogue.core.refs import ActorRef, EvidenceRef, ExecutionContextRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.run_projection import ObservedEvent, ProjectedRun

WorkEvidenceNodeKind = Literal[
    "run",
    "invocation",
    "call",
    "attempt",
    "session-segment",
    "structured-result",
    "claim",
    "artifact",
    "actor",
    "execution-context",
    "effect",
]
WorkEvidenceEdgeKind = Literal[
    "invoked",
    "resumed",
    "retried",
    "represented_by",
    "produced",
    "consumed",
    "mentioned",
    "claimed",
    "superseded",
    "unresolved",
]
WorkEvidenceAuthority = Literal["provider", "operator", "inferred", "unknown"]
WorkEvidenceAssociationState = Literal["resolved", "unresolved", "ambiguous", "contradicted", "superseded"]

_NODE_REF_KINDS: Final[dict[WorkEvidenceNodeKind, frozenset[str]]] = {
    "run": frozenset({"run"}),
    "invocation": frozenset({"work-invocation"}),
    "call": frozenset({"work-call"}),
    "attempt": frozenset({"work-attempt"}),
    "session-segment": frozenset({"work-session-segment"}),
    "structured-result": frozenset({"work-result"}),
    "claim": frozenset({"work-claim"}),
    "artifact": frozenset({"artifact"}),
    "actor": frozenset({"actor"}),
    "execution-context": frozenset({"execution-context"}),
    "effect": frozenset({"commit", "github-issue", "github-pr", "check-run", "artifact"}),
}


class WorkEvidenceNode(ArchiveInsightModel):
    """One evidence-backed, provider-neutral work identity."""

    ref: ObjectRef
    kind: WorkEvidenceNodeKind
    label: str
    evidence_refs: tuple[EvidenceRef, ...]
    corpus_snapshot_ref: ObjectRef
    authority: WorkEvidenceAuthority = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    occurred_at_ms: int | None = Field(default=None, ge=0)
    actor_ref: ActorRef | None = None
    execution_context_ref: ExecutionContextRef | None = None
    association_state: WorkEvidenceAssociationState = "resolved"
    claim_text: str | None = None

    @model_validator(mode="after")
    def _validate_identity_and_claim_boundary(self) -> WorkEvidenceNode:
        if self.ref.kind not in _NODE_REF_KINDS[self.kind]:
            expected = ", ".join(sorted(_NODE_REF_KINDS[self.kind]))
            raise ValueError(f"{self.kind} nodes require a {expected} ref, not {self.ref.kind}")
        if not self.label.strip():
            raise ValueError("work-evidence node label cannot be empty")
        if not self.evidence_refs:
            raise ValueError("work-evidence nodes require source evidence")
        if self.kind == "claim" and not (self.claim_text and self.claim_text.strip()):
            raise ValueError("claim nodes require claim_text")
        if self.kind != "claim" and self.claim_text is not None:
            raise ValueError("only claim nodes may carry claim_text")
        if self.kind == "actor" and (self.actor_ref is None or self.ref.object_id != self.actor_ref.format()):
            raise ValueError("actor nodes must carry their matching ActorRef")
        if self.kind == "execution-context" and (
            self.execution_context_ref is None or self.ref.object_id != self.execution_context_ref.context_id
        ):
            raise ValueError("execution-context nodes must carry their matching ExecutionContextRef")
        return self


class WorkEvidenceEdge(ArchiveInsightModel):
    """Directed, evidence-backed relation between two work identities."""

    ref: ObjectRef
    kind: WorkEvidenceEdgeKind
    source_ref: ObjectRef
    target_ref: ObjectRef
    evidence_refs: tuple[EvidenceRef, ...]
    corpus_snapshot_ref: ObjectRef
    authority: WorkEvidenceAuthority = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    occurred_at_ms: int | None = Field(default=None, ge=0)
    association_state: WorkEvidenceAssociationState = "resolved"

    @model_validator(mode="after")
    def _validate_identity(self) -> WorkEvidenceEdge:
        if self.ref.kind != "work-edge":
            raise ValueError("work-evidence edges require a work-edge ref")
        if self.source_ref == self.target_ref:
            raise ValueError("work-evidence edges cannot self-reference")
        if not self.evidence_refs:
            raise ValueError("work-evidence edges require source evidence")
        return self


class WorkEvidenceGraph(ArchiveInsightModel):
    """Replaceable graph snapshot consumed by generic work-evidence queries."""

    graph_id: str
    corpus_snapshot_ref: ObjectRef
    nodes: tuple[WorkEvidenceNode, ...]
    edges: tuple[WorkEvidenceEdge, ...]

    @model_validator(mode="after")
    def _validate_graph(self) -> WorkEvidenceGraph:
        if not self.graph_id.strip():
            raise ValueError("work-evidence graph_id cannot be empty")
        refs = [node.ref.format() for node in self.nodes]
        if len(refs) != len(set(refs)):
            raise ValueError("work-evidence graph node refs must be unique")
        edge_refs = [edge.ref.format() for edge in self.edges]
        if len(edge_refs) != len(set(edge_refs)):
            raise ValueError("work-evidence graph edge refs must be unique")
        node_refs = set(refs)
        for edge in self.edges:
            if edge.source_ref.format() not in node_refs or edge.target_ref.format() not in node_refs:
                raise ValueError("work-evidence edges must connect graph nodes")
        return self


class WorkEvidenceTraversal(ArchiveInsightModel):
    """Bidirectional neighborhood returned by the production query route."""

    graph_id: str
    focal_ref: ObjectRef
    nodes: tuple[WorkEvidenceNode, ...]
    edges: tuple[WorkEvidenceEdge, ...]


def node_from_projected_run(
    run: ProjectedRun,
    *,
    corpus_snapshot_ref: ObjectRef,
    actor_ref: ActorRef | None = None,
    execution_context_ref: ExecutionContextRef | None = None,
) -> WorkEvidenceNode:
    """Adapt an existing run projection without inventing a second run id."""

    return WorkEvidenceNode(
        ref=run.run_ref,
        kind="run",
        label=run.title,
        evidence_refs=run.evidence_refs,
        corpus_snapshot_ref=corpus_snapshot_ref,
        authority="provider" if run.confidence == "raw" else "inferred",
        confidence=1.0 if run.confidence == "raw" else 0.5,
        actor_ref=actor_ref,
        execution_context_ref=execution_context_ref,
    )


def session_segment_from_observed_events(
    *,
    ref: ObjectRef,
    label: str,
    corpus_snapshot_ref: ObjectRef,
    events: tuple[ObservedEvent, ...],
) -> WorkEvidenceNode:
    """Make a segment from existing observed-event evidence, not prose claims."""

    evidence_refs = tuple(dict.fromkeys(evidence for event in events for evidence in event.evidence_refs))
    return WorkEvidenceNode(
        ref=ref,
        kind="session-segment",
        label=label,
        evidence_refs=evidence_refs,
        corpus_snapshot_ref=corpus_snapshot_ref,
        authority="provider",
        confidence=1.0,
    )


__all__ = [
    "WorkEvidenceAssociationState",
    "WorkEvidenceAuthority",
    "WorkEvidenceEdge",
    "WorkEvidenceEdgeKind",
    "WorkEvidenceGraph",
    "WorkEvidenceNode",
    "WorkEvidenceNodeKind",
    "WorkEvidenceTraversal",
    "node_from_projected_run",
    "session_segment_from_observed_events",
]
