"""Independent repository-effect reconciliation for work-evidence graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceEdge, WorkEvidenceGraph, WorkEvidenceNode

EffectAuthority = Literal["git", "github", "beads", "artifact", "verification"]
Evaluation = Literal["supported", "partial", "contradicted", "unresolved", "superseded"]


@dataclass(frozen=True, slots=True)
class ObservedRepositoryEffect:
    """An independently observed effect; it is never an agent's claim."""

    ref: ObjectRef
    label: str
    authority: EffectAuthority
    evidence_ref: EvidenceRef
    repository_snapshot_ref: ObjectRef
    occurred_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationJudgment:
    """An evaluation over a claim/effect pair, preserving both inputs."""

    claim_ref: ObjectRef
    effect_ref: ObjectRef
    evaluation: Evaluation
    evidence_ref: EvidenceRef


def reconcile_work_effects(
    graph: WorkEvidenceGraph,
    *,
    effects: tuple[ObservedRepositoryEffect, ...],
    judgments: tuple[ReconciliationJudgment, ...],
) -> WorkEvidenceGraph:
    """Attach direct repository observations and explicit evaluations.

    Effects are only linked when a caller supplied a direct identifier-based
    judgment. Temporal/file overlap must be represented upstream as a candidate
    and cannot reach this function as a supported causal relation.
    """
    nodes = {node.ref.format(): node for node in graph.nodes}
    edges = {edge.ref.format(): edge for edge in graph.edges}
    for effect in effects:
        nodes.setdefault(
            effect.ref.format(),
            WorkEvidenceNode(
                ref=effect.ref,
                kind="effect",
                label=effect.label,
                evidence_refs=(effect.evidence_ref,),
                corpus_snapshot_ref=effect.repository_snapshot_ref,
                authority="provider",
                confidence=1.0,
                occurred_at_ms=effect.occurred_at_ms,
            ),
        )
    for judgment in judgments:
        if judgment.claim_ref.format() not in nodes or judgment.effect_ref.format() not in nodes:
            raise ValueError("reconciliation judgments require known claim and effect refs")
        if nodes[judgment.claim_ref.format()].kind != "claim":
            raise ValueError("reconciliation judgment source must be a claim")
        edge_ref = ObjectRef(
            kind="work-edge",
            object_id=f"reconciliation:{judgment.claim_ref.format()}->{judgment.effect_ref.format()}",
        )
        edges[edge_ref.format()] = WorkEvidenceEdge(
            ref=edge_ref,
            kind="claimed",
            source_ref=judgment.claim_ref,
            target_ref=judgment.effect_ref,
            evidence_refs=(judgment.evidence_ref,),
            corpus_snapshot_ref=nodes[judgment.effect_ref.format()].corpus_snapshot_ref,
            authority="operator",
            confidence=1.0,
            association_state=cast(
                Literal["resolved", "unresolved", "ambiguous", "contradicted", "superseded"],
                {
                    "supported": "resolved",
                    "partial": "ambiguous",
                    "contradicted": "contradicted",
                    "unresolved": "unresolved",
                    "superseded": "superseded",
                }[judgment.evaluation],
            ),
        )
    return WorkEvidenceGraph(
        graph_id=graph.graph_id,
        corpus_snapshot_ref=graph.corpus_snapshot_ref,
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
    )


__all__ = ["ObservedRepositoryEffect", "ReconciliationJudgment", "reconcile_work_effects"]
