from __future__ import annotations

import pytest

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.insights.work_reconciliation import (
    ObservedRepositoryEffect,
    ReconciliationJudgment,
    reconcile_work_effects,
)


def test_reconciliation_keeps_claim_and_commit_as_distinct_facts() -> None:
    evidence = EvidenceRef(session_id="codex-session:test", message_id="m", block_index=0)
    snapshot = ObjectRef(kind="context-snapshot", object_id="snapshot")
    claim = WorkEvidenceNode(
        ref=ObjectRef(kind="work-claim", object_id="done"),
        kind="claim",
        label="done",
        claim_text="done",
        evidence_refs=(evidence,),
        corpus_snapshot_ref=snapshot,
        confidence=1,
    )
    graph = WorkEvidenceGraph(graph_id="g", corpus_snapshot_ref=snapshot, nodes=(claim,), edges=())
    effect = ObservedRepositoryEffect(
        ref=ObjectRef(kind="commit", object_id="abc"),
        label="commit abc",
        authority="git",
        evidence_ref=evidence,
        repository_snapshot_ref=snapshot,
    )
    reconciled = reconcile_work_effects(
        graph,
        effects=(effect,),
        judgments=(
            ReconciliationJudgment(
                claim_ref=claim.ref, effect_ref=effect.ref, evaluation="supported", evidence_ref=evidence
            ),
        ),
    )
    assert {node.kind for node in reconciled.nodes} == {"claim", "effect"}
    assert reconciled.edges[0].association_state == "resolved"
    with pytest.raises(ValueError, match="source must be a claim"):
        reconcile_work_effects(
            graph,
            effects=(effect,),
            judgments=(
                ReconciliationJudgment(
                    claim_ref=effect.ref, effect_ref=effect.ref, evaluation="supported", evidence_ref=evidence
                ),
            ),
        )
