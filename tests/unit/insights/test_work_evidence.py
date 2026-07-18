"""Provider-neutral work-evidence graph behavior and real SQLite traversal."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.core.refs import ActorRef, EvidenceRef, ExecutionContextRef, ObjectRef, ObjectRefKind
from polylogue.insights.run_projection import ObservedEvent, ProjectedRun
from polylogue.insights.work_evidence import (
    WorkEvidenceAssociationState,
    WorkEvidenceEdge,
    WorkEvidenceEdgeKind,
    WorkEvidenceGraph,
    WorkEvidenceNode,
    WorkEvidenceNodeKind,
    node_from_projected_run,
    session_segment_from_observed_events,
)
from polylogue.storage.repository import SessionRepository

EVIDENCE = EvidenceRef(session_id="codex-session:work-evidence", message_id="m1", block_index=0)
SNAPSHOT = ObjectRef(kind="context-snapshot", object_id="corpus:2026-07-17")


def _node(
    kind: WorkEvidenceNodeKind,
    object_id: str,
    *,
    label: str | None = None,
    association_state: WorkEvidenceAssociationState = "resolved",
    claim_text: str | None = None,
) -> WorkEvidenceNode:
    ref_kind = {
        "invocation": "work-invocation",
        "call": "work-call",
        "attempt": "work-attempt",
        "structured-result": "work-result",
        "claim": "work-claim",
        "artifact": "artifact",
    }[kind]
    return WorkEvidenceNode(
        ref=ObjectRef(kind=cast(ObjectRefKind, ref_kind), object_id=object_id),
        kind=kind,
        label=label or object_id,
        evidence_refs=(EVIDENCE,),
        corpus_snapshot_ref=SNAPSHOT,
        authority="provider",
        confidence=1.0,
        association_state=association_state,
        claim_text=claim_text,
    )


def _edge(
    kind: WorkEvidenceEdgeKind,
    name: str,
    source: WorkEvidenceNode,
    target: WorkEvidenceNode,
) -> WorkEvidenceEdge:
    return WorkEvidenceEdge(
        ref=ObjectRef(kind="work-edge", object_id=name),
        kind=kind,
        source_ref=source.ref,
        target_ref=target.ref,
        evidence_refs=(EVIDENCE,),
        corpus_snapshot_ref=SNAPSHOT,
        authority="provider",
        confidence=1.0,
    )


def _graph() -> WorkEvidenceGraph:
    actor = ActorRef(kind="agent", identity="codex")
    context = ExecutionContextRef.from_observation(
        {"harness": "codex-cli", "model": "gpt-5.6-terra"}, unknown_fields=("skills",)
    )
    run = node_from_projected_run(
        ProjectedRun(
            run_ref=ObjectRef(kind="run", object_id="codex-session:work-evidence"),
            title="Ship agent task",
            evidence_refs=(EVIDENCE,),
            confidence="raw",
        ),
        corpus_snapshot_ref=SNAPSHOT,
        actor_ref=actor,
        execution_context_ref=context,
    )
    actor_node = WorkEvidenceNode(
        ref=ObjectRef(kind="actor", object_id=actor.format()),
        kind="actor",
        label="Codex agent",
        evidence_refs=(EVIDENCE,),
        corpus_snapshot_ref=SNAPSHOT,
        authority="provider",
        confidence=1.0,
        actor_ref=actor,
    )
    context_node = WorkEvidenceNode(
        ref=ObjectRef(kind="execution-context", object_id=context.context_id),
        kind="execution-context",
        label="Codex execution context",
        evidence_refs=(EVIDENCE,),
        corpus_snapshot_ref=SNAPSHOT,
        authority="provider",
        confidence=1.0,
        execution_context_ref=context,
    )
    invocation_one = _node("invocation", "codex:invoke:1")
    invocation_two = _node("invocation", "codex:invoke:2")
    task = _node("call", "task:ship-work-evidence", label="Agent task")
    retry_task = _node("call", "task:unknown", label="still pending", association_state="unresolved")
    attempt_one = _node("attempt", "attempt:1")
    attempt_two = _node("attempt", "attempt:2")
    attempt_three = _node("attempt", "attempt:3")
    event = ObservedEvent(
        event_ref=ObjectRef(kind="observed-event", object_id="tool-finished"),
        kind="tool_finished",
        run_ref=run.ref,
        summary="Codex ran the verified command",
        evidence_refs=(EVIDENCE,),
    )
    segment_one = session_segment_from_observed_events(
        ref=ObjectRef(kind="work-session-segment", object_id="segment:1"),
        label="Codex transcript segment",
        corpus_snapshot_ref=SNAPSHOT,
        events=(event,),
    )
    segment_two = session_segment_from_observed_events(
        ref=ObjectRef(kind="work-session-segment", object_id="segment:2"),
        label="Codex resume segment",
        corpus_snapshot_ref=SNAPSHOT,
        events=(event,),
    )
    result = _node("structured-result", "result:1")
    original_claim = _node("claim", "claim:1", claim_text="task completed")
    corrected_claim = _node("claim", "claim:2", claim_text="task completed with residual scope")
    artifact = _node("artifact", "archive:result.json").model_copy(
        update={"evidence_refs": (ObjectRef(kind="artifact", object_id="raw:artifact-result"),)}
    )
    nodes = (
        run,
        actor_node,
        context_node,
        invocation_one,
        invocation_two,
        task,
        retry_task,
        attempt_one,
        attempt_two,
        attempt_three,
        segment_one,
        segment_two,
        result,
        original_claim,
        corrected_claim,
        artifact,
    )
    return WorkEvidenceGraph(
        graph_id="codex-agent-task",
        corpus_snapshot_ref=SNAPSHOT,
        nodes=nodes,
        edges=(
            _edge("invoked", "run-invocation-1", run, invocation_one),
            _edge("resumed", "run-invocation-2", run, invocation_two),
            _edge("mentioned", "run-actor", run, actor_node),
            _edge("mentioned", "run-context", run, context_node),
            _edge("invoked", "invocation-task", invocation_one, task),
            _edge("unresolved", "invocation-unknown", invocation_two, retry_task),
            _edge("invoked", "task-attempt-1", task, attempt_one),
            _edge("invoked", "task-attempt-2", task, attempt_two),
            _edge("invoked", "task-attempt-3", task, attempt_three),
            _edge("retried", "attempt-2-retries-1", attempt_two, attempt_one),
            _edge("represented_by", "attempt-1-segment", attempt_one, segment_one),
            _edge("represented_by", "attempt-2-segment-1", attempt_two, segment_one),
            _edge("represented_by", "attempt-2-segment-2", attempt_two, segment_two),
            _edge("produced", "attempt-result", attempt_two, result),
            _edge("claimed", "result-claim", result, original_claim),
            _edge("superseded", "claim-correction", corrected_claim, original_claim),
            _edge("mentioned", "result-artifact", result, artifact),
        ),
    )


@pytest.mark.asyncio
async def test_codex_agent_task_graph_round_trips_and_traverses_bidirectionally(tmp_path: Path) -> None:
    """Anti-vacuity: exercises the repository -> SQLite -> query-store route."""

    graph = _graph()
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)
        task = next(node for node in graph.nodes if node.ref.object_id == "task:ship-work-evidence")
        traversal = await repository.traverse_work_evidence(
            graph_id=graph.graph_id,
            focal_ref=task.ref.format(),
            direction="both",
        )
        incoming = await repository.traverse_work_evidence(
            graph_id=graph.graph_id,
            focal_ref=task.ref.format(),
            direction="incoming",
        )
        actor_traversal = await repository.traverse_work_evidence(
            graph_id=graph.graph_id,
            focal_ref="actor:agent:codex",
            direction="incoming",
        )
        artifact_traversal = await repository.traverse_work_evidence(
            graph_id=graph.graph_id,
            focal_ref="artifact:archive:result.json",
            direction="incoming",
        )

    assert traversal is not None
    assert {node.ref.object_id for node in traversal.nodes} >= {
        "codex:invoke:1",
        "attempt:1",
        "attempt:2",
        "attempt:3",
    }
    assert {edge.kind for edge in traversal.edges} >= {"invoked"}
    assert incoming is not None
    assert [(edge.kind, edge.source_ref.object_id) for edge in incoming.edges] == [("invoked", "codex:invoke:1")]
    assert actor_traversal is not None
    assert [(edge.kind, edge.source_ref.kind) for edge in actor_traversal.edges] == [("mentioned", "run")]
    assert artifact_traversal is not None
    stored_artifact = next(node for node in artifact_traversal.nodes if node.kind == "artifact")
    assert stored_artifact.evidence_refs == (ObjectRef(kind="artifact", object_id="raw:artifact-result"),)


def test_graph_preserves_many_to_many_retries_resumes_and_unresolved_associations() -> None:
    graph = _graph()
    edge_pairs = {(edge.kind, edge.source_ref.object_id, edge.target_ref.object_id) for edge in graph.edges}

    assert sum(edge.kind == "invoked" and edge.target_ref.kind == "work-attempt" for edge in graph.edges) == 3
    assert sum(edge.kind == "represented_by" and edge.source_ref.object_id == "attempt:2" for edge in graph.edges) == 2
    assert ("resumed", "codex-session:work-evidence", "codex:invoke:2") in edge_pairs
    assert ("retried", "attempt:2", "attempt:1") in edge_pairs
    assert next(node for node in graph.nodes if node.ref.object_id == "task:unknown").association_state == "unresolved"
    assert ("superseded", "claim:2", "claim:1") in edge_pairs


def test_mutations_cannot_collapse_task_to_session_or_claim_to_observed_truth() -> None:
    """The node-kind/ref contract rejects both prohibited identity collapses."""

    with pytest.raises(ValueError, match="call nodes require a work-call ref"):
        WorkEvidenceNode(
            ref=ObjectRef(kind="session", object_id="codex-session:work-evidence"),
            kind="call",
            label="incorrect task=session mutation",
            evidence_refs=(EVIDENCE,),
            corpus_snapshot_ref=SNAPSHOT,
            confidence=1.0,
        )
    with pytest.raises(ValueError, match="claim nodes require a work-claim ref"):
        WorkEvidenceNode(
            ref=ObjectRef(kind="observed-event", object_id="external-effect"),
            kind="claim",
            label="incorrect claim=truth mutation",
            claim_text="this would overwrite a project effect",
            evidence_refs=(EVIDENCE,),
            corpus_snapshot_ref=SNAPSHOT,
            confidence=1.0,
        )
