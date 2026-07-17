"""Project admitted Claude Workflow artifacts into the generic evidence graph.

This adapter has no knowledge of session-parent topology: every membership edge
comes from a coordinator invocation or an admitted workflow artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceEdge, WorkEvidenceGraph, WorkEvidenceNode
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.parsers.claude.orchestration import ClaudeOrchestrationArtifact, ClaudeOrchestrationFact


def project_claude_workflow_evidence(
    *,
    graph_id: str,
    corpus_snapshot_ref: ObjectRef,
    artifacts: Iterable[ClaudeOrchestrationArtifact],
    coordinator_events: Iterable[ParsedSessionEvent],
    coordinator_evidence: EvidenceRef,
    evidence_for_path: Callable[[str], EvidenceRef | None],
) -> WorkEvidenceGraph:
    """Create a generic graph from direct Claude Workflow evidence only.

    Missing evidence-bearing counterparts remain absent/unresolved; this never
    promotes an arbitrary child session to a Workflow attempt.
    """
    nodes: dict[str, WorkEvidenceNode] = {}
    edges: dict[str, WorkEvidenceEdge] = {}

    def evidence(path: str) -> EvidenceRef:
        return evidence_for_path(path) or coordinator_evidence

    def add_node(kind: str, object_id: str, label: str, source: EvidenceRef, *, state: str = "resolved") -> ObjectRef:
        ref_kind = {
            "run": "run",
            "invocation": "work-invocation",
            "call": "work-call",
            "attempt": "work-attempt",
            "session-segment": "work-session-segment",
            "structured-result": "work-result",
            "artifact": "artifact",
        }[kind]
        ref = ObjectRef(kind=ref_kind, object_id=object_id)  # type: ignore[arg-type]
        nodes.setdefault(
            ref.format(),
            WorkEvidenceNode(
                ref=ref,
                kind=kind,  # type: ignore[arg-type]
                label=label,
                evidence_refs=(source,),
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
                association_state=state,  # type: ignore[arg-type]
            ),
        )
        return ref

    def add_edge(
        kind: str, source: ObjectRef, target: ObjectRef, source_evidence: EvidenceRef, *, state: str = "resolved"
    ) -> None:
        edge_id = f"claude:{kind}:{source.format()}->{target.format()}"
        ref = ObjectRef(kind="work-edge", object_id=edge_id)
        edges.setdefault(
            ref.format(),
            WorkEvidenceEdge(
                ref=ref,
                kind=kind,  # type: ignore[arg-type]
                source_ref=source,
                target_ref=target,
                evidence_refs=(source_evidence,),
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
                association_state=state,  # type: ignore[arg-type]
            ),
        )

    runs: dict[str, ObjectRef] = {}
    calls: dict[tuple[str, str], ObjectRef] = {}

    def run_ref(run_id: str, source: EvidenceRef) -> ObjectRef:
        return runs.setdefault(
            run_id, add_node("run", f"claude-workflow:{run_id}", f"Claude Workflow {run_id}", source)
        )

    for event in coordinator_events:
        if event.event_type != "claude_workflow_invocation":
            continue
        payload = event.payload
        run_id = _string(payload, "runId", "run_id")
        if run_id is None:
            continue
        run = run_ref(run_id, coordinator_evidence)
        invocation = add_node(
            "invocation",
            f"claude-workflow:{run_id}:{event.source_message_provider_id}",
            "Workflow invocation",
            coordinator_evidence,
        )
        add_edge("invoked", run, invocation, coordinator_evidence)
        resumed = _string(payload, "resumeFromRunId", "resume_from_run_id")
        if resumed is not None:
            add_edge("resumed", run, run_ref(resumed, coordinator_evidence), coordinator_evidence)

    for artifact in artifacts:
        artifact_evidence = evidence(artifact.source_path)
        artifact_ref = add_node("artifact", f"claude-artifact:{artifact.source_path}", artifact.kind, artifact_evidence)
        for fact in artifact.facts:
            _project_fact(fact, artifact_ref, artifact_evidence, run_ref, calls, add_node, add_edge)

    return WorkEvidenceGraph(
        graph_id=graph_id,
        corpus_snapshot_ref=corpus_snapshot_ref,
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
    )


def _project_fact(
    fact: ClaudeOrchestrationFact,
    artifact_ref: ObjectRef,
    evidence: EvidenceRef,
    run_ref: Callable[[str, EvidenceRef], ObjectRef],
    calls: dict[tuple[str, str], ObjectRef],
    add_node: Callable[..., ObjectRef],
    add_edge: Callable[..., None],
) -> None:
    if fact.run_id is None:
        return
    run = run_ref(fact.run_id, evidence)
    add_edge("mentioned", run, artifact_ref, evidence)
    resumed = _string(fact.payload, "resumeFromRunId", "resume_from_run_id")
    if resumed is not None:
        add_edge("resumed", run, run_ref(resumed, evidence), evidence)
    if fact.kind != "workflow_journal_entry" or fact.content_key is None:
        return
    key = (fact.run_id, fact.content_key)
    call = calls.get(key)
    if call is None:
        call = add_node(
            "call",
            f"claude-workflow:{fact.run_id}:call:{fact.content_key}",
            fact.content_key,
            evidence,
            state="unresolved" if fact.payload.get("unresolved") else "resolved",
        )
        calls[key] = call
    add_edge("invoked", run, call, evidence, state="unresolved" if fact.payload.get("unresolved") else "resolved")
    attempt_id = _string(fact.payload, "attemptId", "attempt") or fact.agent_id
    if attempt_id is not None:
        attempt = add_node("attempt", f"claude-workflow:{fact.run_id}:attempt:{attempt_id}", attempt_id, evidence)
        add_edge("invoked", call, attempt, evidence)
    if "structuredResult" in fact.payload or "result" in fact.payload:
        result = add_node(
            "structured-result",
            f"claude-workflow:{fact.run_id}:result:{fact.content_key}:{fact.source_line or 0}",
            "Workflow structured result",
            evidence,
        )
        add_edge("produced", call, result, evidence)


def _string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = ["project_claude_workflow_evidence"]
