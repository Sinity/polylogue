from __future__ import annotations

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.claude_workflow_evidence import (
    ClaudeWorkflowCoordinatorInvocation,
    ClaudeWorkflowPromptEvidence,
    ClaudeWorkflowSessionEvidence,
    project_claude_workflow_evidence,
)
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.parsers.claude.orchestration import ClaudeOrchestrationArtifact, ClaudeOrchestrationFact

RUN_ID = "wf_54d4fb2e-841"


def test_claude_projection_uses_provider_references_not_child_topology() -> None:
    """Provider paths and journal references admit attempts; parent topology cannot.

    Production dependency: the projector consumes OriginSpec-parsed facts plus
    raw artifact and indexed-message evidence. Anti-vacuity mutation: accepting
    every supplied child session would add the 38 ``unrelated`` sessions;
    removing sidecar/transcript matching would drop all 91 represented segments.
    """

    journal_path = f"/fixture/subagents/workflows/{RUN_ID}/journal.jsonl"
    run_path = f"/fixture/workflows/{RUN_ID}.json"
    adopt_path = "/fixture/jobs/coordinator/adopt.json"
    journal_facts: list[ClaudeOrchestrationFact] = []
    sidecars: list[ClaudeOrchestrationArtifact] = []
    sessions: list[ClaudeWorkflowSessionEvidence] = []

    for index in range(91):
        attempt_id = f"attempt-{index:03d}"
        content_key = f"call-{index % 50:02d}"
        transcript_path = f"/fixture/subagents/agent-{attempt_id}.jsonl"
        meta_path = f"/fixture/subagents/agent-{attempt_id}.meta.json"
        journal_facts.append(
            ClaudeOrchestrationFact(
                kind="workflow_journal_entry",
                source_path=journal_path,
                source_line=index + 1,
                run_id=RUN_ID,
                agent_id=attempt_id,
                content_key=content_key,
                payload={
                    "attemptId": attempt_id,
                    "agentId": attempt_id,
                    "contentKey": content_key,
                    "metaPath": meta_path,
                    "transcriptPath": transcript_path,
                    "status": "completed",
                    **({"structuredResult": {"ok": True, "ordinal": index}} if index < 65 else {}),
                    **({"unresolved": True} if content_key == "call-49" else {}),
                },
            )
        )
        sidecars.append(
            ClaudeOrchestrationArtifact(
                kind="agent_sidecar_meta",
                source_path=meta_path,
                parse_policy="fact",
                facts=(
                    ClaudeOrchestrationFact(
                        kind="agent_sidecar_meta",
                        source_path=meta_path,
                        source_line=None,
                        run_id=RUN_ID,
                        agent_id=attempt_id,
                        content_key=content_key,
                        payload={
                            "runId": RUN_ID,
                            "attemptId": attempt_id,
                            "agentId": attempt_id,
                            "contentKey": content_key,
                            "transcriptPath": transcript_path,
                            "model": "claude-sonnet-fixture",
                            "status": "completed",
                            "timing": {"startedAt": index * 10, "completedAt": index * 10 + 5},
                            "tokens": {"input": 100 + index, "output": 20 + index},
                            "tools": ["Read", "Write"],
                        },
                    ),
                ),
            )
        )
        session_id = f"claude-code-session:agent-{index:03d}"
        sessions.append(
            ClaudeWorkflowSessionEvidence(
                source_path=transcript_path,
                raw_artifact_ref=ObjectRef(kind="artifact", object_id=f"raw:transcript-{index:03d}"),
                session_id=session_id,
                session_evidence_ref=EvidenceRef(session_id=session_id),
                prompt_evidence_ref=EvidenceRef(session_id=session_id, message_id=f"prompt-{index:03d}"),
                prompt_material_origin="generated_context_pack",
            )
        )

    for index in range(38):
        session_id = f"claude-code-session:unrelated-{index:03d}"
        sessions.append(
            ClaudeWorkflowSessionEvidence(
                source_path=f"/fixture/subagents/agent-unrelated-{index:03d}.jsonl",
                raw_artifact_ref=ObjectRef(kind="artifact", object_id=f"raw:unrelated-{index:03d}"),
                session_id=session_id,
                session_evidence_ref=EvidenceRef(session_id=session_id),
            )
        )

    artifacts = [
        ClaudeOrchestrationArtifact(
            kind="workflow_run_snapshot",
            source_path=run_path,
            parse_policy="fact",
            facts=(
                ClaudeOrchestrationFact(
                    kind="workflow_run_snapshot",
                    source_path=run_path,
                    source_line=None,
                    run_id=RUN_ID,
                    agent_id=None,
                    content_key=None,
                    payload={
                        "runId": RUN_ID,
                        "taskId": "task-mandate-01",
                        "workflowName": "admission-proof",
                        "scriptPath": "/repo/scripts/workflow.py",
                        "phases": ["plan", "execute", "collect"],
                        "labels": ["mandate", "synthetic"],
                        "status": "completed_with_gap",
                        "finalResult": {"status": "complete", "value": "fixture final"},
                    },
                ),
            ),
        ),
        ClaudeOrchestrationArtifact(
            kind="workflow_journal",
            source_path=journal_path,
            parse_policy="fact",
            facts=tuple(journal_facts),
        ),
        ClaudeOrchestrationArtifact(
            kind="adopt_manifest",
            source_path=adopt_path,
            parse_policy="fact",
            facts=(
                ClaudeOrchestrationFact(
                    kind="adopt_manifest",
                    source_path=adopt_path,
                    source_line=None,
                    run_id=RUN_ID,
                    agent_id="coordinator",
                    content_key=None,
                    payload={"runId": RUN_ID, "resumeFromRunId": RUN_ID, "status": "adopted"},
                ),
            ),
        ),
        *sidecars,
    ]
    artifact_evidence = {
        artifact.source_path: ObjectRef(kind="artifact", object_id=f"raw:fact-{index:03d}")
        for index, artifact in enumerate(artifacts)
    }
    coordinator_evidence = EvidenceRef(
        session_id="claude-code-session:coordinator",
        message_id="coordinator-message",
    )
    invocations = tuple(
        ClaudeWorkflowCoordinatorInvocation(
            session_id="claude-code-session:coordinator",
            event=ParsedSessionEvent(
                event_type="claude_workflow_invocation",
                source_message_provider_id=f"invoke-{index}",
                payload={
                    "runId": RUN_ID,
                    "taskId": "task-mandate-01",
                    "workflowName": "admission-proof",
                    **({"resumeFromRunId": RUN_ID} if index else {}),
                },
            ),
            evidence_ref=coordinator_evidence,
        )
        for index in range(4)
    )

    graph = project_claude_workflow_evidence(
        graph_id=f"claude-workflow:{RUN_ID}",
        run_id=RUN_ID,
        corpus_snapshot_ref=ObjectRef(kind="context-snapshot", object_id="fixture"),
        artifacts=artifacts,
        artifact_evidence=artifact_evidence,
        coordinator_invocations=invocations,
        session_evidence=sessions,
        coordinator_prompt_evidence=(
            ClaudeWorkflowPromptEvidence(
                session_id="claude-code-session:coordinator",
                evidence_ref=EvidenceRef(
                    session_id="claude-code-session:coordinator",
                    message_id="direct-prompt",
                ),
                material_origin="human_authored",
            ),
        ),
    )

    counts = {
        kind: sum(node.kind == kind for node in graph.nodes)
        for kind in {"artifact", "attempt", "call", "invocation", "run", "session-segment", "structured-result"}
    }
    assert counts == {
        "artifact": 94,
        "attempt": 91,
        "call": 50,
        "invocation": 4,
        "run": 1,
        "session-segment": 91,
        "structured-result": 66,
    }
    assert all("unrelated" not in node.ref.object_id for node in graph.nodes)
    assert sum(edge.kind == "resumed" for edge in graph.edges) == 3
    assert sum(node.kind == "call" and node.association_state == "unresolved" for node in graph.nodes) == 1
    assert (
        sum(
            node.kind == "claim"
            and node.claim_text is not None
            and "worker prompt material origin" in node.claim_text
            and "generated_context_pack" in node.claim_text
            for node in graph.nodes
        )
        == 91
    )
    assert any(
        node.kind == "claim"
        and node.association_state == "resolved"
        and node.claim_text is not None
        and "coordinator prompt material origin" in node.claim_text
        and "human_authored" in node.claim_text
        for node in graph.nodes
    )
    assert all(node.evidence_refs for node in graph.nodes)
    assert all(edge.evidence_refs for edge in graph.edges)
    assert any(
        isinstance(ref, ObjectRef) and ref.kind == "artifact" for node in graph.nodes for ref in node.evidence_refs
    )
