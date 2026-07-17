from __future__ import annotations

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.claude_workflow_evidence import project_claude_workflow_evidence
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.parsers.claude.orchestration import ClaudeOrchestrationArtifact, ClaudeOrchestrationFact


def test_claude_projection_uses_provider_facts_not_child_topology() -> None:
    evidence = EvidenceRef(session_id="claude-code-session:coordinator", message_id="m1", block_index=0)
    artifact = ClaudeOrchestrationArtifact(
        kind="workflow_journal",
        source_path="/workflows/wf_54/journal.jsonl",
        parse_policy="fact",
        facts=tuple(
            ClaudeOrchestrationFact(
                kind="workflow_journal_entry",
                source_path="/workflows/wf_54/journal.jsonl",
                source_line=index,
                run_id="wf_54",
                agent_id=f"agent-{index}",
                content_key=f"call-{index % 50}",
                payload={
                    "attemptId": f"attempt-{index}",
                    **({"structuredResult": {"ok": True}} if index < 65 else {}),
                    **({"unresolved": True} if index == 49 else {}),
                },
            )
            for index in range(91)
        ),
    )
    events = tuple(
        ParsedSessionEvent(
            event_type="claude_workflow_invocation",
            timestamp=None,
            source_message_provider_id=f"invoke-{index}",
            payload={"runId": "wf_54", **({"resumeFromRunId": "wf_53"} if index == 3 else {})},
        )
        for index in range(4)
    )
    graph = project_claude_workflow_evidence(
        graph_id="fixture",
        corpus_snapshot_ref=ObjectRef(kind="context-snapshot", object_id="fixture"),
        artifacts=(artifact,),
        coordinator_events=events,
        coordinator_evidence=evidence,
        evidence_for_path=lambda _path: evidence,
    )

    assert sum(node.kind == "invocation" for node in graph.nodes) == 4
    assert sum(node.kind == "call" for node in graph.nodes) == 50
    assert sum(node.kind == "attempt" for node in graph.nodes) == 91
    assert sum(node.kind == "structured-result" for node in graph.nodes) == 65
    assert all("unrelated-child" not in node.ref.object_id for node in graph.nodes)
    assert any(edge.kind == "resumed" for edge in graph.edges)
    assert any(edge.kind == "unresolved" for edge in graph.edges) is False
    assert any(node.association_state == "unresolved" for node in graph.nodes)
