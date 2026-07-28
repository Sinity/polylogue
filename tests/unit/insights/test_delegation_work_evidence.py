"""Delegation-facts -> work-evidence graph projection (polylogue-1vpm.6.1 AC6)."""

from __future__ import annotations

from polylogue.core.refs import ObjectRef
from polylogue.insights.delegation_work_evidence import materialize_delegation_work_evidence_graph
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveDelegationQueryRow

SNAPSHOT = ObjectRef(kind="context-snapshot", object_id="corpus:delegation-2026-07-27")


def _row(
    *,
    parent_session_id: str = "codex-session:parent",
    child_session_id: str | None = "codex-session:child",
    mapping_state: str = "resolved",
    instruction_tool_use_block_id: str | None = "toolu_1",
    instruction_message_id: str | None = "m1",
    artifact_text: str | None = None,
) -> ArchiveDelegationQueryRow:
    return ArchiveDelegationQueryRow(
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        mapping_state=mapping_state,  # type: ignore[arg-type]
        link_confidence=1.0,
        link_method="tool_use_block",
        inheritance="spawned-fresh",
        branch_point_message_id=None,
        instruction_message_id=instruction_message_id,
        instruction_tool_use_block_id=instruction_tool_use_block_id,
        instruction_payload="review the diff",
        dispatch_turn_model="example-model",
        requested_model="example-model",
        artifact_block_id=None,
        artifact_text=artifact_text,
        result_is_error=0,
        result_exit_code=0,
        result_status="ok",
        parent_origin="codex-session",
        parent_session_dominant_model="example-model",
        parent_session_dominant_model_family="example-family",
        parent_terminal_state="completed",
        child_session_dominant_model="example-model",
        child_session_dominant_model_family="example-family",
        child_cost_usd=0.01,
        child_cost_is_estimated=0,
        child_tokens=100,
        child_wall_ms=5000,
        child_terminal_state="completed",
    )


def test_resolved_delegation_projects_call_attempt_and_claim() -> None:
    row = _row(artifact_text="reviewed the diff, no issues found")
    graph = materialize_delegation_work_evidence_graph(
        graph_id="delegation-graph", corpus_snapshot_ref=SNAPSHOT, rows=(row,)
    )

    kinds_by_ref = {node.ref.format(): node.kind for node in graph.nodes}
    assert set(kinds_by_ref.values()) == {"call", "attempt", "claim"}
    call = next(node for node in graph.nodes if node.kind == "call")
    attempt = next(node for node in graph.nodes if node.kind == "attempt")
    claim = next(node for node in graph.nodes if node.kind == "claim")
    assert call.association_state == "resolved"
    assert attempt.association_state == "resolved"
    assert claim.claim_text == "reviewed the diff, no issues found"

    assert any(edge.kind == "produced" and edge.target_ref == claim.ref for edge in graph.edges)
    assert any(
        edge.kind == "invoked" and edge.source_ref == call.ref and edge.target_ref == attempt.ref
        for edge in graph.edges
    )


def test_edge_only_dispatch_with_no_child_session_yields_no_fabricated_attempt() -> None:
    """AC6: an unresolved dispatch never fabricates a one-to-one attempt link."""

    row = _row(child_session_id=None, mapping_state="edge_only", instruction_tool_use_block_id="toolu_2")
    graph = materialize_delegation_work_evidence_graph(
        graph_id="delegation-graph", corpus_snapshot_ref=SNAPSHOT, rows=(row,)
    )

    assert len(graph.nodes) == 1
    call = graph.nodes[0]
    assert call.kind == "call"
    assert call.association_state == "unresolved"
    assert graph.edges == ()


def test_quarantined_mapping_state_becomes_contradicted_not_silently_dropped() -> None:
    row = _row(child_session_id="codex-session:cycle-child", mapping_state="quarantined")
    graph = materialize_delegation_work_evidence_graph(
        graph_id="delegation-graph", corpus_snapshot_ref=SNAPSHOT, rows=(row,)
    )

    call = next(node for node in graph.nodes if node.kind == "call")
    attempt = next(node for node in graph.nodes if node.kind == "attempt")
    assert call.association_state == "contradicted"
    assert attempt.association_state == "contradicted"
    unresolved_edges = [edge for edge in graph.edges if edge.kind == "unresolved"]
    assert len(unresolved_edges) == 1


def test_many_dispatches_from_one_parent_session_retain_distinct_call_identity() -> None:
    """A parent session dispatching two children must not collapse into one call node."""

    rows = (
        _row(instruction_tool_use_block_id="toolu_a", child_session_id="codex-session:child-a"),
        _row(instruction_tool_use_block_id="toolu_b", child_session_id="codex-session:child-b"),
    )
    graph = materialize_delegation_work_evidence_graph(
        graph_id="delegation-graph", corpus_snapshot_ref=SNAPSHOT, rows=rows
    )

    call_ids = {node.ref.object_id for node in graph.nodes if node.kind == "call"}
    assert call_ids == {
        "delegation:codex-session:parent:toolu_a",
        "delegation:codex-session:parent:toolu_b",
    }
