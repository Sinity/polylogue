from __future__ import annotations

import pytest

from polylogue.context.compiler import compile_assertion_context_segment
from polylogue.core.evidence_integrity import (
    EvidenceAuthority,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    evaluate_evidence,
)


def _graph(*, authority: EvidenceAuthority = "raw") -> tuple[dict[str, EvidenceGraphNode], list[EvidenceGraphEdge]]:
    return (
        {
            "finding:f": EvidenceGraphNode("finding:f", "finding", frame_hash="frame", definition_hash="def"),
            "raw:r": EvidenceGraphNode("raw:r", "raw", authority=authority, frame_hash="frame", definition_hash="def"),
        },
        [EvidenceGraphEdge("finding:f", "raw:r")],
    )


def test_one_evaluator_reports_supported_and_distinct_failure_witnesses() -> None:
    nodes, edges = _graph()
    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
    assert verdict.status == "supported"
    assert verdict.supported_paths == (("finding:f", "raw:r"),)

    nodes["raw:r"] = EvidenceGraphNode(
        "raw:r", "raw", authority="raw", ref_state="stale", frame_hash="other", definition_hash="other"
    )
    drifted = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
    assert drifted.status == "stale"
    assert {w.code for w in drifted.witnesses} == {"stale", "definition_drift", "frame_drift"}
    assert all(w.path == ("finding:f", "raw:r") for w in drifted.witnesses)


@pytest.mark.parametrize(
    ("node_state", "expected"),
    [("missing", "unresolved"), ("ambiguous", "unresolved"), ("private", "held_private")],
)
def test_resolution_failures_are_bounded_and_fail_closed(node_state: str, expected: str) -> None:
    nodes, edges = _graph()
    nodes["raw:r"] = EvidenceGraphNode("raw:r", "raw", ref_state=node_state)  # type: ignore[arg-type]
    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
    assert verdict.status == expected
    assert verdict.witnesses[0].path == ("finding:f", "raw:r")


def test_agent_and_assertion_only_ancestry_is_closed_loop() -> None:
    nodes, edges = _graph(authority="agent")
    nodes["raw:r"] = EvidenceGraphNode("raw:r", "assertion", authority="assertion", frame_hash="frame")
    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame")
    assert verdict.status == "closed_loop"


def test_cycle_and_incompatible_transcript_never_support() -> None:
    nodes, edges = _graph(authority="tool")
    edges.append(EvidenceGraphEdge("raw:r", "finding:f"))
    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
    assert verdict.status == "cycle"
    assert any(w.code == "cycle" and w.path[-1] == "finding:f" for w in verdict.witnesses)

    nodes, edges = _graph(authority="agent")
    nodes["raw:r"] = EvidenceGraphNode("raw:r", "transcript", authority="agent", compatible=False, frame_hash="frame")
    incompatible = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
    assert incompatible.status == "not_supported"
    assert any(w.code == "grounding_incompatible" for w in incompatible.witnesses)


def test_context_verdict_overrides_requested_injection() -> None:
    nodes, edges = _graph(authority="agent")
    nodes["raw:r"] = EvidenceGraphNode("raw:r", "assertion", authority="assertion", frame_hash="frame")
    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame")
    segment = compile_assertion_context_segment(
        assertion_id="f",
        kind="finding",
        body_text="candidate",
        target_ref="session:s",
        author_kind="user",
        author_ref="user:operator",
        status="active",
        context_policy={"inject": True},
        integrity_verdict=verdict,
    )
    assert segment.trust_class == "quoted"
    assert "evidence-integrity:closed_loop" in segment.caveats


def test_evaluation_cancellation_is_recorded() -> None:
    nodes, edges = _graph()
    verdict = evaluate_evidence("finding:f", nodes, edges, cancelled=lambda: True)
    assert verdict.status == "unresolved"
    assert any(w.code == "evaluation_cancelled" for w in verdict.witnesses)
