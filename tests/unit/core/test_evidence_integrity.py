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


def test_a_quarantined_leaf_blocks_partial_support() -> None:
    """A quarantined descendant is unresolved, never partially supported.

    Anti-vacuity: without `quarantined` in `_UNRESOLVED_REF_STATES` the witness
    it records matches no ladder branch, the valid sibling leaf populates
    `supported_paths`, and the verdict falls through to `partially_supported`
    whose `supported` property is true -- so quarantined evidence authorizes
    context injection and a public claim.
    """
    nodes, edges = _graph()
    nodes["quarantined:q"] = EvidenceGraphNode(
        "quarantined:q", "raw", authority="raw", ref_state="quarantined", frame_hash="frame", definition_hash="def"
    )
    edges.append(EvidenceGraphEdge("finding:f", "quarantined:q"))

    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")

    assert verdict.status == "unresolved"
    assert verdict.supported is False
    assert any(w.code == "quarantined" for w in verdict.witnesses)


def test_every_ref_state_names_the_verdict_it_forces() -> None:
    """The ref-state partition is total, so no member can fall through."""
    from typing import get_args

    from polylogue.core.evidence_integrity import (
        _PRIVATE_REF_STATES,
        _STALE_REF_STATES,
        _UNRESOLVED_REF_STATES,
        EvidenceRefState,
    )

    named = {"ok"} | _PRIVATE_REF_STATES | _UNRESOLVED_REF_STATES | _STALE_REF_STATES
    assert named == set(get_args(EvidenceRefState))


def test_unknown_authority_never_grounds_a_claim() -> None:
    """An adapter that omits `authority` cannot launder a claim into support.

    Anti-vacuity: without the `unknown_authority` witness the authority set is
    `{"unknown"}`, which fails the `<= {"agent", "assertion"}` closed-loop test,
    so the leaf produces a fully `supported` verdict.
    """
    nodes, edges = _graph(authority="unknown")

    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")

    assert verdict.status == "unresolved"
    assert verdict.supported is False
    assert any(w.code == "unknown_authority" for w in verdict.witnesses)


def test_declared_authority_still_supports_a_claim() -> None:
    """The opposite direction: a named grounding authority is not refused."""
    for authority in ("raw", "human", "tool", "git", "pr"):
        nodes, edges = _graph(authority=authority)
        verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
        assert verdict.status == "supported", authority


@pytest.mark.parametrize("review_state", ["pending", "rejected", "needs_changes"])
def test_unapproved_review_state_is_refused(review_state: str) -> None:
    """Only an approved node grounds a claim; every other value fails closed.

    Anti-vacuity: checking only the two privacy spellings records no witness
    for `pending`/`rejected`, the leaf joins `supported_paths`, and the verdict
    is fully `supported`.
    """
    nodes, edges = _graph()
    nodes["raw:r"] = EvidenceGraphNode(
        "raw:r", "raw", authority="raw", review_state=review_state, frame_hash="frame", definition_hash="def"
    )

    verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")

    assert verdict.status == "not_supported"
    assert verdict.supported is False
    assert any(w.code == "review_unapproved" for w in verdict.witnesses)


def test_private_review_state_is_still_held_private() -> None:
    """The opposite direction: privacy keeps its own, higher-precedence verdict."""
    for review_state in ("private", "held_private"):
        nodes, edges = _graph()
        nodes["raw:r"] = EvidenceGraphNode(
            "raw:r", "raw", authority="raw", review_state=review_state, frame_hash="frame", definition_hash="def"
        )
        verdict = evaluate_evidence("finding:f", nodes, edges, frame_hash="frame", definition_hash="def")
        assert verdict.status == "held_private"
        assert not any(w.code == "review_unapproved" for w in verdict.witnesses)
