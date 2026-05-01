from __future__ import annotations

from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.models import (
    And,
    AttrContains,
    AttrEq,
    AttrIn,
    EvidenceEnvelope,
    Kind,
    Not,
    SourceSpan,
    SubjectRef,
    TrustMetadata,
    subject_query_from_payload,
)
from polylogue.showcase.invariants import SHOWCASE_INVARIANTS


def test_subject_query_ast_matches_and_round_trips() -> None:
    subject = SubjectRef(
        kind="schema.annotation",
        id="codex:v1:field",
        attrs={
            "annotation": "x-polylogue-values",
            "provider": "codex",
            "values": ["assistant", "user"],
        },
    )
    query = And(
        (
            Kind("schema.annotation"),
            AttrEq("annotation", "x-polylogue-values"),
            AttrIn("provider", ("codex", "chatgpt")),
            AttrContains("values", "assistant"),
            Not(AttrEq("provider", "gemini")),
        )
    )

    assert query.matches(subject)
    assert subject_query_from_payload(query.to_payload()).matches(subject)


def test_evidence_envelope_carries_trust_and_stable_fingerprint() -> None:
    trust = TrustMetadata(
        producer="tests",
        reviewed_at="2026-04-22T00:00:00+00:00",
        level="authored",
    )

    first = EvidenceEnvelope.build(
        obligation_id="claim|runner|subject",
        status=OutcomeStatus.OK,
        evidence={"ok": True},
        counterexample=None,
        reproducer=("devtools", "render-verification-catalog", "--check"),
        artifacts=("docs/verification-catalog.md",),
        environment={"python": "3.13"},
        provenance=SourceSpan("tests/unit/proof/test_models.py", line=1),
        trust=trust,
    )
    second = EvidenceEnvelope.build(
        obligation_id="claim|runner|subject",
        status=OutcomeStatus.OK,
        evidence={"ok": True},
        counterexample=None,
        reproducer=("devtools", "render-verification-catalog", "--check"),
        artifacts=("docs/verification-catalog.md",),
        environment={"python": "3.13"},
        provenance=SourceSpan("tests/unit/proof/test_models.py", line=1),
        trust=trust,
    )

    assert first.fingerprint == second.fingerprint
    payload = first.to_payload()
    assert payload["trust"] == trust.to_payload()
    assert payload["environment"] == {"python": "3.13"}


def test_showcase_invariant_to_claim_bridge_preserves_invariant_identity() -> None:
    claim = SHOWCASE_INVARIANTS[0].to_claim()

    assert claim.id == "showcase.invariant.json_valid"
    assert claim.matches(SubjectRef(kind="showcase.exercise", id="json-list"))
    assert claim.bug_classes == ("showcase.invariant.json_valid",)
    assert claim.breaker is not None
