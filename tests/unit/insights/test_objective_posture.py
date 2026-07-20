"""Tests for the objective-posture resumability projection (polylogue-37t.23)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.insights.archive_models import ObjectivePosturePayload
from polylogue.insights.objective_posture import (
    ASSERTION_TIER_KINDS,
    derive_objective_posture,
    resolve_session_objective_posture,
    structural_objective_posture,
)
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope


def _assertion(
    assertion_id: str,
    kind: AssertionKind,
    *,
    target_ref: str = "session:s1",
    evidence_refs: tuple[str, ...] = (),
    confidence: float | None = None,
    updated_at_ms: int = 1_000,
) -> ArchiveAssertionEnvelope:
    return ArchiveAssertionEnvelope(
        assertion_id=assertion_id,
        scope_ref=None,
        target_ref=target_ref,
        key=None,
        kind=kind,
        value={"claimed_resolved": True},
        body_text="body",
        author_ref="agent:test",
        author_kind="agent",
        evidence_refs=list(evidence_refs),
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        confidence=confidence,
        staleness=None,
        context_policy={"inject": False},
        supersedes=[],
        created_at_ms=updated_at_ms,
        updated_at_ms=updated_at_ms,
    )


class TestStructuralObjectivePosture:
    """Tier-4 (structural_inference): bounded terminal_state mapping."""

    @pytest.mark.parametrize(
        ("terminal_state", "expected_posture"),
        [
            ("tool_left", "awaiting_effect"),
            ("error_left", "blocked"),
            ("question_left", "ambiguous"),
            ("unknown", "ambiguous"),
        ],
    )
    def test_never_emits_completed(self, terminal_state: str, expected_posture: str) -> None:
        """AC3/AC4: no structural signal observes a satisfied effect, so this
        tier must never claim "completed" for any terminal_state -- removing
        this mapping (e.g. defaulting unmapped/error-adjacent states to
        "completed") would recreate the exact false-completion bug
        polylogue-9e5.9 measured (50.5% agreement, a coin flip) for the
        deleted `clean_finish` heuristic.
        """

        result = structural_objective_posture(
            terminal_state=terminal_state,
            terminal_state_confidence=0.78,
        )
        assert result.posture == expected_posture
        assert result.posture != "completed"
        assert result.authority == "structural_inference"

    def test_unrecognized_terminal_state_yields_no_authority(self) -> None:
        """A terminal_state this tier does not recognize (e.g. the retired
        `clean_finish`, or any future/unknown label) must fall through to
        authority="none"/"unknown" rather than silently guessing a posture.
        """

        result = structural_objective_posture(terminal_state="clean_finish", terminal_state_confidence=0.68)
        assert result.authority == "none"
        assert result.posture == "unknown"

    def test_confidence_is_capped_and_evidence_refs_formatted(self) -> None:
        result = structural_objective_posture(
            terminal_state="error_left",
            terminal_state_confidence=0.95,
            terminal_state_evidence={"action_id": "act-1", "evidence_class": "raw_evidence"},
            as_of="2026-07-20T00:00:00+00:00",
        )
        # Confidence capped below the raw terminal-state confidence: posture
        # is a further inference layer over terminal_state, never stronger.
        assert result.confidence <= 0.8
        assert result.evidence_refs == ("action_id:act-1",)
        assert result.as_of == "2026-07-20T00:00:00+00:00"

    def test_message_id_evidence_formats_as_object_ref(self) -> None:
        result = structural_objective_posture(
            terminal_state="question_left",
            terminal_state_confidence=0.72,
            terminal_state_evidence={"message_id": "claude-code-session:s1:5", "evidence_class": "raw_evidence"},
        )
        assert result.evidence_refs == ("message:claude-code-session:s1:5",)


class TestDeriveObjectivePosture:
    """Blend the assertion tier over the structural tier per authority order."""

    def test_no_relevant_assertions_returns_structural_unchanged(self) -> None:
        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)
        result = derive_objective_posture(structural, [])
        assert result == structural

        irrelevant = [_assertion("a1", AssertionKind.NOTE)]
        result_irrelevant = derive_objective_posture(structural, irrelevant)
        assert result_irrelevant == structural

    def test_assertion_tier_outranks_structural_inference(self) -> None:
        """AC3/AC4: explicit assertion evidence must outrank the weaker
        structural inference. Mutation: swapping the winner (returning
        `structural` whenever `relevant` is non-empty) makes this fail --
        that is the exact regression AC4 calls out ("removing the authority
        precedence recreates the known false completion/false positive").
        """

        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)
        assertions = [_assertion("decision-1", AssertionKind.DECISION, evidence_refs=("session:s1",))]

        result = derive_objective_posture(structural, assertions)

        assert result.authority == "assertion"
        assert result.posture == "awaiting_operator"
        assert result.posture != "completed"
        assert "assertion:decision-1" in result.evidence_refs
        assert "session:s1" in result.evidence_refs
        assert "structural_inference:ambiguous" in result.contradictions

    def test_blocker_outranks_decision_and_handoff_with_contradictions_preserved(self) -> None:
        """AC2: multiple simultaneous obligations are preserved (not
        silently collapsed) -- the non-winning kinds surface as
        contradictions rather than disappearing.
        """

        structural = structural_objective_posture(terminal_state="tool_left", terminal_state_confidence=0.9)
        assertions = [
            _assertion("handoff-1", AssertionKind.HANDOFF),
            _assertion("decision-1", AssertionKind.DECISION),
            _assertion("blocker-1", AssertionKind.BLOCKER),
        ]

        result = derive_objective_posture(structural, assertions)

        assert result.posture == "blocked"
        assert result.authority == "assertion"
        assert any("decision-1" in item for item in result.contradictions)
        assert any("handoff-1" in item for item in result.contradictions)
        assert any("structural_inference" in item for item in result.contradictions)

    def test_self_reported_assertion_value_cannot_force_completed(self) -> None:
        """AC3: 'a self-reported claim without observed/evaluated effect
        cannot become completed'. Every fixture assertion here carries
        `value={"claimed_resolved": True}` -- freeform author-supplied
        content the blend function must not interpret as a completion
        claim (that requires the work_evidence/goal_graph tiers, not built
        yet).
        """

        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)
        for kind in ASSERTION_TIER_KINDS:
            result = derive_objective_posture(structural, [_assertion(f"{kind.value}-1", kind)])
            assert result.posture != "completed"

    def test_confidence_falls_back_when_assertion_confidence_missing(self) -> None:
        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)
        result = derive_objective_posture(structural, [_assertion("blocker-1", AssertionKind.BLOCKER, confidence=None)])
        assert result.confidence == pytest.approx(0.85)

    def test_as_of_resolves_from_latest_winning_assertion(self) -> None:
        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)
        result = derive_objective_posture(
            structural,
            [_assertion("blocker-1", AssertionKind.BLOCKER, updated_at_ms=1_700_000_000_000)],
        )
        assert result.as_of is not None
        assert result.as_of.startswith("2023-11")


class _FakeOperations:
    def __init__(self, assertions: list[ArchiveAssertionEnvelope]) -> None:
        self._assertions = assertions
        self.calls: list[dict[str, object]] = []

    async def list_assertion_claims(
        self,
        *,
        kinds: Sequence[str | AssertionKind] | None = None,
        target_ref: str | None = None,
        statuses: Sequence[str | AssertionStatus] | None = None,
    ) -> list[ArchiveAssertionEnvelope]:
        self.calls.append({"kinds": kinds, "target_ref": target_ref, "statuses": statuses})
        return [assertion for assertion in self._assertions if assertion.target_ref == target_ref]


class TestResolveSessionObjectivePosture:
    @pytest.mark.asyncio
    async def test_queries_the_session_target_ref_with_assertion_tier_kinds(self) -> None:
        assertions = [_assertion("blocker-1", AssertionKind.BLOCKER, target_ref="session:target-session")]
        operations = _FakeOperations(assertions)
        structural = structural_objective_posture(terminal_state="unknown", terminal_state_confidence=0.2)

        result = await resolve_session_objective_posture(
            operations,
            session_id="target-session",
            structural=structural,
        )

        assert operations.calls == [
            {
                "kinds": ASSERTION_TIER_KINDS,
                "target_ref": "session:target-session",
                "statuses": (AssertionStatus.ACTIVE,),
            }
        ]
        assert result.posture == "blocked"
        assert result.authority == "assertion"

    @pytest.mark.asyncio
    async def test_no_matching_assertions_falls_back_to_structural(self) -> None:
        operations = _FakeOperations([_assertion("blocker-1", AssertionKind.BLOCKER, target_ref="session:other")])
        structural = structural_objective_posture(terminal_state="error_left", terminal_state_confidence=0.78)

        result = await resolve_session_objective_posture(
            operations,
            session_id="target-session",
            structural=structural,
        )

        assert result == structural


def test_objective_posture_payload_defaults_are_unknown_none() -> None:
    payload = ObjectivePosturePayload()
    assert payload.posture == "unknown"
    assert payload.authority == "none"
    assert payload.confidence == 0.0
    assert payload.evidence_refs == ()
    assert payload.contradictions == ()
