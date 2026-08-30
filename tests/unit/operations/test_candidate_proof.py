from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.refs import ObjectRef
from polylogue.maintenance.assertion_transition import SourceIdentityClaims, TransitionBinding
from polylogue.operations.candidate_proof import (
    CandidateCheckPlan,
    CandidateFidelityRequest,
    CandidateProofError,
    ConservationTerm,
    PopulationCoverageRequest,
    TransitionPlanningRequest,
    check_candidate_fidelity,
    plan_candidate_transition,
    reverse_population_coverage,
)


def test_candidate_plan_rejects_omitted_duplicate_and_stale_members() -> None:
    owners = (type("Owner", (), {"name": "source"})(), type("Owner", (), {"name": "schema"})())
    plan = CandidateCheckPlan.compile(owners)
    plan.validate(("source", "schema"))
    with pytest.raises(CandidateProofError):
        plan.validate(("source",))
    with pytest.raises(CandidateProofError):
        CandidateCheckPlan.compile((owners[0], owners[0]))
    with pytest.raises(CandidateProofError):
        CandidateCheckPlan(("source",), plan.digest).validate(("source",))


def test_fidelity_uses_source_denominator_and_rejects_collisions() -> None:
    request = CandidateFidelityRequest(
        source_manifest_digest="seal",
        source_items=("a", "b"),
        candidate_items=("a",),
        terms=(ConservationTerm("sessions", 2, 1),),
    )
    result = check_candidate_fidelity(request)
    assert result.missing == ("b",)
    assert not result.balanced
    with pytest.raises(CandidateProofError):
        check_candidate_fidelity(request.__class__("seal", ("a", "a"), ("a",), request.terms))


def test_reverse_population_coverage_reads_canonical_schema_without_writing(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())
    result = reverse_population_coverage(PopulationCoverageRequest(tmp_path))
    assert result.rows == ()
    assert sorted(tmp_path.iterdir()) == before


def test_transition_planning_is_exact_and_empty_when_preserved() -> None:
    ref = ObjectRef(kind="session", object_id="chatgpt:id").format()
    binding = TransitionBinding("old", "new", "seal", "pkg", (("user", 1),))
    plan = plan_candidate_transition(
        TransitionPlanningRequest(
            user_refs=(ref,),
            candidate_refs=(ref,),
            source_claims=SourceIdentityClaims.from_refs((ref,)),
            binding=binding,
        )
    )
    assert plan.is_empty
