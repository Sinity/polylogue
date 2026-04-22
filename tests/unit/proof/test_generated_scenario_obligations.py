from __future__ import annotations

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import ProofObligation
from polylogue.proof.runners import run_generated_scenario_evidence


def _obligation(claim_id: str, *, subject_id: str | None = None) -> ProofObligation:
    catalog = build_verification_catalog()
    for obligation in catalog.obligations:
        if obligation.claim.id != claim_id:
            continue
        if subject_id is not None and obligation.subject.id != subject_id:
            continue
        return obligation
    raise AssertionError(f"missing obligation for claim={claim_id!r} subject={subject_id!r}")


def test_generated_scenario_subjects_cover_issue_196_families() -> None:
    catalog = build_verification_catalog()
    subjects = tuple(subject for subject in catalog.subjects if subject.kind == "generated.scenario_family")
    subject_ids = {subject.id for subject in subjects}

    assert len(subjects) >= 5
    assert "generated.scenario_family.pathological-raw-rerun" in subject_ids
    assert "generated.scenario_family.destructive-preview-safety" in subject_ids
    assert "generated.scenario_family.archive-substrate-laws" in subject_ids
    assert "generated.scenario_family.cold-doctor-check" in subject_ids
    assert "generated.scenario_family.large-generated-search" in subject_ids
    assert catalog.obligations_by_claim()["generated.scenario.family_registered"] == len(subjects)
    assert catalog.obligations_by_claim()["generated.scenario.semantic_claim_mapping"] == len(subjects)


def test_implemented_generated_scenarios_are_local_and_deterministic() -> None:
    catalog = build_verification_catalog()
    implemented = tuple(
        subject
        for subject in catalog.subjects
        if subject.kind == "generated.scenario_family" and subject.attrs.get("status") == "implemented"
    )

    assert implemented
    assert catalog.obligations_by_claim()["generated.scenario.local_deterministic"] == len(implemented)
    assert all(subject.attrs.get("local_deterministic") is True for subject in implemented)
    assert all(subject.attrs.get("live_archive_dependency") is False for subject in implemented)


def test_generated_scenarios_map_semantic_claim_families() -> None:
    catalog = build_verification_catalog()
    subjects = tuple(subject for subject in catalog.subjects if subject.kind == "generated.scenario_family")
    families: set[str] = set()
    implemented_or_mapped: set[str] = set()

    for subject in subjects:
        semantic_claims = subject.attrs.get("semantic_claims")
        assert isinstance(semantic_claims, list)
        assert semantic_claims
        for claim in semantic_claims:
            assert isinstance(claim, dict)
            family = claim.get("family")
            state = claim.get("state")
            assert isinstance(family, str)
            assert state in {"implemented", "mapped"}
            families.add(family)
            implemented_or_mapped.add(f"{state}:{family}")
            if state == "mapped":
                assert str(claim.get("issue", "")).startswith("#")

    assert len(families) >= 2
    assert any(item.startswith("implemented:") for item in implemented_or_mapped)
    assert any(item.startswith("mapped:") for item in implemented_or_mapped)


def test_generated_scenario_static_runner_accepts_registered_family() -> None:
    obligation = _obligation(
        "generated.scenario.family_registered",
        subject_id="generated.scenario_family.pathological-raw-rerun",
    )

    envelope = run_generated_scenario_evidence(obligation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["status"] == "implemented"
    assert envelope.evidence["generated_world"] == "repo-local malformed raw payload fixtures"
    assert envelope.counterexample is None


def test_generated_scenario_static_runner_accepts_local_deterministic_family() -> None:
    obligation = _obligation(
        "generated.scenario.local_deterministic",
        subject_id="generated.scenario_family.destructive-preview-safety",
    )

    envelope = run_generated_scenario_evidence(obligation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["local_deterministic"] is True
    assert envelope.evidence["live_archive_dependency"] is False
    assert envelope.counterexample is None


def test_generated_scenario_static_runner_accepts_semantic_claim_mapping() -> None:
    obligation = _obligation(
        "generated.scenario.semantic_claim_mapping",
        subject_id="generated.scenario_family.archive-substrate-laws",
    )

    envelope = run_generated_scenario_evidence(obligation)

    assert envelope.status is OutcomeStatus.OK
    implemented_claim_families = envelope.evidence["implemented_claim_families"]
    assert isinstance(implemented_claim_families, list)
    assert any(item == "content hash stability" for item in implemented_claim_families)
    assert envelope.counterexample is None
