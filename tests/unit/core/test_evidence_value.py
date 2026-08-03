"""Survivor laws for the storage-free EvidenceValue protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from polylogue.core.evidence_value import (
    CalibratedConfidence,
    CoverageExclusion,
    EvidenceInvariantError,
    EvidenceValue,
    FactFamilySpec,
    FrameCoverage,
    FreshnessProvenance,
    TemporalProvenance,
    audit_fact_family_completeness,
    refine_evidence_value,
    sum_evidence_values,
)
from polylogue.core.refs import ObjectRef
from polylogue.declarations import DeclarationRegistry, normalized_derivation_bytes

_NOW = "2026-07-17T08:00:00+00:00"
_DEFINITION = ObjectRef(kind="insight", object_id="test-quantitative-family:v1")
_SPEC = FactFamilySpec(
    family="test.quantitative",
    owner="tests exercise polylogue.core.evidence_value",
    source_adapter="tests.unit.core.test_evidence_value._value",
    public_field="value",
    renderer_label="test value",
    value_schema="integer",
    unit="item",
    grain="source",
    denominator="declared source frame",
    definition_ref=_DEFINITION,
    required_axes=frozenset(
        {
            "value_state",
            "measurement_authority",
            "evidence_refs",
            "definition_ref",
            "temporal",
            "enumeration",
            "coverage",
            "freshness",
        }
    ),
    allowed_states=frozenset({"known", "unknown", "unavailable", "skipped", "not_applicable", "redacted"}),
    allowed_authorities=frozenset(
        {
            "structural",
            "provider-reported",
            "catalog-derived",
            "rule-derived",
            "model-derived",
            "agent-declared",
            "judged",
        }
    ),
    authority_precedence=(
        "structural",
        "provider-reported",
        "judged",
        "catalog-derived",
        "rule-derived",
        "model-derived",
        "agent-declared",
    ),
)


def _value(
    name: str,
    value: int | None,
    *,
    state: str = "known",
    authority: str = "structural",
    evidence_suffix: str = "observed",
    time_source: str = "provider_ts",
    freshness: str = "fresh",
) -> EvidenceValue[int]:
    fact_ref = ObjectRef(kind="insight", object_id=f"test-fact:{name}")
    evidence_ref = ObjectRef(kind="run", object_id=f"test-evidence:{name}:{evidence_suffix}")
    coverage = FrameCoverage(
        intended_frame="test frame",
        grain="source",
        denominator="declared source frame",
        intended_count=1,
        observed_count=1,
        supported_count=1 if state == "known" else 0,
        complete=state == "known",
        intended_refs=(fact_ref,),
        observed_refs=(fact_ref,),
    )
    freshness_refs = (evidence_ref,) if freshness != "fresh" else ()
    return EvidenceValue(
        family=_SPEC.family,
        fact_ref=fact_ref,
        value_state=state,  # type: ignore[arg-type]
        value=value,
        measurement_authority=(authority,),  # type: ignore[arg-type]
        weakest_measurement_authority=authority,  # type: ignore[arg-type]
        evidence_refs=(evidence_ref,),
        definition_ref=_DEFINITION,
        temporal=TemporalProvenance.from_source(
            observed_at=_NOW,
            time_source=time_source,  # type: ignore[arg-type]
        ),
        enumeration="census",
        coverage=coverage,
        freshness=FreshnessProvenance(
            state=freshness,  # type: ignore[arg-type]
            evaluated_at=_NOW,
            cause=None if freshness == "fresh" else freshness,
            last_good_at=_NOW if freshness != "fresh" else None,
            last_good_evidence_refs=freshness_refs,
        ),
    )


def _aggregate(values: Sequence[EvidenceValue[int]], expected: Sequence[ObjectRef]) -> EvidenceValue[int]:
    return sum_evidence_values(
        values,
        spec=_SPEC,
        fact_ref=ObjectRef(kind="insight", object_id="test-total"),
        observed_at=_NOW,
        intended_frame="test total frame",
        expected_fact_refs=expected,
    )


def test_known_zero_and_unknown_are_distinct_without_sentinel_coercion() -> None:
    zero = _value("zero", 0)
    unknown = _value("unknown", None, state="unknown")

    assert zero.value_state == "known"
    assert zero.value == 0
    assert unknown.value_state == "unknown"
    assert unknown.value is None
    assert zero.to_dict()["value"] == 0
    assert unknown.to_dict()["value"] is None


def test_every_absence_state_round_trips_independently() -> None:
    for state in ("unknown", "unavailable", "skipped", "not_applicable", "redacted"):
        value = _value(state, None, state=state)
        assert value.to_dict()["value_state"] == state
        assert value.to_dict()["value"] is None


def test_calibrated_confidence_requires_calibration_and_definition_refs() -> None:
    confidence = CalibratedConfidence(
        value=0.8,
        calibration_ref=ObjectRef(kind="insight", object_id="calibration:v2"),
        definition_ref=_DEFINITION,
    )
    assert confidence.to_dict()["calibration_ref"] == "insight:calibration:v2"
    with pytest.raises(EvidenceInvariantError, match="within \\[0, 1\\]"):
        CalibratedConfidence(
            value=1.1,
            calibration_ref=ObjectRef(kind="insight", object_id="calibration:v2"),
            definition_ref=_DEFINITION,
        )


def test_temporal_confidence_cannot_be_strengthened_without_source_provenance() -> None:
    with pytest.raises(EvidenceInvariantError, match="overstates source"):
        TemporalProvenance(
            observed_at=_NOW,
            time_source="materialization_ts",
            time_confidence="recorded",
        )


def test_refinement_adds_known_evidence_without_discarding_unknown_support() -> None:
    unknown = _value("same", None, state="unknown", evidence_suffix="missing")
    known = _value("same", 7, evidence_suffix="provider")

    refined = refine_evidence_value(unknown, known, spec=_SPEC)

    assert refined.value_state == "known"
    assert refined.value == 7
    assert [ref.format() for ref in refined.evidence_refs] == [
        "run:test-evidence:same:missing",
        "run:test-evidence:same:provider",
    ]


def test_authority_cannot_strengthen_without_new_evidence() -> None:
    weak = _value("same", 7, authority="model-derived", evidence_suffix="same")
    strong = _value("same", 7, authority="structural", evidence_suffix="same")

    with pytest.raises(EvidenceInvariantError, match="cannot strengthen without new evidence"):
        refine_evidence_value(weak, strong, spec=_SPEC)


def test_contradictory_equal_authority_is_explicit_and_order_independent() -> None:
    left = _value("same", 7, evidence_suffix="left")
    right = _value("same", 9, evidence_suffix="right")

    first = refine_evidence_value(left, right, spec=_SPEC)
    second = refine_evidence_value(right, left, spec=_SPEC)

    assert first.to_dict() == second.to_dict()
    assert first.value_state == "unknown"
    assert first.value is None
    assert len(first.conflicts) == 1
    assert {item.value for item in first.conflicts[0].observations} == {7, 9}


def test_declared_stronger_observation_supersedes_but_preserves_support() -> None:
    weak = _value("same", 7, authority="model-derived", evidence_suffix="model")
    strong = _value("same", 9, authority="structural", evidence_suffix="filesystem")

    resolved = refine_evidence_value(weak, strong, spec=_SPEC)

    assert resolved.value_state == "known"
    assert resolved.value == 9
    # The selected value is deterministic, while both evidence refs remain.
    assert {ref.format() for ref in resolved.evidence_refs} == {
        "run:test-evidence:same:model",
        "run:test-evidence:same:filesystem",
    }
    assert {item.value for item in resolved.contributions} == {7, 9}


def test_sum_conserves_disjoint_totals_deduplicates_identity_and_is_order_independent() -> None:
    a = _value("a", 3)
    b = _value("b", 5)
    expected = (a.fact_ref, b.fact_ref)

    direct = _aggregate((a, b), expected)
    reversed_and_duplicate = _aggregate((b, a, a), expected)

    assert direct.to_dict() == reversed_and_duplicate.to_dict()
    assert direct.value_state == "known"
    assert direct.value == 8
    assert direct.coverage.observed_count == 2
    assert direct.coverage.supported_count == 2
    assert len(direct.contributions) == 2


def test_nested_grouping_conserves_leaf_contributions() -> None:
    a = _value("a", 2)
    b = _value("b", 3)
    c = _value("c", 7)
    expected = (a.fact_ref, b.fact_ref, c.fact_ref)
    ab = sum_evidence_values(
        (a, b),
        spec=_SPEC,
        fact_ref=ObjectRef(kind="insight", object_id="subtotal-ab"),
        observed_at=_NOW,
        intended_frame="test subtotal frame",
        expected_fact_refs=(a.fact_ref, b.fact_ref),
    )

    grouped = _aggregate((ab, c), expected)
    direct = _aggregate((a, b, c), expected)

    assert grouped.value == direct.value == 12
    assert [item.fact_ref.format() for item in grouped.contributions] == [
        "insight:test-fact:a",
        "insight:test-fact:b",
        "insight:test-fact:c",
    ]


def test_dropping_expected_source_reduces_coverage_and_never_becomes_zero() -> None:
    a = _value("a", 4)
    b = _value("b", 6)

    incomplete = _aggregate((a,), (a.fact_ref, b.fact_ref))

    assert incomplete.value_state == "unknown"
    assert incomplete.value is None
    assert incomplete.coverage.complete is False
    assert incomplete.coverage.observed_count == 1
    assert incomplete.coverage.supported_count == 1
    assert incomplete.coverage.exclusions[0].reason == "missing-contribution"


def test_unknown_contribution_does_not_become_numeric_zero() -> None:
    known = _value("known", 4)
    unknown = _value("unknown", None, state="unknown")

    total = _aggregate((known, unknown), (known.fact_ref, unknown.fact_ref))

    assert total.value_state == "unknown"
    assert total.value is None
    assert total.coverage.observed_count == 2
    assert total.coverage.supported_count == 1


def test_conflicting_duplicate_fact_blocks_total_instead_of_double_counting() -> None:
    first = _value("same", 4, evidence_suffix="first")
    second = _value("same", 6, evidence_suffix="second")

    total = _aggregate((first, second), (first.fact_ref,))

    assert total.value_state == "unknown"
    assert total.value is None
    assert len(total.conflicts) == 1
    assert total.coverage.observed_count == 1
    assert total.coverage.supported_count == 0
    assert {item.reason for item in total.coverage.exclusions} == {"contradictory-contribution"}
    assert {ref.format() for ref in total.evidence_refs} == {
        "run:test-evidence:same:first",
        "run:test-evidence:same:second",
    }
    assert total.measurement_authority == ("structural",)


def test_nested_conflict_grouping_preserves_raw_witnesses_and_unknown_total() -> None:
    first = _value("same", 4, evidence_suffix="first")
    second = _value("same", 6, evidence_suffix="second")
    other = _value("other", 3, evidence_suffix="other")
    conflict = refine_evidence_value(first, second, spec=_SPEC)
    expected = (first.fact_ref, other.fact_ref)

    grouped = _aggregate((conflict, other), expected)
    direct = _aggregate((first, second, other), expected)

    assert grouped.to_dict() == direct.to_dict()
    assert grouped.value_state == "unknown"
    assert {item.value for item in grouped.conflicts[0].observations} == {4, 6}
    assert {ref.format() for ref in grouped.evidence_refs} == {
        "run:test-evidence:other:other",
        "run:test-evidence:same:first",
        "run:test-evidence:same:second",
    }


def test_known_value_with_weaker_coverage_stays_numeric_but_cannot_claim_complete_frame() -> None:
    complete = _value("a", 4)
    incomplete = replace(
        complete,
        coverage=replace(
            complete.coverage,
            complete=False,
            exclusions=(CoverageExclusion(complete.fact_ref, "source-excluded"),),
        ),
    )

    total = _aggregate((incomplete,), (incomplete.fact_ref,))

    assert total.value_state == "known"
    assert total.value == 4
    assert total.coverage.complete is False
    assert {item.reason for item in total.coverage.exclusions} == {
        "source-excluded",
        "contribution-coverage-incomplete",
    }


def test_aggregate_keeps_weakest_temporal_authority_and_freshness() -> None:
    strong = _value("strong", 4, authority="structural", time_source="provider_ts")
    weak = _value(
        "weak",
        6,
        authority="model-derived",
        time_source="materialization_ts",
        freshness="stale",
    )

    total = _aggregate((strong, weak), (strong.fact_ref, weak.fact_ref))

    assert total.value == 10
    assert total.weakest_measurement_authority == "model-derived"
    assert total.temporal.time_source == "materialization_ts"
    assert total.temporal.time_confidence == "unknown"
    assert total.freshness.state == "stale"
    assert total.freshness.last_good_evidence_refs


def test_empty_aggregate_is_unknown_not_known_zero() -> None:
    total = _aggregate((), ())

    assert total.value_state == "unknown"
    assert total.value is None


def test_declared_required_axis_mutations_are_source_locatable_failures() -> None:
    value = _value("a", 1)
    mutations = (
        (replace(value, measurement_authority=(), weakest_measurement_authority=None), "measurement_authority"),
        (replace(value, evidence_refs=()), "evidence_refs"),
        (
            replace(
                value,
                definition_ref=ObjectRef(kind="insight", object_id="wrong-definition:v1"),
            ),
            "definition_ref expected",
        ),
        (
            replace(
                value,
                temporal=TemporalProvenance.from_source(
                    observed_at=None,
                    time_source=None,
                ),
            ),
            "temporal observed_at",
        ),
        (
            replace(
                value,
                coverage=replace(value.coverage, grain="wrong-grain"),
            ),
            "coverage grain expected",
        ),
    )

    for mutated, expected_message in mutations:
        diagnostics = _SPEC.validate(mutated)
        assert any(expected_message in item for item in diagnostics)
        with pytest.raises(EvidenceInvariantError, match=expected_message):
            _SPEC.require(mutated)


def test_value_state_mutation_cannot_pair_unknown_with_numeric_sentinel() -> None:
    value = _value("a", 1)

    with pytest.raises(EvidenceInvariantError, match="cannot carry"):
        replace(value, value_state="unknown")


def test_fact_family_completeness_fails_when_declaration_or_definition_is_removed() -> None:
    value = _value("a", 1)
    assert audit_fact_family_completeness((_SPEC,), (value,), required_families=(_SPEC.family,)) == ()

    no_spec = audit_fact_family_completeness((), (value,), required_families=(_SPEC.family,))
    assert {item.message for item in no_spec} == {
        "required fact-family declaration missing",
        "public value has no fact-family declaration",
    }


def test_fact_family_projects_through_shared_declaration_kernel() -> None:
    """Production dependency: FactFamilySpec uses the shared declaration kernel.

    Anti-vacuity mutation: removing the domain projection makes registration
    and deterministic derivation unavailable to the family completeness path.
    """

    declaration = _SPEC.declaration()
    registry = DeclarationRegistry()
    registry.register(declaration)

    assert declaration.declaration_id == "evidence.test.quantitative"
    assert declaration.owner_path == "tests/unit/core/test_evidence_value.py"
    assert normalized_derivation_bytes(registry)
