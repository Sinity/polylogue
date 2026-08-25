"""Integrity checks for the shared convergence experiment input."""

from __future__ import annotations

import pytest

from tests.infra.convergence_laws import (
    ConvergenceCase,
    ConvergenceLaw,
    ConvergenceMutant,
    SemanticProjection,
    assert_batching_law,
    assert_idempotence_law,
    assert_locality_law,
    assert_permutation_law,
    build_experiment_contract,
    changed_partitions,
    expected_projection,
    generated_convergence_workload,
    semantic_oracle,
)


def test_generated_workload_is_one_six_tier_input() -> None:
    workload = generated_convergence_workload()

    assert generated_convergence_workload() is workload
    assert workload.tiers == ("source", "index", "embeddings", "user", "ops", "audit")
    assert workload.pathology.sessions
    assert workload.digest == generated_convergence_workload().digest
    assert workload.workload_spec.to_payload()["semantic_result"] == "complete"


def test_oracle_is_authoritative_and_has_both_typed_outputs() -> None:
    workload = generated_convergence_workload()
    projection = expected_projection(workload)

    assert tuple(term for term, _members in projection.fts_membership) == workload.probe_terms
    assert any(members for _term, members in projection.fts_membership)
    assert projection.role_counts
    assert projection.affected_partitions == ()

    # The oracle is a pure function of supplied authoritative sessions. A
    # route cannot change it by returning a different derived projection.
    assert semantic_oracle(workload.authoritative_sessions) == projection


def test_contract_records_only_the_closed_experiment_boundary() -> None:
    workload = generated_convergence_workload()
    contract = build_experiment_contract(workload)
    payload = contract.to_payload()

    assert payload["workload_digest"] == workload.digest
    assert payload["law_set"] == [law.value for law in ConvergenceLaw]
    assert payload["mutant_set"] == [mutant.value for mutant in ConvergenceMutant]
    assert payload["route_identities"] == ["04r9f.variant-a", "04r9f.variant-b"]
    assert "scheduler" not in str(payload).lower()
    assert "ledger" not in str(payload).lower()
    assert "artifact" not in str(payload).lower()


def test_all_declared_cases_are_part_of_the_same_law_input() -> None:
    workload = generated_convergence_workload()
    assert workload.cases == tuple(case.value for case in ConvergenceCase)
    assert set(workload.cases) == {case.value for case in ConvergenceCase}


def test_laws_compare_typed_semantics_and_locality() -> None:
    baseline = SemanticProjection(
        fts_membership=(("probe", ("block-a",)),),
        role_counts=(("assistant", 1), ("user", 1)),
    )
    same = SemanticProjection(
        fts_membership=(("probe", ("block-a",)),),
        role_counts=(("assistant", 1), ("user", 1)),
    )
    changed = SemanticProjection(
        fts_membership=(("probe", ("block-a", "block-b")),),
        role_counts=(("assistant", 1), ("user", 2)),
    )

    assert_permutation_law((baseline, same))
    assert_batching_law(baseline, (same,))
    assert_idempotence_law(baseline, same)
    assert changed_partitions(baseline, changed) == ("user",)
    assert_locality_law(baseline, changed, ("user",))

    with pytest.raises(AssertionError, match="permutation law"):
        assert_permutation_law((baseline, changed))
    with pytest.raises(AssertionError, match="batching law"):
        assert_batching_law(baseline, (changed,))
    with pytest.raises(AssertionError, match="idempotence law"):
        assert_idempotence_law(baseline, changed)
    with pytest.raises(AssertionError, match="locality law"):
        assert_locality_law(baseline, changed, ())
