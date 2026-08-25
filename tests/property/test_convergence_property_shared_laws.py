"""The shared oracle and laws exercise the current production route."""

from __future__ import annotations

from pathlib import Path

from tests.infra.convergence_harness import (
    build_converged_archive,
    converge_convergence_archive,
    initialize_active_archive,
)
from tests.infra.convergence_laws import (
    ConvergenceCase,
    assert_batching_law,
    assert_idempotence_law,
    assert_locality_law,
    assert_permutation_law,
    changed_partitions,
    expected_projection,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)


def test_shared_oracle_matches_clean_production_route(tmp_path: Path) -> None:
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "clean", workload.pathology)

    observed = read_semantic_projection(archive.root, probe_terms=workload.probe_terms)
    expected = expected_projection(workload)
    assert observed.without_locality() == expected.without_locality()


def test_permutation_and_incremental_batching_match_the_same_oracle(tmp_path: Path) -> None:
    workload = generated_convergence_workload()
    pathology = workload.pathology
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    permuted = build_converged_archive(
        tmp_path / "permuted",
        pathology,
        session_order=tuple(reversed(range(len(pathology.sessions)))),
    )
    incremental = build_converged_archive(tmp_path / "incremental", pathology, incremental=True)

    projections = tuple(
        read_semantic_projection(archive.root, probe_terms=workload.probe_terms)
        for archive in (canonical, permuted, incremental)
    )
    assert_permutation_law(projections[:2])
    assert_batching_law(projections[0], (projections[2],))
    assert projections[0].without_locality() == expected_projection(workload).without_locality()


def test_unchanged_second_pass_and_empty_output_are_idempotent(tmp_path: Path) -> None:
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "repeat", workload.pathology)
    first = read_semantic_projection(archive.root, probe_terms=workload.probe_terms)

    converge_convergence_archive(archive)
    second = read_semantic_projection(archive.root, probe_terms=workload.probe_terms)
    assert_idempotence_law(first, second)

    empty_root = tmp_path / "empty"
    initialize_active_archive(empty_root)
    empty = read_semantic_projection(empty_root, probe_terms=workload.probe_terms)
    assert empty.without_locality() == semantic_oracle((), probe_terms=workload.probe_terms).without_locality()


def test_replacement_deletion_and_poison_sibling_locality_are_typed(tmp_path: Path) -> None:
    workload = generated_convergence_workload()
    before = expected_projection(workload)
    remaining = workload.authoritative_sessions[1:]
    after = semantic_oracle(remaining, probe_terms=workload.probe_terms)
    declared = changed_partitions(before, after)

    assert_locality_law(before, after, declared)
    assert "orphaned" in {term for term, _members in before.fts_membership}
    assert workload.pathology.components
    assert ConvergenceCase.REPLACEMENT.value in workload.cases
    assert ConvergenceCase.DELETION.value in workload.cases

    # The current production route still retains the old canonical comparator;
    # this check ensures the new law reads the same poison-sibling workload
    # without asking that comparator for expected values.
    archive = build_converged_archive(tmp_path / "poison-sibling", workload.pathology)
    observed = read_semantic_projection(archive.root, probe_terms=("orphaned",))
    assert observed.fts_membership[0][1]


def test_fault_matrix_is_explicit_and_has_no_hidden_route_state() -> None:
    workload = generated_convergence_workload()
    required = {
        "missing",
        "stale",
        "excess",
        "duplicate",
        "bounded-yield",
        "crash-before-publication",
        "crash-after-publication",
        "restart",
        "generation-mismatch",
        "unchanged-second-pass",
    }
    assert required.issubset(set(workload.cases))
    assert "campaign" not in workload.workload_id
