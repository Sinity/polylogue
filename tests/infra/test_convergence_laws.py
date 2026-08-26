"""Integrity checks for the authoritative convergence oracle."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.core.errors import DatabaseError
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, rebuild_fts_index_sync
from polylogue.storage.search import runtime as search_runtime
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.convergence_harness import (
    build_converged_archive,
    ingest_convergence_pathology,
    initialize_active_archive,
)
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    build_convergence_run_plan,
    convergence_declaration,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)


def test_convergence_declaration_compiles_one_bounded_production_plan() -> None:
    declaration = convergence_declaration()
    plan = build_convergence_run_plan()

    assert declaration.owner == "tests.infra.convergence_laws"
    assert declaration.laws == tuple(ConvergenceLaw)
    assert declaration.candidate_applicability.startswith("full-rewrite")
    assert plan.declaration_id == declaration.declaration_id
    assert plan.route_identity == "production-ingest-and-daemon-convergence"
    assert plan.workload_digest.startswith("sha256:")
    assert plan.expected.fts_membership
    assert plan.expected.role_counts
    assert set(plan.mutants) == {
        "order-sensitive-overwrite",
        "omitted-fts-batch-member",
        "unconditional-rewrite",
        "stale-excess-retention",
        "over-broad-invalidation",
    }


def test_generated_workload_has_nonempty_authoritative_fts_probes() -> None:
    workload = generated_convergence_workload()
    projection = semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms)

    assert workload.authoritative_sessions
    assert dict(projection.fts_membership)["toolonly"] == ()
    assert all(members for term, members in projection.fts_membership if term != "toolonly")
    assert projection.role_counts


def test_oracle_contract_handles_multiblock_messages_through_production_reader(tmp_path: Path) -> None:
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "archive", workload.pathology)

    with sqlite3.connect(archive.root / "index.db") as conn:
        block_counts = dict(conn.execute("SELECT block_type, COUNT(*) FROM blocks GROUP BY block_type"))
    assert block_counts["text"] > 0
    assert block_counts["tool_use"] > 0
    assert block_counts["tool_result"] > 0
    assert_projection_matches_oracle(
        read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
        semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms),
    )


def test_projection_rereads_rebuilt_fts_at_the_same_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every projection read bypasses prior-example search cache entries."""
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "archive", workload.pathology)
    readiness_checks = 0

    def count_readiness(readiness: Mapping[str, object]) -> None:
        nonlocal readiness_checks
        readiness_checks += 1
        check_fts_readiness(readiness)

    monkeypatch.setattr(search_runtime, "check_fts_readiness", count_readiness)
    search_runtime.search_messages_cached.cache_clear()

    first = read_semantic_projection(archive.root, probe_terms=workload.probe_terms)
    assert dict(first.fts_membership)["orphaned"]

    with sqlite3.connect(archive.root / "index.db") as conn:
        changed = conn.execute("DELETE FROM blocks WHERE search_text LIKE '%orphaned%'").rowcount
        assert changed > 0
    unconverged_root = tmp_path / "unconverged"
    initialize_active_archive(unconverged_root)
    ingest_convergence_pathology(
        unconverged_root,
        workload.pathology,
        session_indexes=tuple(range(len(workload.pathology.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(DatabaseError, match="Search index is incomplete"):
        read_semantic_projection(unconverged_root, probe_terms=workload.probe_terms)

    with sqlite3.connect(archive.root / "index.db") as conn:
        rebuild_fts_index_sync(conn)
        conn.commit()

    second = read_semantic_projection(archive.root, probe_terms=workload.probe_terms)
    assert dict(second.fts_membership)["orphaned"] == ()
    assert readiness_checks == 2 * len(workload.probe_terms) + 1
    cache_info = search_runtime.search_messages_cached.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == len(workload.probe_terms)


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_role_aggregate_reader_has_constant_mutation_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "archive", workload.pathology)
    if mutated:
        monkeypatch.setattr(ArchiveStore, "query_unit_counts", lambda *_args, **_kwargs: [])
        with pytest.raises(AssertionError, match="production projection differs"):
            assert_projection_matches_oracle(
                read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
                semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms),
            )
    else:
        assert_projection_matches_oracle(
            read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
            semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms),
        )
