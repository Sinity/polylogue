"""Integrity checks for the authoritative convergence oracle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.convergence_harness import build_converged_archive
from tests.infra.convergence_laws import (
    assert_projection_matches_oracle,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)


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
