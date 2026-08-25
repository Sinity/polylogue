"""Integrity checks for the authoritative convergence oracle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.convergence_harness import build_converged_archive
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    expected_projection,
    generated_convergence_workload,
    read_semantic_projection,
)


def test_generated_workload_has_nonempty_authoritative_fts_probes() -> None:
    workload = generated_convergence_workload()
    projection = expected_projection(workload)

    assert workload.laws == tuple(ConvergenceLaw)
    assert workload.authoritative_sessions
    assert not [term for term, members in projection.fts_membership if not members]
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
        expected_projection(workload),
        law=ConvergenceLaw.PERMUTATION,
    )


def test_role_aggregate_reader_constant_mutation_fails_for_nonempty_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "archive", workload.pathology)
    monkeypatch.setattr(ArchiveStore, "query_unit_counts", lambda *_args, **_kwargs: [])

    with pytest.raises(AssertionError, match="production projection differs"):
        assert_projection_matches_oracle(
            read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
            expected_projection(workload),
            law=ConvergenceLaw.IDEMPOTENCE,
        )
