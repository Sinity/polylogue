"""Mutation red twins for the production dependencies of convergence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import tests.infra.convergence_harness as convergence_harness
from polylogue.storage.fts import fts_lifecycle
from polylogue.storage.insights.session import rebuild as insight_rebuild
from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    converge_convergence_archive,
    ingest_convergence_pathology,
    initialize_active_archive,
    rich_convergence_pathology,
)


def test_convergence_property_fts_repair_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bypassed production FTS repair cannot report a converged archive."""
    pathology = rich_convergence_pathology()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(fts_lifecycle, "repair_message_fts_index_sync", lambda *_args, **_kwargs: None)

    archive = ingest_convergence_pathology(
        tmp_path / "mutated",
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="production convergence left pending work"):
        converge_convergence_archive(archive)


def test_convergence_property_insight_repair_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bypassed production insight rebuild cannot report a converged archive."""
    pathology = rich_convergence_pathology()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(insight_rebuild, "rebuild_session_insights_sync", lambda *_args, **_kwargs: None)

    archive = ingest_convergence_pathology(
        tmp_path / "mutated",
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="production convergence left pending work"):
        converge_convergence_archive(archive)


def test_convergence_property_raw_replay_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparator catches a replay path that bypasses durable raw acquisition."""
    pathology = rich_convergence_pathology()
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    mutated_root = tmp_path / "mutated"
    initialize_active_archive(mutated_root)

    monkeypatch.setattr(convergence_harness, "write_source_raw_session", lambda *_args, **_kwargs: "bypassed-raw")
    mutated = ingest_convergence_pathology(
        mutated_root,
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    converge_convergence_archive(mutated)

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical, mutated)


def test_convergence_property_materialized_content_mutation_red_twin(tmp_path: Path) -> None:
    """Changing a materialized insight row cannot pass the semantic comparator."""
    pathology = rich_convergence_pathology()
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    mutated = build_converged_archive(tmp_path / "mutated", pathology)

    with sqlite3.connect(mutated.root / "index.db") as conn:
        cursor = conn.execute(
            """
            UPDATE session_work_events
            SET summary = summary || ' [materialized-content-mutation]'
            WHERE event_id = (SELECT event_id FROM session_work_events ORDER BY event_id LIMIT 1)
            """
        )
        if cursor.rowcount != 1:
            raise AssertionError("materialized-content mutation did not change one work-event row")
        conn.commit()

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical, mutated)
