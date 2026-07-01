"""Targeted contracts for derived-status collection."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest


def test_collect_derived_statuses_skips_retrieval_band_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``collect_derived_model_statuses_sync`` returns exactly the merge of the
    archive-insight and retrieval band builders over the archive `index.db` —
    it does not recompute the retrieval band itself."""
    from polylogue.storage.derived import derived_status as derived_status_mod
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        monkeypatch.setattr(derived_status_mod, "build_archive_insight_statuses", lambda _metrics: {"insight": "ok"})
        monkeypatch.setattr(derived_status_mod, "build_retrieval_statuses", lambda _metrics: {"band": "ok"})

        statuses = cast(
            "dict[str, object]",
            derived_status_mod.collect_derived_model_statuses_sync(conn, verify_full=False),
        )
        assert statuses == {
            "insight": "ok",
            "band": "ok",
        }
    finally:
        conn.close()


def test_collect_derived_statuses_uses_canonical_session_insight_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.derived import derived_status as derived_status_mod
    from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot

    conn = sqlite3.connect(":memory:")
    snapshot = SessionInsightStatusSnapshot(
        total_sessions=2,
        root_threads=0,
        profile_row_count=2,
        work_event_inference_count=1,
        expected_work_event_inference_count=4,
        missing_work_event_materialization_count=3,
        phase_inference_count=2,
        expected_phase_inference_count=5,
        missing_phase_materialization_count=3,
        profile_rows_ready=True,
        work_event_inference_rows_ready=False,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=False,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
    )
    monkeypatch.setattr(
        derived_status_mod,
        "session_insight_status_sync",
        lambda _conn, *, verify_freshness: snapshot,
    )

    try:
        statuses = derived_status_mod.collect_derived_model_statuses_sync(conn, verify_full=True)
    finally:
        conn.close()

    work_events = statuses["session_work_events"]
    phases = statuses["session_phases"]
    assert work_events.ready is False
    assert work_events.source_rows == 4
    assert work_events.materialized_rows == 1
    assert work_events.pending_rows == 3
    assert phases.ready is False
    assert phases.source_rows == 5
    assert phases.materialized_rows == 2
    assert phases.pending_rows == 3


def test_build_timeline_statuses_names_timeline_rows_by_table() -> None:
    from polylogue.storage.derived.insights import build_timeline_statuses

    statuses = build_timeline_statuses(
        {
            "profile_rows": 2,
            "work_event_rows_ready": True,
            "work_event_rows": 3,
            "expected_work_event_rows": 3,
            "stale_work_event_rows": 0,
            "orphan_work_event_rows": 0,
            "work_event_fts_ready": True,
            "work_event_fts_rows": 3,
            "work_event_fts_duplicates": 0,
            "phase_rows_ready": True,
            "phase_rows": 5,
            "expected_phase_rows": 5,
            "stale_phase_rows": 0,
            "orphan_phase_rows": 0,
            "threads_ready": True,
            "thread_rows": 1,
            "total_thread_roots": 1,
            "stale_thread_rows": 0,
            "orphan_thread_rows": 0,
            "thread_fts_ready": True,
            "thread_fts_rows": 1,
            "thread_fts_duplicates": 0,
        }
    )

    assert "session_work_events" in statuses
    assert "session_work_event_inference" not in statuses
    assert statuses["session_work_events"].name == "session_work_events"
    assert statuses["session_work_events"].detail == "Session work events ready (3/3 rows)"
    assert "session_work_events_fts" in statuses
    assert "session_work_event_inference_fts" not in statuses
    assert statuses["session_work_events_fts"].name == "session_work_events_fts"
    assert "session_phases" in statuses
    assert "session_phase_inference" not in statuses
    assert statuses["session_phases"].name == "session_phases"
    assert statuses["session_phases"].detail == "Session phase intervals ready (5/5 rows)"


def test_build_retrieval_statuses_counts_stale_session_insight_rows() -> None:
    from polylogue.storage.derived.derived_status import build_retrieval_statuses

    statuses = build_retrieval_statuses(
        {
            "transcript_embeddings_ready": True,
            "embedded_sessions": 0,
            "embedded_messages": 0,
            "pending_sessions": 0,
            "stale_messages": 0,
            "missing_provenance": 0,
            "total_sessions": 0,
            "evidence_retrieval_ready": False,
            "evidence_retrieval_rows": 13,
            "expected_evidence_retrieval_rows": 10,
            "profile_evidence_fts_rows": 0,
            "profile_evidence_fts_duplicates": 0,
            "profile_rows": 0,
            "orphan_profile_rows": 0,
            "inference_retrieval_ready": True,
            "inference_retrieval_rows": 0,
            "expected_inference_retrieval_rows": 0,
            "profile_inference_fts_rows": 0,
            "profile_inference_fts_duplicates": 0,
            "work_event_fts_rows": 4,
            "work_event_fts_duplicates": 2,
            "work_event_rows": 4,
            "phase_rows": 0,
            "stale_work_event_rows": 1,
            "stale_phase_rows": 2,
            "orphan_work_event_rows": 0,
            "orphan_phase_rows": 0,
            "enrichment_retrieval_ready": True,
            "enrichment_retrieval_rows": 0,
            "expected_enrichment_retrieval_rows": 0,
            "profile_enrichment_fts_rows": 0,
            "profile_enrichment_fts_duplicates": 0,
        }
    )

    inference = statuses["retrieval_inference"]
    assert inference.ready is True
    assert inference.pending_rows == 0
    assert inference.stale_rows == 3
    assert inference.orphan_rows == 0
