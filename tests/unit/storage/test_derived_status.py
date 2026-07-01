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
