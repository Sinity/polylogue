"""Targeted contracts for derived-status collection."""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot


def test_collect_derived_statuses_skips_retrieval_band_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.derived import derived_status as derived_status_mod

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        conn.execute("CREATE TABLE messages (message_id TEXT)")
        conn.commit()

        calls: list[bool] = []

        verify_calls: list[tuple[str, bool]] = []

        def fake_message_fts(_conn: sqlite3.Connection, *, verify_total_rows: bool = True) -> dict[str, object]:
            verify_calls.append(("fts", verify_total_rows))
            return {"exists": True, "indexed_rows": 0, "total_rows": 0, "ready": True}

        def fake_action_status(
            _conn: sqlite3.Connection,
            *,
            verify_source_alignment: bool = True,
        ) -> dict[str, object]:
            verify_calls.append(("action", verify_source_alignment))
            return {
                "count": 0,
                "materialized_conversation_count": 0,
                "valid_source_conversation_count": 0,
                "orphan_tool_block_count": 0,
                "action_fts_count": 0,
                "rows_ready": True,
                "action_fts_ready": True,
                "stale_count": 0,
                "matches_version": True,
            }

        def fake_session_status(
            _conn: sqlite3.Connection,
            *,
            verify_freshness: bool = True,
        ) -> SessionInsightStatusSnapshot:
            verify_calls.append(("session", verify_freshness))
            return SessionInsightStatusSnapshot(
                profile_rows_ready=True,
                profile_merged_fts_ready=True,
                profile_evidence_fts_ready=True,
                profile_inference_fts_ready=True,
                profile_enrichment_fts_ready=True,
                work_event_inference_rows_ready=True,
                work_event_inference_fts_ready=True,
                phase_inference_rows_ready=True,
                threads_ready=True,
                threads_fts_ready=True,
                tag_rollups_ready=True,
                day_summaries_ready=True,
                week_summaries_ready=True,
            )

        monkeypatch.setattr(
            derived_status_mod,
            "message_fts_readiness_sync",
            fake_message_fts,
        )
        monkeypatch.setattr(
            derived_status_mod,
            "action_event_read_model_status_sync",
            fake_action_status,
        )
        monkeypatch.setattr(
            derived_status_mod,
            "session_insight_status_sync",
            fake_session_status,
        )

        def fake_embedding_stats(
            _conn: sqlite3.Connection,
            *,
            include_retrieval_bands: bool = True,
        ) -> EmbeddingStatsSnapshot:
            calls.append(include_retrieval_bands)
            return EmbeddingStatsSnapshot()

        monkeypatch.setattr(derived_status_mod, "read_embedding_stats_sync", fake_embedding_stats)
        monkeypatch.setattr(derived_status_mod, "build_archive_insight_statuses", lambda _metrics: {})
        monkeypatch.setattr(derived_status_mod, "build_retrieval_statuses", lambda _metrics: {})

        assert derived_status_mod.collect_derived_model_statuses_sync(conn, verify_full=False) == {}
    finally:
        conn.close()

    assert calls == [False]
    assert verify_calls == [("fts", False), ("action", False), ("session", False)]


def test_build_retrieval_statuses_counts_stale_action_event_fts_rows() -> None:
    from polylogue.storage.derived.derived_status import build_retrieval_statuses

    statuses = build_retrieval_statuses(
        {
            "transcript_embeddings_ready": True,
            "embedded_conversations": 0,
            "embedded_messages": 0,
            "pending_conversations": 0,
            "stale_messages": 0,
            "missing_provenance": 0,
            "total_conversations": 0,
            "evidence_retrieval_ready": False,
            "evidence_retrieval_rows": 13,
            "expected_evidence_retrieval_rows": 10,
            "profile_evidence_fts_rows": 0,
            "profile_evidence_fts_duplicates": 0,
            "profile_rows": 0,
            "action_fts_rows": 13,
            "action_rows": 10,
            "action_source_documents": 4,
            "action_documents": 4,
            "action_stale_rows": 0,
            "action_fts_stale_rows": 3,
            "action_orphan_rows": 0,
            "orphan_profile_rows": 0,
            "inference_retrieval_ready": True,
            "inference_retrieval_rows": 0,
            "expected_inference_retrieval_rows": 0,
            "profile_inference_fts_rows": 0,
            "profile_inference_fts_duplicates": 0,
            "work_event_fts_rows": 0,
            "work_event_fts_duplicates": 0,
            "work_event_rows": 0,
            "phase_rows": 0,
            "stale_work_event_rows": 0,
            "stale_phase_rows": 0,
            "orphan_work_event_rows": 0,
            "orphan_phase_rows": 0,
            "enrichment_retrieval_ready": True,
            "enrichment_retrieval_rows": 0,
            "expected_enrichment_retrieval_rows": 0,
            "profile_enrichment_fts_rows": 0,
            "profile_enrichment_fts_duplicates": 0,
        }
    )

    evidence = statuses["retrieval_evidence"]
    assert evidence.ready is False
    assert evidence.pending_rows == 0
    assert evidence.stale_rows == 3
