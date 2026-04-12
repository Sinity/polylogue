"""Targeted contracts for derived-status collection."""

from __future__ import annotations

import sqlite3

from polylogue.storage.embedding_stats_models import EmbeddingStatsSnapshot


def test_collect_derived_statuses_skips_retrieval_band_recomputation(
    monkeypatch,
) -> None:
    from polylogue.storage import derived_status as derived_status_mod

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        conn.execute("CREATE TABLE messages (message_id TEXT)")
        conn.commit()

        calls: list[bool] = []

        monkeypatch.setattr(derived_status_mod, "fts_index_status_sync", lambda _conn: {"exists": True, "count": 0})
        monkeypatch.setattr(
            derived_status_mod,
            "action_event_read_model_status_sync",
            lambda _conn: {
                "count": 0,
                "materialized_conversation_count": 0,
                "valid_source_conversation_count": 0,
                "orphan_tool_block_count": 0,
                "action_fts_count": 0,
                "rows_ready": True,
                "action_fts_ready": True,
                "stale_count": 0,
                "matches_version": True,
            },
        )
        monkeypatch.setattr(
            derived_status_mod,
            "session_product_status_sync",
            lambda _conn: {
                "profile_row_count": 0,
                "profile_merged_fts_count": 0,
                "profile_merged_fts_duplicate_count": 0,
                "profile_evidence_fts_count": 0,
                "profile_evidence_fts_duplicate_count": 0,
                "profile_inference_fts_count": 0,
                "profile_inference_fts_duplicate_count": 0,
                "profile_enrichment_fts_count": 0,
                "profile_enrichment_fts_duplicate_count": 0,
                "work_event_inference_count": 0,
                "work_event_inference_fts_count": 0,
                "work_event_inference_fts_duplicate_count": 0,
                "phase_inference_count": 0,
                "thread_count": 0,
                "thread_fts_count": 0,
                "thread_fts_duplicate_count": 0,
                "root_threads": 0,
                "tag_rollup_count": 0,
                "expected_tag_rollup_count": 0,
                "day_summary_count": 0,
                "expected_day_summary_count": 0,
                "missing_profile_row_count": 0,
                "stale_profile_row_count": 0,
                "orphan_profile_row_count": 0,
                "expected_work_event_inference_count": 0,
                "stale_work_event_inference_count": 0,
                "orphan_work_event_inference_count": 0,
                "expected_phase_inference_count": 0,
                "stale_phase_inference_count": 0,
                "orphan_phase_inference_count": 0,
                "stale_thread_count": 0,
                "orphan_thread_count": 0,
                "stale_tag_rollup_count": 0,
                "stale_day_summary_count": 0,
                "profile_rows_ready": True,
                "profile_merged_fts_ready": True,
                "profile_evidence_fts_ready": True,
                "profile_inference_fts_ready": True,
                "profile_enrichment_fts_ready": True,
                "work_event_inference_rows_ready": True,
                "work_event_inference_fts_ready": True,
                "phase_inference_rows_ready": True,
                "threads_ready": True,
                "threads_fts_ready": True,
                "tag_rollups_ready": True,
                "day_summaries_ready": True,
                "week_summaries_ready": True,
            },
        )

        def fake_embedding_stats(_conn, *, include_retrieval_bands: bool = True):
            calls.append(include_retrieval_bands)
            return EmbeddingStatsSnapshot()

        monkeypatch.setattr(derived_status_mod, "read_embedding_stats_sync", fake_embedding_stats)
        monkeypatch.setattr(derived_status_mod, "build_archive_product_statuses", lambda _metrics: {})
        monkeypatch.setattr(derived_status_mod, "build_retrieval_statuses", lambda _metrics: {})

        assert derived_status_mod.collect_derived_model_statuses_sync(conn) == {}
    finally:
        conn.close()

    assert calls == [False]
