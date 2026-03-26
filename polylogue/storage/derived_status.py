"""Canonical readiness/freshness snapshot for durable derived models."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.action_event_lifecycle import action_event_read_model_status_sync
from polylogue.storage.derived_status_products import build_archive_product_statuses
from polylogue.storage.derived_status_retrieval import build_retrieval_statuses
from polylogue.storage.embedding_stats import read_embedding_stats_sync
from polylogue.storage.fts_lifecycle import fts_index_status_sync
from polylogue.storage.session_product_lifecycle import session_product_status_sync


def collect_derived_model_statuses_sync(
    conn: sqlite3.Connection,
) -> dict[str, DerivedModelStatus]:
    total_conversations = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0)
    total_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)

    fts_status = fts_index_status_sync(conn)
    action_status = action_event_read_model_status_sync(conn)
    session_status = session_product_status_sync(conn)
    embedding_stats = read_embedding_stats_sync(conn)

    metrics: dict[str, int | bool] = {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "message_fts_rows": int(fts_status.get("count", 0)),
        "message_fts_ready": bool(fts_status.get("exists", False)) and int(fts_status.get("count", 0)) == total_messages,
        "action_rows": int(action_status["count"]),
        "action_documents": int(action_status["materialized_conversation_count"]),
        "action_source_documents": int(action_status["valid_source_conversation_count"]),
        "action_orphan_rows": int(action_status["orphan_tool_block_count"]),
        "action_fts_rows": int(action_status["action_fts_count"]),
        "action_rows_ready": bool(action_status["rows_ready"]),
        "action_fts_ready": bool(action_status["action_fts_ready"]),
        "action_stale_rows": int(action_status["stale_count"]),
        "action_matches_version": bool(action_status["matches_version"]),
        "profile_rows": int(session_status["profile_row_count"]),
        "profile_merged_fts_rows": int(session_status["profile_merged_fts_count"]),
        "profile_merged_fts_duplicates": int(session_status["profile_merged_fts_duplicate_count"]),
        "profile_evidence_fts_rows": int(session_status["profile_evidence_fts_count"]),
        "profile_evidence_fts_duplicates": int(session_status["profile_evidence_fts_duplicate_count"]),
        "profile_inference_fts_rows": int(session_status["profile_inference_fts_count"]),
        "profile_inference_fts_duplicates": int(session_status["profile_inference_fts_duplicate_count"]),
        "profile_enrichment_fts_rows": int(session_status["profile_enrichment_fts_count"]),
        "profile_enrichment_fts_duplicates": int(session_status["profile_enrichment_fts_duplicate_count"]),
        "work_event_rows": int(session_status["work_event_inference_count"]),
        "work_event_fts_rows": int(session_status["work_event_inference_fts_count"]),
        "work_event_fts_duplicates": int(session_status["work_event_inference_fts_duplicate_count"]),
        "phase_rows": int(session_status["phase_inference_count"]),
        "work_thread_rows": int(session_status["thread_count"]),
        "work_thread_fts_rows": int(session_status["thread_fts_count"]),
        "work_thread_fts_duplicates": int(session_status["thread_fts_duplicate_count"]),
        "total_thread_roots": int(session_status["root_threads"]),
        "tag_rollup_rows": int(session_status["tag_rollup_count"]),
        "expected_tag_rollup_rows": int(session_status["expected_tag_rollup_count"]),
        "day_summary_rows": int(session_status["day_summary_count"]),
        "expected_day_summary_rows": int(session_status["expected_day_summary_count"]),
        "missing_profile_rows": int(session_status["missing_profile_row_count"]),
        "stale_profile_rows": int(session_status["stale_profile_row_count"]),
        "orphan_profile_rows": int(session_status["orphan_profile_row_count"]),
        "expected_work_event_rows": int(session_status["expected_work_event_inference_count"]),
        "stale_work_event_rows": int(session_status["stale_work_event_inference_count"]),
        "orphan_work_event_rows": int(session_status["orphan_work_event_inference_count"]),
        "expected_phase_rows": int(session_status["expected_phase_inference_count"]),
        "stale_phase_rows": int(session_status["stale_phase_inference_count"]),
        "orphan_phase_rows": int(session_status["orphan_phase_inference_count"]),
        "stale_thread_rows": int(session_status["stale_thread_count"]),
        "orphan_thread_rows": int(session_status["orphan_thread_count"]),
        "stale_tag_rollup_rows": int(session_status["stale_tag_rollup_count"]),
        "stale_day_summary_rows": int(session_status["stale_day_summary_count"]),
        "profile_rows_ready": bool(session_status["profile_rows_ready"]),
        "profile_merged_fts_ready": bool(session_status["profile_merged_fts_ready"]),
        "profile_evidence_fts_ready": bool(session_status["profile_evidence_fts_ready"]),
        "profile_inference_fts_ready": bool(session_status["profile_inference_fts_ready"]),
        "profile_enrichment_fts_ready": bool(session_status["profile_enrichment_fts_ready"]),
        "work_event_rows_ready": bool(session_status["work_event_inference_rows_ready"]),
        "work_event_fts_ready": bool(session_status["work_event_inference_fts_ready"]),
        "phase_rows_ready": bool(session_status["phase_inference_rows_ready"]),
        "threads_ready": bool(session_status["threads_ready"]),
        "thread_fts_ready": bool(session_status["threads_fts_ready"]),
        "tag_rollups_ready": bool(session_status["tag_rollups_ready"]),
        "day_summaries_ready": bool(session_status["day_summaries_ready"]),
        "week_summaries_ready": bool(session_status["week_summaries_ready"]),
        "embedded_conversations": embedding_stats.embedded_conversations,
        "embedded_messages": embedding_stats.embedded_messages,
        "pending_conversations": embedding_stats.pending_conversations,
        "stale_messages": embedding_stats.stale_messages,
        "missing_provenance": embedding_stats.messages_missing_provenance,
    }
    metrics["transcript_embeddings_ready"] = (
        total_conversations == 0
        or (
            int(metrics["embedded_conversations"]) == total_conversations
            and int(metrics["pending_conversations"]) == 0
            and int(metrics["stale_messages"]) == 0
            and int(metrics["missing_provenance"]) == 0
        )
    )
    metrics["evidence_retrieval_rows"] = int(metrics["profile_evidence_fts_rows"]) + int(metrics["action_fts_rows"])
    metrics["expected_evidence_retrieval_rows"] = int(metrics["profile_rows"]) + int(metrics["action_rows"])
    metrics["evidence_retrieval_ready"] = bool(session_status["profile_evidence_fts_ready"]) and bool(action_status["action_fts_ready"])
    metrics["inference_retrieval_rows"] = int(metrics["profile_inference_fts_rows"]) + int(metrics["work_event_fts_rows"]) + int(metrics["phase_rows"])
    metrics["expected_inference_retrieval_rows"] = int(metrics["profile_rows"]) + int(metrics["work_event_rows"]) + int(metrics["phase_rows"])
    metrics["inference_retrieval_ready"] = (
        bool(session_status["profile_inference_fts_ready"])
        and bool(session_status["work_event_inference_fts_ready"])
        and bool(session_status["phase_inference_rows_ready"])
    )
    metrics["enrichment_retrieval_rows"] = int(metrics["profile_enrichment_fts_rows"])
    metrics["expected_enrichment_retrieval_rows"] = int(metrics["profile_rows"])
    metrics["enrichment_retrieval_ready"] = bool(session_status["profile_enrichment_fts_ready"])

    return {
        **build_archive_product_statuses(metrics),
        **build_retrieval_statuses(metrics),
    }


__all__ = ["collect_derived_model_statuses_sync"]
