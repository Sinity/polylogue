"""Canonical readiness/freshness snapshot for durable derived models."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.action_event_status import action_event_read_model_status_sync
from polylogue.storage.derived_status_products import build_archive_product_statuses, pending_docs, pending_rows
from polylogue.storage.embedding_stats import read_embedding_stats_sync
from polylogue.storage.fts_lifecycle import message_fts_readiness_sync
from polylogue.storage.session_product_status import session_product_status_sync


def collect_derived_model_statuses_sync(
    conn: sqlite3.Connection,
    *,
    verify_full: bool = True,
) -> dict[str, DerivedModelStatus]:
    total_conversations = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0)

    fts_status = message_fts_readiness_sync(conn, verify_total_rows=verify_full)
    action_status = action_event_read_model_status_sync(conn, verify_source_alignment=verify_full)
    session_status = session_product_status_sync(conn, verify_freshness=verify_full)
    embedding_stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)

    metrics: dict[str, int | bool] = {
        "total_conversations": total_conversations,
        "message_fts_exact_counts": verify_full,
        "message_source_rows": int(fts_status["total_rows"]),
        "message_fts_rows": int(fts_status["indexed_rows"]),
        "message_fts_ready": bool(fts_status["ready"]),
        "action_rows": int(action_status["count"]),
        "action_documents": int(action_status["materialized_conversation_count"]),
        "action_source_documents": int(action_status["valid_source_conversation_count"]),
        "action_orphan_rows": int(action_status["orphan_tool_block_count"]),
        "action_fts_rows": int(action_status["action_fts_count"]),
        "action_fts_stale_rows": int(action_status.get("action_fts_stale_rows", 0)),
        "action_rows_ready": bool(action_status["rows_ready"]),
        "action_fts_ready": bool(action_status["action_fts_ready"]),
        "action_stale_rows": int(action_status["stale_count"]),
        "action_matches_version": bool(action_status["matches_version"]),
        "profile_rows": session_status.profile_row_count,
        "profile_merged_fts_rows": session_status.profile_merged_fts_count,
        "profile_merged_fts_duplicates": session_status.profile_merged_fts_duplicate_count,
        "profile_evidence_fts_rows": session_status.profile_evidence_fts_count,
        "profile_evidence_fts_duplicates": session_status.profile_evidence_fts_duplicate_count,
        "profile_inference_fts_rows": session_status.profile_inference_fts_count,
        "profile_inference_fts_duplicates": session_status.profile_inference_fts_duplicate_count,
        "profile_enrichment_fts_rows": session_status.profile_enrichment_fts_count,
        "profile_enrichment_fts_duplicates": session_status.profile_enrichment_fts_duplicate_count,
        "work_event_rows": session_status.work_event_inference_count,
        "work_event_fts_rows": session_status.work_event_inference_fts_count,
        "work_event_fts_duplicates": session_status.work_event_inference_fts_duplicate_count,
        "phase_rows": session_status.phase_inference_count,
        "work_thread_rows": session_status.thread_count,
        "work_thread_fts_rows": session_status.thread_fts_count,
        "work_thread_fts_duplicates": session_status.thread_fts_duplicate_count,
        "total_thread_roots": session_status.root_threads,
        "tag_rollup_rows": session_status.tag_rollup_count,
        "expected_tag_rollup_rows": session_status.expected_tag_rollup_count,
        "day_summary_rows": session_status.day_summary_count,
        "expected_day_summary_rows": session_status.expected_day_summary_count,
        "missing_profile_rows": session_status.missing_profile_row_count,
        "stale_profile_rows": session_status.stale_profile_row_count,
        "orphan_profile_rows": session_status.orphan_profile_row_count,
        "expected_work_event_rows": session_status.expected_work_event_inference_count,
        "stale_work_event_rows": session_status.stale_work_event_inference_count,
        "orphan_work_event_rows": session_status.orphan_work_event_inference_count,
        "expected_phase_rows": session_status.expected_phase_inference_count,
        "stale_phase_rows": session_status.stale_phase_inference_count,
        "orphan_phase_rows": session_status.orphan_phase_inference_count,
        "stale_thread_rows": session_status.stale_thread_count,
        "orphan_thread_rows": session_status.orphan_thread_count,
        "stale_tag_rollup_rows": session_status.stale_tag_rollup_count,
        "stale_day_summary_rows": session_status.stale_day_summary_count,
        "profile_rows_ready": session_status.profile_rows_ready,
        "profile_merged_fts_ready": session_status.profile_merged_fts_ready,
        "profile_evidence_fts_ready": session_status.profile_evidence_fts_ready,
        "profile_inference_fts_ready": session_status.profile_inference_fts_ready,
        "profile_enrichment_fts_ready": session_status.profile_enrichment_fts_ready,
        "work_event_rows_ready": session_status.work_event_inference_rows_ready,
        "work_event_fts_ready": session_status.work_event_inference_fts_ready,
        "phase_rows_ready": session_status.phase_inference_rows_ready,
        "threads_ready": session_status.threads_ready,
        "thread_fts_ready": session_status.threads_fts_ready,
        "tag_rollups_ready": session_status.tag_rollups_ready,
        "day_summaries_ready": session_status.day_summaries_ready,
        "week_summaries_ready": session_status.week_summaries_ready,
        "embedded_conversations": embedding_stats.embedded_conversations,
        "embedded_messages": embedding_stats.embedded_messages,
        "pending_conversations": embedding_stats.pending_conversations,
        "stale_messages": embedding_stats.stale_messages,
        "missing_provenance": embedding_stats.messages_missing_provenance,
    }
    metrics["transcript_embeddings_ready"] = total_conversations == 0 or (
        int(metrics["embedded_conversations"]) == total_conversations
        and int(metrics["pending_conversations"]) == 0
        and int(metrics["stale_messages"]) == 0
        and int(metrics["missing_provenance"]) == 0
    )
    metrics["evidence_retrieval_rows"] = int(metrics["profile_evidence_fts_rows"]) + int(metrics["action_fts_rows"])
    metrics["expected_evidence_retrieval_rows"] = int(metrics["profile_rows"]) + int(metrics["action_rows"])
    metrics["evidence_retrieval_ready"] = session_status.profile_evidence_fts_ready and bool(
        action_status["action_fts_ready"]
    )
    metrics["inference_retrieval_rows"] = (
        int(metrics["profile_inference_fts_rows"]) + int(metrics["work_event_fts_rows"]) + int(metrics["phase_rows"])
    )
    metrics["expected_inference_retrieval_rows"] = (
        int(metrics["profile_rows"]) + int(metrics["work_event_rows"]) + int(metrics["phase_rows"])
    )
    metrics["inference_retrieval_ready"] = (
        session_status.profile_inference_fts_ready
        and session_status.work_event_inference_fts_ready
        and session_status.phase_inference_rows_ready
    )
    metrics["enrichment_retrieval_rows"] = int(metrics["profile_enrichment_fts_rows"])
    metrics["expected_enrichment_retrieval_rows"] = int(metrics["profile_rows"])
    metrics["enrichment_retrieval_ready"] = session_status.profile_enrichment_fts_ready

    return {
        **build_archive_product_statuses(metrics),
        **build_retrieval_statuses(metrics),
    }


# ---------------------------------------------------------------------------
# Retrieval / embedding statuses
# ---------------------------------------------------------------------------


def build_retrieval_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "transcript_embeddings": DerivedModelStatus(
            name="transcript_embeddings",
            ready=bool(metrics["transcript_embeddings_ready"]),
            detail=(
                f"Transcript embeddings ready ({metrics['embedded_conversations']:,}/{metrics['total_conversations']:,} conversations, {metrics['embedded_messages']:,} messages)"
                if bool(metrics["transcript_embeddings_ready"])
                else (
                    f"Transcript embeddings pending ({metrics['embedded_conversations']:,}/{metrics['total_conversations']:,} conversations, "
                    f"pending {metrics['pending_conversations']:,}, stale {metrics['stale_messages']:,}, missing provenance {metrics['missing_provenance']:,})"
                )
            ),
            source_documents=int(metrics["total_conversations"]),
            materialized_documents=int(metrics["embedded_conversations"]),
            materialized_rows=int(metrics["embedded_messages"]),
            pending_documents=int(metrics["pending_conversations"]),
            stale_rows=int(metrics["stale_messages"]),
            missing_provenance_rows=int(metrics["missing_provenance"]),
        ),
        "retrieval_evidence": DerivedModelStatus(
            name="retrieval_evidence",
            ready=bool(metrics["evidence_retrieval_ready"]),
            detail=(
                f"Evidence retrieval ready ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows)"
                if bool(metrics["evidence_retrieval_ready"])
                else (
                    f"Evidence retrieval pending ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows; "
                    f"profile_evidence_fts={metrics['profile_evidence_fts_rows']:,}/{metrics['profile_rows']:,}, "
                    f"action_event_fts={metrics['action_fts_rows']:,}/{metrics['action_rows']:,})"
                )
            ),
            source_documents=int(metrics["action_source_documents"]) + int(metrics["total_conversations"]),
            materialized_documents=int(metrics["action_documents"]) + int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_evidence_retrieval_rows"]),
            materialized_rows=int(metrics["evidence_retrieval_rows"]),
            pending_documents=(
                pending_docs(int(metrics["action_source_documents"]), int(metrics["action_documents"]))
                + pending_docs(int(metrics["total_conversations"]), int(metrics["profile_rows"]))
            ),
            pending_rows=pending_rows(
                int(metrics["expected_evidence_retrieval_rows"]), int(metrics["evidence_retrieval_rows"])
            ),
            stale_rows=(
                int(metrics["profile_evidence_fts_duplicates"])
                + int(metrics["action_stale_rows"])
                + int(metrics.get("action_fts_stale_rows", 0))
            ),
            orphan_rows=int(metrics["action_orphan_rows"]) + int(metrics["orphan_profile_rows"]),
        ),
        "retrieval_inference": DerivedModelStatus(
            name="retrieval_inference",
            ready=bool(metrics["inference_retrieval_ready"]),
            detail=(
                f"Inference retrieval ready ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows)"
                if bool(metrics["inference_retrieval_ready"])
                else (
                    f"Inference retrieval pending ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows; "
                    f"profile_inference_fts={metrics['profile_inference_fts_rows']:,}/{metrics['profile_rows']:,}, "
                    f"work_event_fts={metrics['work_event_fts_rows']:,}/{metrics['work_event_rows']:,}, "
                    f"phases={metrics['phase_rows']:,}/{metrics['expected_phase_rows']:,})"
                )
            ),
            source_documents=int(metrics["profile_rows"]),
            materialized_documents=int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_inference_retrieval_rows"]),
            materialized_rows=int(metrics["inference_retrieval_rows"]),
            pending_rows=pending_rows(
                int(metrics["expected_inference_retrieval_rows"]), int(metrics["inference_retrieval_rows"])
            ),
            stale_rows=(
                int(metrics["profile_inference_fts_duplicates"])
                + int(metrics["work_event_fts_duplicates"])
                + int(metrics["stale_work_event_rows"])
                + int(metrics["stale_phase_rows"])
            ),
            orphan_rows=(
                int(metrics["orphan_profile_rows"])
                + int(metrics["orphan_work_event_rows"])
                + int(metrics["orphan_phase_rows"])
            ),
        ),
        "retrieval_enrichment": DerivedModelStatus(
            name="retrieval_enrichment",
            ready=bool(metrics["enrichment_retrieval_ready"]),
            detail=(
                f"Enrichment retrieval ready ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows)"
                if bool(metrics["enrichment_retrieval_ready"])
                else (
                    f"Enrichment retrieval pending ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows; "
                    f"profile_enrichment_fts={metrics['profile_enrichment_fts_rows']:,}/{metrics['profile_rows']:,})"
                )
            ),
            source_rows=int(metrics["expected_enrichment_retrieval_rows"]),
            materialized_rows=int(metrics["enrichment_retrieval_rows"]),
            pending_rows=pending_rows(
                int(metrics["expected_enrichment_retrieval_rows"]), int(metrics["enrichment_retrieval_rows"])
            ),
            stale_rows=int(metrics["profile_enrichment_fts_duplicates"]),
        ),
    }


__all__ = ["build_retrieval_statuses", "collect_derived_model_statuses_sync", "pending_docs", "pending_rows"]
