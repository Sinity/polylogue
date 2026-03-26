"""Derived-model statuses for session products and archive read models."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_docs, pending_rows
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION, SESSION_PRODUCT_MATERIALIZER_VERSION


def build_archive_product_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "messages_fts": DerivedModelStatus(
            name="messages_fts",
            ready=bool(metrics["message_fts_ready"]),
            detail=(
                f"Messages FTS ready ({metrics['message_fts_rows']:,}/{metrics['total_messages']:,} rows)"
                if bool(metrics["message_fts_ready"])
                else f"Messages FTS pending ({metrics['message_fts_rows']:,}/{metrics['total_messages']:,} rows)"
            ),
            source_rows=int(metrics["total_messages"]),
            materialized_rows=int(metrics["message_fts_rows"]),
            pending_rows=pending_rows(int(metrics["total_messages"]), int(metrics["message_fts_rows"])),
        ),
        "action_events": DerivedModelStatus(
            name="action_events",
            ready=bool(metrics["action_rows_ready"]),
            detail=(
                f"Action-event rows ready ({metrics['action_documents']:,}/{metrics['action_source_documents']:,} conversations)"
                if bool(metrics["action_rows_ready"])
                else f"Action-event rows pending ({metrics['action_documents']:,}/{metrics['action_source_documents']:,} conversations)"
            ),
            source_documents=int(metrics["action_source_documents"]),
            materialized_documents=int(metrics["action_documents"]),
            materialized_rows=int(metrics["action_rows"]),
            pending_documents=pending_docs(int(metrics["action_source_documents"]), int(metrics["action_documents"])),
            stale_rows=int(metrics["action_stale_rows"]),
            orphan_rows=int(metrics["action_orphan_rows"]),
            materializer_version=ACTION_EVENT_MATERIALIZER_VERSION,
            matches_version=bool(metrics["action_matches_version"]),
        ),
        "action_events_fts": DerivedModelStatus(
            name="action_events_fts",
            ready=bool(metrics["action_fts_ready"]),
            detail=(
                f"Action-event FTS ready ({metrics['action_fts_rows']:,}/{metrics['action_rows']:,} rows)"
                if bool(metrics["action_fts_ready"])
                else f"Action-event FTS pending ({metrics['action_fts_rows']:,}/{metrics['action_rows']:,} rows)"
            ),
            source_rows=int(metrics["action_rows"]),
            materialized_rows=int(metrics["action_fts_rows"]),
            pending_rows=pending_rows(int(metrics["action_rows"]), int(metrics["action_fts_rows"])),
            orphan_rows=int(metrics["action_orphan_rows"]),
        ),
        "session_profile_rows": DerivedModelStatus(
            name="session_profile_rows",
            ready=bool(metrics["profile_rows_ready"]),
            detail=(
                f"Session-profile rows ready ({metrics['profile_rows']:,}/{metrics['total_conversations']:,} conversations)"
                if bool(metrics["profile_rows_ready"])
                else f"Session-profile rows pending ({metrics['profile_rows']:,}/{metrics['total_conversations']:,} conversations)"
            ),
            source_documents=int(metrics["total_conversations"]),
            materialized_documents=int(metrics["profile_rows"]),
            pending_documents=int(metrics["missing_profile_rows"]),
            stale_rows=int(metrics["stale_profile_rows"]),
            orphan_rows=int(metrics["orphan_profile_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_profile_rows"]) == 0 and int(metrics["orphan_profile_rows"]) == 0),
        ),
        "session_profile_merged_fts": DerivedModelStatus(
            name="session_profile_merged_fts",
            ready=bool(metrics["profile_merged_fts_ready"]),
            detail=(
                f"Session-profile merged FTS ready ({metrics['profile_merged_fts_rows']:,}/{metrics['profile_rows']:,} rows)"
                if bool(metrics["profile_merged_fts_ready"])
                else (
                    f"Session-profile merged FTS pending ({metrics['profile_merged_fts_rows']:,}/{metrics['profile_rows']:,} rows, "
                    f"duplicates {metrics['profile_merged_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["profile_rows"]),
            materialized_rows=int(metrics["profile_merged_fts_rows"]),
            pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics["profile_merged_fts_rows"])),
            stale_rows=int(metrics["profile_merged_fts_duplicates"]),
        ),
        "session_profile_evidence_fts": DerivedModelStatus(
            name="session_profile_evidence_fts",
            ready=bool(metrics["profile_evidence_fts_ready"]),
            detail=(
                f"Session-profile evidence FTS ready ({metrics['profile_evidence_fts_rows']:,}/{metrics['profile_rows']:,} rows)"
                if bool(metrics["profile_evidence_fts_ready"])
                else (
                    f"Session-profile evidence FTS pending ({metrics['profile_evidence_fts_rows']:,}/{metrics['profile_rows']:,} rows, "
                    f"duplicates {metrics['profile_evidence_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["profile_rows"]),
            materialized_rows=int(metrics["profile_evidence_fts_rows"]),
            pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics["profile_evidence_fts_rows"])),
            stale_rows=int(metrics["profile_evidence_fts_duplicates"]),
        ),
        "session_profile_inference_fts": DerivedModelStatus(
            name="session_profile_inference_fts",
            ready=bool(metrics["profile_inference_fts_ready"]),
            detail=(
                f"Session-profile inference FTS ready ({metrics['profile_inference_fts_rows']:,}/{metrics['profile_rows']:,} rows)"
                if bool(metrics["profile_inference_fts_ready"])
                else (
                    f"Session-profile inference FTS pending ({metrics['profile_inference_fts_rows']:,}/{metrics['profile_rows']:,} rows, "
                    f"duplicates {metrics['profile_inference_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["profile_rows"]),
            materialized_rows=int(metrics["profile_inference_fts_rows"]),
            pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics["profile_inference_fts_rows"])),
            stale_rows=int(metrics["profile_inference_fts_duplicates"]),
        ),
        "session_profile_enrichment_fts": DerivedModelStatus(
            name="session_profile_enrichment_fts",
            ready=bool(metrics["profile_enrichment_fts_ready"]),
            detail=(
                f"Session-profile enrichment FTS ready ({metrics['profile_enrichment_fts_rows']:,}/{metrics['profile_rows']:,} rows)"
                if bool(metrics["profile_enrichment_fts_ready"])
                else (
                    f"Session-profile enrichment FTS pending ({metrics['profile_enrichment_fts_rows']:,}/{metrics['profile_rows']:,} rows, "
                    f"duplicates {metrics['profile_enrichment_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["profile_rows"]),
            materialized_rows=int(metrics["profile_enrichment_fts_rows"]),
            pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics["profile_enrichment_fts_rows"])),
            stale_rows=int(metrics["profile_enrichment_fts_duplicates"]),
        ),
        "session_work_event_inference": DerivedModelStatus(
            name="session_work_event_inference",
            ready=bool(metrics["work_event_rows_ready"]),
            detail=(
                f"Session work-event inference ready ({metrics['work_event_rows']:,}/{metrics['expected_work_event_rows']:,} rows)"
                if bool(metrics["work_event_rows_ready"])
                else f"Session work-event inference pending ({metrics['work_event_rows']:,}/{metrics['expected_work_event_rows']:,} rows)"
            ),
            source_documents=int(metrics["profile_rows"]),
            materialized_documents=int(metrics["profile_rows"]) if int(metrics["profile_rows"]) else 0,
            source_rows=int(metrics["expected_work_event_rows"]),
            materialized_rows=int(metrics["work_event_rows"]),
            pending_rows=pending_rows(int(metrics["expected_work_event_rows"]), int(metrics["work_event_rows"])),
            stale_rows=int(metrics["stale_work_event_rows"]),
            orphan_rows=int(metrics["orphan_work_event_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_work_event_rows"]) == 0 and int(metrics["orphan_work_event_rows"]) == 0),
        ),
        "session_work_event_inference_fts": DerivedModelStatus(
            name="session_work_event_inference_fts",
            ready=bool(metrics["work_event_fts_ready"]),
            detail=(
                f"Session work-event inference FTS ready ({metrics['work_event_fts_rows']:,}/{metrics['work_event_rows']:,} rows)"
                if bool(metrics["work_event_fts_ready"])
                else (
                    f"Session work-event inference FTS pending ({metrics['work_event_fts_rows']:,}/{metrics['work_event_rows']:,} rows, "
                    f"duplicates {metrics['work_event_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["work_event_rows"]),
            materialized_rows=int(metrics["work_event_fts_rows"]),
            pending_rows=pending_rows(int(metrics["work_event_rows"]), int(metrics["work_event_fts_rows"])),
            stale_rows=int(metrics["work_event_fts_duplicates"]),
        ),
        "session_phase_inference": DerivedModelStatus(
            name="session_phase_inference",
            ready=bool(metrics["phase_rows_ready"]),
            detail=(
                f"Session phase inference ready ({metrics['phase_rows']:,}/{metrics['expected_phase_rows']:,} rows)"
                if bool(metrics["phase_rows_ready"])
                else f"Session phase inference pending ({metrics['phase_rows']:,}/{metrics['expected_phase_rows']:,} rows)"
            ),
            source_documents=int(metrics["profile_rows"]),
            materialized_documents=int(metrics["profile_rows"]) if int(metrics["profile_rows"]) else 0,
            source_rows=int(metrics["expected_phase_rows"]),
            materialized_rows=int(metrics["phase_rows"]),
            pending_rows=pending_rows(int(metrics["expected_phase_rows"]), int(metrics["phase_rows"])),
            stale_rows=int(metrics["stale_phase_rows"]),
            orphan_rows=int(metrics["orphan_phase_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_phase_rows"]) == 0 and int(metrics["orphan_phase_rows"]) == 0),
        ),
        "work_threads": DerivedModelStatus(
            name="work_threads",
            ready=bool(metrics["threads_ready"]),
            detail=(
                f"Work threads ready ({metrics['work_thread_rows']:,}/{metrics['total_thread_roots']:,} roots)"
                if bool(metrics["threads_ready"])
                else f"Work threads pending ({metrics['work_thread_rows']:,}/{metrics['total_thread_roots']:,} roots)"
            ),
            source_documents=int(metrics["total_thread_roots"]),
            materialized_documents=int(metrics["work_thread_rows"]),
            pending_documents=pending_docs(int(metrics["total_thread_roots"]), int(metrics["work_thread_rows"])),
            stale_rows=int(metrics["stale_thread_rows"]),
            orphan_rows=int(metrics["orphan_thread_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_thread_rows"]) == 0 and int(metrics["orphan_thread_rows"]) == 0),
        ),
        "work_threads_fts": DerivedModelStatus(
            name="work_threads_fts",
            ready=bool(metrics["thread_fts_ready"]),
            detail=(
                f"Work-thread FTS ready ({metrics['work_thread_fts_rows']:,}/{metrics['work_thread_rows']:,} rows)"
                if bool(metrics["thread_fts_ready"])
                else (
                    f"Work-thread FTS pending ({metrics['work_thread_fts_rows']:,}/{metrics['work_thread_rows']:,} rows, "
                    f"duplicates {metrics['work_thread_fts_duplicates']:,})"
                )
            ),
            source_rows=int(metrics["work_thread_rows"]),
            materialized_rows=int(metrics["work_thread_fts_rows"]),
            pending_rows=pending_rows(int(metrics["work_thread_rows"]), int(metrics["work_thread_fts_rows"])),
            stale_rows=int(metrics["work_thread_fts_duplicates"]),
        ),
        "session_tag_rollups": DerivedModelStatus(
            name="session_tag_rollups",
            ready=bool(metrics["tag_rollups_ready"]),
            detail=(
                f"Session tag rollups ready ({metrics['tag_rollup_rows']:,}/{metrics['expected_tag_rollup_rows']:,} rows)"
                if bool(metrics["tag_rollups_ready"])
                else f"Session tag rollups pending ({metrics['tag_rollup_rows']:,}/{metrics['expected_tag_rollup_rows']:,} rows)"
            ),
            source_rows=int(metrics["expected_tag_rollup_rows"]),
            materialized_rows=int(metrics["tag_rollup_rows"]),
            pending_rows=pending_rows(int(metrics["expected_tag_rollup_rows"]), int(metrics["tag_rollup_rows"])),
            stale_rows=int(metrics["stale_tag_rollup_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_tag_rollup_rows"]) == 0),
        ),
        "day_session_summaries": DerivedModelStatus(
            name="day_session_summaries",
            ready=bool(metrics["day_summaries_ready"]),
            detail=(
                f"Day session summaries ready ({metrics['day_summary_rows']:,}/{metrics['expected_day_summary_rows']:,} rows)"
                if bool(metrics["day_summaries_ready"])
                else f"Day session summaries pending ({metrics['day_summary_rows']:,}/{metrics['expected_day_summary_rows']:,} rows)"
            ),
            source_rows=int(metrics["expected_day_summary_rows"]),
            materialized_rows=int(metrics["day_summary_rows"]),
            pending_rows=pending_rows(int(metrics["expected_day_summary_rows"]), int(metrics["day_summary_rows"])),
            stale_rows=int(metrics["stale_day_summary_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_day_summary_rows"]) == 0),
        ),
        "week_session_summaries": DerivedModelStatus(
            name="week_session_summaries",
            ready=bool(metrics["week_summaries_ready"]),
            detail=(
                "Week session summaries ready (derived from day-session summaries)"
                if bool(metrics["week_summaries_ready"])
                else "Week session summaries pending (day-session summaries not ready)"
            ),
            source_rows=int(metrics["expected_day_summary_rows"]),
            materialized_rows=int(metrics["day_summary_rows"]),
            pending_rows=pending_rows(int(metrics["expected_day_summary_rows"]), int(metrics["day_summary_rows"])),
            stale_rows=int(metrics["stale_day_summary_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_day_summary_rows"]) == 0),
        ),
    }


__all__ = ["build_archive_product_statuses"]
