"""Derived-model statuses for session products and archive read models."""

from __future__ import annotations

from dataclasses import replace

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.action_event_artifacts import ActionEventArtifactState
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION, SESSION_PRODUCT_MATERIALIZER_VERSION


def pending_rows(source_rows: int, materialized_rows: int) -> int:
    return max(0, source_rows - materialized_rows)


def pending_docs(source_docs: int, materialized_docs: int) -> int:
    return max(0, source_docs - materialized_docs)


# ---------------------------------------------------------------------------
# Action/search statuses
# ---------------------------------------------------------------------------


def build_action_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    state = ActionEventArtifactState.from_metrics(metrics)
    action_rows_status = state.row_status()
    action_rows_status = replace(action_rows_status, materializer_version=ACTION_EVENT_MATERIALIZER_VERSION)
    message_fts_exact_counts = bool(metrics.get("message_fts_exact_counts", True))
    return {
        "messages_fts": DerivedModelStatus(
            name="messages_fts",
            ready=bool(metrics["message_fts_ready"]),
            detail=(
                (
                    f"Messages FTS ready ({metrics['message_fts_rows']:,}/{metrics['message_source_rows']:,} rows)"
                    if message_fts_exact_counts
                    else "Messages FTS present"
                )
                if bool(metrics["message_fts_ready"])
                else (
                    f"Messages FTS pending ({metrics['message_fts_rows']:,}/{metrics['message_source_rows']:,} rows)"
                    if message_fts_exact_counts
                    else "Messages FTS missing or empty; use --deep to verify full coverage"
                )
            ),
            source_rows=int(metrics["message_source_rows"]),
            materialized_rows=int(metrics["message_fts_rows"]),
            pending_rows=pending_rows(int(metrics["message_source_rows"]), int(metrics["message_fts_rows"])),
        ),
        "action_events": action_rows_status,
        "action_events_fts": state.fts_status(),
    }


# ---------------------------------------------------------------------------
# Profile/status statuses
# ---------------------------------------------------------------------------


def build_profile_fts_status(
    metrics: dict[str, int | bool],
    *,
    key_prefix: str,
    name: str,
    label: str,
) -> DerivedModelStatus:
    ready_key = f"{key_prefix}_ready"
    rows_key = f"{key_prefix}_rows"
    duplicate_key = f"{key_prefix}_duplicates"
    return DerivedModelStatus(
        name=name,
        ready=bool(metrics[ready_key]),
        detail=(
            f"{label} ready ({metrics[rows_key]:,}/{metrics['profile_rows']:,} rows)"
            if bool(metrics[ready_key])
            else (
                f"{label} pending ({metrics[rows_key]:,}/{metrics['profile_rows']:,} rows, "
                f"duplicates {metrics[duplicate_key]:,})"
            )
        ),
        source_rows=int(metrics["profile_rows"]),
        materialized_rows=int(metrics[rows_key]),
        pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics[rows_key])),
        stale_rows=int(metrics[duplicate_key]),
    )


def build_profile_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
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
        "session_profile_merged_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_merged_fts",
            name="session_profile_merged_fts",
            label="Session-profile merged FTS",
        ),
        "session_profile_evidence_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_evidence_fts",
            name="session_profile_evidence_fts",
            label="Session-profile evidence FTS",
        ),
        "session_profile_inference_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_inference_fts",
            name="session_profile_inference_fts",
            label="Session-profile inference FTS",
        ),
        "session_profile_enrichment_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_enrichment_fts",
            name="session_profile_enrichment_fts",
            label="Session-profile enrichment FTS",
        ),
    }


# ---------------------------------------------------------------------------
# Timeline/work-product statuses
# ---------------------------------------------------------------------------


def build_timeline_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
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
            matches_version=bool(
                int(metrics["stale_work_event_rows"]) == 0 and int(metrics["orphan_work_event_rows"]) == 0
            ),
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
    }


# ---------------------------------------------------------------------------
# Aggregate statuses
# ---------------------------------------------------------------------------


def build_aggregate_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
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


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def build_archive_product_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        **build_action_statuses(metrics),
        **build_profile_statuses(metrics),
        **build_timeline_statuses(metrics),
        **build_aggregate_statuses(metrics),
    }


__all__ = ["build_archive_product_statuses"]
