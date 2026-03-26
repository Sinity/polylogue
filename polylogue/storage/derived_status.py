"""Shared derived-model readiness/freshness status helpers."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.action_event_lifecycle import action_event_read_model_status_sync
from polylogue.storage.embedding_stats import read_embedding_stats_sync
from polylogue.storage.fts_lifecycle import fts_index_status_sync
from polylogue.storage.session_product_lifecycle import session_product_status_sync
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION, SESSION_PRODUCT_MATERIALIZER_VERSION


def _pending_rows(source_rows: int, materialized_rows: int) -> int:
    return max(0, source_rows - materialized_rows)


def _pending_docs(source_docs: int, materialized_docs: int) -> int:
    return max(0, source_docs - materialized_docs)


def collect_derived_model_statuses_sync(
    conn: sqlite3.Connection,
) -> dict[str, DerivedModelStatus]:
    """Return a canonical readiness/freshness snapshot for durable derived models."""
    total_conversations = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0)
    total_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)

    fts_status = fts_index_status_sync(conn)
    message_fts_rows = int(fts_status.get("count", 0))
    message_fts_ready = bool(fts_status.get("exists", False)) and message_fts_rows == total_messages

    action_status = action_event_read_model_status_sync(conn)
    action_rows = int(action_status["count"])
    action_documents = int(action_status["materialized_conversation_count"])
    action_source_documents = int(action_status["valid_source_conversation_count"])
    action_orphan_rows = int(action_status["orphan_tool_block_count"])
    action_fts_rows = int(action_status["action_fts_count"])

    session_status = session_product_status_sync(conn)
    profile_rows = int(session_status["profile_row_count"])
    profile_merged_fts_rows = int(session_status["profile_merged_fts_count"])
    profile_merged_fts_duplicates = int(session_status["profile_merged_fts_duplicate_count"])
    profile_evidence_fts_rows = int(session_status["profile_evidence_fts_count"])
    profile_evidence_fts_duplicates = int(session_status["profile_evidence_fts_duplicate_count"])
    profile_inference_fts_rows = int(session_status["profile_inference_fts_count"])
    profile_inference_fts_duplicates = int(session_status["profile_inference_fts_duplicate_count"])
    expected_work_event_rows = int(session_status["expected_work_event_inference_count"])
    work_event_rows = int(session_status["work_event_inference_count"])
    work_event_fts_rows = int(session_status["work_event_inference_fts_count"])
    work_event_fts_duplicates = int(session_status["work_event_inference_fts_duplicate_count"])
    expected_phase_rows = int(session_status["expected_phase_inference_count"])
    phase_rows = int(session_status["phase_inference_count"])
    work_thread_rows = int(session_status["thread_count"])
    work_thread_fts_rows = int(session_status["thread_fts_count"])
    work_thread_fts_duplicates = int(session_status["thread_fts_duplicate_count"])
    total_thread_roots = int(session_status["root_threads"])
    tag_rollup_rows = int(session_status["tag_rollup_count"])
    expected_tag_rollup_rows = int(session_status["expected_tag_rollup_count"])
    day_summary_rows = int(session_status["day_summary_count"])
    expected_day_summary_rows = int(session_status["expected_day_summary_count"])

    embedding_stats = read_embedding_stats_sync(conn)
    embedded_conversations = embedding_stats.embedded_conversations
    embedded_messages = embedding_stats.embedded_messages
    pending_conversations = embedding_stats.pending_conversations
    stale_messages = embedding_stats.stale_messages
    missing_provenance = embedding_stats.messages_missing_provenance
    transcript_embeddings_ready = (
        total_conversations == 0
        or (
            embedded_conversations == total_conversations
            and pending_conversations == 0
            and stale_messages == 0
            and missing_provenance == 0
        )
    )
    evidence_retrieval_rows = profile_evidence_fts_rows + action_fts_rows
    expected_evidence_retrieval_rows = profile_rows + action_rows
    evidence_retrieval_ready = (
        bool(session_status["profile_evidence_fts_ready"])
        and bool(action_status["action_fts_ready"])
    )
    inference_retrieval_rows = profile_inference_fts_rows + work_event_fts_rows + phase_rows
    expected_inference_retrieval_rows = profile_rows + work_event_rows + phase_rows
    inference_retrieval_ready = (
        bool(session_status["profile_inference_fts_ready"])
        and bool(session_status["work_event_inference_fts_ready"])
        and bool(session_status["phase_inference_rows_ready"])
    )

    return {
        "messages_fts": DerivedModelStatus(
            name="messages_fts",
            ready=message_fts_ready,
            detail=(
                f"Messages FTS ready ({message_fts_rows:,}/{total_messages:,} rows)"
                if message_fts_ready
                else f"Messages FTS pending ({message_fts_rows:,}/{total_messages:,} rows)"
            ),
            source_rows=total_messages,
            materialized_rows=message_fts_rows,
            pending_rows=_pending_rows(total_messages, message_fts_rows),
        ),
        "action_events": DerivedModelStatus(
            name="action_events",
            ready=bool(action_status["rows_ready"]),
            detail=(
                f"Action-event rows ready ({action_documents:,}/{action_source_documents:,} conversations)"
                if bool(action_status["rows_ready"])
                else f"Action-event rows pending ({action_documents:,}/{action_source_documents:,} conversations)"
            ),
            source_documents=action_source_documents,
            materialized_documents=action_documents,
            materialized_rows=action_rows,
            pending_documents=_pending_docs(action_source_documents, action_documents),
            stale_rows=int(action_status["stale_count"]),
            orphan_rows=action_orphan_rows,
            materializer_version=ACTION_EVENT_MATERIALIZER_VERSION,
            matches_version=bool(action_status["matches_version"]),
        ),
        "action_events_fts": DerivedModelStatus(
            name="action_events_fts",
            ready=bool(action_status["action_fts_ready"]),
            detail=(
                f"Action-event FTS ready ({action_fts_rows:,}/{action_rows:,} rows)"
                if bool(action_status["action_fts_ready"])
                else f"Action-event FTS pending ({action_fts_rows:,}/{action_rows:,} rows)"
            ),
            source_rows=action_rows,
            materialized_rows=action_fts_rows,
            pending_rows=_pending_rows(action_rows, action_fts_rows),
            orphan_rows=action_orphan_rows,
        ),
        "session_profile_rows": DerivedModelStatus(
            name="session_profile_rows",
            ready=bool(session_status["profile_rows_ready"]),
            detail=(
                f"Session-profile rows ready ({profile_rows:,}/{total_conversations:,} conversations)"
                if bool(session_status["profile_rows_ready"])
                else f"Session-profile rows pending ({profile_rows:,}/{total_conversations:,} conversations)"
            ),
            source_documents=total_conversations,
            materialized_documents=profile_rows,
            pending_documents=int(session_status["missing_profile_row_count"]),
            stale_rows=int(session_status["stale_profile_row_count"]),
            orphan_rows=int(session_status["orphan_profile_row_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_status["stale_profile_row_count"]) == 0
                and int(session_status["orphan_profile_row_count"]) == 0
            ),
        ),
        "session_profile_merged_fts": DerivedModelStatus(
            name="session_profile_merged_fts",
            ready=bool(session_status["profile_merged_fts_ready"]),
            detail=(
                f"Session-profile merged FTS ready ({profile_merged_fts_rows:,}/{profile_rows:,} rows)"
                if bool(session_status["profile_merged_fts_ready"])
                else (
                    f"Session-profile merged FTS pending ({profile_merged_fts_rows:,}/{profile_rows:,} rows, "
                    f"duplicates {profile_merged_fts_duplicates:,})"
                )
            ),
            source_rows=profile_rows,
            materialized_rows=profile_merged_fts_rows,
            pending_rows=_pending_rows(profile_rows, profile_merged_fts_rows),
            stale_rows=profile_merged_fts_duplicates,
        ),
        "session_profile_evidence_fts": DerivedModelStatus(
            name="session_profile_evidence_fts",
            ready=bool(session_status["profile_evidence_fts_ready"]),
            detail=(
                f"Session-profile evidence FTS ready ({profile_evidence_fts_rows:,}/{profile_rows:,} rows)"
                if bool(session_status["profile_evidence_fts_ready"])
                else (
                    f"Session-profile evidence FTS pending ({profile_evidence_fts_rows:,}/{profile_rows:,} rows, "
                    f"duplicates {profile_evidence_fts_duplicates:,})"
                )
            ),
            source_rows=profile_rows,
            materialized_rows=profile_evidence_fts_rows,
            pending_rows=_pending_rows(profile_rows, profile_evidence_fts_rows),
            stale_rows=profile_evidence_fts_duplicates,
        ),
        "session_profile_inference_fts": DerivedModelStatus(
            name="session_profile_inference_fts",
            ready=bool(session_status["profile_inference_fts_ready"]),
            detail=(
                f"Session-profile inference FTS ready ({profile_inference_fts_rows:,}/{profile_rows:,} rows)"
                if bool(session_status["profile_inference_fts_ready"])
                else (
                    f"Session-profile inference FTS pending ({profile_inference_fts_rows:,}/{profile_rows:,} rows, "
                    f"duplicates {profile_inference_fts_duplicates:,})"
                )
            ),
            source_rows=profile_rows,
            materialized_rows=profile_inference_fts_rows,
            pending_rows=_pending_rows(profile_rows, profile_inference_fts_rows),
            stale_rows=profile_inference_fts_duplicates,
        ),
        "session_work_event_inference": DerivedModelStatus(
            name="session_work_event_inference",
            ready=bool(session_status["work_event_inference_rows_ready"]),
            detail=(
                f"Session work-event inference ready ({work_event_rows:,}/{expected_work_event_rows:,} rows)"
                if bool(session_status["work_event_inference_rows_ready"])
                else f"Session work-event inference pending ({work_event_rows:,}/{expected_work_event_rows:,} rows)"
            ),
            source_documents=profile_rows,
            materialized_documents=profile_rows if profile_rows else 0,
            source_rows=expected_work_event_rows,
            materialized_rows=work_event_rows,
            pending_rows=_pending_rows(expected_work_event_rows, work_event_rows),
            stale_rows=int(session_status["stale_work_event_inference_count"]),
            orphan_rows=int(session_status["orphan_work_event_inference_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_status["stale_work_event_inference_count"]) == 0
                and int(session_status["orphan_work_event_inference_count"]) == 0
            ),
        ),
        "session_work_event_inference_fts": DerivedModelStatus(
            name="session_work_event_inference_fts",
            ready=bool(session_status["work_event_inference_fts_ready"]),
            detail=(
                f"Session work-event inference FTS ready ({work_event_fts_rows:,}/{work_event_rows:,} rows)"
                if bool(session_status["work_event_inference_fts_ready"])
                else (
                    f"Session work-event inference FTS pending ({work_event_fts_rows:,}/{work_event_rows:,} rows, "
                    f"duplicates {work_event_fts_duplicates:,})"
                )
            ),
            source_rows=work_event_rows,
            materialized_rows=work_event_fts_rows,
            pending_rows=_pending_rows(work_event_rows, work_event_fts_rows),
            stale_rows=work_event_fts_duplicates,
        ),
        "session_phase_inference": DerivedModelStatus(
            name="session_phase_inference",
            ready=bool(session_status["phase_inference_rows_ready"]),
            detail=(
                f"Session phase inference ready ({phase_rows:,}/{expected_phase_rows:,} rows)"
                if bool(session_status["phase_inference_rows_ready"])
                else f"Session phase inference pending ({phase_rows:,}/{expected_phase_rows:,} rows)"
            ),
            source_documents=profile_rows,
            materialized_documents=profile_rows if profile_rows else 0,
            source_rows=expected_phase_rows,
            materialized_rows=phase_rows,
            pending_rows=_pending_rows(expected_phase_rows, phase_rows),
            stale_rows=int(session_status["stale_phase_inference_count"]),
            orphan_rows=int(session_status["orphan_phase_inference_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_status["stale_phase_inference_count"]) == 0
                and int(session_status["orphan_phase_inference_count"]) == 0
            ),
        ),
        "work_threads": DerivedModelStatus(
            name="work_threads",
            ready=bool(session_status["threads_ready"]),
            detail=(
                f"Work threads ready ({work_thread_rows:,}/{total_thread_roots:,} roots)"
                if bool(session_status["threads_ready"])
                else f"Work threads pending ({work_thread_rows:,}/{total_thread_roots:,} roots)"
            ),
            source_documents=total_thread_roots,
            materialized_documents=work_thread_rows,
            pending_documents=_pending_docs(total_thread_roots, work_thread_rows),
            stale_rows=int(session_status["stale_thread_count"]),
            orphan_rows=int(session_status["orphan_thread_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_status["stale_thread_count"]) == 0
                and int(session_status["orphan_thread_count"]) == 0
            ),
        ),
        "work_threads_fts": DerivedModelStatus(
            name="work_threads_fts",
            ready=bool(session_status["threads_fts_ready"]),
            detail=(
                f"Work-thread FTS ready ({work_thread_fts_rows:,}/{work_thread_rows:,} rows)"
                if bool(session_status["threads_fts_ready"])
                else (
                    f"Work-thread FTS pending ({work_thread_fts_rows:,}/{work_thread_rows:,} rows, "
                    f"duplicates {work_thread_fts_duplicates:,})"
                )
            ),
            source_rows=work_thread_rows,
            materialized_rows=work_thread_fts_rows,
            pending_rows=_pending_rows(work_thread_rows, work_thread_fts_rows),
            stale_rows=work_thread_fts_duplicates,
        ),
        "session_tag_rollups": DerivedModelStatus(
            name="session_tag_rollups",
            ready=bool(session_status["tag_rollups_ready"]),
            detail=(
                f"Session tag rollups ready ({tag_rollup_rows:,}/{expected_tag_rollup_rows:,} rows)"
                if bool(session_status["tag_rollups_ready"])
                else f"Session tag rollups pending ({tag_rollup_rows:,}/{expected_tag_rollup_rows:,} rows)"
            ),
            source_rows=expected_tag_rollup_rows,
            materialized_rows=tag_rollup_rows,
            pending_rows=_pending_rows(expected_tag_rollup_rows, tag_rollup_rows),
            stale_rows=int(session_status["stale_tag_rollup_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_status["stale_tag_rollup_count"]) == 0,
        ),
        "day_session_summaries": DerivedModelStatus(
            name="day_session_summaries",
            ready=bool(session_status["day_summaries_ready"]),
            detail=(
                f"Day session summaries ready ({day_summary_rows:,}/{expected_day_summary_rows:,} rows)"
                if bool(session_status["day_summaries_ready"])
                else f"Day session summaries pending ({day_summary_rows:,}/{expected_day_summary_rows:,} rows)"
            ),
            source_rows=expected_day_summary_rows,
            materialized_rows=day_summary_rows,
            pending_rows=_pending_rows(expected_day_summary_rows, day_summary_rows),
            stale_rows=int(session_status["stale_day_summary_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_status["stale_day_summary_count"]) == 0,
        ),
        "week_session_summaries": DerivedModelStatus(
            name="week_session_summaries",
            ready=bool(session_status["week_summaries_ready"]),
            detail=(
                "Week session summaries ready (derived from day-session summaries)"
                if bool(session_status["week_summaries_ready"])
                else "Week session summaries pending (day-session summaries not ready)"
            ),
            source_rows=expected_day_summary_rows,
            materialized_rows=day_summary_rows,
            pending_rows=_pending_rows(expected_day_summary_rows, day_summary_rows),
            stale_rows=int(session_status["stale_day_summary_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_status["stale_day_summary_count"]) == 0,
        ),
        "transcript_embeddings": DerivedModelStatus(
            name="transcript_embeddings",
            ready=transcript_embeddings_ready,
            detail=(
                f"Transcript embeddings ready ({embedded_conversations:,}/{total_conversations:,} conversations, {embedded_messages:,} messages)"
                if transcript_embeddings_ready
                else (
                    f"Transcript embeddings pending ({embedded_conversations:,}/{total_conversations:,} conversations, "
                    f"pending {pending_conversations:,}, stale {stale_messages:,}, missing provenance {missing_provenance:,})"
                )
            ),
            source_documents=total_conversations,
            materialized_documents=embedded_conversations,
            materialized_rows=embedded_messages,
            pending_documents=pending_conversations,
            stale_rows=stale_messages,
            missing_provenance_rows=missing_provenance,
        ),
        "retrieval_evidence": DerivedModelStatus(
            name="retrieval_evidence",
            ready=evidence_retrieval_ready,
            detail=(
                f"Evidence retrieval ready ({evidence_retrieval_rows:,}/{expected_evidence_retrieval_rows:,} supporting rows)"
                if evidence_retrieval_ready
                else (
                    f"Evidence retrieval pending ({evidence_retrieval_rows:,}/{expected_evidence_retrieval_rows:,} supporting rows; "
                    f"profile_evidence_fts={profile_evidence_fts_rows:,}/{profile_rows:,}, "
                    f"action_event_fts={action_fts_rows:,}/{action_rows:,})"
                )
            ),
            source_documents=action_source_documents + total_conversations,
            materialized_documents=action_documents + profile_rows,
            source_rows=expected_evidence_retrieval_rows,
            materialized_rows=evidence_retrieval_rows,
            pending_documents=_pending_docs(action_source_documents, action_documents) + _pending_docs(total_conversations, profile_rows),
            pending_rows=_pending_rows(expected_evidence_retrieval_rows, evidence_retrieval_rows),
            stale_rows=profile_evidence_fts_duplicates + int(action_status["stale_count"]),
            orphan_rows=action_orphan_rows + int(session_status["orphan_profile_row_count"]),
        ),
        "retrieval_inference": DerivedModelStatus(
            name="retrieval_inference",
            ready=inference_retrieval_ready,
            detail=(
                f"Inference retrieval ready ({inference_retrieval_rows:,}/{expected_inference_retrieval_rows:,} supporting rows)"
                if inference_retrieval_ready
                else (
                    f"Inference retrieval pending ({inference_retrieval_rows:,}/{expected_inference_retrieval_rows:,} supporting rows; "
                    f"profile_inference_fts={profile_inference_fts_rows:,}/{profile_rows:,}, "
                    f"work_event_fts={work_event_fts_rows:,}/{work_event_rows:,}, "
                    f"phases={phase_rows:,}/{expected_phase_rows:,})"
                )
            ),
            source_documents=profile_rows,
            materialized_documents=profile_rows,
            source_rows=expected_inference_retrieval_rows,
            materialized_rows=inference_retrieval_rows,
            pending_rows=_pending_rows(expected_inference_retrieval_rows, inference_retrieval_rows),
            stale_rows=(
                profile_inference_fts_duplicates
                + work_event_fts_duplicates
                + int(session_status["stale_work_event_inference_count"])
                + int(session_status["stale_phase_inference_count"])
            ),
            orphan_rows=(
                int(session_status["orphan_profile_row_count"])
                + int(session_status["orphan_work_event_inference_count"])
                + int(session_status["orphan_phase_inference_count"])
            ),
        ),
    }


__all__ = ["collect_derived_model_statuses_sync"]
