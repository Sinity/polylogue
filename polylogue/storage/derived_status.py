"""Shared derived-model readiness/freshness status helpers."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.action_event_lifecycle import action_event_read_model_status_sync
from polylogue.storage.embedding_stats import read_embedding_stats_sync
from polylogue.storage.fts_lifecycle import fts_index_status_sync
from polylogue.storage.session_product_lifecycle import session_product_status_sync
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION, SESSION_PRODUCT_MATERIALIZER_VERSION


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

    session_product_status = session_product_status_sync(conn)
    session_profile_count = int(session_product_status["profile_count"])
    session_profile_fts_count = int(session_product_status["profile_fts_count"])
    session_profile_fts_duplicate_count = int(session_product_status.get("profile_fts_duplicate_count", 0))
    expected_work_event_count = int(session_product_status["expected_work_event_count"])
    session_work_event_count = int(session_product_status["work_event_count"])
    session_work_event_fts_count = int(session_product_status["work_event_fts_count"])
    session_work_event_fts_duplicate_count = int(session_product_status.get("work_event_fts_duplicate_count", 0))
    work_thread_count = int(session_product_status["thread_count"])
    work_thread_fts_count = int(session_product_status["thread_fts_count"])
    work_thread_fts_duplicate_count = int(session_product_status.get("thread_fts_duplicate_count", 0))
    total_thread_roots = int(session_product_status["root_threads"])
    tag_rollup_count = int(session_product_status["tag_rollup_count"])
    expected_tag_rollup_count = int(session_product_status["expected_tag_rollup_count"])
    day_summary_count = int(session_product_status["day_summary_count"])
    expected_day_summary_count = int(session_product_status["expected_day_summary_count"])

    embedding_stats = read_embedding_stats_sync(conn)
    embedded_conversations = embedding_stats.embedded_conversations
    embedded_messages = embedding_stats.embedded_messages
    pending_conversations = embedding_stats.pending_conversations
    stale_messages = embedding_stats.stale_messages
    missing_provenance = embedding_stats.messages_missing_provenance
    embeddings_ready = (
        total_conversations == 0
        or (
            embedded_conversations == total_conversations
            and pending_conversations == 0
            and stale_messages == 0
            and missing_provenance == 0
        )
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
            pending_rows=max(0, total_messages - message_fts_rows),
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
            pending_documents=max(0, action_source_documents - action_documents),
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
            pending_rows=max(0, action_rows - action_fts_rows),
            orphan_rows=action_orphan_rows,
        ),
        "session_profiles": DerivedModelStatus(
            name="session_profiles",
            ready=bool(session_product_status["profiles_ready"]),
            detail=(
                f"Session profiles ready ({session_profile_count:,}/{total_conversations:,} conversations)"
                if bool(session_product_status["profiles_ready"])
                else f"Session profiles pending ({session_profile_count:,}/{total_conversations:,} conversations)"
            ),
            source_documents=total_conversations,
            materialized_documents=session_profile_count,
            pending_documents=max(0, int(session_product_status["missing_profile_count"])),
            stale_rows=int(session_product_status["stale_profile_count"]),
            orphan_rows=int(session_product_status["orphan_profile_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_product_status["stale_profile_count"]) == 0
                and int(session_product_status["orphan_profile_count"]) == 0
            ),
        ),
        "session_profiles_fts": DerivedModelStatus(
            name="session_profiles_fts",
            ready=bool(session_product_status["profiles_fts_ready"]),
            detail=(
                f"Session-profile FTS ready ({session_profile_fts_count:,}/{session_profile_count:,} rows)"
                if bool(session_product_status["profiles_fts_ready"])
                else (
                    f"Session-profile FTS pending ({session_profile_fts_count:,}/{session_profile_count:,} rows, "
                    f"duplicates {session_profile_fts_duplicate_count:,})"
                )
            ),
            source_rows=session_profile_count,
            materialized_rows=session_profile_fts_count,
            pending_rows=max(0, session_profile_count - session_profile_fts_count),
            stale_rows=session_profile_fts_duplicate_count,
        ),
        "session_work_events": DerivedModelStatus(
            name="session_work_events",
            ready=bool(session_product_status["work_events_ready"]),
            detail=(
                f"Session work events ready ({session_work_event_count:,}/{expected_work_event_count:,} rows)"
                if bool(session_product_status["work_events_ready"])
                else f"Session work events pending ({session_work_event_count:,}/{expected_work_event_count:,} rows)"
            ),
            source_documents=session_profile_count,
            materialized_documents=session_profile_count if session_profile_count else 0,
            source_rows=expected_work_event_count,
            materialized_rows=session_work_event_count,
            pending_rows=max(0, expected_work_event_count - session_work_event_count),
            stale_rows=int(session_product_status["stale_work_event_count"]),
            orphan_rows=int(session_product_status["orphan_work_event_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_product_status["stale_work_event_count"]) == 0
                and int(session_product_status["orphan_work_event_count"]) == 0
            ),
        ),
        "session_work_events_fts": DerivedModelStatus(
            name="session_work_events_fts",
            ready=bool(session_product_status["work_events_fts_ready"]),
            detail=(
                f"Session work-event FTS ready ({session_work_event_fts_count:,}/{session_work_event_count:,} rows)"
                if bool(session_product_status["work_events_fts_ready"])
                else (
                    f"Session work-event FTS pending ({session_work_event_fts_count:,}/{session_work_event_count:,} rows, "
                    f"duplicates {session_work_event_fts_duplicate_count:,})"
                )
            ),
            source_rows=session_work_event_count,
            materialized_rows=session_work_event_fts_count,
            pending_rows=max(0, session_work_event_count - session_work_event_fts_count),
            stale_rows=session_work_event_fts_duplicate_count,
        ),
        "work_threads": DerivedModelStatus(
            name="work_threads",
            ready=bool(session_product_status["threads_ready"]),
            detail=(
                f"Work threads ready ({work_thread_count:,}/{total_thread_roots:,} roots)"
                if bool(session_product_status["threads_ready"])
                else f"Work threads pending ({work_thread_count:,}/{total_thread_roots:,} roots)"
            ),
            source_documents=total_thread_roots,
            materialized_documents=work_thread_count,
            pending_documents=max(0, total_thread_roots - work_thread_count),
            stale_rows=int(session_product_status["stale_thread_count"]),
            orphan_rows=int(session_product_status["orphan_thread_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=(
                int(session_product_status["stale_thread_count"]) == 0
                and int(session_product_status["orphan_thread_count"]) == 0
            ),
        ),
        "work_threads_fts": DerivedModelStatus(
            name="work_threads_fts",
            ready=bool(session_product_status["threads_fts_ready"]),
            detail=(
                f"Work-thread FTS ready ({work_thread_fts_count:,}/{work_thread_count:,} rows)"
                if bool(session_product_status["threads_fts_ready"])
                else (
                    f"Work-thread FTS pending ({work_thread_fts_count:,}/{work_thread_count:,} rows, "
                    f"duplicates {work_thread_fts_duplicate_count:,})"
                )
            ),
            source_rows=work_thread_count,
            materialized_rows=work_thread_fts_count,
            pending_rows=max(0, work_thread_count - work_thread_fts_count),
            stale_rows=work_thread_fts_duplicate_count,
        ),
        "session_tag_rollups": DerivedModelStatus(
            name="session_tag_rollups",
            ready=bool(session_product_status["tag_rollups_ready"]),
            detail=(
                f"Session tag rollups ready ({tag_rollup_count:,}/{expected_tag_rollup_count:,} rows)"
                if bool(session_product_status["tag_rollups_ready"])
                else f"Session tag rollups pending ({tag_rollup_count:,}/{expected_tag_rollup_count:,} rows)"
            ),
            source_rows=expected_tag_rollup_count,
            materialized_rows=tag_rollup_count,
            pending_rows=max(0, expected_tag_rollup_count - tag_rollup_count),
            stale_rows=int(session_product_status["stale_tag_rollup_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_product_status["stale_tag_rollup_count"]) == 0,
        ),
        "day_session_summaries": DerivedModelStatus(
            name="day_session_summaries",
            ready=bool(session_product_status["day_summaries_ready"]),
            detail=(
                f"Day session summaries ready ({day_summary_count:,}/{expected_day_summary_count:,} rows)"
                if bool(session_product_status["day_summaries_ready"])
                else f"Day session summaries pending ({day_summary_count:,}/{expected_day_summary_count:,} rows)"
            ),
            source_rows=expected_day_summary_count,
            materialized_rows=day_summary_count,
            pending_rows=max(0, expected_day_summary_count - day_summary_count),
            stale_rows=int(session_product_status["stale_day_summary_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_product_status["stale_day_summary_count"]) == 0,
        ),
        "week_session_summaries": DerivedModelStatus(
            name="week_session_summaries",
            ready=bool(session_product_status["week_summaries_ready"]),
            detail=(
                "Week session summaries ready (derived from day-session summaries)"
                if bool(session_product_status["week_summaries_ready"])
                else "Week session summaries pending (day-session summaries not ready)"
            ),
            source_rows=expected_day_summary_count,
            materialized_rows=day_summary_count,
            pending_rows=max(0, expected_day_summary_count - day_summary_count),
            stale_rows=int(session_product_status["stale_day_summary_count"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=int(session_product_status["stale_day_summary_count"]) == 0,
        ),
        "embeddings": DerivedModelStatus(
            name="embeddings",
            ready=embeddings_ready,
            detail=(
                f"Embeddings ready ({embedded_conversations:,}/{total_conversations:,} conversations, {embedded_messages:,} messages)"
                if embeddings_ready
                else (
                    f"Embeddings pending ({embedded_conversations:,}/{total_conversations:,} conversations, "
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
    }


__all__ = ["collect_derived_model_statuses_sync"]
