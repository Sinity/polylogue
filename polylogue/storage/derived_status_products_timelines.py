"""Timeline/work-product derived-model builders."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_docs, pending_rows
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION


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
    }


__all__ = ["build_timeline_statuses"]
