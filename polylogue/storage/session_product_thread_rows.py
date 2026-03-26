"""Work-thread row builders for session products."""

from __future__ import annotations

from polylogue.lib.threads import WorkThread
from polylogue.storage.session_product_row_support import now_iso
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION, WorkThreadRecord


def thread_search_text(thread: WorkThread) -> str:
    parts = [
        thread.thread_id,
        thread.root_id,
        thread.dominant_project or "",
        *thread.session_ids,
        *thread.work_event_breakdown.keys(),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or thread.thread_id


def build_work_thread_record(
    thread: WorkThread,
    *,
    materialized_at: str | None = None,
) -> WorkThreadRecord:
    built_at = materialized_at or now_iso()
    return WorkThreadRecord(
        thread_id=thread.thread_id,
        root_id=thread.root_id,
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        start_time=thread.start_time.isoformat() if thread.start_time else None,
        end_time=thread.end_time.isoformat() if thread.end_time else None,
        dominant_project=thread.dominant_project,
        session_ids=thread.session_ids,
        session_count=len(thread.session_ids),
        depth=thread.depth,
        branch_count=thread.branch_count,
        total_messages=thread.total_messages,
        total_cost_usd=thread.total_cost_usd,
        wall_duration_ms=thread.wall_duration_ms,
        work_event_breakdown=thread.work_event_breakdown,
        payload=thread.to_dict(),
        search_text=thread_search_text(thread),
    )


def hydrate_work_thread(record: WorkThreadRecord) -> WorkThread:
    return WorkThread.from_dict(record.payload)


__all__ = [
    "build_work_thread_record",
    "hydrate_work_thread",
    "thread_search_text",
]
