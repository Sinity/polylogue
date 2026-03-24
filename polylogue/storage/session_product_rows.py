"""Row builders and hydrators for durable session-product read models."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from polylogue.lib.hashing import hash_text
from polylogue.lib.session_profile import SessionProfile, build_session_analysis, build_session_profile
from polylogue.lib.threads import WorkThread
from polylogue.lib.work_events import WorkEvent
from polylogue.storage.store import (
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _event_id(conversation_id: str, event_index: int, event: WorkEvent) -> str:
    seed = f"{conversation_id}:{event_index}:{event.kind.value}:{event.start_index}:{event.end_index}:{event.summary}"
    return f"wev-{hash_text(seed)[:16]}"


def _primary_work_kind(profile: SessionProfile) -> str | None:
    if not profile.work_events:
        return None
    counts = Counter(event.kind.value for event in profile.work_events)
    return counts.most_common(1)[0][0]


def _profile_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.canonical_projects,
        *profile.repo_paths,
        *profile.file_paths_touched,
        *profile.tags,
        *profile.auto_tags,
        *(event.summary for event in profile.work_events),
        *(event.kind.value for event in profile.work_events),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def _event_search_text(profile: SessionProfile, event: WorkEvent) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        event.kind.value,
        event.summary,
        *profile.canonical_projects,
        *event.file_paths,
        *event.tools_used,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or f"{profile.conversation_id}:{event.kind.value}"


def _thread_search_text(thread: WorkThread) -> str:
    parts = [
        thread.thread_id,
        thread.root_id,
        thread.dominant_project or "",
        *thread.session_ids,
        *thread.work_event_breakdown.keys(),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or thread.thread_id


def build_session_profile_record(
    profile: SessionProfile,
    *,
    materialized_at: str | None = None,
) -> SessionProfileRecord:
    built_at = materialized_at or _now_iso()
    return SessionProfileRecord(
        conversation_id=profile.conversation_id,
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
        provider_name=profile.provider,
        title=profile.title,
        first_message_at=profile.first_message_at.isoformat() if profile.first_message_at else None,
        last_message_at=profile.last_message_at.isoformat() if profile.last_message_at else None,
        primary_work_kind=_primary_work_kind(profile),
        repo_paths=profile.repo_paths,
        canonical_projects=profile.canonical_projects,
        tags=profile.tags,
        auto_tags=profile.auto_tags,
        message_count=profile.message_count,
        work_event_count=len(profile.work_events),
        word_count=profile.word_count,
        tool_use_count=profile.tool_use_count,
        thinking_count=profile.thinking_count,
        total_cost_usd=profile.total_cost_usd,
        total_duration_ms=profile.total_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        payload=profile.to_dict(),
        search_text=_profile_search_text(profile),
    )


def build_session_work_event_records(
    profile: SessionProfile,
    *,
    materialized_at: str | None = None,
) -> list[SessionWorkEventRecord]:
    built_at = materialized_at or _now_iso()
    rows: list[SessionWorkEventRecord] = []
    for index, event in enumerate(profile.work_events):
        rows.append(
            SessionWorkEventRecord(
                event_id=_event_id(profile.conversation_id, index, event),
                conversation_id=profile.conversation_id,
                materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
                materialized_at=built_at,
                source_updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
                source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
                provider_name=profile.provider,
                event_index=index,
                kind=event.kind.value,
                confidence=event.confidence,
                start_index=event.start_index,
                end_index=event.end_index,
                summary=event.summary,
                file_paths=event.file_paths,
                tools_used=event.tools_used,
                payload=event.to_dict(),
                search_text=_event_search_text(profile, event),
            )
        )
    return rows


def build_work_thread_record(
    thread: WorkThread,
    *,
    materialized_at: str | None = None,
) -> WorkThreadRecord:
    built_at = materialized_at or _now_iso()
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
        search_text=_thread_search_text(thread),
    )


def hydrate_session_profile(record: SessionProfileRecord) -> SessionProfile:
    return SessionProfile.from_dict(record.payload)


def hydrate_work_event(record: SessionWorkEventRecord) -> WorkEvent:
    return WorkEvent.from_dict(record.payload)


def hydrate_work_thread(record: WorkThreadRecord) -> WorkThread:
    return WorkThread.from_dict(record.payload)


def build_session_product_records(conversation) -> tuple[SessionProfileRecord, list[SessionWorkEventRecord]]:
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)
    materialized_at = _now_iso()
    return (
        build_session_profile_record(profile, materialized_at=materialized_at),
        build_session_work_event_records(profile, materialized_at=materialized_at),
    )


__all__ = [
    "build_session_product_records",
    "build_session_profile_record",
    "build_session_work_event_records",
    "build_work_thread_record",
    "hydrate_session_profile",
    "hydrate_work_event",
    "hydrate_work_thread",
]
