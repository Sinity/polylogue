"""Row builders and hydrators for durable session-product read models."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from polylogue.lib.hashing import hash_text
from polylogue.lib.phases import SessionPhase
from polylogue.lib.session_profile import SessionProfile, build_session_analysis, build_session_profile
from polylogue.lib.threads import WorkThread
from polylogue.lib.work_events import WorkEvent
from polylogue.storage.store import (
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _event_id(conversation_id: str, event_index: int, event: WorkEvent) -> str:
    seed = (
        f"{conversation_id}:{event_index}:{event.kind.value}:{event.start_index}:"
        f"{event.end_index}:{_event_summary(event)}"
    )
    return f"wev-{hash_text(seed)[:16]}"


def _phase_id(conversation_id: str, phase_index: int, phase: SessionPhase) -> str:
    seed = (
        f"{conversation_id}:{phase_index}:{phase.kind}:{phase.message_range[0]}:"
        f"{phase.message_range[1]}:{phase.start_time.isoformat() if phase.start_time else ''}:"
        f"{phase.end_time.isoformat() if phase.end_time else ''}"
    )
    return f"sph-{hash_text(seed)[:16]}"


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
        _event_summary(event),
        *profile.canonical_projects,
        *event.file_paths,
        *event.tools_used,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or f"{profile.conversation_id}:{event.kind.value}"


def _event_summary(event: WorkEvent) -> str:
    summary = str(event.summary or "").strip()
    return summary or event.kind.value


def _phase_search_text(profile: SessionProfile, phase: SessionPhase) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        phase.kind,
        *profile.canonical_projects,
        *profile.repo_paths,
        *phase.tool_counts.keys(),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or f"{profile.conversation_id}:{phase.kind}"


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
        phase_count=len(profile.phases),
        word_count=profile.word_count,
        tool_use_count=profile.tool_use_count,
        thinking_count=profile.thinking_count,
        total_cost_usd=profile.total_cost_usd,
        total_duration_ms=profile.total_duration_ms,
        engaged_duration_ms=profile.engaged_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        canonical_session_date=(
            profile.canonical_session_date.isoformat()
            if profile.canonical_session_date
            else None
        ),
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
                summary=_event_summary(event),
                file_paths=event.file_paths,
                tools_used=event.tools_used,
                payload=event.to_dict(),
                search_text=_event_search_text(profile, event),
            )
        )
    return rows


def build_session_phase_records(
    profile: SessionProfile,
    *,
    materialized_at: str | None = None,
) -> list[SessionPhaseRecord]:
    built_at = materialized_at or _now_iso()
    rows: list[SessionPhaseRecord] = []
    for index, phase in enumerate(profile.phases):
        rows.append(
            SessionPhaseRecord(
                phase_id=_phase_id(profile.conversation_id, index, phase),
                conversation_id=profile.conversation_id,
                materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
                materialized_at=built_at,
                source_updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
                source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
                provider_name=profile.provider,
                phase_index=index,
                kind=phase.kind,
                start_index=phase.message_range[0],
                end_index=phase.message_range[1],
                start_time=phase.start_time.isoformat() if phase.start_time else None,
                end_time=phase.end_time.isoformat() if phase.end_time else None,
                duration_ms=phase.duration_ms,
                canonical_session_date=(
                    phase.canonical_session_date.isoformat()
                    if phase.canonical_session_date
                    else None
                ),
                tool_counts=phase.tool_counts,
                word_count=phase.word_count,
                payload={
                    "kind": phase.kind,
                    "start_time": phase.start_time.isoformat() if phase.start_time else None,
                    "end_time": phase.end_time.isoformat() if phase.end_time else None,
                    "canonical_session_date": (
                        phase.canonical_session_date.isoformat()
                        if phase.canonical_session_date
                        else None
                    ),
                    "message_range": list(phase.message_range),
                    "duration_ms": phase.duration_ms,
                    "tool_counts": phase.tool_counts,
                    "word_count": phase.word_count,
                },
                search_text=_phase_search_text(profile, phase),
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


def hydrate_session_phase(record: SessionPhaseRecord) -> SessionPhase:
    payload = record.payload
    return SessionPhase(
        kind=str(payload.get("kind") or record.kind),
        start_time=(
            datetime.fromisoformat(str(payload["start_time"]))
            if payload.get("start_time")
            else None
        ),
        end_time=(
            datetime.fromisoformat(str(payload["end_time"]))
            if payload.get("end_time")
            else None
        ),
        canonical_session_date=(
            datetime.fromisoformat(str(payload["canonical_session_date"])).date()
            if payload.get("canonical_session_date")
            else None
        ),
        message_range=tuple(int(v) for v in payload.get("message_range", [record.start_index, record.end_index])),
        duration_ms=int(payload.get("duration_ms", record.duration_ms) or 0),
        tool_counts={
            str(key): int(value or 0)
            for key, value in (payload.get("tool_counts", record.tool_counts) or {}).items()
        },
        word_count=int(payload.get("word_count", record.word_count) or 0),
    )


def hydrate_work_thread(record: WorkThreadRecord) -> WorkThread:
    return WorkThread.from_dict(record.payload)


def build_session_product_records(
    conversation,
) -> tuple[SessionProfileRecord, list[SessionWorkEventRecord], list[SessionPhaseRecord]]:
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)
    materialized_at = _now_iso()
    return (
        build_session_profile_record(profile, materialized_at=materialized_at),
        build_session_work_event_records(profile, materialized_at=materialized_at),
        build_session_phase_records(profile, materialized_at=materialized_at),
    )


__all__ = [
    "build_session_product_records",
    "build_session_phase_records",
    "build_session_profile_record",
    "build_session_work_event_records",
    "build_work_thread_record",
    "hydrate_session_phase",
    "hydrate_session_profile",
    "hydrate_work_event",
    "hydrate_work_thread",
]
