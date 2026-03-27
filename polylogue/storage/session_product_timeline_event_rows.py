"""Work-event row builders and hydration helpers."""

from __future__ import annotations

from polylogue.lib.hashing import hash_text
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.work_events import WorkEvent
from polylogue.storage.session_product_row_signal_support import (
    event_fallback,
    event_summary,
    event_support_signals,
    now_iso,
    support_level,
)
from polylogue.storage.store import (
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionWorkEventRecord,
)


def event_id(conversation_id: str, event_index: int, event: WorkEvent) -> str:
    seed = (
        f"{conversation_id}:{event_index}:{event.kind.value}:{event.start_index}:"
        f"{event.end_index}:{event_summary(event)}"
    )
    return f"wev-{hash_text(seed)[:16]}"


def event_evidence_payload(event: WorkEvent) -> dict[str, object]:
    return {
        "start_index": event.start_index,
        "end_index": event.end_index,
        "start_time": event.start_time.isoformat() if event.start_time else None,
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "canonical_session_date": (
            event.canonical_session_date.isoformat()
            if event.canonical_session_date
            else None
        ),
        "duration_ms": event.duration_ms,
        "file_paths": list(event.file_paths),
        "tools_used": list(event.tools_used),
    }


def event_inference_payload(event: WorkEvent) -> dict[str, object]:
    signals = event_support_signals(event)
    fallback = event_fallback(event)
    return {
        "kind": event.kind.value,
        "summary": event_summary(event),
        "confidence": event.confidence,
        "evidence": list(event.evidence),
        "support_level": support_level(
            float(event.confidence or 0.0),
            support_signals=signals,
            fallback=fallback,
        ),
        "support_signals": list(signals),
        "fallback_inference": fallback,
    }


def event_search_text(profile: SessionProfile, event: WorkEvent) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        event.kind.value,
        event_summary(event),
        *profile.canonical_projects,
        *event.file_paths,
        *event.tools_used,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or f"{profile.conversation_id}:{event.kind.value}"


def build_session_work_event_records(
    profile: SessionProfile,
    *,
    materialized_at: str | None = None,
) -> list[SessionWorkEventRecord]:
    built_at = materialized_at or now_iso()
    return [
        SessionWorkEventRecord(
            event_id=event_id(profile.conversation_id, index, event),
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
            start_time=event.start_time.isoformat() if event.start_time else None,
            end_time=event.end_time.isoformat() if event.end_time else None,
            duration_ms=event.duration_ms,
            canonical_session_date=(
                event.canonical_session_date.isoformat()
                if event.canonical_session_date
                else None
            ),
            summary=event_summary(event),
            file_paths=event.file_paths,
            tools_used=event.tools_used,
            evidence_payload=event_evidence_payload(event),
            inference_payload=event_inference_payload(event),
            search_text=event_search_text(profile, event),
            inference_version=SESSION_INFERENCE_VERSION,
            inference_family=SESSION_INFERENCE_FAMILY,
        )
        for index, event in enumerate(profile.work_events)
    ]


def hydrate_work_event(record: SessionWorkEventRecord) -> WorkEvent:
    return WorkEvent.from_dict({**record.evidence_payload, **record.inference_payload})


__all__ = [
    "build_session_work_event_records",
    "event_evidence_payload",
    "event_id",
    "event_inference_payload",
    "event_search_text",
    "hydrate_work_event",
]
