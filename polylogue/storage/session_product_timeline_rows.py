"""Work-event and session-phase row builders and hydration helpers."""

from __future__ import annotations

from datetime import datetime

from polylogue.archive_product_models import (
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.lib.hashing import hash_text
from polylogue.lib.phase_extraction import SessionPhase
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.work_event_extraction import WorkEvent
from polylogue.storage.session_product_profiles import (
    event_fallback,
    event_summary,
    event_support_signals,
    now_iso,
    phase_fallback,
    phase_support_signals,
    support_level,
)
from polylogue.storage.store import (
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionPhaseRecord,
    SessionWorkEventRecord,
)
from polylogue.types import ConversationId

# ---------------------------------------------------------------------------
# Work-event row builders and hydration
# ---------------------------------------------------------------------------


def event_id(conversation_id: str, event_index: int, event: WorkEvent) -> str:
    return _event_id(conversation_id, event_index, event, summary=event_summary(event))


def _event_id(
    conversation_id: str,
    event_index: int,
    event: WorkEvent,
    *,
    summary: str,
) -> str:
    seed = f"{conversation_id}:{event_index}:{event.kind.value}:{event.start_index}:{event.end_index}:{summary}"
    return f"wev-{hash_text(seed)[:16]}"


def event_evidence_payload(event: WorkEvent) -> WorkEventEvidencePayload:
    return WorkEventEvidencePayload(
        start_index=event.start_index,
        end_index=event.end_index,
        start_time=event.start_time.isoformat() if event.start_time else None,
        end_time=event.end_time.isoformat() if event.end_time else None,
        canonical_session_date=event.canonical_session_date.isoformat() if event.canonical_session_date else None,
        duration_ms=event.duration_ms,
        file_paths=event.file_paths,
        tools_used=event.tools_used,
    )


def event_inference_payload(event: WorkEvent) -> WorkEventInferencePayload:
    summary = event_summary(event)
    signals = event_support_signals(event)
    fallback = event_fallback(event)
    return _event_inference_payload(
        event,
        summary=summary,
        signals=signals,
        fallback=fallback,
    )


def _event_inference_payload(
    event: WorkEvent,
    *,
    summary: str,
    signals: tuple[str, ...],
    fallback: bool,
) -> WorkEventInferencePayload:
    return WorkEventInferencePayload(
        kind=event.kind.value,
        summary=summary,
        confidence=event.confidence,
        evidence=event.evidence,
        support_level=support_level(
            float(event.confidence or 0.0),
            support_signals=signals,
            fallback=fallback,
        ),
        support_signals=signals,
        fallback_inference=fallback,
    )


def event_search_text(profile: SessionProfile, event: WorkEvent) -> str:
    return _event_search_text(profile, event, summary=event_summary(event))


def _event_search_text(
    profile: SessionProfile,
    event: WorkEvent,
    *,
    summary: str,
) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        event.kind.value,
        summary,
        *profile.repo_names,
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
    source_updated_at = profile.updated_at.isoformat() if profile.updated_at else None
    source_sort_key = profile.updated_at.timestamp() if profile.updated_at else None
    records: list[SessionWorkEventRecord] = []
    for index, event in enumerate(profile.work_events):
        summary = event_summary(event)
        signals = event_support_signals(event)
        fallback = event_fallback(event)
        start_time = event.start_time.isoformat() if event.start_time else None
        end_time = event.end_time.isoformat() if event.end_time else None
        canonical_session_date = event.canonical_session_date.isoformat() if event.canonical_session_date else None
        records.append(
            SessionWorkEventRecord(
                event_id=_event_id(profile.conversation_id, index, event, summary=summary),
                conversation_id=ConversationId(profile.conversation_id),
                materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
                materialized_at=built_at,
                source_updated_at=source_updated_at,
                source_sort_key=source_sort_key,
                provider_name=profile.provider,
                event_index=index,
                kind=event.kind.value,
                confidence=event.confidence,
                start_index=event.start_index,
                end_index=event.end_index,
                start_time=start_time,
                end_time=end_time,
                duration_ms=event.duration_ms,
                canonical_session_date=canonical_session_date,
                summary=summary,
                file_paths=event.file_paths,
                tools_used=event.tools_used,
                evidence_payload=event_evidence_payload(event),
                inference_payload=_event_inference_payload(
                    event,
                    summary=summary,
                    signals=signals,
                    fallback=fallback,
                ),
                search_text=_event_search_text(profile, event, summary=summary),
                inference_version=SESSION_INFERENCE_VERSION,
                inference_family=SESSION_INFERENCE_FAMILY,
            )
        )
    return records


def hydrate_work_event(record: SessionWorkEventRecord) -> WorkEvent:
    return WorkEvent.from_dict(
        {
            **record.evidence_payload.model_dump(mode="json"),
            **record.inference_payload.model_dump(mode="json"),
        }
    )


# ---------------------------------------------------------------------------
# Session-phase row builders and hydration
# ---------------------------------------------------------------------------


def phase_id(conversation_id: str, phase_index: int, phase: SessionPhase) -> str:
    seed = (
        f"{conversation_id}:{phase_index}:{phase.message_range[0]}:"
        f"{phase.message_range[1]}:{phase.start_time.isoformat() if phase.start_time else ''}:"
        f"{phase.end_time.isoformat() if phase.end_time else ''}"
    )
    return f"sph-{hash_text(seed)[:16]}"


def phase_search_text(profile: SessionProfile, phase: SessionPhase) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.repo_names,
        *profile.repo_paths,
        *phase.tool_counts.keys(),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def phase_evidence_payload(phase: SessionPhase) -> SessionPhaseEvidencePayload:
    return SessionPhaseEvidencePayload(
        start_time=phase.start_time.isoformat() if phase.start_time else None,
        end_time=phase.end_time.isoformat() if phase.end_time else None,
        canonical_session_date=phase.canonical_session_date.isoformat() if phase.canonical_session_date else None,
        message_range=phase.message_range,
        duration_ms=phase.duration_ms,
        tool_counts=dict(phase.tool_counts),
        word_count=phase.word_count,
    )


def phase_inference_payload(phase: SessionPhase) -> SessionPhaseInferencePayload:
    signals = phase_support_signals(phase)
    fallback = phase_fallback(phase)
    return _phase_inference_payload(
        phase,
        signals=signals,
        fallback=fallback,
    )


def _phase_inference_payload(
    phase: SessionPhase,
    *,
    signals: tuple[str, ...],
    fallback: bool,
) -> SessionPhaseInferencePayload:
    return SessionPhaseInferencePayload(
        confidence=phase.confidence,
        evidence=phase.evidence,
        support_level=support_level(
            float(phase.confidence or 0.0),
            support_signals=signals,
            fallback=fallback,
        ),
        support_signals=signals,
        fallback_inference=fallback,
    )


def build_session_phase_records(
    profile: SessionProfile,
    *,
    materialized_at: str | None = None,
) -> list[SessionPhaseRecord]:
    built_at = materialized_at or now_iso()
    source_updated_at = profile.updated_at.isoformat() if profile.updated_at else None
    source_sort_key = profile.updated_at.timestamp() if profile.updated_at else None
    records: list[SessionPhaseRecord] = []
    for index, phase in enumerate(profile.phases):
        start_time = phase.start_time.isoformat() if phase.start_time else None
        end_time = phase.end_time.isoformat() if phase.end_time else None
        canonical_session_date = phase.canonical_session_date.isoformat() if phase.canonical_session_date else None
        signals = phase_support_signals(phase)
        fallback = phase_fallback(phase)
        records.append(
            SessionPhaseRecord(
                phase_id=phase_id(profile.conversation_id, index, phase),
                conversation_id=ConversationId(profile.conversation_id),
                materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
                materialized_at=built_at,
                source_updated_at=source_updated_at,
                source_sort_key=source_sort_key,
                provider_name=profile.provider,
                phase_index=index,
                kind="phase",
                start_index=phase.message_range[0],
                end_index=phase.message_range[1],
                start_time=start_time,
                end_time=end_time,
                duration_ms=phase.duration_ms,
                canonical_session_date=canonical_session_date,
                confidence=phase.confidence,
                evidence_reasons=phase.evidence,
                tool_counts=phase.tool_counts,
                word_count=phase.word_count,
                evidence_payload=phase_evidence_payload(phase),
                inference_payload=_phase_inference_payload(
                    phase,
                    signals=signals,
                    fallback=fallback,
                ),
                search_text=phase_search_text(profile, phase),
                inference_version=SESSION_INFERENCE_VERSION,
                inference_family=SESSION_INFERENCE_FAMILY,
            )
        )
    return records


def hydrate_session_phase(record: SessionPhaseRecord) -> SessionPhase:
    payload = {
        **record.evidence_payload.model_dump(mode="json"),
        **record.inference_payload.model_dump(mode="json"),
    }
    return SessionPhase(
        start_time=(datetime.fromisoformat(str(payload["start_time"])) if payload.get("start_time") else None),
        end_time=(datetime.fromisoformat(str(payload["end_time"])) if payload.get("end_time") else None),
        canonical_session_date=(
            datetime.fromisoformat(str(payload["canonical_session_date"])).date()
            if payload.get("canonical_session_date")
            else None
        ),
        message_range=(
            int(payload.get("message_range", [record.start_index, record.end_index])[0]),
            int(payload.get("message_range", [record.start_index, record.end_index])[1]),
        ),
        duration_ms=int(payload.get("duration_ms", record.duration_ms) or 0),
        tool_counts={
            str(key): int(value or 0) for key, value in (payload.get("tool_counts", record.tool_counts) or {}).items()
        },
        word_count=int(payload.get("word_count", record.word_count) or 0),
        confidence=float(payload.get("confidence", record.confidence) or 0.0),
        evidence=tuple(
            str(value) for value in (payload.get("evidence", record.evidence_reasons) or record.evidence_reasons)
        ),
    )


__all__ = [
    "build_session_phase_records",
    "build_session_work_event_records",
    "event_evidence_payload",
    "event_id",
    "event_inference_payload",
    "event_search_text",
    "hydrate_session_phase",
    "hydrate_work_event",
    "phase_evidence_payload",
    "phase_id",
    "phase_inference_payload",
    "phase_search_text",
]
