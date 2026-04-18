"""Session-profile row builders, payloads, search-text, and hydration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from polylogue.lib.phase_extraction import SessionPhase
from polylogue.lib.session_profile import SessionAnalysis, SessionProfile
from polylogue.lib.work_event_extraction import WorkEvent
from polylogue.storage.store import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionProfileRecord,
)
from polylogue.types import ConversationId

# ---------------------------------------------------------------------------
# Search-text builders
# ---------------------------------------------------------------------------


def profile_evidence_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.repo_paths,
        *profile.cwd_paths,
        *profile.file_paths_touched,
        *profile.tags,
        *profile.branch_names,
        *profile.languages_detected,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def profile_inference_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.repo_names,
        *profile.auto_tags,
        *(event.summary for event in profile.work_events),
        *(event.kind.value for event in profile.work_events),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def profile_search_text(profile: SessionProfile) -> str:
    return (
        " \n".join(
            part
            for part in (
                profile_evidence_search_text(profile),
                profile_inference_search_text(profile),
            )
            if part
        )
        or profile.conversation_id
    )


def _payload_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def profile_enrichment_search_text(profile: SessionProfile, enrichment_payload: dict[str, object]) -> str:
    blockers = _payload_strings(enrichment_payload.get("blockers"))
    support_signals_list = _payload_strings(enrichment_payload.get("support_signals"))
    parts = [
        profile.provider,
        profile.title or "",
        str(enrichment_payload.get("refined_work_kind") or ""),
        str(enrichment_payload.get("intent_summary") or ""),
        str(enrichment_payload.get("outcome_summary") or ""),
        *profile.repo_names,
        *profile.repo_paths,
        *blockers,
        *support_signals_list,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def session_enrichment_payload(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> dict[str, object]:
    text_bands = _collect_enrichment_text_bands(analysis)
    user_turns = text_bands.user_turns
    assistant_turns = text_bands.assistant_turns
    blockers_val = text_bands.blockers
    support_signals_val = enrichment_support_signals(profile, analysis, text_bands=text_bands)
    input_band_summary = {
        "user_turns": len(user_turns),
        "assistant_turns": len(assistant_turns),
        "action_events": len(analysis.facts.action_events) if analysis is not None else 0,
        "touched_paths": len(profile.file_paths_touched),
        "repo_names": len(profile.repo_names),
    }
    confidence = min(
        0.95,
        0.2
        + (0.2 if user_turns else 0.0)
        + (0.15 if analysis is not None and analysis.facts.action_events else 0.0)
        + (0.15 if profile.file_paths_touched else 0.0)
        + (0.15 if profile.repo_names else 0.0)
        + (0.1 if profile.work_events else 0.0)
        + (0.05 if blockers_val else 0.0),
    )
    intent_summary = user_turns[0] if user_turns else (profile.title or None)
    outcome_summary = assistant_turns[-1] if assistant_turns else (user_turns[-1] if user_turns else None)
    return {
        "intent_summary": intent_summary,
        "outcome_summary": outcome_summary,
        "blockers": list(blockers_val),
        "confidence": round(confidence, 3),
        "support_level": support_level(confidence, support_signals=support_signals_val),
        "support_signals": list(support_signals_val),
        "input_band_summary": input_band_summary,
    }


def profile_evidence_payload(profile: SessionProfile) -> dict[str, object]:
    return {
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        "first_message_at": profile.first_message_at.isoformat() if profile.first_message_at else None,
        "last_message_at": profile.last_message_at.isoformat() if profile.last_message_at else None,
        "canonical_session_date": (
            profile.canonical_session_date.isoformat() if profile.canonical_session_date else None
        ),
        "message_count": profile.message_count,
        "substantive_count": profile.substantive_count,
        "attachment_count": profile.attachment_count,
        "tool_use_count": profile.tool_use_count,
        "thinking_count": profile.thinking_count,
        "word_count": profile.word_count,
        "total_cost_usd": profile.total_cost_usd,
        "total_duration_ms": profile.total_duration_ms,
        "wall_duration_ms": profile.wall_duration_ms,
        "cost_is_estimated": profile.cost_is_estimated,
        "compaction_count": profile.compaction_count,
        "has_compaction": profile.compaction_count > 0,
        "tool_categories": dict(profile.tool_categories),
        "repo_paths": list(profile.repo_paths),
        "cwd_paths": list(profile.cwd_paths),
        "branch_names": list(profile.branch_names),
        "file_paths_touched": list(profile.file_paths_touched),
        "languages_detected": list(profile.languages_detected),
        "tags": list(profile.tags),
        "is_continuation": profile.is_continuation,
        "parent_id": profile.parent_id,
    }


def profile_inference_payload(profile: SessionProfile) -> dict[str, object]:
    signals = profile_support_signals(profile)
    return {
        "repo_names": list(profile.repo_names),
        "work_event_count": len(profile.work_events),
        "phase_count": len(profile.phases),
        "engaged_duration_ms": profile.engaged_duration_ms,
        "engaged_minutes": round(profile.engaged_duration_ms / 60_000.0, 4),
        "support_level": profile_support_level(profile),
        "support_signals": list(signals),
        "engaged_duration_source": engaged_duration_source(profile),
        "repo_inference_strength": repo_inference_strength(profile),
        "auto_tags": list(profile.auto_tags),
        "work_events": [event.to_dict() for event in profile.work_events],
        "phases": [
            {
                "start_time": phase.start_time.isoformat() if phase.start_time else None,
                "end_time": phase.end_time.isoformat() if phase.end_time else None,
                "canonical_session_date": (
                    phase.canonical_session_date.isoformat() if phase.canonical_session_date else None
                ),
                "message_range": list(phase.message_range),
                "duration_ms": phase.duration_ms,
                "tool_counts": dict(phase.tool_counts),
                "word_count": phase.word_count,
                "confidence": phase.confidence,
                "evidence": list(phase.evidence),
            }
            for phase in profile.phases
        ],
    }


# ---------------------------------------------------------------------------
# Build and hydrate session-profile storage rows
# ---------------------------------------------------------------------------


def build_session_profile_record(
    profile: SessionProfile,
    *,
    analysis: SessionAnalysis | None = None,
    materialized_at: str | None = None,
) -> SessionProfileRecord:
    built_at = materialized_at or now_iso()
    evidence = profile_evidence_payload(profile)
    inference = profile_inference_payload(profile)
    enrichment = session_enrichment_payload(profile, analysis)
    evidence_search_text = profile_evidence_search_text(profile)
    inference_search_text = profile_inference_search_text(profile)
    return SessionProfileRecord(
        conversation_id=ConversationId(profile.conversation_id),
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
        provider_name=profile.provider,
        title=profile.title,
        first_message_at=profile.first_message_at.isoformat() if profile.first_message_at else None,
        last_message_at=profile.last_message_at.isoformat() if profile.last_message_at else None,
        repo_paths=profile.repo_paths,
        repo_names=profile.repo_names,
        tags=profile.tags,
        auto_tags=profile.auto_tags,
        message_count=profile.message_count,
        substantive_count=profile.substantive_count,
        attachment_count=profile.attachment_count,
        work_event_count=len(profile.work_events),
        phase_count=len(profile.phases),
        word_count=profile.word_count,
        tool_use_count=profile.tool_use_count,
        thinking_count=profile.thinking_count,
        total_cost_usd=profile.total_cost_usd,
        total_duration_ms=profile.total_duration_ms,
        engaged_duration_ms=profile.engaged_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        cost_is_estimated=profile.cost_is_estimated,
        canonical_session_date=(profile.canonical_session_date.isoformat() if profile.canonical_session_date else None),
        evidence_payload=evidence,
        inference_payload=inference,
        enrichment_payload=enrichment,
        search_text=" \n".join(part for part in (evidence_search_text, inference_search_text) if part)
        or profile.conversation_id,
        evidence_search_text=evidence_search_text,
        inference_search_text=inference_search_text,
        enrichment_search_text=profile_enrichment_search_text(profile, enrichment),
        enrichment_version=SESSION_ENRICHMENT_VERSION,
        enrichment_family=SESSION_ENRICHMENT_FAMILY,
        inference_version=SESSION_INFERENCE_VERSION,
        inference_family=SESSION_INFERENCE_FAMILY,
    )


def hydrate_session_profile(record: SessionProfileRecord) -> SessionProfile:
    merged_payload = {
        **record.evidence_payload,
        **record.inference_payload,
        "conversation_id": str(record.conversation_id),
        "provider": record.provider_name,
        "title": record.title,
        "first_message_at": record.first_message_at,
        "last_message_at": record.last_message_at,
        "canonical_session_date": record.canonical_session_date,
        "repo_paths": list(record.repo_paths),
        "repo_names": list(record.repo_names),
        "tags": list(record.tags),
        "auto_tags": list(record.auto_tags),
        "message_count": record.message_count,
        "substantive_count": record.substantive_count,
        "attachment_count": record.attachment_count,
        "work_event_count": record.work_event_count,
        "phase_count": record.phase_count,
        "word_count": record.word_count,
        "tool_use_count": record.tool_use_count,
        "thinking_count": record.thinking_count,
        "total_cost_usd": record.total_cost_usd,
        "total_duration_ms": record.total_duration_ms,
        "engaged_duration_ms": record.engaged_duration_ms,
        "wall_duration_ms": record.wall_duration_ms,
        "cost_is_estimated": record.cost_is_estimated,
    }
    return SessionProfile.from_dict(merged_payload)


# ---------------------------------------------------------------------------
# Signal/support helpers (merged from session_product_row_signal_support)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def event_summary(event: WorkEvent) -> str:
    summary = str(event.summary or "").strip()
    return summary or event.kind.value


def support_level(confidence: float, *, support_signals: tuple[str, ...], fallback: bool = False) -> str:
    if fallback or confidence < 0.55 or not support_signals:
        return "weak"
    if confidence >= 0.78 and len(support_signals) >= 2:
        return "strong"
    return "moderate"


def event_support_signals(event: WorkEvent) -> tuple[str, ...]:
    signals = [str(item) for item in event.evidence if str(item).strip()]
    if event.file_paths:
        signals.append("touched_paths")
    if event.tools_used:
        signals.append("tool_calls")
    if event.start_time and event.end_time:
        signals.append("timestamped_range")
    return tuple(dict.fromkeys(signals))


def event_fallback(event: WorkEvent) -> bool:
    weak_markers = {"weak_signal", "no_tools", "shell_default"}
    return not event.evidence or bool(set(event.evidence) & weak_markers)


def phase_support_signals(phase: SessionPhase) -> tuple[str, ...]:
    signals = [str(item) for item in phase.evidence if str(item).strip()]
    if phase.tool_counts:
        signals.append("tool_counts")
    if phase.start_time and phase.end_time:
        signals.append("timestamped_range")
    if phase.word_count > 0:
        signals.append("word_count")
    return tuple(dict.fromkeys(signals))


def phase_fallback(phase: SessionPhase) -> bool:
    return not phase.tool_counts


def engaged_duration_source(profile: SessionProfile) -> str:
    return "phase_sum" if any(int(phase.duration_ms or 0) > 0 for phase in profile.phases) else "session_total_fallback"


def repo_inference_strength(profile: SessionProfile) -> str:
    if profile.repo_paths and profile.repo_names:
        return "strong"
    if profile.repo_names and (profile.file_paths_touched or profile.cwd_paths):
        return "moderate"
    if profile.repo_names:
        return "weak"
    return "none"


def profile_support_signals(profile: SessionProfile) -> tuple[str, ...]:
    signals: list[str] = []
    if profile.repo_paths:
        signals.append("repo_paths")
    if profile.file_paths_touched:
        signals.append("touched_paths")
    if profile.repo_names:
        signals.append("repo_names")
    if profile.work_events:
        signals.append("work_events")
    if profile.phases:
        signals.append("phases")
    if engaged_duration_source(profile) == "phase_sum":
        signals.append("phase_duration_sum")
    return tuple(signals)


def profile_support_level(profile: SessionProfile) -> str:
    signals = profile_support_signals(profile)
    work_confidence = max((float(event.confidence or 0.0) for event in profile.work_events), default=0.0)
    phase_confidence = max((float(phase.confidence or 0.0) for phase in profile.phases), default=0.0)
    confidence = max(work_confidence, phase_confidence)
    fallback = engaged_duration_source(profile) != "phase_sum" and not profile.work_events and not profile.phases
    return support_level(confidence, support_signals=signals, fallback=fallback)


# ---------------------------------------------------------------------------
# Text extraction helpers (merged from session_product_row_text_support)
# ---------------------------------------------------------------------------


def dedupe_texts(values: list[str], *, limit: int | None = None, width: int = 180) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = " ".join(str(value or "").split()).strip()
        if not candidate:
            continue
        shortened = candidate[:width]
        if shortened in seen:
            continue
        seen.add(shortened)
        items.append(shortened)
        if limit is not None and len(items) >= limit:
            break
    return tuple(items)


_BLOCKER_MARKERS = (
    "error",
    "failed",
    "failure",
    "blocked",
    "cannot",
    "can't",
    "exception",
    "traceback",
    "panic",
)


@dataclass(frozen=True)
class _EnrichmentTextBands:
    user_turns: tuple[str, ...] = ()
    assistant_turns: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


def _collect_enrichment_text_bands(analysis: SessionAnalysis | None) -> _EnrichmentTextBands:
    if analysis is None:
        return _EnrichmentTextBands()

    user_texts: list[str] = []
    assistant_texts: list[str] = []
    blocker_candidates: list[str] = []
    for message in analysis.facts.message_facts:
        text = str(message.text or "").strip()
        if not text:
            continue
        if message.is_user and not message.is_context_dump:
            user_texts.append(message.text)
            text_lower = message.text.lower()
            if any(marker in text_lower for marker in _BLOCKER_MARKERS):
                blocker_candidates.append(message.text)
        if message.is_assistant and message.is_substantive:
            assistant_texts.append(message.text)

    return _EnrichmentTextBands(
        user_turns=dedupe_texts(user_texts, width=220),
        assistant_turns=dedupe_texts(assistant_texts, width=220),
        blockers=dedupe_texts(blocker_candidates, limit=4, width=140),
    )


def user_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    return _collect_enrichment_text_bands(analysis).user_turns


def assistant_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    return _collect_enrichment_text_bands(analysis).assistant_turns


def blocker_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    return _collect_enrichment_text_bands(analysis).blockers


def enrichment_support_signals(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
    *,
    text_bands: _EnrichmentTextBands | None = None,
) -> tuple[str, ...]:
    signals: list[str] = []
    bands = text_bands or _collect_enrichment_text_bands(analysis)
    if bands.user_turns:
        signals.append("user_turns")
    if analysis is not None and analysis.facts.action_events:
        signals.append("action_events")
    if profile.file_paths_touched:
        signals.append("touched_paths")
    if profile.repo_names:
        signals.append("repo_names")
    if profile.work_events:
        signals.append("heuristic_work_events")
    if bands.assistant_turns:
        signals.append("assistant_outcome_text")
    return tuple(signals)


__all__ = [
    "assistant_turn_texts",
    "blocker_texts",
    "build_session_profile_record",
    "dedupe_texts",
    "engaged_duration_source",
    "enrichment_support_signals",
    "event_fallback",
    "event_summary",
    "event_support_signals",
    "hydrate_session_profile",
    "now_iso",
    "phase_fallback",
    "phase_support_signals",
    "profile_enrichment_search_text",
    "profile_evidence_payload",
    "profile_evidence_search_text",
    "profile_inference_payload",
    "profile_inference_search_text",
    "profile_search_text",
    "profile_support_level",
    "profile_support_signals",
    "repo_inference_strength",
    "session_enrichment_payload",
    "support_level",
    "user_turn_texts",
]
