"""Session-profile row builders, payloads, search-text, and hydration."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from datetime import UTC, datetime

from polylogue.archive.phase.extraction import SessionPhase
from polylogue.archive.session.documents import (
    SessionPhaseDocument,
    SessionProfileDocument,
    WorkEventDocument,
)
from polylogue.archive.session.extraction import WorkEvent, WorkEventPayload
from polylogue.archive.session.models import SessionPhasePayload
from polylogue.archive.session.session_profile import SessionAnalysis, SessionProfile
from polylogue.core.payload_coercion import (
    coerce_float,
    coerce_int,
    int_pair,
    optional_date,
    optional_datetime,
    string_int_mapping,
    string_sequence,
)
from polylogue.insights.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.insights.confidence import ConfidenceBand, from_signals
from polylogue.insights.fallback import FallbackReason
from polylogue.insights.temporal_source import classify_profile_hwm_source
from polylogue.storage.runtime import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
    SessionProfileRecord,
)
from polylogue.types import SessionId


def _serialize_percentiles(percentiles: dict[str, int]) -> str:
    return _json.dumps(percentiles, sort_keys=True) if percentiles else "{}"


# ---------------------------------------------------------------------------
# Search-text builders
# ---------------------------------------------------------------------------


def profile_evidence_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.origin,
        profile.title or "",
        *profile.repo_paths,
        *profile.cwd_paths,
        *profile.file_paths_touched,
        *profile.tags,
        *profile.branch_names,
        *profile.languages_detected,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.session_id


def profile_inference_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.origin,
        profile.title or "",
        profile.inferred_topic or "",
        profile.workflow_shape,
        profile.terminal_state,
        *profile.repo_names,
        *profile.auto_tags,
        *(event.summary for event in profile.work_events),
        *(event.heuristic_label.value for event in profile.work_events),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.session_id


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
        or profile.session_id
    )


def profile_enrichment_search_text(
    profile: SessionProfile,
    enrichment_payload: SessionEnrichmentPayload,
) -> str:
    blockers = enrichment_payload.blockers
    support_signals_list = enrichment_payload.support_signals
    parts = [
        profile.origin,
        profile.title or "",
        profile.inferred_topic or "",
        enrichment_payload.intent_summary or "",
        enrichment_payload.outcome_summary or "",
        enrichment_payload.goal_text or "",
        enrichment_payload.goal_outcome or "",
        *profile.repo_names,
        *profile.repo_paths,
        *blockers,
        *support_signals_list,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.session_id


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def session_enrichment_payload(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> SessionEnrichmentPayload:
    text_bands = _collect_enrichment_text_bands(analysis)
    user_turns = text_bands.user_turns
    assistant_turns = text_bands.assistant_turns
    blockers_val = text_bands.blockers if profile.terminal_state in _UNRESOLVED_BLOCKER_TERMINAL_STATES else ()
    support_signals_val = enrichment_support_signals(profile, analysis, text_bands=text_bands)
    input_band_summary = {
        "user_turns": len(user_turns),
        "assistant_turns": len(assistant_turns),
        "actions": len(analysis.facts.actions) if analysis is not None else 0,
        "touched_paths": len(profile.file_paths_touched),
        "repo_names": len(profile.repo_names),
    }
    confidence = min(
        0.95,
        0.2
        + (0.2 if user_turns else 0.0)
        + (0.15 if analysis is not None and analysis.facts.actions else 0.0)
        + (0.15 if profile.file_paths_touched else 0.0)
        + (0.15 if profile.repo_names else 0.0)
        + (0.1 if profile.work_events else 0.0)
        + (0.05 if blockers_val else 0.0),
    )
    from polylogue.archive.session.runtime import _clean_topic_text

    # Strip the Claude Code "Caveat: The messages below..." preamble and
    # similar known noise prefixes from the heuristic intent/outcome — they
    # otherwise contaminate the field with system text rather than the
    # user's actual ask. Same cleaner used for inferred_topic.
    raw_intent = user_turns[0] if user_turns else (profile.title or None)
    intent_summary = _clean_topic_text(raw_intent, width=220) if raw_intent else None
    raw_outcome = assistant_turns[-1] if assistant_turns else (user_turns[-1] if user_turns else None)
    outcome_summary = _clean_topic_text(raw_outcome, width=220) if raw_outcome else None
    fallback_reasons = enrichment_fallback_reasons(analysis, user_turns=user_turns)

    # #1687: detect /goal-driven autonomous sessions from first user message.
    import re

    _goal_re = re.compile(r"^/goal\s+", re.IGNORECASE)
    is_goal_session = False
    goal_text: str | None = None
    goal_outcome: str | None = None
    if raw_intent and _goal_re.search(raw_intent):
        is_goal_session = True
        goal_text = _clean_topic_text(raw_intent, width=500) if raw_intent else None
        # Classify boundary posture from the current terminal-state vocabulary.
        # These are not claims about whether the goal itself succeeded.
        match profile.terminal_state:
            case "clean_finish":
                goal_outcome = "ended_cleanly"
            case "error_left":
                goal_outcome = "ended_with_error"
            case "question_left":
                goal_outcome = "awaiting_user"
            case "tool_left":
                goal_outcome = "pending_tool"
            case "agent_hanging":
                goal_outcome = "inactive_pending"
            case _:
                goal_outcome = None

    return SessionEnrichmentPayload(
        intent_summary=intent_summary,
        outcome_summary=outcome_summary,
        blockers=blockers_val,
        confidence=round(confidence, 3),
        support_level=support_level(confidence, support_signals=support_signals_val),
        support_signals=support_signals_val,
        input_band_summary=input_band_summary,
        fallback_reasons=fallback_reasons,
        is_goal_session=is_goal_session,
        goal_text=goal_text,
        goal_outcome=goal_outcome,
    )


def enrichment_fallback_reasons(
    analysis: SessionAnalysis | None,
    *,
    user_turns: tuple[str, ...],
) -> tuple[FallbackReason, ...]:
    reasons: list[FallbackReason] = []
    if analysis is None:
        reasons.append(FallbackReason.MISSING_SESSION_ANALYSIS)
    if not user_turns:
        reasons.append(FallbackReason.NO_USER_TURNS)
    return tuple(reasons)


def profile_evidence_payload(profile: SessionProfile) -> SessionEvidencePayload:
    return SessionEvidencePayload(
        created_at=profile.created_at.isoformat() if profile.created_at else None,
        updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        first_message_at=profile.first_message_at.isoformat() if profile.first_message_at else None,
        last_message_at=profile.last_message_at.isoformat() if profile.last_message_at else None,
        session_timestamp=profile.first_message_at.isoformat() if profile.first_message_at else None,
        timestamp_source=profile.timestamp_source,
        timestamped_message_count=profile.timestamped_message_count,
        untimestamped_message_count=profile.untimestamped_message_count,
        timestamp_coverage=profile.timestamp_coverage,
        canonical_session_date=profile.canonical_session_date.isoformat() if profile.canonical_session_date else None,
        message_count=profile.message_count,
        substantive_count=profile.substantive_count,
        attachment_count=profile.attachment_count,
        tool_use_count=profile.tool_use_count,
        thinking_count=profile.thinking_count,
        word_count=profile.word_count,
        total_cost_usd=profile.total_cost_usd,
        total_duration_ms=profile.total_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        tool_active_duration_ms=profile.tool_active_duration_ms,
        workflow_shape_features=dict(profile.workflow_shape_features),
        terminal_state_evidence=dict(profile.terminal_state_evidence),
        cost_is_estimated=profile.cost_is_estimated,
        compaction_count=profile.compaction_count,
        has_compaction=profile.compaction_count > 0,
        tool_categories=dict(profile.tool_categories),
        repo_paths=profile.repo_paths,
        cwd_paths=profile.cwd_paths,
        branch_names=profile.branch_names,
        file_paths_touched=profile.file_paths_touched,
        languages_detected=profile.languages_detected,
        tags=profile.tags,
        is_continuation=profile.is_continuation,
        parent_id=profile.parent_id,
        logical_session_id=profile.logical_session_id,
        thinking_duration_ms=profile.thinking_duration_ms,
        output_duration_ms=profile.output_duration_ms,
        tool_duration_ms=profile.tool_duration_ms,
        latency_percentiles_ms=dict(profile.latency_percentiles_ms),
        tool_calls_per_minute=profile.tool_calls_per_minute,
        timing_provenance=profile.timing_provenance,
        total_input_tokens=profile.total_input_tokens,
        total_output_tokens=profile.total_output_tokens,
        total_cache_read_tokens=profile.total_cache_read_tokens,
        total_cache_write_tokens=profile.total_cache_write_tokens,
        total_credit_cost=profile.total_credit_cost,
        cost_provenance=profile.cost_provenance,
    )


def profile_inference_payload(profile: SessionProfile) -> SessionInferencePayload:
    signals = profile_support_signals(profile)
    return SessionInferencePayload(
        inferred_topic=profile.inferred_topic,
        inferred_topic_source=profile.inferred_topic_source,
        repo_names=profile.repo_names,
        work_event_count=len(profile.work_events),
        phase_count=len(profile.phases),
        engaged_duration_ms=profile.engaged_duration_ms,
        engaged_minutes=round(profile.engaged_duration_ms / 60_000.0, 4),
        tool_active_duration_ms=profile.tool_active_duration_ms,
        tool_active_minutes=round(profile.tool_active_duration_ms / 60_000.0, 4),
        workflow_shape=profile.workflow_shape,
        workflow_shape_confidence=profile.workflow_shape_confidence,
        terminal_state=profile.terminal_state,
        terminal_state_confidence=profile.terminal_state_confidence,
        support_level=profile_support_level(profile),
        support_signals=signals,
        engaged_duration_source=engaged_duration_source(profile),
        repo_inference_strength=repo_inference_strength(profile),
        auto_tags=profile.auto_tags,
        work_events=tuple(_work_event_document(event.to_dict()) for event in profile.work_events),
        phases=tuple(_phase_document(_phase_payload_from_phase(phase)) for phase in profile.phases),
        fallback_reasons=profile_inference_fallback_reasons(profile),
    )


def profile_inference_fallback_reasons(profile: SessionProfile) -> tuple[FallbackReason, ...]:
    reasons: list[FallbackReason] = []
    if engaged_duration_source(profile) == "session_total_fallback":
        reasons.append(FallbackReason.ENGAGED_DURATION_SESSION_TOTAL)
    if not profile.work_events and not profile.phases:
        reasons.append(FallbackReason.NO_WORK_EVENTS_AND_NO_PHASES)
    elif profile.work_events and all(event_fallback(event) for event in profile.work_events):
        reasons.append(FallbackReason.ALL_WORK_EVENTS_WEAK)
    return tuple(reasons)


# ---------------------------------------------------------------------------
# Build and hydrate session-profile storage rows
# ---------------------------------------------------------------------------


def build_session_profile_record(
    profile: SessionProfile,
    *,
    analysis: SessionAnalysis | None = None,
    logical_session_id: str | None = None,
    materialized_at: str | None = None,
) -> SessionProfileRecord:
    built_at = materialized_at or now_iso()
    evidence = profile_evidence_payload(profile)
    resolved_logical_session_id = logical_session_id or profile.logical_session_id or profile.session_id
    evidence = evidence.model_copy(update={"logical_session_id": resolved_logical_session_id})
    inference = profile_inference_payload(profile)
    enrichment = session_enrichment_payload(profile, analysis)
    evidence_search_text = profile_evidence_search_text(profile)
    inference_search_text = profile_inference_search_text(profile)
    source_updated_at = profile.updated_at.isoformat() if profile.updated_at else None
    return SessionProfileRecord(
        session_id=SessionId(profile.session_id),
        logical_session_id=SessionId(resolved_logical_session_id),
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=source_updated_at,
        source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
        input_high_water_mark=source_updated_at,
        input_high_water_mark_source=classify_profile_hwm_source(profile.updated_at),
        input_row_count=profile.message_count,
        source_name=profile.origin,
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
        tool_active_duration_ms=profile.tool_active_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        workflow_shape=profile.workflow_shape,
        workflow_shape_confidence=profile.workflow_shape_confidence,
        workflow_shape_features_json=_json.dumps(profile.workflow_shape_features, sort_keys=True),
        terminal_state=profile.terminal_state,
        terminal_state_confidence=profile.terminal_state_confidence,
        terminal_state_evidence_json=_json.dumps(profile.terminal_state_evidence, sort_keys=True),
        cost_is_estimated=profile.cost_is_estimated,
        total_input_tokens=profile.total_input_tokens,
        total_output_tokens=profile.total_output_tokens,
        total_cache_read_tokens=profile.total_cache_read_tokens,
        total_cache_write_tokens=profile.total_cache_write_tokens,
        total_credit_cost=profile.total_credit_cost,
        cost_provenance=profile.cost_provenance,
        per_model_cost_json=profile.per_model_cost_json,
        thinking_duration_ms=profile.thinking_duration_ms,
        output_duration_ms=profile.output_duration_ms,
        tool_duration_ms=profile.tool_duration_ms,
        latency_percentiles_ms_json=_serialize_percentiles(profile.latency_percentiles_ms),
        tool_calls_per_minute=profile.tool_calls_per_minute,
        timing_provenance=profile.timing_provenance,
        canonical_session_date=(profile.canonical_session_date.isoformat() if profile.canonical_session_date else None),
        evidence_payload=evidence,
        inference_payload=inference,
        enrichment_payload=enrichment,
        search_text=" \n".join(part for part in (evidence_search_text, inference_search_text) if part)
        or profile.session_id,
        evidence_search_text=evidence_search_text,
        inference_search_text=inference_search_text,
        enrichment_search_text=profile_enrichment_search_text(profile, enrichment),
        enrichment_version=SESSION_ENRICHMENT_VERSION,
        enrichment_family=SESSION_ENRICHMENT_FAMILY,
        inference_version=SESSION_INFERENCE_VERSION,
        inference_family=SESSION_INFERENCE_FAMILY,
    )


def hydrate_session_profile(record: SessionProfileRecord) -> SessionProfile:
    merged_payload: SessionProfileDocument = {
        "session_id": str(record.session_id),
        "logical_session_id": str(record.logical_session_id),
        "origin": record.source_name,
        "title": record.title,
        "inferred_topic": record.inference_payload.inferred_topic,
        "inferred_topic_source": record.inference_payload.inferred_topic_source,
        "created_at": record.evidence_payload.created_at,
        "updated_at": record.evidence_payload.updated_at,
        "tool_categories": dict(record.evidence_payload.tool_categories),
        "cwd_paths": list(record.evidence_payload.cwd_paths),
        "branch_names": list(record.evidence_payload.branch_names),
        "file_paths_touched": list(record.evidence_payload.file_paths_touched),
        "languages_detected": list(record.evidence_payload.languages_detected),
        "first_message_at": record.first_message_at,
        "last_message_at": record.last_message_at,
        "timestamp_source": record.evidence_payload.timestamp_source,
        "canonical_session_date": record.canonical_session_date,
        "repo_paths": list(record.repo_paths),
        "repo_names": list(record.repo_names),
        "tags": list(record.tags),
        "auto_tags": list(record.auto_tags),
        "message_count": record.message_count,
        "substantive_count": record.substantive_count,
        "attachment_count": record.attachment_count,
        "word_count": record.word_count,
        "tool_use_count": record.tool_use_count,
        "thinking_count": record.thinking_count,
        "total_cost_usd": record.total_cost_usd,
        "total_duration_ms": record.total_duration_ms,
        "engaged_duration_ms": record.engaged_duration_ms,
        "engaged_minutes": record.inference_payload.engaged_minutes,
        "tool_active_duration_ms": record.tool_active_duration_ms,
        "tool_active_minutes": record.inference_payload.tool_active_minutes,
        "wall_duration_ms": record.wall_duration_ms,
        "workflow_shape": record.workflow_shape,
        "workflow_shape_confidence": record.workflow_shape_confidence,
        "workflow_shape_features": record.evidence_payload.workflow_shape_features,
        "terminal_state": record.terminal_state,
        "terminal_state_confidence": record.terminal_state_confidence,
        "terminal_state_evidence": record.evidence_payload.terminal_state_evidence,
        "cost_is_estimated": record.cost_is_estimated,
        "compaction_count": record.evidence_payload.compaction_count,
        "work_events": [_work_event_payload(event) for event in record.inference_payload.work_events],
        "phases": [_phase_payload(phase) for phase in record.inference_payload.phases],
        "thread_id": None,
        "continuation_depth": 0,
        "timestamped_message_count": record.evidence_payload.timestamped_message_count,
        "untimestamped_message_count": record.evidence_payload.untimestamped_message_count,
        "timestamp_coverage": record.evidence_payload.timestamp_coverage,
        "is_continuation": record.evidence_payload.is_continuation,
        "parent_id": record.evidence_payload.parent_id,
        "thinking_duration_ms": record.thinking_duration_ms,
        "output_duration_ms": record.output_duration_ms,
        "tool_duration_ms": record.tool_duration_ms,
        "latency_percentiles_ms": record.evidence_payload.latency_percentiles_ms,
        "tool_calls_per_minute": record.tool_calls_per_minute,
        "timing_provenance": record.timing_provenance,
        "total_input_tokens": record.total_input_tokens,
        "total_output_tokens": record.total_output_tokens,
        "total_cache_read_tokens": record.total_cache_read_tokens,
        "total_cache_write_tokens": record.total_cache_write_tokens,
        "total_credit_cost": record.total_credit_cost,
        "cost_provenance": record.cost_provenance,
        "per_model_cost_json": record.per_model_cost_json,
    }
    return SessionProfile.from_dict(merged_payload)


def _phase_document(payload: SessionPhasePayload) -> SessionPhaseDocument:
    start_time = payload["start_time"]
    end_time = payload["end_time"]
    canonical_session_date = payload["canonical_session_date"]
    return {
        "start_time": start_time,
        "end_time": end_time,
        "canonical_session_date": canonical_session_date,
        "timing_provenance": _range_timing_provenance(start_time, end_time),
        "date_provenance": _date_provenance(canonical_session_date, start_time, end_time),
        "message_range": list(payload["message_range"]),
        "duration_ms": payload["duration_ms"],
        "phase_idle_threshold_ms": payload.get("phase_idle_threshold_ms", 300_000),
        "tool_counts": dict(payload["tool_counts"]),
        "word_count": payload["word_count"],
        "confidence": payload["confidence"],
        "evidence": list(payload["evidence"]),
    }


def _phase_payload_from_phase(phase: SessionPhase) -> SessionPhasePayload:
    start_time = phase.start_time.isoformat() if phase.start_time else None
    end_time = phase.end_time.isoformat() if phase.end_time else None
    canonical_session_date = phase.canonical_session_date.isoformat() if phase.canonical_session_date else None
    return {
        "start_time": start_time,
        "end_time": end_time,
        "canonical_session_date": canonical_session_date,
        "timing_provenance": _range_timing_provenance(start_time, end_time),
        "date_provenance": _date_provenance(canonical_session_date, start_time, end_time),
        "message_range": list(phase.message_range),
        "duration_ms": phase.duration_ms,
        "phase_idle_threshold_ms": phase.phase_idle_threshold_ms,
        "tool_counts": dict(phase.tool_counts),
        "word_count": phase.word_count,
        "confidence": phase.confidence,
        "evidence": list(phase.evidence),
    }


def _phase_from_payload_mapping(payload: SessionPhaseDocument | dict[str, object]) -> SessionPhase:
    return SessionPhase(
        start_time=optional_datetime(payload.get("start_time")),
        end_time=optional_datetime(payload.get("end_time")),
        canonical_session_date=optional_date(payload.get("canonical_session_date")),
        message_range=int_pair(payload.get("message_range")),
        duration_ms=coerce_int(payload.get("duration_ms"), 0),
        phase_idle_threshold_ms=coerce_int(payload.get("phase_idle_threshold_ms"), 300_000),
        tool_counts=string_int_mapping(payload.get("tool_counts")),
        word_count=coerce_int(payload.get("word_count"), 0),
        confidence=coerce_float(payload.get("confidence"), 0.0),
        evidence=string_sequence(payload.get("evidence")),
    )


def _work_event_document(payload: WorkEventPayload) -> WorkEventDocument:
    start_time = payload["start_time"]
    end_time = payload["end_time"]
    canonical_session_date = payload["canonical_session_date"]
    return {
        "heuristic_label": payload["heuristic_label"],
        "start_index": payload["start_index"],
        "end_index": payload["end_index"],
        "start_time": start_time,
        "end_time": end_time,
        "canonical_session_date": canonical_session_date,
        "timing_provenance": _range_timing_provenance(start_time, end_time),
        "date_provenance": _date_provenance(canonical_session_date, start_time, end_time),
        "duration_ms": payload["duration_ms"],
        "confidence": payload["confidence"],
        "evidence": list(payload["evidence"]),
        "file_paths": list(payload["file_paths"]),
        "tools_used": list(payload["tools_used"]),
        "summary": payload["summary"],
    }


def _phase_payload(payload: SessionPhaseDocument | dict[str, object]) -> SessionPhasePayload:
    return _phase_payload_from_phase(_phase_from_payload_mapping(payload))


def _work_event_payload(payload: WorkEventDocument | dict[str, object]) -> WorkEventPayload:
    event = WorkEvent.from_dict(payload)
    return event.to_dict()


# ---------------------------------------------------------------------------
# Signal/support helpers (merged from session_insight_row_signal_support)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def event_summary(event: WorkEvent) -> str:
    summary = str(event.summary or "").strip()
    return summary or event.heuristic_label.value


def support_level(
    confidence: float,
    *,
    support_signals: tuple[str, ...],
    fallback: bool = False,
) -> ConfidenceBand:
    """Map a confidence + supporting signals into the shared ConfidenceBand vocab (#1277)."""

    return from_signals(confidence, support_signals=support_signals, fallback=fallback)


def event_support_signals(event: WorkEvent) -> tuple[str, ...]:
    signals = [str(item) for item in event.evidence if str(item).strip()]
    if event.file_paths:
        signals.append("touched_paths")
    if event.tools_used:
        signals.append("tool_calls")
    if event.start_time and event.end_time:
        signals.append("timestamped_range")
    elif event.start_time or event.end_time:
        signals.append("partial_timestamp_range")
    elif event.canonical_session_date:
        signals.append("date_only_range")
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
    elif phase.start_time or phase.end_time:
        signals.append("partial_timestamp_range")
    elif phase.canonical_session_date:
        signals.append("date_only_range")
    if phase.word_count > 0:
        signals.append("word_count")
    return tuple(dict.fromkeys(signals))


def _range_timing_provenance(start_time: str | None, end_time: str | None) -> str:
    if start_time is not None and end_time is not None:
        return "timestamped_range"
    if start_time is not None:
        return "start_timestamp_only"
    if end_time is not None:
        return "end_timestamp_only"
    return "untimestamped"


def _date_provenance(canonical_session_date: str | None, start_time: str | None, end_time: str | None) -> str:
    if canonical_session_date is None:
        return "none"
    if start_time is not None or end_time is not None:
        return "event_timestamp"
    return "date_only"


def phase_fallback(phase: SessionPhase) -> bool:
    return not phase.tool_counts


def engaged_duration_source(profile: SessionProfile) -> str:
    return "phase_sum" if any(int(phase.duration_ms or 0) > 0 for phase in profile.phases) else "session_total_fallback"


def repo_inference_strength(profile: SessionProfile) -> ConfidenceBand:
    """Score repo-detection rigor using the shared vocab (#1277).

    ``NONE`` distinguishes "no repo evidence at all" from ``WEAK`` (we
    detected a name but nothing corroborating).
    """

    if profile.repo_paths and profile.repo_names:
        return ConfidenceBand.STRONG
    if profile.repo_names and (profile.file_paths_touched or profile.cwd_paths):
        return ConfidenceBand.MODERATE
    if profile.repo_names:
        return ConfidenceBand.WEAK
    return ConfidenceBand.NONE


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


def profile_support_level(profile: SessionProfile) -> ConfidenceBand:
    signals = profile_support_signals(profile)
    work_confidence = max((float(event.confidence or 0.0) for event in profile.work_events), default=0.0)
    fallback = engaged_duration_source(profile) != "phase_sum" and not profile.work_events and not profile.phases
    return support_level(work_confidence, support_signals=signals, fallback=fallback)


# ---------------------------------------------------------------------------
# Text extraction helpers (merged from session_insight_row_text_support)
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

_UNRESOLVED_BLOCKER_TERMINAL_STATES = frozenset(
    {
        "error_left",
        "question_left",
        "tool_left",
        "agent_hanging",
    }
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
        if message.is_candidate_human_authored and not message.is_noise:
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
    if analysis is not None and analysis.facts.actions:
        signals.append("actions")
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
    "enrichment_fallback_reasons",
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
    "profile_inference_fallback_reasons",
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
