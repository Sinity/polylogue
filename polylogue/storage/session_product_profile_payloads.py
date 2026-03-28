"""Payload builders for session-profile products."""

from __future__ import annotations

from polylogue.lib.session_profile import SessionAnalysis, SessionProfile
from polylogue.storage.session_product_row_support import (
    assistant_turn_texts,
    blocker_texts,
    decision_signal_strength,
    engaged_duration_source,
    enrichment_support_signals,
    keyword_work_kind,
    primary_work_kind,
    profile_support_level,
    profile_support_signals,
    project_inference_strength,
    support_level,
    user_turn_texts,
)


def session_enrichment_payload(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> dict[str, object]:
    user_turns = user_turn_texts(analysis)
    assistant_turns = assistant_turn_texts(analysis)
    blockers = blocker_texts(analysis)
    refined_work_kind = keyword_work_kind(user_turns, profile) or primary_work_kind(profile)
    support_signals = enrichment_support_signals(profile, analysis)
    input_band_summary = {
        "user_turns": len(user_turns),
        "assistant_turns": len(assistant_turns),
        "action_events": len(analysis.facts.action_events) if analysis is not None else 0,
        "touched_paths": len(profile.file_paths_touched),
        "canonical_projects": len(profile.canonical_projects),
        "decisions": len(profile.decisions),
    }
    confidence = min(
        0.95,
        0.2
        + (0.2 if user_turns else 0.0)
        + (0.15 if analysis is not None and analysis.facts.action_events else 0.0)
        + (0.15 if profile.file_paths_touched else 0.0)
        + (0.15 if profile.canonical_projects else 0.0)
        + (0.1 if profile.work_events else 0.0)
        + (0.05 if blockers else 0.0),
    )
    intent_summary = user_turns[0] if user_turns else (profile.title or None)
    outcome_summary = assistant_turns[-1] if assistant_turns else (user_turns[-1] if user_turns else None)
    return {
        "intent_summary": intent_summary,
        "outcome_summary": outcome_summary,
        "blockers": list(blockers),
        "refined_work_kind": refined_work_kind,
        "confidence": round(confidence, 3),
        "support_level": support_level(confidence, support_signals=support_signals),
        "support_signals": list(support_signals),
        "input_band_summary": input_band_summary,
    }


def profile_evidence_payload(profile: SessionProfile) -> dict[str, object]:
    return {
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        "first_message_at": profile.first_message_at.isoformat() if profile.first_message_at else None,
        "last_message_at": profile.last_message_at.isoformat() if profile.last_message_at else None,
        "canonical_session_date": (
            profile.canonical_session_date.isoformat()
            if profile.canonical_session_date
            else None
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
        "canonical_projects": list(profile.canonical_projects),
        "primary_work_kind": primary_work_kind(profile),
        "work_event_count": len(profile.work_events),
        "phase_count": len(profile.phases),
        "engaged_duration_ms": profile.engaged_duration_ms,
        "engaged_minutes": round(profile.engaged_duration_ms / 60_000.0, 4),
        "support_level": profile_support_level(profile),
        "support_signals": list(signals),
        "engaged_duration_source": engaged_duration_source(profile),
        "project_inference_strength": project_inference_strength(profile),
        "decision_signal_strength": decision_signal_strength(profile),
        "auto_tags": list(profile.auto_tags),
        "work_events": [event.to_dict() for event in profile.work_events],
        "phases": [
            {
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
                "tool_counts": dict(phase.tool_counts),
                "word_count": phase.word_count,
                "confidence": phase.confidence,
                "evidence": list(phase.evidence),
            }
            for phase in profile.phases
        ],
        "decisions": [
            {
                "index": decision.index,
                "summary": decision.summary,
                "confidence": decision.confidence,
                "context": decision.context,
            }
            for decision in profile.decisions
        ],
    }
