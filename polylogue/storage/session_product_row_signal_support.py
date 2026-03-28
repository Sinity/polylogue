"""Signal/support helpers for session-product rows."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from polylogue.lib.phases import SessionPhase
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.work_events import WorkEvent


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def event_summary(event: WorkEvent) -> str:
    summary = str(event.summary or "").strip()
    return summary or event.kind.value


def primary_work_kind(profile: SessionProfile) -> str | None:
    if not profile.work_events:
        return None
    counts = Counter(event.kind.value for event in profile.work_events)
    return counts.most_common(1)[0][0]


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
    return not phase.tool_counts or phase.kind in {"mixed", "conversation"}


def engaged_duration_source(profile: SessionProfile) -> str:
    return "phase_sum" if any(int(phase.duration_ms or 0) > 0 for phase in profile.phases) else "session_total_fallback"


def project_inference_strength(profile: SessionProfile) -> str:
    if profile.repo_paths and profile.canonical_projects:
        return "strong"
    if profile.canonical_projects and (profile.file_paths_touched or profile.cwd_paths):
        return "moderate"
    if profile.canonical_projects:
        return "weak"
    return "none"


def decision_signal_strength(profile: SessionProfile) -> str:
    if not profile.decisions:
        return "none"
    best_confidence = max(float(decision.confidence or 0.0) for decision in profile.decisions)
    if best_confidence >= 0.8:
        return "strong"
    if best_confidence >= 0.6:
        return "moderate"
    return "weak"


def profile_support_signals(profile: SessionProfile) -> tuple[str, ...]:
    signals: list[str] = []
    if profile.repo_paths:
        signals.append("repo_paths")
    if profile.file_paths_touched:
        signals.append("touched_paths")
    if profile.canonical_projects:
        signals.append("canonical_projects")
    if profile.work_events:
        signals.append("work_events")
    if profile.phases:
        signals.append("phases")
    if profile.decisions:
        signals.append("decisions")
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


__all__ = [
    "decision_signal_strength",
    "engaged_duration_source",
    "event_fallback",
    "event_summary",
    "event_support_signals",
    "now_iso",
    "phase_fallback",
    "phase_support_signals",
    "primary_work_kind",
    "profile_support_level",
    "profile_support_signals",
    "project_inference_strength",
    "support_level",
]
