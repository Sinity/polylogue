"""Row builders and hydrators for durable session-product read models."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from polylogue.lib.hashing import hash_text
from polylogue.lib.phases import SessionPhase
from polylogue.lib.session_profile import (
    SessionAnalysis,
    SessionProfile,
    build_session_analysis,
    build_session_profile,
)
from polylogue.lib.threads import WorkThread
from polylogue.lib.work_events import WorkEvent
from polylogue.storage.store import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
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


def _dedupe_texts(values: list[str], *, limit: int | None = None, width: int = 180) -> tuple[str, ...]:
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


def _support_level(confidence: float, *, support_signals: tuple[str, ...], fallback: bool = False) -> str:
    if fallback or confidence < 0.55 or not support_signals:
        return "weak"
    if confidence >= 0.78 and len(support_signals) >= 2:
        return "strong"
    return "moderate"


def _event_support_signals(event: WorkEvent) -> tuple[str, ...]:
    signals = [str(item) for item in event.evidence if str(item).strip()]
    if event.file_paths:
        signals.append("touched_paths")
    if event.tools_used:
        signals.append("tool_calls")
    if event.start_time and event.end_time:
        signals.append("timestamped_range")
    return tuple(dict.fromkeys(signals))


def _event_fallback(event: WorkEvent) -> bool:
    weak_markers = {"weak_signal", "no_tools", "shell_default"}
    return not event.evidence or bool(set(event.evidence) & weak_markers)


def _phase_support_signals(phase: SessionPhase) -> tuple[str, ...]:
    signals = [str(item) for item in phase.evidence if str(item).strip()]
    if phase.tool_counts:
        signals.append("tool_counts")
    if phase.start_time and phase.end_time:
        signals.append("timestamped_range")
    if phase.word_count > 0:
        signals.append("word_count")
    return tuple(dict.fromkeys(signals))


def _phase_fallback(phase: SessionPhase) -> bool:
    return not phase.tool_counts or phase.kind in {"mixed", "conversation"}


def _engaged_duration_source(profile: SessionProfile) -> str:
    return "phase_sum" if any(int(phase.duration_ms or 0) > 0 for phase in profile.phases) else "session_total_fallback"


def _project_inference_strength(profile: SessionProfile) -> str:
    if profile.repo_paths and profile.canonical_projects:
        return "strong"
    if profile.canonical_projects and (profile.file_paths_touched or profile.cwd_paths):
        return "moderate"
    if profile.canonical_projects:
        return "weak"
    return "none"


def _decision_signal_strength(profile: SessionProfile) -> str:
    if not profile.decisions:
        return "none"
    best_confidence = max(float(decision.confidence or 0.0) for decision in profile.decisions)
    if best_confidence >= 0.8:
        return "strong"
    if best_confidence >= 0.6:
        return "moderate"
    return "weak"


def _profile_support_signals(profile: SessionProfile) -> tuple[str, ...]:
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
    if _engaged_duration_source(profile) == "phase_sum":
        signals.append("phase_duration_sum")
    return tuple(signals)


def _profile_support_level(profile: SessionProfile) -> str:
    support_signals = _profile_support_signals(profile)
    work_confidence = max((float(event.confidence or 0.0) for event in profile.work_events), default=0.0)
    phase_confidence = max((float(phase.confidence or 0.0) for phase in profile.phases), default=0.0)
    confidence = max(work_confidence, phase_confidence)
    fallback = _engaged_duration_source(profile) != "phase_sum" and not profile.work_events and not profile.phases
    return _support_level(confidence, support_signals=support_signals, fallback=fallback)


def _user_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    if analysis is None:
        return ()
    return _dedupe_texts(
        [
            message.text
            for message in analysis.facts.message_facts
            if message.is_user and not message.is_context_dump and message.text.strip()
        ],
        width=220,
    )


def _assistant_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    if analysis is None:
        return ()
    return _dedupe_texts(
        [
            message.text
            for message in analysis.facts.message_facts
            if message.is_assistant and message.is_substantive and message.text.strip()
        ],
        width=220,
    )


def _blocker_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    if analysis is None:
        return ()
    blocker_markers = (
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
    texts = [
        message.text
        for message in analysis.facts.message_facts
        if message.is_user
        and not message.is_context_dump
        and message.text.strip()
        and any(marker in message.text.lower() for marker in blocker_markers)
    ]
    return _dedupe_texts(texts, limit=4, width=140)


def _keyword_work_kind(user_turns: tuple[str, ...], profile: SessionProfile) -> str | None:
    joined = " ".join(turn.lower() for turn in user_turns)
    mapping = (
        ("debugging", ("error", "failed", "traceback", "bug", "fix")),
        ("testing", ("test", "pytest", "assert", "spec")),
        ("planning", ("plan", "approach", "architecture", "design")),
        ("documentation", ("readme", "document", "docs", "docstring")),
        ("configuration", ("config", "nix", "flake", "settings", "toml", "yaml")),
        ("refactoring", ("refactor", "cleanup", "rename", "extract", "restructure")),
        ("data_analysis", ("duckdb", "sql", "analysis", "csv", "plot", "pandas")),
        ("research", ("search", "compare", "investigate", "look up", "browse")),
    )
    for kind, markers in mapping:
        if any(marker in joined for marker in markers):
            return kind
    if profile.work_events:
        counts = Counter(event.kind.value for event in profile.work_events)
        return counts.most_common(1)[0][0]
    return None


def _enrichment_support_signals(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> tuple[str, ...]:
    signals: list[str] = []
    user_turns = _user_turn_texts(analysis)
    if user_turns:
        signals.append("user_turns")
    if analysis is not None and analysis.facts.action_events:
        signals.append("action_events")
    if profile.file_paths_touched:
        signals.append("touched_paths")
    if profile.canonical_projects:
        signals.append("canonical_projects")
    if profile.work_events:
        signals.append("heuristic_work_events")
    if profile.decisions:
        signals.append("decisions")
    if _assistant_turn_texts(analysis):
        signals.append("assistant_outcome_text")
    return tuple(signals)


def _session_enrichment_payload(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> dict[str, object]:
    user_turns = _user_turn_texts(analysis)
    assistant_turns = _assistant_turn_texts(analysis)
    blockers = _blocker_texts(analysis)
    refined_work_kind = _keyword_work_kind(user_turns, profile) or _primary_work_kind(profile)
    support_signals = _enrichment_support_signals(profile, analysis)
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
    support_level = _support_level(confidence, support_signals=support_signals)
    intent_summary = user_turns[0] if user_turns else (profile.title or None)
    outcome_summary = assistant_turns[-1] if assistant_turns else (user_turns[-1] if user_turns else None)
    return {
        "intent_summary": intent_summary,
        "outcome_summary": outcome_summary,
        "blockers": list(blockers),
        "refined_work_kind": refined_work_kind,
        "confidence": round(confidence, 3),
        "support_level": support_level,
        "support_signals": list(support_signals),
        "input_band_summary": input_band_summary,
    }


def _profile_evidence_payload(profile: SessionProfile) -> dict[str, object]:
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


def _profile_inference_payload(profile: SessionProfile) -> dict[str, object]:
    support_signals = _profile_support_signals(profile)
    return {
        "canonical_projects": list(profile.canonical_projects),
        "primary_work_kind": _primary_work_kind(profile),
        "work_event_count": len(profile.work_events),
        "phase_count": len(profile.phases),
        "engaged_duration_ms": profile.engaged_duration_ms,
        "engaged_minutes": round(profile.engaged_duration_ms / 60_000.0, 4),
        "support_level": _profile_support_level(profile),
        "support_signals": list(support_signals),
        "engaged_duration_source": _engaged_duration_source(profile),
        "project_inference_strength": _project_inference_strength(profile),
        "decision_signal_strength": _decision_signal_strength(profile),
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


def _profile_evidence_search_text(profile: SessionProfile) -> str:
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


def _profile_inference_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.canonical_projects,
        *profile.auto_tags,
        *(event.summary for event in profile.work_events),
        *(event.kind.value for event in profile.work_events),
        *(phase.kind for phase in profile.phases),
        *(decision.summary for decision in profile.decisions),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def _profile_search_text(profile: SessionProfile) -> str:
    return " \n".join(
        part
        for part in (
            _profile_evidence_search_text(profile),
            _profile_inference_search_text(profile),
        )
        if part
    ) or profile.conversation_id


def _event_evidence_payload(event: WorkEvent) -> dict[str, object]:
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


def _event_inference_payload(event: WorkEvent) -> dict[str, object]:
    support_signals = _event_support_signals(event)
    fallback_inference = _event_fallback(event)
    return {
        "kind": event.kind.value,
        "summary": _event_summary(event),
        "confidence": event.confidence,
        "evidence": list(event.evidence),
        "support_level": _support_level(
            float(event.confidence or 0.0),
            support_signals=support_signals,
            fallback=fallback_inference,
        ),
        "support_signals": list(support_signals),
        "fallback_inference": fallback_inference,
    }


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


def _phase_evidence_payload(phase: SessionPhase) -> dict[str, object]:
    return {
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
    }


def _phase_inference_payload(phase: SessionPhase) -> dict[str, object]:
    support_signals = _phase_support_signals(phase)
    fallback_inference = _phase_fallback(phase)
    return {
        "kind": phase.kind,
        "confidence": phase.confidence,
        "evidence": list(phase.evidence),
        "support_level": _support_level(
            float(phase.confidence or 0.0),
            support_signals=support_signals,
            fallback=fallback_inference,
        ),
        "support_signals": list(support_signals),
        "fallback_inference": fallback_inference,
    }


def _profile_enrichment_search_text(profile: SessionProfile, enrichment_payload: dict[str, object]) -> str:
    blockers = tuple(str(item) for item in enrichment_payload.get("blockers", []) or [])
    support_signals = tuple(str(item) for item in enrichment_payload.get("support_signals", []) or [])
    parts = [
        profile.provider,
        profile.title or "",
        str(enrichment_payload.get("refined_work_kind") or ""),
        str(enrichment_payload.get("intent_summary") or ""),
        str(enrichment_payload.get("outcome_summary") or ""),
        *profile.canonical_projects,
        *profile.repo_paths,
        *blockers,
        *support_signals,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


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
    analysis: SessionAnalysis | None = None,
    materialized_at: str | None = None,
) -> SessionProfileRecord:
    built_at = materialized_at or _now_iso()
    evidence_payload = _profile_evidence_payload(profile)
    inference_payload = _profile_inference_payload(profile)
    enrichment_payload = _session_enrichment_payload(profile, analysis)
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
        canonical_session_date=(
            profile.canonical_session_date.isoformat()
            if profile.canonical_session_date
            else None
        ),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        enrichment_payload=enrichment_payload,
        search_text=_profile_search_text(profile),
        evidence_search_text=_profile_evidence_search_text(profile),
        inference_search_text=_profile_inference_search_text(profile),
        enrichment_search_text=_profile_enrichment_search_text(profile, enrichment_payload),
        enrichment_version=SESSION_ENRICHMENT_VERSION,
        enrichment_family=SESSION_ENRICHMENT_FAMILY,
        inference_version=SESSION_INFERENCE_VERSION,
        inference_family=SESSION_INFERENCE_FAMILY,
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
                start_time=event.start_time.isoformat() if event.start_time else None,
                end_time=event.end_time.isoformat() if event.end_time else None,
                duration_ms=event.duration_ms,
                canonical_session_date=(
                    event.canonical_session_date.isoformat()
                    if event.canonical_session_date
                    else None
                ),
                summary=_event_summary(event),
                file_paths=event.file_paths,
                tools_used=event.tools_used,
                evidence_payload=_event_evidence_payload(event),
                inference_payload=_event_inference_payload(event),
                search_text=_event_search_text(profile, event),
                inference_version=SESSION_INFERENCE_VERSION,
                inference_family=SESSION_INFERENCE_FAMILY,
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
                confidence=phase.confidence,
                evidence_reasons=phase.evidence,
                tool_counts=phase.tool_counts,
                word_count=phase.word_count,
                evidence_payload=_phase_evidence_payload(phase),
                inference_payload=_phase_inference_payload(phase),
                search_text=_phase_search_text(profile, phase),
                inference_version=SESSION_INFERENCE_VERSION,
                inference_family=SESSION_INFERENCE_FAMILY,
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
        "canonical_projects": list(record.canonical_projects),
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
        "primary_work_kind": record.primary_work_kind,
    }
    return SessionProfile.from_dict(merged_payload)


def hydrate_work_event(record: SessionWorkEventRecord) -> WorkEvent:
    return WorkEvent.from_dict({**record.evidence_payload, **record.inference_payload})


def hydrate_session_phase(record: SessionPhaseRecord) -> SessionPhase:
    payload = {**record.evidence_payload, **record.inference_payload}
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
        confidence=float(payload.get("confidence", record.confidence) or 0.0),
        evidence=tuple(
            str(value)
            for value in (
                payload.get("evidence", record.evidence_reasons) or record.evidence_reasons
            )
        ),
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
        build_session_profile_record(profile, analysis=analysis, materialized_at=materialized_at),
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
