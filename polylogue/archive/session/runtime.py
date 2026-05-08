"""Session-profile inference and builder runtime."""

from __future__ import annotations

import json as _json
from dataclasses import replace
from typing import TYPE_CHECKING

from polylogue.archive.conversation.attribution import extract_attribution
from polylogue.archive.conversation.extraction import extract_work_events
from polylogue.archive.phase.extraction import extract_phases
from polylogue.archive.semantic.facts import build_conversation_semantic_facts
from polylogue.archive.semantic.timing import compute_session_timing
from polylogue.archive.session.models import SessionAnalysis, SessionProfile

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation
    from polylogue.archive.semantic.facts import ConversationSemanticFacts


def build_session_analysis(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> SessionAnalysis:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    phases = tuple(extract_phases(conversation, facts=semantic_facts))
    return SessionAnalysis(
        facts=semantic_facts,
        attribution=extract_attribution(conversation, facts=semantic_facts),
        work_events=tuple(extract_work_events(conversation, facts=semantic_facts, phases=phases)),
        phases=phases,
    )


def infer_auto_tags(profile: SessionProfile) -> tuple[str, ...]:
    tags: list[str] = [f"provider:{profile.provider}"]
    for repo_name in list(profile.repo_names)[:3]:
        tags.append(f"repo:{repo_name}")
    if profile.is_continuation:
        tags.append("continuation")
    if profile.continuation_depth >= 3:
        tags.append("deep-thread")
    if len(profile.repo_names) > 1:
        tags.append("multi-repo")
    if profile.total_cost_usd >= 1.0:
        tags.append("costly")
    return tuple(sorted(set(tags)))


def build_session_profile(
    conversation: Conversation,
    *,
    analysis: SessionAnalysis | None = None,
    compaction_count: int | None = None,
) -> SessionProfile:
    from polylogue.archive.semantic.cost_compute import compute_session_cost
    from polylogue.archive.semantic.pricing import harmonize_session_cost

    session_analysis = analysis or build_session_analysis(conversation)
    facts = session_analysis.facts
    attribution = session_analysis.attribution
    cost_usd, cost_is_estimated = harmonize_session_cost(conversation)
    cost_summary = compute_session_cost(conversation)
    resolved_compaction_count = (
        compaction_count
        if compaction_count is not None
        else sum(1 for event in conversation.provider_events if event.event_type == "compaction")
    )
    engaged_duration_ms = sum(int(phase.duration_ms or 0) for phase in session_analysis.phases)
    if engaged_duration_ms <= 0:
        engaged_duration_ms = max(int(conversation.total_duration_ms or 0), 0)
    canonical_session_at = (
        facts.first_message_at or conversation.created_at or conversation.updated_at or facts.last_message_at
    )
    timing = compute_session_timing(
        list(conversation.messages),
        tool_use_count=facts.tool_messages,
        wall_duration_ms=facts.wall_duration_ms,
    )
    partial = SessionProfile(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=facts.total_messages,
        substantive_count=facts.substantive_messages,
        tool_use_count=facts.tool_messages,
        thinking_count=facts.thinking_messages,
        attachment_count=facts.attachment_count,
        word_count=facts.word_count,
        total_cost_usd=cost_usd,
        total_duration_ms=conversation.total_duration_ms,
        tool_categories=facts.tool_category_counts,
        repo_paths=attribution.repo_paths,
        cwd_paths=attribution.cwd_paths,
        branch_names=attribution.branch_names,
        file_paths_touched=attribution.file_paths_touched,
        languages_detected=attribution.languages_detected,
        repo_names=attribution.repo_names,
        work_events=session_analysis.work_events,
        phases=session_analysis.phases,
        first_message_at=facts.first_message_at,
        last_message_at=facts.last_message_at,
        timestamped_message_count=facts.timestamped_messages,
        untimestamped_message_count=facts.untimestamped_messages,
        timestamp_coverage=facts.timestamp_coverage,
        canonical_session_date=canonical_session_at.date() if canonical_session_at else None,
        engaged_duration_ms=engaged_duration_ms,
        wall_duration_ms=facts.wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        total_input_tokens=cost_summary.total_input_tokens,
        total_output_tokens=cost_summary.total_output_tokens,
        total_cache_read_tokens=cost_summary.total_cache_read_tokens,
        total_cache_write_tokens=cost_summary.total_cache_write_tokens,
        total_credit_cost=cost_summary.total_credit_cost,
        cost_provenance=cost_summary.cost_provenance,
        per_model_cost_json=_json.dumps(
            [b.model_dump(mode="json") for b in cost_summary.per_model] if cost_summary.per_model else [],
            default=str,
        ),
        compaction_count=resolved_compaction_count,
        tags=tuple(conversation.tags),
        is_continuation=conversation.is_continuation,
        parent_id=str(conversation.parent_id) if conversation.parent_id else None,
        thinking_duration_ms=timing.thinking_duration_ms,
        output_duration_ms=timing.output_duration_ms,
        tool_duration_ms=timing.tool_duration_ms,
        latency_percentiles_ms=timing.latency_percentiles_ms,
        tool_calls_per_minute=timing.tool_calls_per_minute,
        timing_provenance=timing.timing_provenance,
    )
    return replace(partial, auto_tags=infer_auto_tags(partial))


__all__ = ["build_session_analysis", "build_session_profile", "infer_auto_tags"]
