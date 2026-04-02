"""Session-profile inference and builder runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from polylogue.lib.attribution import extract_attribution
from polylogue.lib.phase_extraction import extract_phases
from polylogue.lib.semantic_facts import build_conversation_semantic_facts
from polylogue.lib.session_profile_models import SessionAnalysis, SessionProfile
from polylogue.lib.work_event_extraction import extract_work_events

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.semantic_facts import ConversationSemanticFacts


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
    for project in list(profile.canonical_projects)[:3]:
        tags.append(f"project:{project}")
    if profile.is_continuation:
        tags.append("continuation")
    if profile.continuation_depth >= 3:
        tags.append("deep-thread")
    if len(profile.canonical_projects) > 1:
        tags.append("multi-project")
    if profile.total_cost_usd >= 1.0:
        tags.append("costly")
    return tuple(sorted(set(tags)))


def build_session_profile(
    conversation: Conversation,
    *,
    analysis: SessionAnalysis | None = None,
) -> SessionProfile:
    from polylogue.lib.pricing import harmonize_session_cost

    session_analysis = analysis or build_session_analysis(conversation)
    facts = session_analysis.facts
    attribution = session_analysis.attribution
    cost_usd, cost_is_estimated = harmonize_session_cost(conversation)
    engaged_duration_ms = sum(int(phase.duration_ms or 0) for phase in session_analysis.phases)
    if engaged_duration_ms <= 0:
        engaged_duration_ms = max(int(conversation.total_duration_ms or 0), 0)
    canonical_session_at = (
        facts.first_message_at
        or conversation.created_at
        or conversation.updated_at
        or facts.last_message_at
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
        canonical_projects=attribution.canonical_projects,
        work_events=session_analysis.work_events,
        phases=session_analysis.phases,
        first_message_at=facts.first_message_at,
        last_message_at=facts.last_message_at,
        canonical_session_date=canonical_session_at.date() if canonical_session_at else None,
        engaged_duration_ms=engaged_duration_ms,
        wall_duration_ms=facts.wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        tags=tuple(conversation.tags),
        is_continuation=conversation.is_continuation,
        parent_id=str(conversation.parent_id) if conversation.parent_id else None,
    )
    return replace(partial, auto_tags=infer_auto_tags(partial))


__all__ = ["build_session_analysis", "build_session_profile", "infer_auto_tags"]
