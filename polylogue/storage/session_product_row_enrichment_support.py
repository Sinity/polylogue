"""Enrichment helpers for session-product rows."""

from __future__ import annotations

from polylogue.lib.session_profile import SessionAnalysis, SessionProfile

from .session_product_row_text_support import assistant_turn_texts, user_turn_texts


def enrichment_support_signals(
    profile: SessionProfile,
    analysis: SessionAnalysis | None,
) -> tuple[str, ...]:
    signals: list[str] = []
    user_turns = user_turn_texts(analysis)
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
    if assistant_turn_texts(analysis):
        signals.append("assistant_outcome_text")
    return tuple(signals)


__all__ = ["enrichment_support_signals"]
