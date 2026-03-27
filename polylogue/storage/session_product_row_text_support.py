"""Text extraction helpers for session-product rows."""

from __future__ import annotations

from collections import Counter

from polylogue.lib.session_profile import SessionAnalysis, SessionProfile


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


def user_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    if analysis is None:
        return ()
    return dedupe_texts(
        [
            message.text
            for message in analysis.facts.message_facts
            if message.is_user and not message.is_context_dump and message.text.strip()
        ],
        width=220,
    )


def assistant_turn_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
    if analysis is None:
        return ()
    return dedupe_texts(
        [
            message.text
            for message in analysis.facts.message_facts
            if message.is_assistant and message.is_substantive and message.text.strip()
        ],
        width=220,
    )


def blocker_texts(analysis: SessionAnalysis | None) -> tuple[str, ...]:
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
    return dedupe_texts(texts, limit=4, width=140)


def keyword_work_kind(user_turns: tuple[str, ...], profile: SessionProfile) -> str | None:
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


__all__ = [
    "assistant_turn_texts",
    "blocker_texts",
    "dedupe_texts",
    "enrichment_support_signals",
    "keyword_work_kind",
    "user_turn_texts",
]
