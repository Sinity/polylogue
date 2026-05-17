"""Heuristic session classification — typed taxonomy with confidence + evidence.

Classifies a :class:`SessionProfile` into a closed :class:`SessionCategory`
taxonomy using only deterministic, evidence-bearing signals already present on
the profile (work-event distribution, tool categories, repo paths, message
counts, thinking ratio).

Design goals (issue #1130):

- **Typed taxonomy**: ``SessionCategory`` is a closed ``str`` enum. Adding a
  new variant requires an explicit code change. The classifier never returns
  a string label outside this enum.
- **Confidence + evidence**: every classification carries a ``confidence``
  float in ``[0.0, 1.0]`` plus a tuple of ``EvidenceCite`` entries naming the
  specific profile fields and values that drove the decision.
- **Deterministic and versioned**: a fixed ``CLASSIFIER_VERSION`` pins the
  behavior. The same input must always produce the same output.
- **Suggestion-grade**: the classifier is heuristic. ``support_level``
  ("strong"/"moderate"/"weak") communicates how much the evidence supports
  the label; user-authored tags remain authoritative downstream.

This module is a pure function over ``SessionProfile`` — no I/O, no DB. The
materialization path (storing the classification alongside the session
profile, surfacing it as M2M auto-tags) is a separate concern tracked in a
follow-up issue.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.conversation.extraction import WorkEvent, WorkEventKind
from polylogue.archive.session.session_profile import SessionProfile

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

CLASSIFIER_VERSION: int = 1
"""Classifier version. Bump on every behavior change so callers can detect
stale classifications and rebuild them. Pinned per #1130 AC."""

CLASSIFIER_FAMILY: str = "session-classification-heuristic"


# ---------------------------------------------------------------------------
# Closed taxonomy
# ---------------------------------------------------------------------------


class SessionCategory(str, Enum):
    """Closed, typed classification taxonomy for AI conversation sessions.

    Each variant represents a distinct shape of session work. Variants are
    intentionally coarse-grained: finer distinctions belong in tags. Adding
    a new variant is an explicit code change (no string fall-through).
    """

    DEBUGGING = "debugging"
    """Iterative diagnosis of an error or unexpected behavior."""

    REFACTORING = "refactoring"
    """Restructuring existing code without changing observed behavior."""

    FEATURE = "feature"
    """Building a new capability — implementation-led."""

    EXPLORATION = "exploration"
    """Open-ended research/learning with no fixed implementation target."""

    REVIEW = "review"
    """Reading, reviewing, or analyzing existing code or artifacts."""

    PLANNING = "planning"
    """Designing or planning work without implementing it."""

    TESTING = "testing"
    """Test-focused work: writing tests, fixing test failures, running suites."""

    DOCUMENTATION = "documentation"
    """Writing or editing docs, READMEs, comments."""

    CONFIGURATION = "configuration"
    """Configuration, environment, dependency, or build-system work."""

    CONVERSATION = "conversation"
    """Pure conversation/chat with no code-bearing signals."""

    UNCLASSIFIED = "unclassified"
    """Insufficient signal to choose a more specific category."""


# Mapping from the existing ``WorkEventKind`` enum (used by work-event
# extraction) into the broader classification taxonomy. WorkEventKind is
# per-event; SessionCategory is per-session. Multiple work events of the
# same kind reinforce the session-level category.
_WORK_EVENT_TO_CATEGORY: dict[WorkEventKind, SessionCategory] = {
    WorkEventKind.DEBUGGING: SessionCategory.DEBUGGING,
    WorkEventKind.REFACTORING: SessionCategory.REFACTORING,
    WorkEventKind.IMPLEMENTATION: SessionCategory.FEATURE,
    WorkEventKind.RESEARCH: SessionCategory.EXPLORATION,
    WorkEventKind.REVIEW: SessionCategory.REVIEW,
    WorkEventKind.PLANNING: SessionCategory.PLANNING,
    WorkEventKind.TESTING: SessionCategory.TESTING,
    WorkEventKind.DOCUMENTATION: SessionCategory.DOCUMENTATION,
    WorkEventKind.CONFIGURATION: SessionCategory.CONFIGURATION,
    WorkEventKind.CONVERSATION: SessionCategory.CONVERSATION,
    WorkEventKind.DATA_ANALYSIS: SessionCategory.EXPLORATION,
}


# ---------------------------------------------------------------------------
# Typed evidence + result models
# ---------------------------------------------------------------------------


class EvidenceCite(BaseModel):
    """One piece of evidence supporting a classification.

    Names the profile field consulted and the value observed. Downstream
    consumers can reconstruct why a label was chosen and audit drift.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    """SessionProfile field name the evidence comes from."""

    value: str
    """Stringified observation. Already-redacted by virtue of coming from
    counts/category-names rather than user content."""

    weight: float = Field(ge=0.0, le=1.0)
    """How strongly this evidence votes for the chosen category (0..1)."""


class SessionClassification(BaseModel):
    """Typed session-classification result with confidence and evidence.

    Stable contract: the classifier always returns this shape. Tests pin
    determinism (same input → same output), evidence non-emptiness (every
    non-unclassified label has at least one cited field), and taxonomy
    completeness (the chosen ``category`` is always a defined enum value).
    """

    model_config = ConfigDict(frozen=True)

    category: SessionCategory
    """The classification verdict — always a closed-enum value."""

    confidence: float = Field(ge=0.0, le=1.0)
    """How confident the heuristic is (0..1). Independent of support_level —
    confidence reflects within-classifier vote margin; support_level
    summarizes evidence breadth."""

    support_level: str
    """Coarse evidence-strength tag: 'strong', 'moderate', or 'weak'."""

    evidence: tuple[EvidenceCite, ...]
    """Ordered evidence citations (highest-weight first). Always non-empty
    when ``category`` is not ``UNCLASSIFIED``."""

    classifier_version: int = CLASSIFIER_VERSION
    classifier_family: str = CLASSIFIER_FAMILY

    @property
    def auto_tag(self) -> str:
        """Suggestion-grade auto-tag string for the M2M tag system.

        Distinct from user-authored tags by the ``auto:`` prefix so the tag
        system can keep user tags authoritative (#817 surface).
        """

        return f"auto:category:{self.category.value}"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _support_for(confidence: float, evidence_count: int) -> str:
    """Map (confidence, evidence breadth) to a coarse support tag."""

    if confidence >= 0.75 and evidence_count >= 3:
        return "strong"
    if confidence >= 0.5 and evidence_count >= 2:
        return "moderate"
    return "weak"


def _work_event_votes(work_events: Iterable[WorkEvent]) -> dict[SessionCategory, float]:
    """Weighted votes from work-event extraction.

    Each event contributes its confidence as a vote toward the
    corresponding ``SessionCategory``. Unmapped kinds are skipped.
    """

    votes: dict[SessionCategory, float] = {}
    for event in work_events:
        category = _WORK_EVENT_TO_CATEGORY.get(event.kind)
        if category is None:
            continue
        votes[category] = votes.get(category, 0.0) + max(0.0, float(event.confidence or 0.0))
    return votes


def _tool_category_signals(tool_categories: dict[str, int]) -> dict[SessionCategory, float]:
    """Auxiliary votes from aggregated tool-category usage."""

    # Keys here are the categories produced by the tool-usage classifier;
    # they correlate loosely with session categories but are not authoritative.
    mapping: dict[str, SessionCategory] = {
        "test": SessionCategory.TESTING,
        "testing": SessionCategory.TESTING,
        "debug": SessionCategory.DEBUGGING,
        "edit": SessionCategory.FEATURE,
        "write": SessionCategory.FEATURE,
        "search": SessionCategory.EXPLORATION,
        "read": SessionCategory.REVIEW,
        "config": SessionCategory.CONFIGURATION,
        "docs": SessionCategory.DOCUMENTATION,
        "documentation": SessionCategory.DOCUMENTATION,
    }
    votes: dict[SessionCategory, float] = {}
    for raw_key, count in tool_categories.items():
        if count <= 0:
            continue
        category = mapping.get(raw_key.lower())
        if category is None:
            continue
        # Tool-category counts are noisy; weight them sublinearly relative
        # to direct work-event evidence.
        votes[category] = votes.get(category, 0.0) + min(2.0, float(count) ** 0.5)
    return votes


def classify_session(profile: SessionProfile) -> SessionClassification:
    """Classify a :class:`SessionProfile` into a typed :class:`SessionCategory`.

    Deterministic and pure: no I/O, no randomness. Same input → same output.

    Strategy:

    1. Gather weighted votes from work events (primary signal) and
       aggregated tool categories (secondary signal).
    2. Pick the winning category by total weight. Ties resolve in
       taxonomy-declaration order (Python enum iteration order).
    3. Compute confidence as the winner's share of the total vote mass,
       attenuated when the absolute mass is small (few events).
    4. Build typed evidence citations for each contributing signal.
    5. Fall back to ``CONVERSATION`` for chat-only sessions or
       ``UNCLASSIFIED`` when no signal at all is present.
    """

    work_votes = _work_event_votes(profile.work_events)
    tool_votes = _tool_category_signals(dict(profile.tool_categories))

    combined: dict[SessionCategory, float] = {}
    for source in (work_votes, tool_votes):
        for category, weight in source.items():
            combined[category] = combined.get(category, 0.0) + weight

    evidence_list: list[EvidenceCite] = []

    if not combined:
        # No category-bearing signal. Decide between conversation and
        # unclassified based on whether there's any message content at all.
        if profile.message_count > 0 and profile.tool_use_count == 0 and not profile.file_paths_touched:
            evidence_list.append(
                EvidenceCite(
                    field="tool_use_count",
                    value=str(profile.tool_use_count),
                    weight=0.5,
                )
            )
            evidence_list.append(
                EvidenceCite(
                    field="message_count",
                    value=str(profile.message_count),
                    weight=0.5,
                )
            )
            return SessionClassification(
                category=SessionCategory.CONVERSATION,
                confidence=0.6,
                support_level=_support_for(0.6, len(evidence_list)),
                evidence=tuple(evidence_list),
            )
        # Truly empty: no messages, no tool use, no files. Unclassified
        # carries an empty evidence tuple by contract.
        return SessionClassification(
            category=SessionCategory.UNCLASSIFIED,
            confidence=0.0,
            support_level="weak",
            evidence=(),
        )

    # Pick winner. Iterate taxonomy in enum-declaration order so ties are
    # deterministic regardless of dict insertion order.
    total_weight = sum(combined.values())
    winner: SessionCategory | None = None
    winning_weight = -1.0
    for category in SessionCategory:
        weight = combined.get(category, 0.0)
        if weight > winning_weight:
            winner = category
            winning_weight = weight
    assert winner is not None  # combined is non-empty here

    # Confidence: share of vote mass, attenuated by absolute mass so a
    # single low-confidence event doesn't yield confidence=1.0.
    share = winning_weight / total_weight if total_weight > 0 else 0.0
    mass_factor = min(1.0, total_weight / 3.0)  # saturates at ~3 strong votes
    confidence = round(share * (0.5 + 0.5 * mass_factor), 4)

    # Build evidence: cite work-event votes first (primary), then
    # tool-category votes (secondary). Order by weight desc.
    if winner in work_votes:
        evidence_list.append(
            EvidenceCite(
                field="work_events",
                value=f"{winner.value}:weight={work_votes[winner]:.2f}",
                weight=min(1.0, work_votes[winner] / 3.0),
            )
        )
    if winner in tool_votes:
        evidence_list.append(
            EvidenceCite(
                field="tool_categories",
                value=f"{winner.value}:weight={tool_votes[winner]:.2f}",
                weight=min(1.0, tool_votes[winner] / 3.0),
            )
        )
    # Always cite breadth-of-evidence signals when present.
    if profile.repo_names:
        evidence_list.append(
            EvidenceCite(
                field="repo_names",
                value=",".join(profile.repo_names[:3]),
                weight=0.2,
            )
        )
    if profile.file_paths_touched:
        evidence_list.append(
            EvidenceCite(
                field="file_paths_touched",
                value=str(len(profile.file_paths_touched)),
                weight=0.2,
            )
        )

    evidence_list.sort(key=lambda cite: cite.weight, reverse=True)

    return SessionClassification(
        category=winner,
        confidence=confidence,
        support_level=_support_for(confidence, len(evidence_list)),
        evidence=tuple(evidence_list),
    )


__all__ = [
    "CLASSIFIER_FAMILY",
    "CLASSIFIER_VERSION",
    "EvidenceCite",
    "SessionCategory",
    "SessionClassification",
    "classify_session",
]
