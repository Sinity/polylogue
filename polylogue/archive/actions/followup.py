"""Failure follow-up classification for action query surfaces."""

from __future__ import annotations

from typing import Literal, TypedDict

FollowupClass = Literal["acknowledged", "silent_proceed", "wordless_continuation", "ambiguous"]


class FollowupEvidence(TypedDict):
    classification: FollowupClass
    reason: str
    matched_marker: str | None


ACKNOWLEDGMENT_MARKERS: tuple[str, ...] = (
    "failed",
    "failure",
    "error",
    "errored",
    "exit code",
    "non-zero",
    "nonzero",
    "exception",
    "traceback",
    "bug",
    "fails",
    "permission denied",
    "not found",
    "don't exist",
    "timed out",
    "timeout",
    "isn't working",
    "doesn't work",
    "out of sync",
    "locked",
    "blocked",
    "hook blocks",
    "hanging",
    "modified during processing",
    "have been modified",
    "were modified",
    "conflict",
    "conflicting",
    "doesn't support",
    "being modified",
    "duplicate",
    "what went wrong",
    "the issue is",
    "formatting issue",
    "offline issue",
    "staging issue",
    "escaping issue",
    "couple of issues",
    "found some issues",
    "remaining three issues",
    "remaining issues",
    "additional issues",
    "many issues",
    "more issues",
    "unclosed function body",
    "need a clean database",
    "failing",
)


def classify_failed_followup_evidence(text: str | None) -> FollowupEvidence:
    """Classify the next assistant turn after a structured action failure.

    The structured failure is the anchor. This classifier only asks whether the
    next assistant turn explicitly acknowledges that failure. Missing or very
    short follow-up stays ambiguous instead of being counted as silent proceed.
    """

    if text is None:
        return {"classification": "ambiguous", "reason": "missing_next_assistant_message", "matched_marker": None}
    normalized = " ".join(text.lower().split())
    if len(normalized) < 20:
        return {"classification": "ambiguous", "reason": "short_next_assistant_message", "matched_marker": None}
    matched_marker = next((marker for marker in ACKNOWLEDGMENT_MARKERS if marker in normalized), None)
    if matched_marker is not None:
        return {
            "classification": "acknowledged",
            "reason": "explicit_acknowledgment_marker",
            "matched_marker": matched_marker,
        }
    return {"classification": "silent_proceed", "reason": "no_explicit_acknowledgment_marker", "matched_marker": None}


def classify_failed_followup(text: str | None) -> FollowupClass:
    """Return only the class label for a failed-action follow-up."""

    return classify_failed_followup_evidence(text)["classification"]


__all__ = [
    "ACKNOWLEDGMENT_MARKERS",
    "FollowupClass",
    "FollowupEvidence",
    "classify_failed_followup",
    "classify_failed_followup_evidence",
]
