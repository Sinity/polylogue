"""Failure follow-up classification for action query surfaces.

polylogue-b0b heuristic-tier inventory: ``ACKNOWLEDGMENT_MARKERS`` (and the
classifiers below that consume it) determine whether the *next* assistant
turn after a structurally confirmed action failure (``is_error:true`` --
already 100% structural, see ``tool_result_is_error``/``tool_result_exit_code``)
explicitly acknowledges that failure, versus silently proceeding. There is
no structural equivalent to convert to: "did the assistant say something
acknowledging this" is inherently a judgment about prose content, not a
fact recoverable from provider-emitted columns. This stays LABEL/
heuristic-tier (evidence class ``text_derived``), not CONVERT -- the caveat
is that ``silent_proceed`` is only as good as this keyword list's recall,
and it undercounts acknowledgments phrased without any of the listed
markers.

Two independent consumers apply this same marker list: the live
``followup_class`` SQL CASE expression in
``polylogue/storage/sqlite/archive_tiers/archive.py``
(``_ACTION_FOLLOWUP_RELATION_SQL``, the public query-surface path) and
``devtools/claim_vs_evidence.py``'s ``devtools claim-vs-evidence`` analysis
command (which calls :func:`classify_failed_followup_evidence` directly).
The SQL path reimplements the same short-follow-up/marker-match precedence
in SQL rather than calling this module, so a change to one does not
automatically apply to the other -- keep them in sync by hand if the
classification rule changes.
"""

from __future__ import annotations

import re
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

# Some runtimes retain an internal-reasoning envelope as a text block.  It is
# not a reader-visible follow-up, so its absence of an acknowledgement marker
# cannot support a claim that the assistant silently proceeded.
_PROTOCOL_ONLY_FOLLOWUP = re.compile(
    r"^\s*<(?:thinking|analysis|reasoning)(?:\s[^>]*)?>.*?</(?:thinking|analysis|reasoning)>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def classify_failed_followup_evidence(text: str | None) -> FollowupEvidence:
    """Classify the next assistant turn after a structured action failure.

    The structured failure is the anchor. This classifier only asks whether the
    next assistant turn explicitly acknowledges that failure. Missing or very
    short follow-up stays ambiguous instead of being counted as silent proceed.
    """

    if text is None:
        return {"classification": "ambiguous", "reason": "missing_next_assistant_message", "matched_marker": None}
    if _PROTOCOL_ONLY_FOLLOWUP.fullmatch(text):
        return {"classification": "ambiguous", "reason": "protocol_only_next_assistant_message", "matched_marker": None}
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
