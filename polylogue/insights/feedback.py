"""Learning-feedback loop — corrections as deterministic rebuild signals (#1131).

This module defines the typed shape of *user corrections* applied to
heuristic insights. A correction is a user override that:

- lives **outside** the content-hashed session payload, so applying or
  removing a correction never alters ``session_content_hash()``;
- is keyed by ``(session_id, insight_kind)``: at most one correction
  of each kind per session, so deterministic rebuilds always produce the
  same merged output;
- is consulted by the insight materialization path **after** the heuristic
  has produced its base suggestion. The correction wins.

Three correction kinds are wired:

- ``TAG_REJECT`` — user rejects an auto-suggested tag for a session, so
  rebuilds must suppress it from the auto-tag output.
- ``TAG_ACCEPT`` — user confirms an auto-suggested tag so a surface can
  preserve that choice as user-authored state.
- ``SUMMARY_OVERRIDE`` — user provides a replacement summary for a session.

The set of recognized kinds is intentionally closed: surfaces that record
unknown kinds raise :class:`UnknownCorrectionKindError` immediately. New
kinds require an explicit code change so the merge semantics for that
kind are part of the same review as the new value.

The merge helpers (:func:`apply_corrections_to_auto_tags`,
:func:`apply_correction_to_summary`) are pure functions over
already-computed insight values. They never touch the database. Storage
and surface wiring lives in
:mod:`polylogue.storage.insights.feedback` and
:mod:`polylogue.api.archive` respectively.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

FEEDBACK_VERSION: int = 1
"""Bump on every behavior change to the correction merge semantics so
callers can detect stale rebuilds. Pinned per #1131 AC."""


# ---------------------------------------------------------------------------
# Typed enum of supported correction kinds
# ---------------------------------------------------------------------------


class CorrectionKind(str, Enum):
    """Closed taxonomy of recognized correction kinds.

    The string value is what gets persisted in the ``insight_kind`` column
    of ``user_corrections``. Adding a kind requires:

    - a new enum variant here;
    - matching merge semantics in either an ``apply_correction_to_*``
      helper or a follow-up materialization path;
    - tests that pin the new merge behavior.
    """

    TAG_REJECT = "tag_reject"
    """Suppress a specific auto-suggested tag from the rebuilt tag set."""

    TAG_ACCEPT = "tag_accept"
    """Confirm an auto-suggested tag so it is treated as user-authoritative."""

    SUMMARY_OVERRIDE = "summary_override"
    """Replace a heuristic session summary with user-provided text."""


class UnknownCorrectionKindError(ValueError):
    """Raised when a string cannot be mapped to a :class:`CorrectionKind`."""


def parse_correction_kind(value: str) -> CorrectionKind:
    """Map a string to a :class:`CorrectionKind` or raise.

    Surfaces (CLI/MCP/API) accept the value as a plain string and route
    through this function so the rejection of unknown kinds happens in
    exactly one place.
    """

    try:
        return CorrectionKind(value)
    except ValueError as exc:
        accepted = ", ".join(sorted(k.value for k in CorrectionKind))
        raise UnknownCorrectionKindError(f"Unknown correction kind: {value!r}. Accepted kinds: {accepted}") from exc


# ---------------------------------------------------------------------------
# Typed correction record
# ---------------------------------------------------------------------------


class LearningCorrection(BaseModel):
    """One typed user correction targeting a single insight on a session.

    The ``payload`` shape depends on ``kind``:

    - ``TAG_REJECT`` / ``TAG_ACCEPT``: ``{"tag": "<tag-string>"}``.
    - ``SUMMARY_OVERRIDE``: ``{"summary": "<text>"}``.

    The payload is intentionally a plain ``dict[str, str]``: corrections
    are user-supplied and do not need to carry numeric or nested data in
    this slice. Extending the payload shape for a future kind requires a
    typed schema there, not a generic widening here.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(min_length=1)
    """Target session. Stored after the surface layer resolves it."""

    kind: CorrectionKind
    """Closed-enum correction kind. See :class:`CorrectionKind`."""

    payload: dict[str, str]
    """Correction-specific data — see class docstring for each kind."""

    note: str | None = None
    """Optional free-form reason text from the user."""

    created_at: datetime
    """ISO-8601 timestamp of the latest write (storage upserts overwrite)."""

    feedback_version: int = FEEDBACK_VERSION

    @field_validator("payload")
    @classmethod
    def _payload_not_empty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("correction payload must not be empty")
        return dict(value)


def now_utc() -> datetime:
    """Return the current UTC timestamp. Factored out for test seam clarity."""

    return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Merge helpers — pure functions consulted by materialization paths
# ---------------------------------------------------------------------------


def select_correction(
    corrections: Iterable[LearningCorrection],
    kind: CorrectionKind,
) -> LearningCorrection | None:
    """Pick the correction of ``kind`` from a session's correction set.

    The ``(session_id, insight_kind)`` uniqueness invariant means
    there can be at most one match, but callers pass in a generic
    iterable, so the helper iterates defensively and returns the first
    one. The DB-level UNIQUE constraint is the durable source of truth.
    """

    for correction in corrections:
        if correction.kind == kind:
            return correction
    return None


def apply_corrections_to_auto_tags(
    base: Sequence[str],
    corrections: Sequence[LearningCorrection],
) -> tuple[str, ...]:
    """Apply ``TAG_REJECT`` corrections to a heuristic auto-tag list.

    A ``TAG_REJECT`` correction removes its specific tag from the rebuilt
    list; tag-accept corrections are returned unchanged here because tag
    acceptance only matters once the surface stops separating "auto" and
    "user-authoritative" tags.

    The function is pure and stable: tags are returned in their original
    order, with rejected tags filtered out.
    """

    rejected = {
        correction.payload.get("tag", "") for correction in corrections if correction.kind == CorrectionKind.TAG_REJECT
    }
    rejected.discard("")
    return tuple(tag for tag in base if tag not in rejected)


def apply_correction_to_summary(
    base: str,
    corrections: Sequence[LearningCorrection],
) -> str:
    """Apply a ``SUMMARY_OVERRIDE`` correction to a heuristic summary."""

    correction = select_correction(corrections, CorrectionKind.SUMMARY_OVERRIDE)
    if correction is None:
        return base
    replacement = correction.payload.get("summary", "")
    if not replacement:
        return base
    return replacement


__all__ = [
    "CorrectionKind",
    "FEEDBACK_VERSION",
    "LearningCorrection",
    "UnknownCorrectionKindError",
    "apply_corrections_to_auto_tags",
    "apply_correction_to_summary",
    "now_utc",
    "parse_correction_kind",
    "select_correction",
]
