"""One typed terminal outcome shared by every action envelope.

The operation boundary decides the outcome once, from facts only it holds:
how many rows its declared scope produced, which named gaps it had to accept,
and whether it could answer at all. CLI, MCP, HTTP, and the Python API
serialize that decision and map it to their transport's exit or status
convention; none of them re-infers it from the shape of the payload.

``empty`` -- the operation completed over its declared scope and that scope
holds zero rows. ``degraded`` -- it answered, but named gaps shaped the
answer, so zero rows may be an artifact of the gap rather than the archive.
``error`` -- it produced no valid answer. Inferring these from ``not rows``
is what makes a broken surface indistinguishable from an empty archive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TerminalOutcomeState = Literal["ok", "empty", "degraded", "error"]
"""The closed terminal-outcome vocabulary every action envelope carries."""

#: Reason recorded when a scope completed and simply holds no rows.
NO_ROWS_IN_SCOPE = "no_rows_in_scope"


class OutcomeEnvelope(BaseModel):
    """The terminal outcome of one operation, decided at the operation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: TerminalOutcomeState
    reason: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)

    @property
    def rows_are_authoritative(self) -> bool:
        """Whether an absent row set means the scope really holds nothing."""

        return self.state in {"ok", "empty"}

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def decide_outcome(
    *,
    matched: int,
    degraded: Sequence[str] = (),
    error: str | None = None,
    empty_reason: str = NO_ROWS_IN_SCOPE,
    detail: Mapping[str, object] | None = None,
) -> OutcomeEnvelope:
    """Decide one terminal outcome from the operation's own facts.

    Precedence is error, then degraded, then empty. A degraded answer with
    zero rows stays ``degraded``: the gap, not the archive, may be why the
    rows are absent, and collapsing it to ``empty`` is the confusion this
    type exists to prevent.
    """

    gaps = tuple(dict.fromkeys(reason for reason in degraded if reason))
    payload = dict(detail or {})
    if gaps:
        payload.setdefault("gaps", list(gaps))
    if error is not None:
        return OutcomeEnvelope(state="error", reason=error, detail=payload)
    if gaps:
        return OutcomeEnvelope(state="degraded", reason=gaps[0], detail=payload)
    if matched <= 0:
        return OutcomeEnvelope(state="empty", reason=empty_reason, detail=payload)
    return OutcomeEnvelope(state="ok", reason=None, detail=payload)


def combine_outcomes(outcomes: Sequence[OutcomeEnvelope], *, empty_reason: str = NO_ROWS_IN_SCOPE) -> OutcomeEnvelope:
    """Roll per-part outcomes into the outcome of the envelope that holds them.

    A composite answer is only as honest as its worst part: one errored part
    makes the whole answer degraded, because the caller received real data for
    the other parts alongside a named gap.
    """

    if not outcomes:
        return OutcomeEnvelope(state="empty", reason=empty_reason)
    gaps = tuple(
        dict.fromkeys(
            entry.reason or entry.state for entry in outcomes if entry.state in {"error", "degraded"} and entry.reason
        )
    )
    if gaps:
        return OutcomeEnvelope(state="degraded", reason=gaps[0], detail={"gaps": list(gaps)})
    if all(entry.state == "empty" for entry in outcomes):
        return OutcomeEnvelope(state="empty", reason=empty_reason)
    return OutcomeEnvelope(state="ok")


def lineage_page_outcome(*, matched: int, complete: bool, truncation_reason: str | None) -> OutcomeEnvelope:
    """Decide the outcome of one page of a composed lineage transcript.

    An incomplete composition is a named gap: the page may be short because
    the transcript was truncated, not because the session holds fewer
    messages.
    """

    gaps = () if complete else (f"lineage_truncated:{truncation_reason or 'unknown'}",)
    return decide_outcome(matched=matched, degraded=gaps)


#: The one CLI exit convention for a terminal action. ``2`` says the action
#: ran and its scope held nothing, which a shell can branch on without
#: parsing output; ``1`` says the answer is missing or gap-shaped.
OUTCOME_EXIT_CODES: dict[TerminalOutcomeState, int] = {"ok": 0, "empty": 2, "degraded": 1, "error": 1}


def outcome_exit_code(outcome: OutcomeEnvelope) -> int:
    """Map a terminal outcome to the CLI process exit code."""

    return OUTCOME_EXIT_CODES[outcome.state]


def outcome_http_status(outcome: OutcomeEnvelope) -> HTTPStatus:
    """Map a terminal outcome to the HTTP status of a served envelope.

    Every state that produced an envelope is a 200: the envelope itself
    carries the distinction. ``error`` means the route could not answer, so
    the response is a server-side failure rather than a body to be read as
    data. Input rejected before an envelope exists never reaches here.
    """

    return HTTPStatus.INTERNAL_SERVER_ERROR if outcome.state == "error" else HTTPStatus.OK


def render_outcome_line(outcome: OutcomeEnvelope) -> str | None:
    """Return the one human-readable line a terminal surface prints, if any.

    ``ok`` renders nothing: the rows are the answer. Every other state is
    stated explicitly so a zero-row terminal result is never bare.
    """

    if outcome.state == "ok":
        return None
    gaps = outcome.detail.get("gaps")
    suffix = ""
    if isinstance(gaps, list) and len(gaps) > 1:
        suffix = f" ({len(gaps)} gaps: {', '.join(str(gap) for gap in gaps)})"
    elif outcome.reason:
        suffix = f" ({outcome.reason})"
    label = {"empty": "empty", "degraded": "DEGRADED", "error": "ERROR"}[outcome.state]
    return f"outcome: {label}{suffix}"


__all__ = [
    "NO_ROWS_IN_SCOPE",
    "OUTCOME_EXIT_CODES",
    "OutcomeEnvelope",
    "TerminalOutcomeState",
    "combine_outcomes",
    "decide_outcome",
    "lineage_page_outcome",
    "outcome_exit_code",
    "outcome_http_status",
    "render_outcome_line",
]
