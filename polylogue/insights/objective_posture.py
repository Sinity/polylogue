"""Session objective-posture projection (polylogue-37t.23).

Derives whether a session's underlying objective remains open from
authority-bearing evidence, distinct from ``SessionInferencePayload
.terminal_state`` -- which describes how the process *ended* (structural
tool/turn evidence: ``tool_left`` / ``error_left`` / ``question_left`` /
``unknown``), not whether the goal was resolved. Process termination and
objective posture are orthogonal by design: polylogue-9e5.9 measured the old
prose-derived "clean finish" signal at 50.5% agreement (a coin flip) with
structural truth, and it was deleted (polylogue-ve9z,
``archive/session/runtime.py::_terminal_state``). A session that ends on a
normal assistant reply now gets ``terminal_state="unknown"`` -- silence, not
a completion claim -- which is exactly why a *separate* posture projection is
needed: "unknown how it ended" must never collapse into "done".

Authority order, highest first:

1. ``goal_graph`` -- declared ``::goal``/``::question`` open/close/block
   events (polylogue-7yk5). NOT YET BUILT; reserved authority tier, never
   produced by this module today.
2. ``work_evidence`` -- provider/work-evidence graph claims, observed
   effects, and evaluated AC satisfaction (polylogue-1vpm.6). NOT YET
   BUILT; reserved authority tier, never produced by this module today.
3. ``assertion`` -- durable ``decision``/``blocker``/``handoff``
   ``AssertionKind`` rows targeting the session (``user.db``, available
   now). Outranks structural inference: an operator- or agent-declared
   obligation is stronger evidence than a heuristic read of how the
   session ended.
4. ``structural_inference`` -- bounded ``terminal_state``-derived signal
   (``index.db``, available now). Never emits ``"completed"`` -- no tier
   available today observes a satisfied effect, only how the tool/turn
   stream ended.
5. ``none`` -- no evidence at any tier -> ``"unknown"``.

``derive_objective_posture`` is the one blending function every consumer
(profile enrichment, blocker extraction, resume ranking, context
compilation) goes through; there is no parallel terminal-state completion
heuristic. Tiers 1-2 are reserved slots that always defer today (there is
nothing yet to query) -- wiring them in is the follow-on once 7yk5/1vpm.6
land, and the precedence order and payload shape do not need to change to
accommodate them.

Architecture note: the ``assertion`` tier is deliberately a *read-time*
overlay (``resolve_session_objective_posture``), never baked into the
materialized ``session_profiles`` row. ``index.db`` must stay independently
rebuildable from ``source.db`` alone (see the schema-regimes note in
``CLAUDE.md``); entangling index materialization with a ``user.db`` read
would break that invariant. Only the ``structural_inference`` tier
(``structural_objective_posture``), which needs no durable-tier read, is
safe to compute at materialization time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.refs import ObjectRef
from polylogue.insights.archive_models import ObjectivePosturePayload
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope

ObjectivePosture = Literal[
    "completed",
    "blocked",
    "awaiting_operator",
    "awaiting_effect",
    "abandoned_inactive",
    "ambiguous",
    "unknown",
]

ObjectivePostureAuthority = Literal[
    "goal_graph",
    "work_evidence",
    "assertion",
    "structural_inference",
    "none",
]

AUTHORITY_ORDER: tuple[ObjectivePostureAuthority, ...] = (
    "goal_graph",
    "work_evidence",
    "assertion",
    "structural_inference",
    "none",
)

# Durable AssertionKinds the `assertion` tier reads. HANDOFF is deliberately
# included even though it is absent from
# `user_write.ASSERTION_CLAIM_KINDS` (that tuple serves a broader, unrelated
# "claims" surface) -- objective posture needs it explicitly.
ASSERTION_TIER_KINDS: tuple[AssertionKind, ...] = (
    AssertionKind.BLOCKER,
    AssertionKind.DECISION,
    AssertionKind.HANDOFF,
)

# Bounded structural mapping from `terminal_state` (process-termination
# evidence) to a posture. Deliberately never maps to "completed": no
# structural signal at this authority level observes that the underlying
# objective was satisfied, only how the tool/turn stream ended (every
# populated `terminal_state_evidence` entry is `raw_evidence` -- see
# `_terminal_state`). Claiming completion needs the assertion tier or above.
_STRUCTURAL_POSTURE_BY_TERMINAL_STATE: dict[str, ObjectivePosture] = {
    "tool_left": "awaiting_effect",
    "error_left": "blocked",
    "question_left": "ambiguous",
    "unknown": "ambiguous",
}

# Assertion-kind -> implied posture, in the sub-priority applied when more
# than one kind is present simultaneously (AC2: multiple simultaneous
# obligations are preserved rather than silently collapsed -- the
# non-winning kinds surface in `contradictions`).
_ASSERTION_POSTURE_PRIORITY: tuple[tuple[AssertionKind, ObjectivePosture], ...] = (
    (AssertionKind.BLOCKER, "blocked"),
    (AssertionKind.DECISION, "awaiting_operator"),
    (AssertionKind.HANDOFF, "awaiting_operator"),
)


def _format_evidence_ref(key: str, value: object) -> str:
    if key == "message_id" and isinstance(value, str) and value:
        return ObjectRef(kind="message", object_id=value).format()
    return f"{key}:{value}"


def structural_objective_posture(
    *,
    terminal_state: str,
    terminal_state_confidence: float,
    terminal_state_evidence: dict[str, object] | None = None,
    as_of: str | None = None,
) -> ObjectivePosturePayload:
    """Tier-4 (``structural_inference``) posture, from raw terminal-state
    evidence alone. index.db-only inputs -- safe to compute at
    index-materialization time (unlike the assertion tier).
    """

    posture = _STRUCTURAL_POSTURE_BY_TERMINAL_STATE.get(terminal_state)
    if posture is None:
        return ObjectivePosturePayload(posture="unknown", authority="none", as_of=as_of)
    evidence = terminal_state_evidence or {}
    evidence_refs = tuple(
        _format_evidence_ref(key, value) for key, value in sorted(evidence.items()) if key != "evidence_class"
    )
    return ObjectivePosturePayload(
        posture=posture,
        confidence=round(min(max(terminal_state_confidence, 0.0), 0.8), 4),
        authority="structural_inference",
        as_of=as_of,
        evidence_refs=evidence_refs,
    )


def derive_objective_posture(
    structural: ObjectivePosturePayload,
    assertions: Sequence[ArchiveAssertionEnvelope],
    *,
    as_of: str | None = None,
) -> ObjectivePosturePayload:
    """Blend the ``assertion`` tier over the ``structural_inference`` tier
    per the authority order.

    ``goal_graph``/``work_evidence`` are reserved tiers with nothing to
    query yet; this function accepts only the two currently-available
    tiers. A future caller adds the higher-authority arguments without
    changing this contract's shape or the precedence semantics for the
    tiers below them (AC3: explicit higher-authority evidence outranks
    weaker inference).
    """

    relevant = [assertion for assertion in assertions if assertion.kind in ASSERTION_TIER_KINDS]
    if not relevant:
        return structural

    contradictions: list[str] = list(structural.contradictions)
    if structural.authority != "none" and structural.posture != "unknown":
        contradictions.append(f"structural_inference:{structural.posture}")

    by_kind: dict[AssertionKind, list[ArchiveAssertionEnvelope]] = {}
    for assertion in relevant:
        by_kind.setdefault(assertion.kind, []).append(assertion)

    winning_kind: AssertionKind | None = None
    winning_posture: ObjectivePosture | None = None
    for kind, posture in _ASSERTION_POSTURE_PRIORITY:
        if kind in by_kind:
            winning_kind, winning_posture = kind, posture
            break

    if winning_kind is None or winning_posture is None:
        # Assertions of a kind this tier doesn't map (should not happen
        # given ASSERTION_TIER_KINDS == the priority table's kinds, but
        # fail safe rather than silently claiming a posture).
        return structural

    for kind, posture in _ASSERTION_POSTURE_PRIORITY:
        if kind == winning_kind or kind not in by_kind:
            continue
        for assertion in by_kind[kind]:
            contradictions.append(f"assertion:{assertion.assertion_id}:{kind.value}->{posture}")

    winning_assertions = by_kind[winning_kind]
    evidence_refs: list[str] = []
    for assertion in winning_assertions:
        evidence_refs.append(f"assertion:{assertion.assertion_id}")
        evidence_refs.extend(assertion.evidence_refs)

    confidences = [assertion.confidence for assertion in winning_assertions if assertion.confidence is not None]
    confidence = max(confidences) if confidences else 0.85
    latest_updated_ms = max((assertion.updated_at_ms for assertion in winning_assertions), default=None)
    resolved_as_of = (
        as_of
        or (
            datetime.fromtimestamp(latest_updated_ms / 1000, tz=UTC).isoformat()
            if latest_updated_ms is not None
            else None
        )
        or structural.as_of
    )

    return ObjectivePosturePayload(
        posture=winning_posture,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        authority="assertion",
        as_of=resolved_as_of,
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        contradictions=tuple(dict.fromkeys(contradictions)),
    )


class ObjectivePostureOperations(Protocol):
    """Minimal read surface `resolve_session_objective_posture` needs.

    Satisfied structurally by the ``Polylogue`` API facade and by
    ``ResumeOperations`` -- no explicit inheritance required.
    """

    async def list_assertion_claims(
        self,
        *,
        kinds: Sequence[str | AssertionKind] | None = None,
        target_ref: str | None = None,
        statuses: Sequence[str | AssertionStatus] | None = None,
    ) -> list[ArchiveAssertionEnvelope]: ...


async def resolve_session_objective_posture(
    operations: ObjectivePostureOperations,
    *,
    session_id: str,
    structural: ObjectivePosturePayload,
) -> ObjectivePosturePayload:
    """Read-time projection: overlay the live ``assertion`` tier on top of
    the ``structural_inference`` tier already materialized on the session
    profile.

    Scoped to the single physical ``session_id`` supplied (matching how
    assertions are authored today, e.g. ``scenarios/corpus.py``'s
    ``target_ref=f"session:{session_id}"``) -- it does not yet fan out
    across a logical session's full lineage. Widening to the logical-session
    scope is a bounded follow-on, not a redesign: it only changes which
    ``target_ref``s get queried.
    """

    target_ref = ObjectRef(kind="session", object_id=session_id).format()
    assertions = await operations.list_assertion_claims(
        kinds=ASSERTION_TIER_KINDS,
        target_ref=target_ref,
        statuses=(AssertionStatus.ACTIVE,),
    )
    return derive_objective_posture(structural, assertions)


__all__ = [
    "ASSERTION_TIER_KINDS",
    "AUTHORITY_ORDER",
    "ObjectivePosture",
    "ObjectivePostureAuthority",
    "ObjectivePostureOperations",
    "derive_objective_posture",
    "resolve_session_objective_posture",
    "structural_objective_posture",
]
