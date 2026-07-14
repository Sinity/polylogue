"""Hermes runtime lifecycle-event taxonomy and snapshot reconciliation.

Hermes lifecycle hooks are best-effort: a synchronous delivery can disappear
during an outage, so events reach ``source.db`` through the durable spool in
``polylogue.sources.hooks`` (extended for the ``hermes`` provider). This module
owns the vocabulary those events are expected to use and the reconciliation
that renders an incomplete event stream *visible* against the Hermes
``state.db`` snapshot rather than silently accepting gaps (fs1.7 AC).

Event bodies are evidence records (ids, hashes, timings, outcomes), never a
second copy of message text — the durable spool itself enforces that
boundary (see ``sources.hooks._reject_duplicated_transcript``); this module
only classifies and correlates what already passed that gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .base import ParsedSession

# --- Taxonomy -----------------------------------------------------------
#
# Distinguishing PER_TURN_END from DURABLE_FINALIZE is a hard requirement
# (fs1.7 AC): a Hermes turn can end many times in one session (each reply),
# but a session is durably finalized once. Conflating the two would make
# "session ended" ambiguous evidence.

MODEL_ATTEMPT = "model_attempt"
MODEL_FAILURE = "model_failure"
MODEL_RETRY = "model_retry"
MODEL_FALLBACK = "model_fallback"
TOOL_START = "tool_start"
TOOL_FINISH = "tool_finish"
TOOL_FAILURE = "tool_failure"
TOOL_DENIAL = "tool_denial"
APPROVAL_REQUEST = "approval_request"
APPROVAL_RESPONSE = "approval_response"
SUBAGENT_START = "subagent_start"
SUBAGENT_FINISH = "subagent_finish"
COMPACTION = "compaction"
REWIND = "rewind"
PER_TURN_END = "on_session_end"
"""Emitted once per assistant turn. NOT a durable-session signal."""
DURABLE_FINALIZE = "on_session_finalize"
"""Emitted exactly once, when the Hermes session is durably closed."""
CONTEXT_INJECTED = "context_injected"
"""A Polylogue-compiled context pack was delivered into a live Hermes turn."""

HERMES_LIFECYCLE_EVENT_TYPES: tuple[str, ...] = (
    MODEL_ATTEMPT,
    MODEL_FAILURE,
    MODEL_RETRY,
    MODEL_FALLBACK,
    TOOL_START,
    TOOL_FINISH,
    TOOL_FAILURE,
    TOOL_DENIAL,
    APPROVAL_REQUEST,
    APPROVAL_RESPONSE,
    SUBAGENT_START,
    SUBAGENT_FINISH,
    COMPACTION,
    REWIND,
    PER_TURN_END,
    DURABLE_FINALIZE,
    CONTEXT_INJECTED,
)

# Event types that must carry a `tool_call_id`/`message_id` pairing partner
# somewhere else in the same stream. Used to compute reconciliation gaps.
_PAIRED_EVENT_TYPES: tuple[tuple[str, str], ...] = (
    (TOOL_START, TOOL_FINISH),
    (APPROVAL_REQUEST, APPROVAL_RESPONSE),
    (SUBAGENT_START, SUBAGENT_FINISH),
)


@dataclass(frozen=True, slots=True)
class HermesLifecycleEvent:
    """A minimal typed view over one drained ``raw_hook_events`` row."""

    event_id: str
    event_type: str
    session_native_id: str
    observed_at_ms: int
    payload: dict[str, object]

    @property
    def correlation_id(self) -> str | None:
        """The tool/turn/subagent id this event should pair against, if any."""
        for key in ("tool_call_id", "turn_id", "subagent_id", "approval_id"):
            value = self.payload.get(key)
            if isinstance(value, str) and value:
                return value
        return None


@dataclass(frozen=True, slots=True)
class HermesLifecycleReconciliation:
    """Explicit coverage report for one Hermes session's lifecycle-event stream.

    ``unpaired`` events are not dropped anywhere — they remain durable rows in
    ``raw_hook_events`` — this report only renders the gap visible instead of
    letting a caller assume the stream is complete.
    """

    session_native_id: str
    total_events: int
    events_by_type: dict[str, int]
    unpaired_event_ids: tuple[str, ...]
    known_message_ids: frozenset[str]
    events_referencing_unknown_messages: tuple[str, ...]
    finalized: bool
    caveats: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unpaired_event_ids and not self.events_referencing_unknown_messages


def known_message_ids(session: ParsedSession) -> frozenset[str]:
    """Return every provider-native message id a Hermes snapshot retains."""
    return frozenset(message.provider_message_id for message in session.messages if message.provider_message_id)


def reconcile_lifecycle_events(
    session_native_id: str,
    events: list[HermesLifecycleEvent],
    *,
    snapshot_message_ids: frozenset[str] = frozenset(),
) -> HermesLifecycleReconciliation:
    """Reconcile a drained lifecycle-event stream against a session snapshot.

    Detects two distinct kinds of incompleteness, both rendered explicitly
    rather than swallowed:

    1. Unpaired start/finish, approval-request/response, or
       subagent-start/finish events (a hook fired but its counterpart never
       arrived — plausible during an outage the durable spool is designed to
       survive at the transport level, but not at the application level).
    2. Events whose payload references a ``message_id`` the snapshot does not
       retain (the runtime event arrived, but the corresponding conversational
       revision has not been ingested yet, or was rewound).
    """
    ordered = sorted(events, key=lambda event: event.observed_at_ms)
    counts = Counter(event.event_type for event in ordered)

    open_by_start_type: dict[str, dict[str, str]] = {start: {} for start, _finish in _PAIRED_EVENT_TYPES}
    unpaired: list[str] = []
    for event in ordered:
        for start_type, finish_type in _PAIRED_EVENT_TYPES:
            correlation = event.correlation_id
            if correlation is None:
                continue
            if event.event_type == start_type:
                open_by_start_type[start_type][correlation] = event.event_id
            elif event.event_type == finish_type and correlation in open_by_start_type[start_type]:
                del open_by_start_type[start_type][correlation]
    for pending in open_by_start_type.values():
        unpaired.extend(pending.values())

    unknown_message_refs: list[str] = []
    if snapshot_message_ids:
        for event in ordered:
            referenced = event.payload.get("message_id")
            if isinstance(referenced, str) and referenced and referenced not in snapshot_message_ids:
                unknown_message_refs.append(event.event_id)

    finalized = counts.get(DURABLE_FINALIZE, 0) > 0
    caveats: list[str] = []
    if unpaired:
        caveats.append(f"{len(unpaired)} lifecycle event(s) never observed their paired counterpart.")
    if unknown_message_refs:
        caveats.append(
            f"{len(unknown_message_refs)} lifecycle event(s) reference a message id absent from the retained snapshot."
        )
    if not finalized and counts.get(PER_TURN_END, 0) > 0:
        caveats.append("Session observed per-turn ends but no durable on_session_finalize event.")

    return HermesLifecycleReconciliation(
        session_native_id=session_native_id,
        total_events=len(ordered),
        events_by_type=dict(counts),
        unpaired_event_ids=tuple(unpaired),
        known_message_ids=snapshot_message_ids,
        events_referencing_unknown_messages=tuple(unknown_message_refs),
        finalized=finalized,
        caveats=tuple(caveats),
    )


__all__ = [
    "APPROVAL_REQUEST",
    "APPROVAL_RESPONSE",
    "COMPACTION",
    "CONTEXT_INJECTED",
    "DURABLE_FINALIZE",
    "HERMES_LIFECYCLE_EVENT_TYPES",
    "HermesLifecycleEvent",
    "HermesLifecycleReconciliation",
    "MODEL_ATTEMPT",
    "MODEL_FAILURE",
    "MODEL_FALLBACK",
    "MODEL_RETRY",
    "PER_TURN_END",
    "REWIND",
    "SUBAGENT_FINISH",
    "SUBAGENT_START",
    "TOOL_DENIAL",
    "TOOL_FAILURE",
    "TOOL_FINISH",
    "TOOL_START",
    "known_message_ids",
    "reconcile_lifecycle_events",
]
