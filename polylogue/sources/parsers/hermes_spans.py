"""Hermes NeMo Relay ATOF/ATIF observer-layer trace importer (fs1.2).

Imports Hermes's observer-layer trace export as runtime-span evidence:
``pre/post_api_request`` pairs become LLM request spans; ``pre/post_tool_call``
pairs become tool-execution spans with duration/status; approval hooks become
decision-point events; subagent hooks become delegation evidence; error hooks
become retry/fallback taxonomy events.

Honesty note (verify-first gap): the exact wire shape of Hermes's own NeMo
Relay ATOF JSONL / ATIF JSON export was not independently verifiable from
this workspace (no local checkout of the Hermes observer plugin source was
available). The marker-based document shape and per-``hook_type`` field
mapping below are therefore a *documented, testable, best-effort schema*
derived from the fs1.2 design notes and the shared lifecycle-event taxonomy
in ``hermes_lifecycle`` -- not a shape confirmed against Hermes's own code.
``import_fidelity_declaration`` below marks every capability as at most
``inferred`` for exactly this reason, and a follow-up bead
(``polylogue-fs1.2.1``, filed alongside this change) tracks re-verifying the
real shape once the Hermes plugin source is available and adjusting the
parser/fixtures without changing the public contract if possible.

Session correlation (design's "join key: Hermes session id ->
sessions.native_id"): an observer trace does not carry conversational content
of its own (spans carry ids/timings/outcomes, never a duplicated transcript,
matching the ``sources.hooks`` payload-hygiene rule), so this parser produces
its own observer-evidence session identity (``observer:<hermes_session_id>``)
rather than physically merging span events into the state-db-ingested
conversational session's message tree -- a physical merge across two
independently-acquired artifacts for the same logical Hermes session is a
session-identity/lineage design decision (topology_edges / session_links)
that is explicitly deferred, not silently assumed. Read-side correlation by
the shared Hermes session id remains possible today via
``hermes_observer_session_id_for`` and is exercised by
``tests/unit/sources/test_hermes_spans.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from .hermes_state import HermesFidelityCapability, HermesFidelityStatus, HermesImportFidelity

HERMES_ATIF_TRACE_MARKER = "hermes_atif_trace"

# The observer-layer hook vocabulary this parser understands. Anything else
# is preserved as a generic "hermes_observer_span" event (ambiguous input is
# handled deterministically -- never dropped, never guessed into one of the
# known kinds) per fs1.2 AC.
_PRE_API_REQUEST = "pre_api_request"
_POST_API_REQUEST = "post_api_request"
_PRE_TOOL_CALL = "pre_tool_call"
_POST_TOOL_CALL = "post_tool_call"
_APPROVAL_REQUEST = "approval_request"
_APPROVAL_RESPONSE = "approval_response"
_SUBAGENT_START = "subagent_start"
_SUBAGENT_STOP = "subagent_stop"
_ERROR = "error"

_PAIRED_HOOK_TYPES: tuple[tuple[str, str], ...] = (
    (_PRE_API_REQUEST, _POST_API_REQUEST),
    (_PRE_TOOL_CALL, _POST_TOOL_CALL),
    (_APPROVAL_REQUEST, _APPROVAL_RESPONSE),
    (_SUBAGENT_START, _SUBAGENT_STOP),
)
_KNOWN_HOOK_TYPES = frozenset(
    {
        _PRE_API_REQUEST,
        _POST_API_REQUEST,
        _PRE_TOOL_CALL,
        _POST_TOOL_CALL,
        _APPROVAL_REQUEST,
        _APPROVAL_RESPONSE,
        _SUBAGENT_START,
        _SUBAGENT_STOP,
        _ERROR,
    }
)

HermesSpanEventType: TypeAlias = Literal[
    "hermes_llm_request_span",
    "hermes_tool_execution_span",
    "hermes_decision_point",
    "hermes_subagent_span",
    "hermes_error_taxonomy",
    "hermes_observer_span",
]

_EVENT_TYPE_BY_HOOK_TYPE: dict[str, HermesSpanEventType] = {
    _PRE_API_REQUEST: "hermes_llm_request_span",
    _POST_API_REQUEST: "hermes_llm_request_span",
    _PRE_TOOL_CALL: "hermes_tool_execution_span",
    _POST_TOOL_CALL: "hermes_tool_execution_span",
    _APPROVAL_REQUEST: "hermes_decision_point",
    _APPROVAL_RESPONSE: "hermes_decision_point",
    _SUBAGENT_START: "hermes_subagent_span",
    _SUBAGENT_STOP: "hermes_subagent_span",
    _ERROR: "hermes_error_taxonomy",
}


def observer_session_provider_id(hermes_session_id: str) -> str:
    """Return the observer-evidence session identity for a Hermes session id.

    Deliberately distinct from the state-db-ingested conversational session
    id (``<raw_id>@profile-<key>``, see ``hermes_state.py``): this parser
    never claims to physically merge into that session's message tree (see
    module docstring). Consumers correlate the two via
    ``hermes_observer_session_id_for``.
    """
    return f"observer:{hermes_session_id}"


def hermes_observer_session_id_for(conversational_session_id: str) -> str:
    """Map a state-db-ingested Hermes session id to its observer counterpart.

    Strips the ``@profile-<key>`` qualifier (observer traces carry no
    profile-root context of their own) so a reader holding the qualified
    conversational session id can look up its observer evidence.
    """
    raw_id = conversational_session_id.split("@profile-", 1)[0]
    return observer_session_provider_id(raw_id)


def marker_payload(hermes_session_id: str, spans: Sequence[JSONValue]) -> JSONDocument:
    """Return a minimal valid ATIF trace document for producers/tests.

    ``spans`` accepts any ``JSONValue`` sequence (``Sequence`` is covariant,
    unlike ``list``) rather than requiring ``list[JSONDocument]`` on purpose:
    a real acquisition can hand this a malformed stream (a non-object entry),
    and :func:`parse_atif_document` must handle that deterministically rather
    than assuming a caller always validates first.
    """
    return {
        "polylogue_artifact": HERMES_ATIF_TRACE_MARKER,
        "schema": "hermes-observer-trace",
        "version": 1,
        "session_id": hermes_session_id,
        "spans": list(spans),
    }


def looks_like_atif_payload(payload: JSONDocument) -> bool:
    return (
        payload.get("polylogue_artifact") == HERMES_ATIF_TRACE_MARKER
        and isinstance(payload.get("session_id"), str)
        and isinstance(payload.get("spans"), list)
    )


def parse_atif_document(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    """Parse one Hermes ATIF/ATOF observer trace document into a session.

    Ambiguous input is handled deterministically: an unrecognized
    ``hook_type`` becomes a generic ``hermes_observer_span`` event rather
    than being dropped or mis-classified as a known kind; a span missing its
    required ``hook_type``/``span_id`` is skipped and counted (see
    :func:`import_fidelity_declaration`), never silently coerced.
    """
    session_id = str(payload.get("session_id") or fallback_id)
    raw_spans = cast("list[JSONValue]", payload.get("spans") or [])
    events: list[ParsedSessionEvent] = []
    skipped = 0
    unknown_hook_types = 0
    for raw_span in raw_spans:
        span = json_document(raw_span)
        if span is None:
            skipped += 1
            continue
        hook_type = span.get("hook_type")
        span_id = span.get("span_id")
        if not isinstance(hook_type, str) or not hook_type or not isinstance(span_id, str) or not span_id:
            skipped += 1
            continue
        if hook_type not in _KNOWN_HOOK_TYPES:
            unknown_hook_types += 1
        event_type = _EVENT_TYPE_BY_HOOK_TYPE.get(hook_type, "hermes_observer_span")
        events.append(
            ParsedSessionEvent(
                event_type=event_type,
                timestamp=_optional_str(span.get("timestamp")),
                payload={
                    "hook_type": hook_type,
                    "span_id": span_id,
                    **{k: v for k, v in span.items() if k not in {"hook_type", "span_id", "timestamp"}},
                },
            )
        )
    unpaired = _unpaired_span_ids(events)
    # A trace with zero usable spans still lands as a session (never a
    # silent drop of the whole document): its own fidelity declaration marks
    # runtime_spans absent/degraded rather than raising.
    summary_text = (
        f"Hermes observer trace: {len(events)} span event(s) "
        f"({len(unpaired)} unpaired, {skipped} unparseable, {unknown_hook_types} unrecognized hook_type)."
    )
    provider_session_id = observer_session_provider_id(session_id)
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=provider_session_id,
        title=f"Hermes observer trace: {session_id}",
        created_at=_earliest_timestamp(events),
        updated_at=_latest_timestamp(events),
        messages=[
            ParsedMessage(
                provider_message_id=f"{provider_session_id}:trace-summary",
                role=Role.SYSTEM,
                text=summary_text,
                timestamp=_earliest_timestamp(events),
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)],
                position=0,
                variant_index=0,
                is_active_path=True,
                material_origin=MaterialOrigin.RUNTIME_CONTEXT,
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="hermes_observer_trace_correlation",
                payload={
                    "hermes_conversation_session_id_prefix": session_id,
                    "join_key": "sessions.native_id",
                    "note": (
                        "This session carries observer-layer evidence only; correlate with the "
                        "state-db-ingested conversational session sharing this raw Hermes session id."
                    ),
                },
            ),
            *events,
        ],
        ingest_flags=["hermes:atif-observer-trace"],
    )


@dataclass(frozen=True, slots=True)
class HermesSpanReconciliation:
    """Per-document pairing/parse-quality summary, independent of any snapshot."""

    total_spans: int
    unpaired_span_ids: tuple[str, ...]
    skipped: int
    unknown_hook_types: int

    @property
    def complete(self) -> bool:
        return not self.unpaired_span_ids and not self.skipped and not self.unknown_hook_types


def import_fidelity_declaration(session: ParsedSession) -> HermesImportFidelity:
    """Declare the fidelity this best-effort ATIF schema can substantiate.

    Every capability tops out at ``inferred`` (never ``exact``): the wire
    shape itself is unverified against Hermes's own observer-plugin source
    (see module docstring), so claiming exactness here would overstate what
    is actually known.
    """
    span_events = [
        event
        for event in session.session_events
        if event.event_type in set(_EVENT_TYPE_BY_HOOK_TYPE.values()) | {"hermes_observer_span"}
    ]
    unpaired = _unpaired_span_ids(span_events)
    total = len(span_events)
    llm_spans = sum(1 for event in span_events if event.event_type == "hermes_llm_request_span")
    tool_spans = sum(1 for event in span_events if event.event_type == "hermes_tool_execution_span")
    decision_spans = sum(1 for event in span_events if event.event_type == "hermes_decision_point")
    subagent_spans = sum(1 for event in span_events if event.event_type == "hermes_subagent_span")
    error_spans = sum(1 for event in span_events if event.event_type == "hermes_error_taxonomy")

    def capability(observed: int, detail: str) -> HermesFidelityCapability:
        status: HermesFidelityStatus = (
            "inferred" if observed and not unpaired else ("absent" if not observed else "degraded")
        )
        return HermesFidelityCapability(
            status=status, observed=observed, expected=max(total, 1), counts={}, detail=detail
        )

    capabilities = {
        "llm_request_spans": capability(llm_spans, "pre/post_api_request pairs mapped to LLM request spans."),
        "tool_execution_spans": capability(tool_spans, "pre/post_tool_call pairs mapped to tool execution spans."),
        "decision_points": capability(decision_spans, "approval_request/response pairs mapped to decision points."),
        "subagent_delegation": capability(
            subagent_spans,
            "subagent_start/stop pairs recorded as evidence; NOT materialized into topology_edges/session_links "
            "in this pass -- deferred, see module docstring.",
        ),
        "error_taxonomy": capability(error_spans, "error hooks recorded as retry/fallback taxonomy evidence."),
        "topology_edges": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=max(subagent_spans, 1),
            counts={},
            detail="Physical topology_edges materialization from observer spans is out of scope for this pass.",
        ),
    }
    caveats = tuple(f"{name}: {cap.detail}" for name, cap in capabilities.items() if cap.status != "exact")
    if unpaired:
        caveats = (*caveats, f"{len(unpaired)} span(s) never observed their paired counterpart (acquisition debt).")
    return HermesImportFidelity(
        producer="Hermes NeMo Relay ATIF/ATOF trace (unverified wire shape)",
        schema_version=1,
        profile_namespace=None,
        acquisition_method="json_fallback",
        retained_blob_reproducibility=HermesFidelityCapability(
            status="exact",
            observed=1,
            expected=1,
            counts={},
            detail="The raw ATIF document bytes are retained before parsing, like every other origin.",
        ),
        capabilities=capabilities,
        caveats=caveats,
    )


def _unpaired_span_ids(events: list[ParsedSessionEvent]) -> tuple[str, ...]:
    open_by_start: dict[str, dict[str, str]] = {start: {} for start, _finish in _PAIRED_HOOK_TYPES}
    for event in events:
        hook_type = event.payload.get("hook_type")
        span_id = event.payload.get("span_id")
        if not isinstance(hook_type, str) or not isinstance(span_id, str):
            continue
        for start_type, finish_type in _PAIRED_HOOK_TYPES:
            if hook_type == start_type:
                open_by_start[start_type][span_id] = span_id
            elif hook_type == finish_type:
                open_by_start[start_type].pop(span_id, None)
    unpaired: list[str] = []
    for pending in open_by_start.values():
        unpaired.extend(sorted(pending.values()))
    return tuple(unpaired)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _earliest_timestamp(events: list[ParsedSessionEvent]) -> str | None:
    timestamps = sorted(event.timestamp for event in events if event.timestamp)
    return timestamps[0] if timestamps else None


def _latest_timestamp(events: list[ParsedSessionEvent]) -> str | None:
    timestamps = sorted(event.timestamp for event in events if event.timestamp)
    return timestamps[-1] if timestamps else None


__all__ = [
    "HERMES_ATIF_TRACE_MARKER",
    "HermesSpanEventType",
    "HermesSpanReconciliation",
    "hermes_observer_session_id_for",
    "import_fidelity_declaration",
    "looks_like_atif_payload",
    "marker_payload",
    "observer_session_provider_id",
    "parse_atif_document",
]
