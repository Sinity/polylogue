"""Shared parser extraction helpers."""

from __future__ import annotations

import base64
import binascii
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, MessageType, WebConstructType
from polylogue.core.hashing import hash_payload, hash_text
from polylogue.core.types import AttachmentDirection
from polylogue.sources.tool_result_reasons import unknown_reason

from .base_models import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionRefusalReason,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
)

_SessionParser = TypeVar("_SessionParser", bound=Callable[..., ParsedSession])


class AdmissionLedger:
    """Small mutable builder for the immutable parsed-session admission proof."""

    # polylogue-ro922: a session file is untrusted input, so per-record
    # retention here is a memory amplifier -- 200k records of an unhandled
    # record type used to cost one Pydantic model each for the whole parse.
    # Materialized outcomes carry no evidence beyond their own ordinal, so
    # they are accumulated as half-open ordinal runs and only the
    # evidence-bearing dispositions are kept as models.
    def __init__(self) -> None:
        self._expected: dict[AdmissionUnit, int] = {}
        self._outcomes: list[AdmissionOutcome] = []
        self._counts: dict[AdmissionUnit, int] = {}
        self._materialized: dict[AdmissionUnit, list[list[int]]] = {}

    def expect(self, unit: AdmissionUnit, count: int) -> None:
        if count < 0:
            raise ValueError("admission denominators cannot be negative")
        self._expected[unit] = self._expected.get(unit, 0) + count

    def next_ordinal(self, unit: AdmissionUnit) -> int:
        return self._counts.get(unit, 0)

    def _record(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        disposition: AdmissionDisposition,
        reason: AdmissionUnknownReason | AdmissionRefusalReason | None = None,
    ) -> AdmissionOutcome:
        outcome = AdmissionOutcome(
            unit=unit,
            ordinal=ordinal,
            key=key,
            disposition=disposition,
            reason=reason,
        )
        self._counts[unit] = self._counts.get(unit, 0) + 1
        if disposition is AdmissionDisposition.MATERIALIZED:
            runs = self._materialized.setdefault(unit, [])
            if runs and runs[-1][1] == ordinal:
                runs[-1][1] = ordinal + 1
            else:
                runs.append([ordinal, ordinal + 1])
        else:
            self._outcomes.append(outcome)
        return outcome

    def materialized(self, unit: AdmissionUnit, ordinal: int, key: str) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.MATERIALIZED)

    def unknown(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        reason: AdmissionUnknownReason = AdmissionUnknownReason.UNRECOGNIZED_TYPE,
    ) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.TYPED_UNKNOWN, reason)

    def refusal(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        reason: AdmissionRefusalReason,
    ) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.TYPED_REFUSAL, reason)

    def close(self) -> ParseAccounting:
        accounting = ParseAccounting(
            expected=dict(self._expected),
            outcomes=list(self._outcomes),
            materialized_ordinals={
                unit: [(start, end) for start, end in runs] for unit, runs in self._materialized.items()
            },
        )
        accounting.assert_conserved()
        return accounting


def _is_unknown_sentinel(candidate: object) -> bool:
    return isinstance(candidate, str) and (
        candidate.startswith(("future_", "unknown_", "unsupported_"))
        or candidate in {"future", "unknown", "unsupported"}
    )


def claude_code_unknown_wire_type(value: object) -> str | None:
    """Return an unknown Claude Code wire type read only from its discriminators.

    A Claude Code record's discriminators are its own ``type`` and the
    ``type`` of each ``message.content`` block. Everything beneath a block --
    a tool call's ``input``, a tool result's body -- is user-controlled data,
    so a tool argument ``{"type": "unknown"}`` is not a provider wire type.
    """
    if not isinstance(value, Mapping):
        return None
    record_type = value.get("type")
    if _is_unknown_sentinel(record_type):
        return cast(str, record_type)
    message = value.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        for block in content:
            block_type = block.get("type") if isinstance(block, Mapping) else None
            if _is_unknown_sentinel(block_type):
                return cast(str, block_type)
    return None


def codex_unknown_wire_type(value: object) -> str | None:
    """Return an unknown Codex wire type read only from its discriminators.

    A Codex rollout record's discriminators are its envelope ``type`` and its
    payload's ``type``. Everything beneath -- tool-call arguments, MCP
    invocation inputs, outputs -- is user-controlled data, so an argument
    ``{"type": "unknown"}`` is not a provider wire type.
    """
    if not isinstance(value, Mapping):
        return None
    record_type = value.get("type")
    if _is_unknown_sentinel(record_type):
        return cast(str, record_type)
    payload = value.get("payload")
    payload_type = payload.get("type") if isinstance(payload, Mapping) else None
    if _is_unknown_sentinel(payload_type):
        return cast(str, payload_type)
    return None


def hermes_unknown_wire_type(value: object) -> str | None:
    """Return an unknown Hermes wire type read only from its discriminators.

    An ATOF record's discriminators are its own ``type``/``kind``; an ATIF
    document's are its own and each step's. Tool-call ``arguments`` and
    every other nested value are user data, so ``{"type": "unknown"}`` inside
    them is not a provider wire type.
    """
    if not isinstance(value, Mapping):
        return None
    for key in ("type", "kind", "record_type"):
        if _is_unknown_sentinel(value.get(key)):
            return cast(str, value.get(key))
    steps = value.get("steps")
    for step in steps if isinstance(steps, list) else ():
        if isinstance(step, Mapping):
            for key in ("type", "kind"):
                if _is_unknown_sentinel(step.get(key)):
                    return cast(str, step.get(key))
    return None


def otel_genai_unknown_wire_type(value: object) -> str | None:
    """Return an unknown OTLP wire type read only from its structural fields.

    An OTLP document's discriminators are each span's ``kind`` and the
    wrapper keys naming its resource and scope lists. Attribute values --
    tool-call arguments, message content, resource attributes -- are user
    data, so ``{"type": "unknown"}`` inside them is not a provider wire type.
    """
    if not isinstance(value, Mapping):
        return None
    resource_spans = value.get("resourceSpans", value.get("resource_spans"))
    for resource in resource_spans if isinstance(resource_spans, list) else ():
        if not isinstance(resource, Mapping):
            continue
        scopes = resource.get("scopeSpans", resource.get("instrumentationLibrarySpans"))
        for scope in scopes if isinstance(scopes, list) else ():
            spans = scope.get("spans") if isinstance(scope, Mapping) else None
            for span in spans if isinstance(spans, list) else ():
                kind = span.get("kind") if isinstance(span, Mapping) else None
                if _is_unknown_sentinel(kind):
                    return cast(str, kind)
    return None


#: Keys whose values are user or tool data in every origin: a tool call's
#: arguments or input, its output or result, span attributes. The default
#: scan never reads a wire type from beneath them -- a tool argument
#: ``{"type": "unknown"}`` is data, not a provider discriminator.
_USER_DATA_KEYS = frozenset(
    {
        "arguments",
        "args",
        "input",
        "tool_input",
        "toolInput",
        "parameters",
        "params",
        "output",
        "result",
        "results",
        "attributes",
    }
)


def _unknown_wire_type(value: object) -> str | None:
    """Return a deliberately future-shaped wire type, if one is visible.

    This is intentionally narrow.  Admission must not classify ordinary
    provider metadata as unknown merely because it contains a ``type`` field;
    the parser-specific lowering remains authoritative for known shapes. It
    does not descend into user data (:data:`_USER_DATA_KEYS`).
    """
    if isinstance(value, dict):
        for key in ("type", "content_type", "kind", "record_type"):
            candidate = value.get(key)
            if _is_unknown_sentinel(candidate):
                return cast(str, candidate)
        for key, child in value.items():
            if key in _USER_DATA_KEYS:
                continue
            found = _unknown_wire_type(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _unknown_wire_type(child)
            if found is not None:
                return found
    return None


class AdmissionObserver:
    """Classify each outer record of one session's input exactly once.

    The single conservation boundary every production parse route shares:
    the decorated leaf parsers (:func:`parser_admission`), the Claude Code
    multi-session stream (one observer per session group) and the dispatch
    routes that call undecorated entry points. Each record gets one terminal
    disposition. A record that is not a JSON object is a typed refusal, never
    counted as materialized: every origin's outer records are objects, and a
    parser that skips a scalar produced no material from it (fail closed).

    Only the parsing owner can say a record was lowered. The observer never
    infers it from a record's shape: an ordinary-looking record the parser
    discarded is not material. A record observed before the parser ran is
    *pending*. When the input was one whole document the parser consumed, its
    returned session settles that one record. When the input was a sequence
    or a stream of records, the parser settles them: either the caller passes
    each record's disposition (``lowered``/``malformed``), or the session
    carries the parser's own ledger, whose outer-record denominator must equal
    the complete count observed here. A stream is drained to its end after
    the parser returns, so a parser that stops early cannot shrink the
    denominator (polylogue-mg7jx).
    """

    def __init__(self, scan: Callable[[object], str | None] | None = None, *, record_stream: bool = False) -> None:
        """``record_stream`` declares up front that the caller observes a sequence of records."""
        #: How one record's unknown wire type is found. The default scans the
        #: whole record; an origin with declared discriminators passes a scan
        #: that reads only those, so nested user data is never a wire type.
        self._scan = scan if scan is not None else _unknown_wire_type
        self._ledger = AdmissionLedger()
        self._count = 0
        #: Records observed with no parser disposition yet, as half-open
        #: ordinal runs (compact: a stream of records costs one run).
        self._pending: list[list[int]] = []
        #: Whether the observed input was a sequence or stream of records
        #: rather than one whole document.
        self._record_stream = record_stream
        #: The first source index of each unknown wire type: one event per
        #: type is emitted, so later occurrences are not retained.
        self._unknowns: dict[str, int] = {}
        #: The closed proof over every observed record, shared by each
        #: session drawn from them that carries no ledger of its own.
        self._proof: ParseAccounting | None = None

    def observe(
        self,
        item: object,
        source_index: int | None = None,
        *,
        lowered: bool | None = None,
        malformed: bool = False,
    ) -> None:
        """Classify one record at dense ledger ordinal ``self._count``.

        ``source_index`` is the record's 1-based position in the source file
        when that differs from its position in this session (an interleaved
        multi-session stream); the typed unknown event names that position.

        ``lowered`` is the parsing owner's disposition when the caller has it:
        ``True`` when the parser lowered the record into the session,
        ``False`` when it folded (or tried to fold) the record and produced
        nothing from it -- e.g. a Claude Code record with a missing or
        non-string ``type`` that ``_fold_code_record`` drops. ``None`` leaves
        the record pending for the parser's result to settle (see the class
        docstring). ``malformed`` is the parser's refusal of a known kind it
        skips for a missing required field.
        """
        ordinal = self._count
        self._count += 1
        if not isinstance(item, Mapping) or malformed:
            self._ledger.refusal(
                AdmissionUnit.OUTER_RECORD, ordinal, type(item).__name__, AdmissionRefusalReason.MALFORMED
            )
            return
        wire_type = self._scan(item)
        if wire_type is None and lowered is False:
            wire_type = "unrecognized_record_type"
        if wire_type is not None:
            self._ledger.unknown(AdmissionUnit.OUTER_RECORD, ordinal, wire_type)
            self._unknowns.setdefault(wire_type, source_index if source_index is not None else ordinal + 1)
        elif lowered:
            self._ledger.materialized(AdmissionUnit.OUTER_RECORD, ordinal, "parsed")
        elif self._pending and self._pending[-1][1] == ordinal:
            self._pending[-1][1] = ordinal + 1
        else:
            self._pending.append([ordinal, ordinal + 1])

    def observe_input(self, payload: object, *, recognizes: Callable[[Any], bool] | None = None) -> None:
        """Observe a payload that is fully in memory: one document or a record sequence.

        ``recognizes`` is the parser's own per-record recognition for a
        record sequence: a record it recognizes is lowered, one it does not
        is refused as malformed. Without it, sequence records stay pending.
        """
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            self._record_stream = True
            for item in payload:
                if recognizes is None:
                    self.observe(item)
                    continue
                recognized = isinstance(item, Mapping) and recognizes(item)
                # A future wire type stays a typed unknown, as the scan
                # classifies it; only a known-shaped record the parser does
                # not recognize is refused.
                self.observe(
                    item,
                    lowered=recognized,
                    malformed=isinstance(item, Mapping) and not recognized and self._scan(item) is None,
                )
        else:
            self.observe(payload)

    def observing(self, payload: Iterable[Any]) -> Iterator[Any]:
        """Yield a one-pass record stream through, observing each record as it is pulled.

        The caller drains the rest with :meth:`drain` once the parser returns.
        """
        self._record_stream = True
        for item in payload:
            self.observe(item)
            yield item

    @staticmethod
    def drain(stream: Iterator[Any]) -> None:
        """Pull whatever the parser left of an :meth:`observing` stream, so every record is counted."""
        for _ in stream:
            pass

    def apply(self, session: ParsedSession, provider: str) -> ParsedSession:
        """Attach the typed unknown events and, absent a parser ledger, the proof.

        A disk-backed event sink (the prepared-JSONL route's
        ``SqliteSessionEventSink``) is appended to in place and never copied
        into a list: an event-heavy stream keeps its events on disk.
        """
        events = session.session_events
        if self._unknowns:
            existing_types = {
                str(event.payload.get("wire_type")) for event in events if event.payload.get("wire_type") is not None
            }
            if isinstance(events, list) or not isinstance(events, MutableSequence):
                events = list(events)
            self._append_unknown_events(events, existing_types, provider)
            if provider == "claude_code":
                # Claude declares its event order (missing timestamps first);
                # the untimestamped admission events take their place in it.
                from polylogue.sources.parsers.claude.code_parser import order_session_events

                events = cast(list[ParsedSessionEvent], order_session_events(events))
        accounting = session.unit_accounting
        if accounting is None:
            accounting = self._closed_proof(provider)
        elif AdmissionUnit.OUTER_RECORD not in accounting.expected:
            outer = self._closed_proof(provider)
            accounting = ParseAccounting(
                expected={**outer.expected, **accounting.expected},
                outcomes=[*outer.outcomes, *accounting.outcomes],
                materialized_ordinals={**outer.materialized_ordinals, **accounting.materialized_ordinals},
            )
        elif self._record_stream:
            # The parser accounted for the records it consumed; the complete
            # input is what was observed here, drained to its end.
            accounted = accounting.expected.get(AdmissionUnit.OUTER_RECORD, 0)
            if accounted != self._count:
                raise ValueError(f"{provider} parser accounted for {accounted} of {self._count} outer records")
        accounting.assert_conserved()
        return session.model_copy(update={"session_events": events, "unit_accounting": accounting})

    def apply_each(self, sessions: Sequence[ParsedSession], provider: str) -> list[ParsedSession]:
        """``apply`` for every session one observed input produced.

        Each session without its parser's own ledger gets the input's proof,
        closed once and shared.
        """
        return [self.apply(session, provider) for session in sessions]

    def _closed_proof(self, provider: str) -> ParseAccounting:
        if self._proof is None:
            if self._pending:
                if self._record_stream or self._count > 1:
                    # Records of a sequence or stream that no parser
                    # disposition settled: the observer cannot certify them.
                    pending = sum(end - start for start, end in self._pending)
                    raise ValueError(
                        f"{provider} parser gave no disposition for {pending} of {self._count} outer records"
                    )
                # One whole document, consumed by the parser that returned a
                # session from it.
                for start, end in self._pending:
                    for ordinal in range(start, end):
                        self._ledger.materialized(AdmissionUnit.OUTER_RECORD, ordinal, "parsed")
                self._pending = []
            self._ledger.expect(AdmissionUnit.OUTER_RECORD, self._count)
            self._proof = self._ledger.close()
        return self._proof

    def _append_unknown_events(
        self, events: MutableSequence[ParsedSessionEvent], existing_types: set[str], provider: str
    ) -> None:
        for wire_type, index in self._unknowns.items():
            if wire_type in existing_types:
                continue
            events.append(
                ParsedSessionEvent(
                    event_type=f"{provider}_unknown_input",
                    payload={"source_index": index, "wire_type": wire_type},
                )
            )
            existing_types.add(wire_type)


def admit_parsed_sessions(
    provider: str,
    payload: object,
    sessions: list[ParsedSession],
    *,
    recognizes: Callable[[Any], bool] | None = None,
) -> list[ParsedSession]:
    """Apply the admission boundary to a dispatch route's result.

    For routes that reach an undecorated entry point. A session that already
    carries its parser's own ledger keeps it (the Claude Code stream admits
    each of its sessions against its own records). Every other session --
    one of several conversations in an OTLP document, a Hermes parent and its
    materialized subagent trajectories -- is proven against the whole
    document it was drawn from -- observed once and the closed proof shared,
    so an N-session document costs one scan -- and no emitted session
    reaches the writer without a conservation proof.

    A record-sequence payload needs the parser's own recognition for each
    record: ``recognizes`` returns whether the parser lowers that record, and
    a record it does not lower is refused as malformed. Without it, a
    sequence the sessions carry no ledger for is refused (see
    :class:`AdmissionObserver`).
    """
    if all(session.unit_accounting is not None for session in sessions):
        return sessions
    observer = AdmissionObserver(_ADMISSION_SCANS.get(provider))
    observer.observe_input(payload, recognizes=recognizes)
    return [
        session if session.unit_accounting is not None else observer.apply(session, provider) for session in sessions
    ]


#: Origins whose outer records have declared discriminators; any other origin
#: uses the conservative whole-record scan.
_ADMISSION_SCANS: dict[str, Callable[[object], str | None]] = {
    "hermes": hermes_unknown_wire_type,
    "codex": codex_unknown_wire_type,
    "claude_code": claude_code_unknown_wire_type,
    "opentelemetry": otel_genai_unknown_wire_type,
}


def _payload_parameter(parser: Callable[..., ParsedSession]) -> tuple[int, str]:
    """Locate the wire-payload parameter of one decorated session parser.

    ``drive.parse_chunked_prompt`` takes ``(provider, payload, fallback_id)``,
    so binding ``args[0]`` classified the provider token instead of the
    document: no unknown record in the real payload was ever observed and the
    synthesized ledger recorded one parsed outer record regardless of how many
    chunks arrived. Resolve the parameter by name so a parser's own signature
    decides which argument the conservation boundary reads, and fall back to
    the first positional for parsers that name it something else
    (``chatgpt_codex_sidecar.parse_codex_task`` takes ``task``).
    """
    parameters = [
        parameter
        for parameter in inspect.signature(parser).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if not parameters:
        raise TypeError(f"{parser.__qualname__} has no payload parameter to admit")
    names = [parameter.name for parameter in parameters]
    name = "payload" if "payload" in names else names[0]
    return names.index(name), name


def parser_admission(
    provider: str, *, scan: Callable[[object], str | None] | None = None
) -> Callable[[_SessionParser], _SessionParser]:
    """Put a common conservation boundary around every session parser.

    Provider implementations are still responsible for their known wire
    shapes.  This boundary handles the class-killer case: a future-shaped
    outer envelope that the implementation does not recognize must become a
    typed session event, never vanish beside otherwise-valid content.
    """

    def decorate(parser: _SessionParser) -> _SessionParser:
        payload_index, payload_name = _payload_parameter(parser)

        @wraps(parser)
        def wrapped(*args: Any, **kwargs: Any) -> ParsedSession:
            payload = args[payload_index] if len(args) > payload_index else kwargs.get(payload_name)
            observer = AdmissionObserver(scan)
            if isinstance(payload, Iterator):
                # A one-pass payload can only be classified while the parser
                # pulls it; re-reading it afterwards would see an exhausted
                # iterator and undercount every record. Whatever the parser
                # leaves is drained after it returns, so the denominator is
                # the whole stream.
                instrumented = observer.observing(payload)
                if len(args) > payload_index:
                    args = (*args[:payload_index], instrumented, *args[payload_index + 1 :])
                else:
                    kwargs = {**kwargs, payload_name: instrumented}
                session = parser(*args, **kwargs)
                observer.drain(instrumented)
            else:
                observer.observe_input(payload)
                session = parser(*args, **kwargs)
            return observer.apply(session, provider)

        return wrapped  # type: ignore[return-value]

    return decorate


def typed_unknown(
    unit: AdmissionUnit,
    ordinal: int,
    key: str,
    *,
    ledger: AdmissionLedger | None = None,
    reason: AdmissionUnknownReason = AdmissionUnknownReason.UNRECOGNIZED_TYPE,
) -> AdmissionOutcome:
    """Record an unknown through the one shared parser admission combinator."""
    if ledger is None:
        ledger = AdmissionLedger()
        ledger.expect(unit, 1)
    return ledger.unknown(unit, ordinal, key, reason)


def typed_unknown_block(
    value: object,
    *,
    wire_type: str | None = None,
    ledger: AdmissionLedger | None = None,
    ordinal: int = 0,
) -> ParsedContentBlock:
    """Retain an unrecognized structured segment without guessing its meaning."""
    observed_type = wire_type
    if observed_type is None and isinstance(value, dict):
        raw_type = value.get("type") or value.get("content_type")
        observed_type = raw_type if isinstance(raw_type, str) and raw_type else "unknown"
    observed_type = observed_type or "unknown"
    typed_unknown(
        AdmissionUnit.BLOCK,
        ordinal,
        observed_type,
        ledger=ledger,
        reason=AdmissionUnknownReason.UNRECOGNIZED_TYPE,
    )
    return ParsedContentBlock(
        type=BlockType.DOCUMENT,
        metadata={
            "admission_disposition": AdmissionDisposition.TYPED_UNKNOWN.value,
            "unknown_reason": AdmissionUnknownReason.UNRECOGNIZED_TYPE.value,
            "wire_type": observed_type,
        },
    )


def human_authored_override(
    role: Role,
    message_type: MessageType,
    material_origin: MaterialOrigin,
) -> MaterialOrigin:
    """Honor caller-qualified human input while retaining structural provenance.

    The caller has established the human-input channel independently of role
    and text. Pasted runtime or generated-pack markers cannot refute that
    evidence. Tool, summary and assistant evidence remains authoritative.
    """
    if (
        role is Role.USER
        and message_type in (MessageType.MESSAGE, MessageType.CONTEXT, MessageType.PROTOCOL)
        and material_origin not in (MaterialOrigin.TOOL_RESULT, MaterialOrigin.ASSISTANT_AUTHORED)
    ):
        return MaterialOrigin.HUMAN_AUTHORED
    return material_origin


def text_blocks_prose(blocks: Sequence[ParsedContentBlock]) -> str | None:
    """Join only TEXT-type block text, in position order, with ``\\n``.

    This is the parse-time twin of
    ``polylogue.storage.embeddings.materialization.message_prose_sql``
    (``block_types=("text",)``, ``separator="'\\n'"``), keeping the
    classifier input aligned with the persisted TEXT blocks.

    Anything that folds THINKING/TOOL_USE/TOOL_RESULT segments into the
    same string handed to ``classify_text_message_type`` (e.g. a combined
    "full record text" built before blocks are split apart) can see markers
    that are not part of the message's authored prose. Build the input from
    the message's own already-split content blocks instead.
    Callers that classify a message's runtime-artifact type from its text
    must build that text from the message's own already-split
    ``ParsedContentBlock`` list via this helper, not from a separately
    reconstructed "all segment types" string.
    """
    parts = [block.text for block in blocks if block.type is BlockType.TEXT and block.text]
    return "\n".join(parts) if parts else None


def synthetic_message_id(
    *,
    role: Role,
    text: str | None,
    timestamp: str | None,
    namespace: str = "",
    kind: str = "",
) -> str:
    """Build a reorder-stable id for a message with no provider id.

    This is reserved for parser-produced rows that are inherently synthetic,
    such as an exported summary or a transcript section. Native-id fallback
    paths must pass an empty string instead, so ``pipeline.ids`` can use its
    role/timestamp/text comparison anchor.
    """
    seed = "\x1f".join((namespace, str(role), timestamp or "", text or "", kind))
    return f"synthetic-{hash_text(seed)[:24]}"


def fill_linear_parent_chain(messages: Sequence[ParsedMessage]) -> list[ParsedMessage]:
    """Backfill linear parent evidence for a strictly linear message list.

    Gemini CLI, Grok, and AI Studio Drive's non-branch path never assert an
    explicit reply-to edge, because their session shape is a plain ordered
    turn sequence with no fork/retry concept at the message level -- there is
    no ``variant_index>0`` row in any of them today. Leaving
    ``parent_message_provider_id`` at ``None`` for every message makes
    position-order the ONLY way to reconstruct the conversation shape for
    these paths, unlike Claude Code / ChatGPT where a real parent chain is
    carried end to end.

    This fills the trivial, unambiguous case -- chaining each message to the
    previous message on the same active path -- without fabricating branch
    structure: a message that already carries real parent evidence (e.g.
    ``parsers/drive.py``'s explicit Gemini branch chunks) is left untouched,
    and only a message with ``parent_message_provider_id is None`` is
    chained to the nearest preceding *active-path* message. A session's
    first message (and any message with no preceding active-path message)
    keeps ``parent_message_provider_id=None`` -- there is nothing to chain
    it to.
    """
    filled: list[ParsedMessage] = []
    previous_active_message: ParsedMessage | None = None
    for message in messages:
        if message.parent_message_provider_id is None and previous_active_message is not None:
            if previous_active_message.provider_message_id:
                message = message.model_copy(
                    update={"parent_message_provider_id": previous_active_message.provider_message_id}
                )
            elif previous_active_message.position is not None:
                message = message.model_copy(update={"parent_message_position": previous_active_message.position})
        filled.append(message)
        if message.is_active_path is not False:
            previous_active_message = message
    return filled


def _tool_result_nested_parts(content: object) -> Iterator[tuple[str | None, object]]:
    if not isinstance(content, list):
        return
    for segment in content:
        if not isinstance(segment, Mapping) or segment.get("type") != "tool_result":
            continue
        parts = segment.get("content")
        if isinstance(parts, list):
            tool_id = segment.get("tool_use_id")
            for part in parts:
                yield tool_id if isinstance(tool_id, str) else None, part


def _tool_result_media_block(segment: Mapping[str, object], tool_id: str | None) -> ParsedContentBlock:
    source = segment.get("source")
    source = source if isinstance(source, Mapping) else {}
    metadata: dict[str, object] = {"tool_result_media": True, "tool_result_id": tool_id}
    # Inline bytes belong to attachment acquisition. The block keeps a stable
    # witness, so two different images cannot collapse to the same empty block.
    metadata["source_digest"] = hash_payload(source)
    for key in ("type", "url", "file_id"):
        if isinstance(source.get(key), str):
            metadata[f"source_{key}"] = source[key]
    media_type = segment.get("media_type") or source.get("media_type")
    text = source.get("data") if source.get("type") == "text" else None
    return ParsedContentBlock(
        type=BlockType.from_string(str(segment["type"])),
        text=text if isinstance(text, str) else None,
        media_type=media_type if isinstance(media_type, str) else None,
        metadata=metadata,
    )


def tool_result_media_attachments(content: object, message_id: str | None, *, role: Role) -> list[ParsedAttachment]:
    """Conserve Anthropic tool-result media without putting binary data in block metadata."""
    attachments: list[ParsedAttachment] = []
    for tool_id, segment in _tool_result_nested_parts(content):
        if not isinstance(segment, Mapping) or segment.get("type") not in {"image", "document"}:
            continue
        source = segment.get("source")
        if not isinstance(source, Mapping):
            continue
        inline_bytes: bytes | None = None
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            inline_bytes = decode_attachment_base64(source["data"], field_name="source.data")
        elif source.get("type") == "text" and isinstance(source.get("data"), str):
            inline_bytes = source["data"].encode("utf-8")
        file_id = source.get("file_id")
        url = source.get("url")
        media_type = segment.get("media_type") or source.get("media_type")
        title = segment.get("title")
        attachments.append(
            ParsedAttachment(
                provider_attachment_id="tool-result-media:"
                + hash_payload({"message_id": message_id, "tool_id": tool_id, "segment": segment}),
                message_provider_id=message_id,
                name=title if isinstance(title, str) else None,
                mime_type=media_type if isinstance(media_type, str) else None,
                size_bytes=len(inline_bytes) if inline_bytes is not None else None,
                provider_file_id=file_id if isinstance(file_id, str) else None,
                source_url=url if isinstance(url, str) else None,
                attachment_kind=str(segment["type"]),
                direction="model_output",
                producer_ref=f"tool:{tool_id}" if tool_id else None,
                inline_bytes=inline_bytes,
            )
        )
    return attachments


def content_blocks_from_segments(
    content: object,
    *,
    admission: AdmissionLedger | None = None,
    lower_transport_text: bool = False,
) -> list[ParsedContentBlock]:
    """Convert raw API content (str, list, dict) to ParsedContentBlock list.

    Provider parsers normally lower transport-only text items themselves. A
    parser that needs the shared converter to retain those items can opt in;
    this keeps the default behavior for providers whose overlay has richer
    transport-item handling.
    """
    if isinstance(content, str):
        if admission is not None:
            part_ordinal = admission.next_ordinal(AdmissionUnit.PART)
            admission.expect(AdmissionUnit.PART, 1)
            admission.materialized(AdmissionUnit.PART, part_ordinal, "text")
            admission.expect(AdmissionUnit.BLOCK, 1 if content else 0)
            if content:
                admission.materialized(AdmissionUnit.BLOCK, admission.next_ordinal(AdmissionUnit.BLOCK), "text")
        return [ParsedContentBlock(type=BlockType.TEXT, text=content)] if content else []
    if not isinstance(content, list):
        return []
    part_offset = admission.next_ordinal(AdmissionUnit.PART) if admission is not None else 0
    if admission is not None:
        admission.expect(AdmissionUnit.PART, len(content))
    blocks: list[ParsedContentBlock] = []
    known_types = {
        "thinking",
        "tool_use",
        "tool_result",
        "text",
        "image",
        "document",
        "token_budget",
        "voice_note",
        "code",
        "input_text",
        "output_text",
        "input_image",
    }
    for part_ordinal, seg in enumerate(content):
        if (
            isinstance(seg, dict)
            and seg.get("type") == "tool_use"
            and not (seg.get("name") or seg.get("id") or (isinstance(seg.get("input"), dict) and seg["input"]))
        ):
            if admission is not None:
                admission.refusal(
                    AdmissionUnit.PART, part_offset + part_ordinal, "tool_use", AdmissionRefusalReason.MALFORMED
                )
            continue
        if admission is not None:
            if isinstance(seg, str):
                admission.materialized(AdmissionUnit.PART, part_offset + part_ordinal, "text")
            elif (
                isinstance(seg, dict)
                and isinstance(seg.get("type", "text"), str)
                and seg.get("type", "text") in known_types
            ):
                admission.materialized(AdmissionUnit.PART, part_offset + part_ordinal, str(seg.get("type", "text")))
            else:
                admission.unknown(
                    AdmissionUnit.PART,
                    part_offset + part_ordinal,
                    "unsupported",
                    AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                )
        if isinstance(seg, str):
            if seg:
                blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=seg))
            continue
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type", "text")
        if seg_type == "text":
            text = seg.get("text") or ""
            if text:
                blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=str(text)))
        elif seg_type == "thinking":
            text = seg.get("thinking") or seg.get("text") or ""
            signature = seg.get("signature")
            # polylogue-vf9x: since roughly 2026-06 the wire ships thinking
            # blocks with an empty `thinking` body and only a `signature` --
            # the reasoning genuinely occurred but its text is not on the
            # wire (verified against raw ~/.claude/projects JSONL: Feb-2026
            # sessions carry non-empty text, Jul-2026 sessions are 100%
            # empty-body/signature-only). Previously this `if text:` guard
            # dropped the block outright, silently zeroing thinking_count
            # and making the archive look like reasoning stopped -- record
            # the block regardless so the fact that the model reasoned here
            # (and the signature, for provenance) survives even without text.
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=text or None,
                    signature=signature if isinstance(signature, str) and signature else None,
                )
            )
        elif seg_type == "tool_use":
            tool_name = seg.get("name")
            tool_id = seg.get("id")
            tool_input = seg.get("input") if isinstance(seg.get("input"), dict) else None
            if tool_name or tool_id or tool_input:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name=tool_name,
                        tool_id=tool_id,
                        tool_input=tool_input,
                    )
                )
        elif seg_type == "tool_result":
            result_content = seg.get("content")
            result_text = None
            knowledge_constructs: list[ParsedWebConstruct] = []
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                text_parts = [
                    block.get("text", "")
                    for block in result_content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                result_text = "\n".join(part for part in text_parts if part) or None
                # polylogue-zocm: web_search tool_result content also carries
                # retrieved-source entries the provider read but did not
                # necessarily cite in the answer text -- {type: knowledge,
                # title, url, metadata: {site_domain, site_name,
                # favicon_url}, text} (1,514 measured). Distinct from the
                # answer-text `citations` anchors ``_citation_construct``
                # (claude/common.py) projects as CONTENT_REFERENCE:
                # SEARCH_RESULT here means "retrieved", not "cited", so
                # "sources read" stays queryable separately from "sources
                # cited" instead of being flattened together.
                for rank, block in enumerate(result_content):
                    if not isinstance(block, dict) or block.get("type") != "knowledge":
                        continue
                    knowledge_title = block.get("title")
                    knowledge_url = block.get("url")
                    knowledge_text = block.get("text")
                    knowledge_metadata = block.get("metadata")
                    site_name = (
                        knowledge_metadata.get("site_name") or knowledge_metadata.get("site_domain")
                        if isinstance(knowledge_metadata, dict)
                        else None
                    )
                    knowledge_constructs.append(
                        ParsedWebConstruct(
                            construct_type=WebConstructType.SEARCH_RESULT,
                            provider_key="web_search_knowledge",
                            title=knowledge_title if isinstance(knowledge_title, str) else None,
                            url=knowledge_url if isinstance(knowledge_url, str) else None,
                            text=knowledge_text if isinstance(knowledge_text, str) else None,
                            group_title=site_name if isinstance(site_name, str) else None,
                            rank=rank,
                        )
                    )
            raw_is_error = seg.get("is_error")
            is_error = raw_is_error if isinstance(raw_is_error, bool) else None
            raw_exit_code = seg.get("exit_code")
            exit_code = (
                raw_exit_code if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool) else None
            )
            # The shared Anthropic-protocol tool_result segment shape (Claude
            # Code, Claude common, Codex). A present-but-unreadable
            # ``is_error``/``exit_code`` is a structure carrying a verdict this
            # mapping does not cover; both keys absent is an unreported
            # outcome. Origin-specific overlays (e.g. Claude Code's own
            # toolUseResult verdicts) may resolve or override this afterward.
            outcome_unknown_reason = unknown_reason(
                is_error=is_error,
                exit_code=exit_code,
                outcome_field_present=any(seg.get(key) is not None for key in ("is_error", "exit_code")),
            )
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id=seg.get("tool_use_id"),
                    text=result_text,
                    is_error=is_error,
                    exit_code=exit_code,
                    outcome_unknown_reason=outcome_unknown_reason,
                    web_constructs=knowledge_constructs,
                )
            )
        elif seg_type in ("image", "document"):
            block_type = BlockType.from_string(seg_type)
            # bd polylogue-9x22: this ``metadata`` dict is never persisted --
            # the ``blocks`` table has no metadata column and the write path
            # only reads a ``language`` key back out of it -- but unlike
            # every other polylogue-9x22 site, it is NOT routed to
            # session_events here. The Anthropic-protocol image/document
            # segment shape's remaining keys after `type`/`media_type` are
            # dominated by `source` (the inline base64 payload itself, or a
            # file/url reference already captured by the attachment
            # pipeline) -- verbatim-copying this dict the way the other
            # sites do would duplicate large binary/attachment data into a
            # durable evidence table meant for small JSON payloads. Re-audit
            # with real corpus evidence if a genuinely small, non-blob,
            # non-attachment-duplicate field is ever found on these segments.
            blocks.append(
                ParsedContentBlock(
                    type=block_type,
                    media_type=seg.get("media_type"),
                )
            )
        elif seg_type == "token_budget":
            remaining = seg.get("remaining")
            if remaining is not None:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=f"[Claude token budget remaining: {remaining}]",
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.TOKEN_BUDGET,
                                provider_key="token_budget",
                                text=str(remaining),
                            )
                        ],
                    )
                )
        elif seg_type == "voice_note":
            text = seg.get("text") or ""
            title = seg.get("title")
            if text or title:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=str(text or title),
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.VOICE_NOTE,
                                provider_key="voice_note",
                                title=str(title) if title else None,
                                text=str(text) if text else None,
                            )
                        ],
                    )
                )
        elif seg_type == "code":
            text = seg.get("text") or seg.get("code") or ""
            if text:
                metadata: dict[str, object] | None = None
                language = seg.get("language")
                if isinstance(language, str) and language:
                    metadata = {"language": language}
                blocks.append(ParsedContentBlock(type=BlockType.CODE, text=str(text), metadata=metadata))
        elif seg_type in ("input_text", "output_text", "input_image", "image"):
            # Provider transport items are lowered by the owning parser:
            # Codex supplies text through ``extract_codex_text`` and bounded
            # image evidence through ``_codex_inline_image_blocks``.
            if lower_transport_text and seg_type in {"input_text", "output_text"}:
                text = seg.get("text") or ""
                if text:
                    blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=str(text)))
            continue
        else:
            # Unknown structured content is still an input unit. Preserve its
            # existence as an opaque DOCUMENT block with a typed disposition;
            # callers must never infer prose from a future wire shape.
            unknown_block = typed_unknown_block(seg, wire_type=str(seg_type))
            if isinstance(seg.get("text"), str) and seg["text"]:
                unknown_block = unknown_block.model_copy(update={"type": BlockType.TEXT, "text": seg["text"]})
            blocks.append(unknown_block)
    for tool_id, part in _tool_result_nested_parts(content):
        part_type = part.get("type") if isinstance(part, Mapping) else None
        if admission is not None:
            ordinal = admission.next_ordinal(AdmissionUnit.PART)
            admission.expect(AdmissionUnit.PART, 1)
            if part_type in {"text", "knowledge", "image", "document"}:
                admission.materialized(AdmissionUnit.PART, ordinal, str(part_type))
            else:
                admission.unknown(AdmissionUnit.PART, ordinal, str(part_type or "unsupported"))
        if isinstance(part, Mapping) and part_type in {"image", "document"}:
            blocks.append(_tool_result_media_block(part, tool_id))
        elif part_type not in {"text", "knowledge"}:
            block = typed_unknown_block(part, wire_type=str(part_type or "unsupported"))
            blocks.append(
                block.model_copy(
                    update={
                        "metadata": {
                            **(block.metadata or {}),
                            "tool_result_media": True,
                            "tool_result_id": tool_id,
                        }
                    }
                )
            )
    if admission is not None:
        admission.expect(AdmissionUnit.BLOCK, len(blocks))
        for block in blocks:
            block_ordinal = admission.next_ordinal(AdmissionUnit.BLOCK)
            if (
                block.metadata
                and block.metadata.get("admission_disposition") == AdmissionDisposition.TYPED_UNKNOWN.value
            ):
                admission.unknown(
                    AdmissionUnit.BLOCK,
                    block_ordinal,
                    str(block.metadata.get("wire_type") or "unknown"),
                )
            else:
                admission.materialized(AdmissionUnit.BLOCK, block_ordinal, block.type.value)
    return blocks


def _make_attachment_id(seed: str) -> str:
    return f"att-{hash_text(seed)[:12]}"


def derive_attachment_provenance(
    role: Role | str | None,
    message_id: str | None,
) -> tuple[AttachmentDirection | None, str | None]:
    """Derive attachment direction and producer from the owning turn role."""
    if role is None:
        return None, None
    normalized_role = role if isinstance(role, Role) else Role.normalize(role)
    if normalized_role is Role.USER:
        return "user_input", None
    if normalized_role in {Role.ASSISTANT, Role.TOOL}:
        producer_ref = f"message:{message_id}" if message_id else None
        return "model_output", producer_ref
    return None, None


#: The metadata keys an export may carry a provider-assigned attachment
#: identity under. When none is present, ``attachment_from_meta`` seeds a
#: synthetic identity from the owning message id, so the same physical file
#: recorded under two different message ids -- or under none -- mints two
#: identities. A caller that reconciles such records must ask
#: :func:`meta_carries_provider_attachment_id` first.
PROVIDER_ATTACHMENT_ID_KEYS = ("id", "file_id", "fileId", "uuid", "file_uuid")


def meta_carries_provider_attachment_id(meta: object) -> bool:
    """Report whether attachment metadata names its own provider identity."""
    if not isinstance(meta, dict):
        return False
    return any(meta.get(key) for key in PROVIDER_ATTACHMENT_ID_KEYS)


def decode_attachment_base64(value: object, *, field_name: str = "content_base64") -> bytes:
    """Decode a declared attachment byte carrier or fail explicitly."""
    if not isinstance(value, str):
        raise ValueError(f"invalid base64 for attachment field {field_name!r}")
    data = value
    if value.startswith("data:") and ";base64," in value:
        _, data = value.split(";base64,", 1)
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid base64 for attachment field {field_name!r}") from exc


def attachment_from_meta(
    meta: object,
    message_id: str | None,
    *,
    role: Role | str | None = None,
) -> ParsedAttachment | None:
    if not isinstance(meta, dict):
        return None
    attachment_id = next((meta.get(key) for key in PROVIDER_ATTACHMENT_ID_KEYS if meta.get(key)), None)
    name = meta.get("name") or meta.get("filename") or meta.get("file_name")
    mime_type = meta.get("mimeType") or meta.get("mime_type") or meta.get("content_type") or meta.get("file_type")
    if not attachment_id:
        if not name:
            return None
        # polylogue-hith: identity must be a property of the attachment
        # itself, not of its position in whichever bucket ("attachments" vs
        # "files") or order the export happened to walk them in. `index` used
        # to be part of this seed, so re-ordering an export's attachment list
        # -- observed happening between vintages of the SAME conversation --
        # minted a different id for the same physical attachment even though
        # nothing about it changed. `mime_type` is included instead: it is
        # read directly from the export's own metadata (never lazily
        # acquired the way `size`/inline bytes can be), so it adds real
        # disambiguation without reintroducing acquisition-state instability.
        # Two un-identified attachments sharing both name and mime_type on
        # the same message are genuinely indistinguishable from the metadata
        # available here; collapsing them to one identity is the honest
        # outcome of that, not a regression -- see polylogue-hith for the
        # full trade-off discussion (a real-id-havingness axis is a separate,
        # unfixed failure mode filed as a follow-up there).
        seed = f"{message_id or 'msg'}:{name}:{mime_type or ''}"
        attachment_id = _make_attachment_id(seed)
    size_raw = meta.get("size") or meta.get("size_bytes") or meta.get("sizeBytes") or meta.get("file_size")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    inline_bytes = None
    binary_carrier = meta.get("content_base64")
    if binary_carrier is not None:
        inline_bytes = decode_attachment_base64(binary_carrier)
    else:
        extracted_content = meta.get("extracted_content")
        if isinstance(extracted_content, str):
            inline_bytes = extracted_content.encode("utf-8")
    if inline_bytes is not None and size_bytes is None:
        size_bytes = len(inline_bytes)
    # #1252: promote native identifiers when present. claude-code/codex
    # attachments arrive via OAuth-authenticated session/export.
    file_id_raw = meta.get("file_id") or meta.get("fileId") or meta.get("file_uuid")
    drive_id_raw = meta.get("drive_id") or meta.get("driveId")
    direction, producer_ref = derive_attachment_provenance(role, message_id)
    return ParsedAttachment(
        provider_attachment_id=str(attachment_id),
        message_provider_id=message_id,
        name=name,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_file_id=str(file_id_raw) if isinstance(file_id_raw, str) and file_id_raw else None,
        provider_drive_id=str(drive_id_raw) if isinstance(drive_id_raw, str) and drive_id_raw else None,
        upload_origin="oauth",
        direction=direction,
        producer_ref=producer_ref,
        inline_bytes=inline_bytes,
    )


def iter_messages_from_list(items: Iterable[object]) -> Iterator[ParsedMessage]:
    """Normalize independent generic message records without retaining the cohort."""
    for _idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        message_val = item.get("message")
        payload = message_val if isinstance(message_val, dict) else item

        role = Role.normalize(
            str(
                payload.get("role")
                or item.get("role")
                or payload.get("sender")
                or item.get("sender")
                or payload.get("author")
                or item.get("author")
                or "unknown"
            )
        )

        timestamp = (
            item.get("timestamp")
            or payload.get("timestamp")
            or payload.get("created_at")
            or item.get("created_at")
            or payload.get("create_time")
            or item.get("create_time")
        )

        text = None
        content_blocks: list[ParsedContentBlock] = []
        text_val = payload.get("text")
        if text_val is not None and isinstance(text_val, str):
            text = text_val
            if text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
        else:
            content = payload.get("content")
            if isinstance(content, str):
                text = content
                if text:
                    content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts: list[str] = []
                    for part in parts:
                        if isinstance(part, str) and part:
                            texts.append(part)
                            content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=part))
                        elif isinstance(part, dict):
                            part_text = part.get("text")
                            if isinstance(part_text, str) and part_text:
                                texts.append(part_text)
                                content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=part_text))
                    text = "\n".join(texts) or None
                else:
                    text_dict_val = content.get("text")
                    if text_dict_val is not None and isinstance(text_dict_val, str):
                        text = text_dict_val
                        if text:
                            content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            elif isinstance(content, list):
                content_blocks = content_blocks_from_segments(content)
                text = "\n".join(block.text for block in content_blocks if block.text) or None

        if text:
            # polylogue-slshy: no positional fallback -- empty id lets
            # _message_revision_match_id's content-derived anchor fallback run
            # instead of a position-derived string that would change identity
            # when array order shifts across re-acquisitions.
            msg_id = str(payload.get("id") or payload.get("uuid") or item.get("uuid") or item.get("id") or "")
            yield ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=str(timestamp) if timestamp is not None else None,
                blocks=content_blocks,
            )


def extract_messages_from_list(items: Sequence[object]) -> list[ParsedMessage]:
    return list(iter_messages_from_list(items))


def mark_last_occurrence_as_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Flag exactly one message as ``is_active_leaf``: the LAST message in
    the list, by position -- never by comparing ``provider_message_id``.

    ``provider_message_id`` is not guaranteed unique across a flat message
    list assembled by concatenating streaming chunks or markdown sections --
    retries/variants/regenerations can legitimately reuse the same native id
    at more than one position (bd polylogue-2hwl). Comparing every message's
    id against ``messages[-1].provider_message_id`` (the naive approach)
    flags EVERY matching position, not just the true leaf, which then lets
    more than one ``is_active_leaf=True`` message reach MCP payloads and
    archive_query message output -- an invariant violation (at most one
    active leaf per session). Matching by exact list position instead of by
    id equality alone keeps the flag unique regardless of duplicate ids.
    """
    if not messages:
        return messages
    leaf_index = len(messages) - 1
    return [
        message.model_copy(update={"is_active_leaf": index == leaf_index}) for index, message in enumerate(messages)
    ]
