"""Parse configured OTLP-JSON files that contain GenAI span attributes.

This is an import adapter for retained OTLP trace exports.  It does not start
an OTel receiver or infer a complete conversation from a trace.  Unsupported
fields remain in a span-evidence session event rather than being discarded.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from contextlib import closing
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.json import JSONDocument
from polylogue.core.payload_coercion import optional_string
from polylogue.core.timestamps import iso_from_epoch_ms, to_epoch_ms
from polylogue.sources import origin_specs
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _decode_any(value: object) -> object:
    """Decode the OTLP JSON AnyValue mapping, retaining plain JSON too."""
    item = _mapping(value)
    for key in ("stringValue", "boolValue", "intValue", "doubleValue", "bytesValue"):
        if key in item:
            return item[key]
    if "arrayValue" in item:
        values = _mapping(item["arrayValue"]).get("values")
        return [_decode_any(child) for child in values] if isinstance(values, list) else []
    if "kvlistValue" in item:
        return _attributes(_mapping(item["kvlistValue"]).get("values"))
    return value


def _attributes(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): _decode_any(item) for key, item in value.items()}
    if not isinstance(value, list):
        return {}
    result: dict[str, object] = {}
    for entry in value:
        item = _mapping(entry)
        key = item.get("key")
        if isinstance(key, str) and "value" in item:
            result[key] = _decode_any(item["value"])
    return result


#: Span fields ``otel_span_evidence`` carries in their own projected keys.
_PROJECTED_SPAN_FIELDS = frozenset(
    {
        "traceId",
        "trace_id",
        "spanId",
        "span_id",
        "parentSpanId",
        "parent_span_id",
        "name",
        "kind",
        "status",
        "attributes",
        "events",
    }
)


def _json_value(value: object) -> object:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timestamp(span: dict[str, object]) -> tuple[str | None, int | None]:
    value = span.get("startTimeUnixNano", span.get("start_time_unix_nano"))
    try:
        epoch_ms = to_epoch_ms(int(str(value)) // 1_000_000, numeric_unit="milliseconds")
    except (TypeError, ValueError):
        return None, None
    return iso_from_epoch_ms(epoch_ms), epoch_ms


def _token_count(value: object) -> int | None:
    try:
        count = int(str(value))
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _usage_counts(attrs: dict[str, object]) -> tuple[int | None, int | None, int | None]:
    return (
        _token_count(attrs.get("gen_ai.usage.input_tokens")),
        _token_count(attrs.get("gen_ai.usage.output_tokens")),
        _token_count(attrs.get("gen_ai.usage.cache_read.input_tokens")),
    )


def _messages(value: object) -> tuple[list[dict[str, object]], str]:
    """Decode a GenAI messages attribute and expose its source state."""
    if value is None:
        return [], "missing"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [], "unsupported"
    if not isinstance(value, list):
        return [], "unsupported"
    if not value:
        return [], "empty"
    messages = [item for item in value if isinstance(item, dict)]
    return messages, "present" if messages else "unsupported"


def _message_fidelity(attrs: dict[str, object], field: str) -> str:
    if attrs.get("polylogue.evidence.redacted") is True or attrs.get(f"{field}.redacted") is True:
        return "redacted"
    if attrs.get("polylogue.evidence.truncated") is True or attrs.get(f"{field}.truncated") is True:
        return "truncated"
    return _messages(attrs.get(field))[1]


def _usage_fidelity(attrs: dict[str, object]) -> str:
    if attrs.get("polylogue.usage.redacted") is True:
        return "redacted"
    if attrs.get("polylogue.usage.truncated") is True:
        return "truncated"
    return "present" if any(key.startswith("gen_ai.usage.") for key in attrs) else "missing"


def _message_text(message: dict[str, object]) -> str | None:
    value = message.get("content")
    if isinstance(value, str):
        return value
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    text = [part.get("content") for part in parts if isinstance(part, dict) and part.get("type") == "text"]
    return "".join(item for item in text if isinstance(item, str)) or None


def _role(value: object, default: Role) -> Role:
    role = optional_string(value)
    return Role.normalize(role) if role in {"user", "assistant", "system", "developer", "tool"} else default


def _material_origin(role: Role) -> MaterialOrigin:
    if role is Role.USER:
        return MaterialOrigin.HUMAN_AUTHORED
    if role is Role.ASSISTANT:
        return MaterialOrigin.ASSISTANT_AUTHORED
    if role is Role.TOOL:
        return MaterialOrigin.TOOL_RESULT
    return MaterialOrigin.RUNTIME_CONTEXT


def _tool_outcome(span: dict[str, object]) -> tuple[ToolOutcome, bool | None, str | None]:
    code = _mapping(span.get("status")).get("code")
    if code in (1, "1", "STATUS_CODE_OK"):
        return ToolOutcome.OK, False, None
    if code in (2, "2", "STATUS_CODE_ERROR"):
        return ToolOutcome.ERROR, True, None
    reason = (
        ToolResultUnknownReason.NOT_REPORTED.value
        if code in (None, 0, "0", "STATUS_CODE_UNSET")
        else ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT.value
    )
    return ToolOutcome.UNKNOWN, None, reason


def scope_schema_url(scope: dict[str, object]) -> str | None:
    return optional_string(scope.get("schemaUrl")) or optional_string(scope.get("schema_url"))


def _span_key(span: dict[str, object]) -> tuple[int, str]:
    try:
        start = int(str(span.get("startTimeUnixNano", span.get("start_time_unix_nano", "0"))))
    except (TypeError, ValueError):
        start = 0
    return start, str(span.get("spanId", span.get("span_id", "")))


def _span_coordinate(resource_id: str, span: dict[str, object]) -> tuple[str, str, str]:
    return (
        resource_id,
        optional_string(span.get("traceId")) or optional_string(span.get("trace_id")) or "",
        optional_string(span.get("spanId")) or optional_string(span.get("span_id")) or "",
    )


def _span_variant_key(item: tuple[dict[str, object], str | None]) -> tuple[int, int, str, str]:
    span, schema_url = item
    schema_rank = 0 if schema_url == origin_specs.SEMCONV_SCHEMA_URL else 1 if schema_url is None else 2
    return (
        schema_rank,
        _span_key(span)[0],
        json.dumps(span, sort_keys=True, separators=(",", ":")),
        schema_url or "",
    )


#: Resource attributes that name one running instance (a process, host,
#: container or SDK build) rather than the resource. They change on every
#: restart or upgrade, so they never contribute to a session's identity.
_INSTANCE_RESOURCE_PREFIXES = (
    "process.",
    "host.",
    "container.",
    "k8s.pod.",
    "k8s.container.",
    "os.",
    "telemetry.",
    "service.instance.",
)


def _is_deployment_version(key: str) -> bool:
    """Name a version attribute (``service.version``, ``deployment.version`` ...).

    An upgrade changes it independently of the conversation, so it never
    contributes to a session's identity.
    """
    return key == "version" or key.endswith((".version", "_version"))


def _stable_resource_attributes(resource_attrs: dict[str, object]) -> str:
    """The canonical JSON of a resource's identity-bearing attributes."""
    stable = {
        key: _json_value(value)
        for key, value in resource_attrs.items()
        if not key.startswith(_INSTANCE_RESOURCE_PREFIXES) and not _is_deployment_version(key)
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def _resource_id(resource_attrs: dict[str, object]) -> str:
    """Name one OTLP resource by its service, or by its stable attributes.

    Two ``resourceSpans`` entries without ``service.name`` are still distinct
    resources when their configured attributes differ; collapsing both onto
    one shared fallback grouped their spans into a single provider session.
    Per-instance attributes (``process.pid``, ``service.instance.id`` ...)
    and version attributes (``service.version`` ...) are left out, so a
    restarted or upgraded process exports the same conversation under the
    same identity. The name depends on this resource alone, never on how many
    other resources share its document: a later export that adds a second
    resource keeps the first one's sessions.
    """
    service_name = optional_string(resource_attrs.get("service.name"))
    if service_name:
        return service_name
    canonical = _stable_resource_attributes(resource_attrs)
    if canonical == "{}":
        return "resource"
    return f"resource-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _session_identity(resource_id: str, kind: str, group_identity: str) -> str:
    """Join a session's resource, grouping kind and group into one native id.

    ``service.name`` and conversation ids are free text and may themselves
    contain ``:``. Each component escapes ``%`` and ``:`` before the join, so
    service ``svc:conversation`` with conversation ``x`` and service ``svc``
    with conversation ``conversation:x`` stay two sessions. A component
    without either character keeps its plain spelling.
    """
    return ":".join(part.replace("%", "%25").replace(":", "%3A") for part in (resource_id, kind, group_identity))


def _iter_spans(payload: dict[str, object]) -> Iterable[tuple[str, dict[str, object], str | None]]:
    resource_spans = payload.get("resourceSpans", payload.get("resource_spans"))
    if not isinstance(resource_spans, list):
        return
    for resource_span in resource_spans:
        resource = _mapping(resource_span)
        resource_id = resource_id_for(resource)
        scopes = resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", ()))
        if not isinstance(scopes, list):
            continue
        for raw_scope in scopes:
            scope = _mapping(raw_scope)
            spans = scope.get("spans")
            if not isinstance(spans, list):
                continue
            for raw_span in spans:
                span = _mapping(raw_span)
                if has_span_identity(span):
                    yield resource_id, span, scope_schema_url(scope)


def looks_like(payload: object) -> bool:
    """Recognize an OTLP JSON document with a normalizable GenAI span."""
    record = _mapping(payload)
    for _resource, span, schema_url in _iter_spans(record):
        if schema_url not in (None, origin_specs.SEMCONV_SCHEMA_URL):
            continue
        if any(key.startswith("gen_ai.") for key in _attributes(span.get("attributes"))):
            return True
    return False


_TranscriptEntry = tuple[str, str | None, tuple[str, ...]]


def _transcript_entry(raw_message: dict[str, object], default_role: Role) -> _TranscriptEntry:
    """Identify one GenAI message by role, text and the tool calls it carries.

    Tool-call and tool-response parts carry no text, so their call ids are
    what distinguishes one tool exchange from another in replayed history.
    """
    parts = raw_message.get("parts")
    tool_ids = tuple(
        str(part.get("id"))
        for part in (parts if isinstance(parts, list) else ())
        if isinstance(part, dict)
        and part.get("type") in {"tool_call", "tool_call_response"}
        and part.get("id") is not None
    )
    return (_role(raw_message.get("role"), default_role).value, _message_text(raw_message), tool_ids)


class _Transcript:
    """One session's conversation transcript, kept in SQLite.

    The history-overlap check reads only the transcript's first and last
    entries, never more of either than one span's input count, and the tool
    exchange check is an indexed lookup, so no Python list grows with the
    session.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._next = 0
        conn.execute("CREATE TABLE otel_transcript (ordinal INTEGER PRIMARY KEY, entry TEXT NOT NULL)")
        conn.execute("CREATE TABLE otel_transcript_tool (tool_id TEXT PRIMARY KEY) WITHOUT ROWID")

    @staticmethod
    def _entry(entry_json: str) -> _TranscriptEntry:
        role, text, tool_ids = json.loads(entry_json)
        return (role, text, tuple(tool_ids))

    def head(self, count: int) -> list[_TranscriptEntry]:
        return [
            self._entry(entry)
            for (entry,) in self._conn.execute("SELECT entry FROM otel_transcript ORDER BY ordinal LIMIT ?", (count,))
        ]

    def tail(self, count: int) -> list[_TranscriptEntry]:
        rows = self._conn.execute(
            "SELECT entry FROM otel_transcript ORDER BY ordinal DESC LIMIT ?", (count,)
        ).fetchall()
        return [self._entry(entry) for (entry,) in reversed(rows)]

    def append(self, entry: _TranscriptEntry) -> None:
        self._conn.execute("INSERT INTO otel_transcript VALUES (?, ?)", (self._next, json.dumps(list(entry))))
        self._next += 1
        if entry[0] == Role.ASSISTANT.value:
            self._conn.executemany(
                "INSERT OR IGNORE INTO otel_transcript_tool VALUES (?)",
                ((json.dumps(tool_id),) for tool_id in entry[2]),
            )

    def has_assistant_tool(self, tool_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM otel_transcript_tool WHERE tool_id = ?", (json.dumps(tool_id),)
            ).fetchone()
            is not None
        )

    def clear(self) -> None:
        self._conn.execute("DELETE FROM otel_transcript")
        self._conn.execute("DELETE FROM otel_transcript_tool")
        self._next = 0


def _history_overlap(inputs: list[_TranscriptEntry], transcript: _Transcript) -> int:
    """Count leading ``inputs`` the conversation transcript already covers.

    A GenAI request's ``gen_ai.input.messages`` is the history sent with that
    request, so the second turn of a chat carries ``[Q1, A1, Q2]`` after the
    first carried ``[Q1]`` and produced ``A1``. Only the unseen suffix is new
    material; re-emitting the prefix duplicated every earlier message once
    per later span.

    A pinned prefix (typically a system prompt) survives context truncation
    verbatim even once the middle of the history is dropped: transcript
    ``[S, Q1, A1, Q2, A2]`` truncates to inputs ``[S, Q2, A2, Q3]``, where no
    leading slice of ``inputs`` equals a trailing slice of ``transcript``
    because ``S`` interrupts the suffix match. Match the stable leading
    prefix first, then look for the retained-history suffix in what remains,
    so the two overlaps compose instead of the pinned prefix defeating the
    suffix match.
    """
    # A request carries at least one new message: the current turn. Only a
    # trailing tool entry (a result the tool span already recorded) may be
    # replayed whole; an identical input-only turn repeated by the user is a
    # second occurrence, not history.
    largest = len(inputs) if inputs and inputs[-1][2] else len(inputs) - 1
    if largest <= 0:
        return 0
    # Only a transcript's first ``largest`` entries can match the pinned
    # prefix, and only its last ``len(pattern)`` can end the overlap.
    head = transcript.head(largest)
    pinned = 0
    while pinned < largest and pinned < len(head) and inputs[pinned] == head[pinned]:
        pinned += 1
    pattern = inputs[pinned:largest]
    return pinned + _prefix_suffix_overlap(pattern, transcript.tail(len(pattern)))


def _prefix_suffix_overlap(pattern: list[_TranscriptEntry], text: list[_TranscriptEntry]) -> int:
    """Length of the longest prefix of ``pattern`` that is a suffix of ``text``.

    Knuth-Morris-Pratt over ``text`` with ``pattern``'s failure function:
    linear in both lengths, where trying every overlap size and slicing
    each candidate was quadratic in a long retained history.
    """
    if not pattern or not text:
        return 0
    failure = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = failure[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        failure[index] = matched
    matched = 0
    for entry in text:
        if matched == len(pattern):
            matched = failure[matched - 1]
        while matched and entry != pattern[matched]:
            matched = failure[matched - 1]
        if entry == pattern[matched]:
            matched += 1
    return matched


def _messages_for_span(
    span: dict[str, object],
    attrs: dict[str, object],
    trace_id: str,
    transcript: _Transcript,
) -> list[ParsedMessage]:
    span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id")) or "span"
    timestamp, occurred_at_ms = _timestamp(span)
    parent_span_id = optional_string(span.get("parentSpanId")) or optional_string(span.get("parent_span_id"))
    parent = f"{trace_id}:{parent_span_id}:output:0" if parent_span_id else None
    messages: list[ParsedMessage] = []
    usage = _usage_counts(attrs)
    usage_attached = False
    # The model that produced the output: model usage rollups aggregate
    # message tokens by ``model_name``, so a token-bearing message without one
    # contributes nothing to usage or cost.
    span_model = optional_string(attrs.get("gen_ai.response.model")) or optional_string(
        attrs.get("gen_ai.request.model")
    )
    for field, direction, default_role in (
        ("gen_ai.input.messages", "input", Role.USER),
        ("gen_ai.output.messages", "output", Role.ASSISTANT),
    ):
        raw_messages = _messages(attrs.get(field))[0]
        entries = [_transcript_entry(raw_message, default_role) for raw_message in raw_messages]
        already_seen = _history_overlap(entries, transcript) if direction == "input" else 0
        for index, raw_message in enumerate(raw_messages):
            if index < already_seen:
                continue
            transcript.append(entries[index])
            role = _role(raw_message.get("role"), default_role)
            carries_usage = direction == "output" and role is Role.ASSISTANT and not usage_attached
            messages.append(
                ParsedMessage(
                    provider_message_id=f"{trace_id}:{span_id}:{direction}:{index}",
                    role=role,
                    text=_message_text(raw_message),
                    timestamp=timestamp,
                    occurred_at_ms=occurred_at_ms,
                    material_origin=_material_origin(role),
                    parent_message_provider_id=parent,
                    input_tokens=usage[0] if carries_usage else None,
                    output_tokens=usage[1] if carries_usage else None,
                    cache_read_tokens=usage[2] if carries_usage else None,
                    model_name=span_model if direction == "output" and role is Role.ASSISTANT else None,
                )
            )
            if carries_usage:
                usage_attached = True
    if optional_string(attrs.get("gen_ai.operation.name")) != "execute_tool" and "gen_ai.tool.name" not in attrs:
        return messages
    tool_id = optional_string(attrs.get("gen_ai.tool.call.id")) or span_id
    tool_name = optional_string(attrs.get("gen_ai.tool.name")) or optional_string(span.get("name")) or "unknown"
    arguments = attrs.get("gen_ai.tool.call.arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    tool_input = arguments if isinstance(arguments, dict) else {"raw": arguments} if arguments is not None else {}
    outcome, is_error, unknown_reason = _tool_outcome(span)
    tool_result = attrs.get("gen_ai.tool.call.result")
    # The tool exchange is conversation history too: the next request's
    # ``gen_ai.input.messages`` replays the call and its result, and the
    # overlap check must recognise them. A chat span whose output already
    # carried this call recorded the call entry; only the result is new then.
    # Matched by tool-call id alone: an output message may carry text beside
    # the call.
    if not transcript.has_assistant_tool(tool_id):
        transcript.append((Role.ASSISTANT.value, None, (tool_id,)))
    transcript.append((Role.TOOL.value, None, (tool_id,)))
    # A structured, numeric or boolean result is still the provider's result:
    # serialize it rather than leave the result block empty. An absent
    # attribute stays absent.
    result_text = (
        tool_result
        if isinstance(tool_result, str)
        else json.dumps(_json_value(tool_result), ensure_ascii=False, sort_keys=True)
        if "gen_ai.tool.call.result" in attrs
        else None
    )
    messages.extend(
        (
            ParsedMessage(
                provider_message_id=f"{trace_id}:{span_id}:tool-use",
                role=Role.ASSISTANT,
                timestamp=timestamp,
                occurred_at_ms=occurred_at_ms,
                material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                parent_message_provider_id=parent,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE, tool_name=tool_name, tool_id=tool_id, tool_input=tool_input
                    )
                ],
            ),
            ParsedMessage(
                provider_message_id=f"{trace_id}:{span_id}:tool-result",
                role=Role.TOOL,
                timestamp=timestamp,
                occurred_at_ms=occurred_at_ms,
                material_origin=MaterialOrigin.TOOL_RESULT,
                parent_message_provider_id=f"{trace_id}:{span_id}:tool-use",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id=tool_id,
                        text=result_text,
                        tool_outcome=outcome,
                        is_error=is_error,
                        outcome_unknown_reason=unknown_reason,
                    )
                ],
            ),
        )
    )
    return messages


def _text_key(value: str) -> bytes:
    """Order a string by code point under SQLite's bytewise comparison.

    ``surrogatepass`` keeps a lone surrogate from a JSON escape encodable and
    places it where Python's string order does.
    """
    return value.encode("utf-8", "surrogatepass")


def _from_text_key(value: bytes) -> str:
    return value.decode("utf-8", "surrogatepass")


def _int_key(value: int) -> bytes:
    """Encode an integer so that bytewise order is numeric order."""
    digits = str(abs(value))
    if value >= 0:
        return b"1" + f"{len(digits):010d}".encode() + digits.encode()
    complement = digits.translate(str.maketrans("0123456789", "9876543210"))
    return b"0" + f"{9_999_999_999 - len(digits):010d}".encode() + complement.encode()


def resource_id_for(resource: dict[str, object]) -> str:
    """The session scope of one ``resourceSpans`` entry."""
    return _resource_id(_attributes(_mapping(resource.get("resource")).get("attributes")))


def has_span_identity(span: dict[str, object]) -> bool:
    trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
    span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id"))
    return bool(trace_id and span_id)


def _span_evidence_event(
    span: dict[str, object],
    attrs: dict[str, object],
    trace_id: str,
    schema_url: str | None,
    foreign_resource_id: str | None,
) -> ParsedSessionEvent:
    timestamp, _occurred_at_ms = _timestamp(span)
    return ParsedSessionEvent(
        event_type="otel_span_evidence",
        timestamp=timestamp,
        payload={
            "trace_id": trace_id,
            "span_id": span.get("spanId", span.get("span_id")),
            "parent_span_id": span.get("parentSpanId", span.get("parent_span_id")),
            "span_name": span.get("name"),
            "span_kind": span.get("kind"),
            "status": _json_value(span.get("status")),
            "attributes": {key: _json_value(value) for key, value in attrs.items()},
            "schema_url": schema_url,
            "schema_url_status": "missing"
            if schema_url is None
            else "supported"
            if schema_url == origin_specs.SEMCONV_SCHEMA_URL
            else "unsupported",
            "dialect": origin_specs.OTLP_JSON_DIALECT,
            "message_fidelity": {
                field: _message_fidelity(attrs, field) for field in ("gen_ai.input.messages", "gen_ai.output.messages")
            },
            "usage_fidelity": _usage_fidelity(attrs),
            "events": _json_value(span.get("events", [])),
            # Every span field this payload does not project (end time,
            # links, trace state, flags, dropped counts, fields a later OTLP
            # revision adds) stays recoverable from the evidence.
            "unprojected_span_fields": {
                key: _json_value(value) for key, value in span.items() if key not in _PROJECTED_SPAN_FIELDS
            },
            # A span from another resource of the same trace keeps its own
            # resource identity.
            **({"resource_id": foreign_resource_id} if foreign_resource_id is not None else {}),
        },
    )


def _span_model(attrs: dict[str, object]) -> str | None:
    return optional_string(attrs.get("gen_ai.response.model")) or optional_string(attrs.get("gen_ai.request.model"))


def _append_span(
    span: dict[str, object],
    schema_url: str | None,
    conflicts: Iterable[tuple[dict[str, object], str | None]],
    messages: MutableSequence[ParsedMessage],
    events: MutableSequence[ParsedSessionEvent],
    transcript: _Transcript,
    foreign_resource_id: str | None,
) -> None:
    """Append one selected span's evidence, conflicts, messages and usage."""
    attrs = _attributes(span.get("attributes"))
    trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
    if trace_id is None:
        return
    timestamp, _occurred_at_ms = _timestamp(span)
    events.append(_span_evidence_event(span, attrs, trace_id, schema_url, foreign_resource_id))
    for conflicting_span, conflicting_schema_url in conflicts:
        events.append(
            ParsedSessionEvent(
                event_type="otel_conflicting_span_id",
                payload={
                    "trace_id": trace_id,
                    "span_id": span.get("spanId", span.get("span_id")),
                    "conflicting_span": conflicting_span,
                    "schema_url": conflicting_schema_url,
                },
            )
        )
    if schema_url not in (None, origin_specs.SEMCONV_SCHEMA_URL):
        return
    span_messages = _messages_for_span(span, attrs, trace_id, transcript)
    messages.extend(span_messages)
    usage = _usage_counts(attrs)
    # Any GenAI operation that reports usage counters keeps them:
    # ``text_completion`` and ``generate_content`` spans whose message
    # bodies were not exported still carry billable tokens.
    if any(count is not None for count in usage) and not any(
        message.input_tokens is not None or message.output_tokens is not None or message.cache_read_tokens is not None
        for message in span_messages
    ):
        events.append(
            ParsedSessionEvent(
                event_type="message_usage",
                timestamp=timestamp,
                payload={
                    "last_token_usage": {
                        key: count
                        for key, count in zip(
                            ("input_tokens", "output_tokens", "cached_input_tokens"), usage, strict=True
                        )
                        if count is not None
                    },
                    "model": _span_model(attrs),
                },
            )
        )


_SpanKey = tuple[bytes, bytes, bytes]
#: ``(start, span id, conversation id, conversation resource)``: the order in
#: which conversations claim a shared ancestor.
_Claim = tuple[bytes, bytes, bytes, bytes]


class OtelSpanIndex:
    """Select, deduplicate and group one OTLP document's spans in SQLite.

    Span copies, conflicting variants, trace membership, the parent walks
    that find each span's conversation and adopt conversation-less
    ancestors, the session grouping and each session's transcript all live
    in ``conn``; Python holds one span, one walk step and one span's inputs
    at a time whatever the document's span count. The object parser runs
    this same index over an in-memory connection.

    Keys are stored as ``_text_key``/``_int_key`` bytes, whose bytewise order
    is the Python order of the strings and integers they encode.
    """

    _TABLES = (
        "otel_span",
        "otel_seen",
        "otel_conflict",
        "otel_selected",
        "otel_genai_trace",
        "otel_trace_only",
        "otel_walk",
        "otel_transcript",
        "otel_transcript_tool",
    )

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._next_seq = 0
        #: Whether any span is one ``looks_like`` would accept.
        self.normalizable = False
        conn.execute(
            "CREATE TABLE otel_span (seq INTEGER PRIMARY KEY, resource_id BLOB NOT NULL, trace_id BLOB NOT NULL, "
            "span_id BLOB NOT NULL, schema_url TEXT NOT NULL, schema_rank INTEGER NOT NULL, "
            "start_key BLOB NOT NULL, canonical TEXT NOT NULL, schema_sort BLOB NOT NULL, span_json TEXT NOT NULL, "
            "genai INTEGER NOT NULL, conversation_id BLOB)"
        )

    def add(self, resource_id: str, span: dict[str, object], schema_url: str | None) -> None:
        """Record one span copy, in document order."""
        _resource, trace_id, span_id = _span_coordinate(resource_id, span)
        schema_rank, start, canonical, schema_sort = _span_variant_key((span, schema_url))
        attrs = _attributes(span.get("attributes"))
        genai = any(key.startswith("gen_ai.") for key in attrs)
        if genai and schema_url in (None, origin_specs.SEMCONV_SCHEMA_URL):
            self.normalizable = True
        conversation_id = optional_string(attrs.get("gen_ai.conversation.id"))
        self._conn.execute(
            "INSERT INTO otel_span VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._next_seq,
                _text_key(resource_id),
                _text_key(trace_id),
                _text_key(span_id),
                json.dumps(schema_url),
                schema_rank,
                _int_key(start),
                canonical,
                _text_key(schema_sort),
                json.dumps(span),
                int(genai),
                _text_key(conversation_id) if conversation_id else None,
            ),
        )
        self._next_seq += 1

    def _span(self, seq: int) -> dict[str, object]:
        row = self._conn.execute("SELECT span_json FROM otel_span WHERE seq = ?", (seq,)).fetchone()
        span = json.loads(row[0])
        assert isinstance(span, dict)
        return span

    def _select_variants(self) -> None:
        """Keep the first variant of each coordinate; record distinct others.

        A coordinate's conversation is read from every copy: a conversation
        id surviving only in a conflicting copy still names the session, so a
        later clean export of that copy keys the same one. Copies naming
        different conversations leave the coordinate identity-ambiguous.
        """
        conn = self._conn
        conn.execute("CREATE INDEX otel_span_coordinate ON otel_span(resource_id, trace_id, span_id)")
        conn.execute(
            "CREATE TABLE otel_seen (resource_id BLOB, trace_id BLOB, span_id BLOB, schema_url TEXT, "
            "canonical_digest BLOB, PRIMARY KEY (resource_id, trace_id, span_id, schema_url, canonical_digest)) "
            "WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TABLE otel_conflict (seq INTEGER PRIMARY KEY, resource_id BLOB NOT NULL, "
            "trace_id BLOB NOT NULL, span_id BLOB NOT NULL, span_seq INTEGER NOT NULL)"
        )
        conn.execute("CREATE INDEX otel_conflict_coordinate ON otel_conflict(resource_id, trace_id, span_id, seq)")
        conn.execute(
            "CREATE TABLE otel_selected (resource_id BLOB NOT NULL, trace_id BLOB NOT NULL, span_id BLOB NOT NULL, "
            "span_seq INTEGER NOT NULL, conversation_id BLOB, parent_id BLOB, model BLOB, start_key BLOB NOT NULL, "
            "span_key BLOB NOT NULL, resolved INTEGER NOT NULL DEFAULT 0, conversation BLOB, "
            "conversation_resource BLOB, claim_start BLOB, claim_span BLOB, claim_conversation BLOB, "
            "claim_resource BLOB, group_resource BLOB, group_kind TEXT, group_identity BLOB, "
            "PRIMARY KEY (resource_id, trace_id, span_id)) WITHOUT ROWID"
        )
        current: _SpanKey | None = None
        conflict_seq = 0
        for seq, resource_id, trace_id, span_id, schema_url_json, canonical in conn.execute(
            "SELECT seq, resource_id, trace_id, span_id, schema_url, canonical FROM otel_span "
            "ORDER BY resource_id, trace_id, span_id, schema_rank, start_key, canonical, schema_sort, seq"
        ):
            coordinate = (resource_id, trace_id, span_id)
            digest = hashlib.sha256(canonical.encode("ascii")).digest()
            fresh = (
                conn.execute(
                    "INSERT OR IGNORE INTO otel_seen VALUES (?, ?, ?, ?, ?)", (*coordinate, schema_url_json, digest)
                ).rowcount
                == 1
            )
            if coordinate != current:
                current = coordinate
                span = self._span(seq)
                attrs = _attributes(span.get("attributes"))
                schema_url = json.loads(schema_url_json)
                parent_id = optional_string(span.get("parentSpanId")) or optional_string(span.get("parent_span_id"))
                model = _span_model(attrs) if schema_url in (None, origin_specs.SEMCONV_SCHEMA_URL) else None
                start, span_key = _span_key(span)
                conn.execute(
                    "INSERT INTO otel_selected (resource_id, trace_id, span_id, span_seq, parent_id, model, "
                    "start_key, span_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *coordinate,
                        seq,
                        _text_key(parent_id) if parent_id is not None else None,
                        _text_key(model) if model else None,
                        _int_key(start),
                        _text_key(span_key),
                    ),
                )
            elif fresh:
                conn.execute("INSERT INTO otel_conflict VALUES (?, ?, ?, ?, ?)", (conflict_seq, *coordinate, seq))
                conflict_seq += 1
        conn.execute("DROP TABLE otel_seen")
        conn.execute(
            "UPDATE otel_selected SET conversation_id = ("
            "SELECT CASE WHEN COUNT(DISTINCT s.conversation_id) = 1 THEN MIN(s.conversation_id) END "
            "FROM otel_span s WHERE s.resource_id = otel_selected.resource_id "
            "AND s.trace_id = otel_selected.trace_id AND s.span_id = otel_selected.span_id "
            "AND s.conversation_id IS NOT NULL)"
        )
        # A GenAI export may carry ordinary HTTP/database traces beside the
        # GenAI one. Only traces that contain a GenAI span -- in any copy of
        # any coordinate, under any resource -- become sessions; their
        # non-GenAI spans stay as topology evidence inside that session.
        conn.execute("CREATE TABLE otel_genai_trace (trace_id BLOB PRIMARY KEY) WITHOUT ROWID")
        conn.execute("INSERT OR IGNORE INTO otel_genai_trace SELECT trace_id FROM otel_span WHERE genai = 1")
        conn.execute("DELETE FROM otel_selected WHERE trace_id NOT IN (SELECT trace_id FROM otel_genai_trace)")
        conn.execute("CREATE INDEX otel_selected_location ON otel_selected(trace_id, span_id, resource_id)")

    def _keys(self, where: str) -> Iterator[_SpanKey]:
        """Page through selected span keys matching ``where``, in key order.

        Paging by key keeps the walks below free to update the rows they
        visit while the iteration continues.
        """
        last: _SpanKey = (b"", b"", b"")
        while True:
            rows = self._conn.execute(
                f"SELECT resource_id, trace_id, span_id FROM otel_selected WHERE ({where}) "
                "AND (resource_id, trace_id, span_id) > (?, ?, ?) ORDER BY resource_id, trace_id, span_id LIMIT 512",
                last,
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield (row[0], row[1], row[2])
            last = rows[-1]

    def _parent_of(self, key: _SpanKey) -> _SpanKey | None:
        """A span's parent: in its own resource first, else its trace's only holder."""
        resource_id, trace_id, _span_id = key
        row = self._conn.execute(
            "SELECT parent_id FROM otel_selected WHERE resource_id = ? AND trace_id = ? AND span_id = ?", key
        ).fetchone()
        parent_id = row[0] if row is not None else None
        if parent_id is None:
            return None
        holders = self._conn.execute(
            "SELECT resource_id FROM otel_selected WHERE trace_id = ? AND span_id = ? "
            "ORDER BY resource_id != ?, resource_id LIMIT 2",
            (trace_id, parent_id, resource_id),
        ).fetchall()
        if not holders:
            return None
        if holders[0][0] == resource_id or len(holders) == 1:
            return (holders[0][0], trace_id, parent_id)
        return None

    def _start_walk(self, key: _SpanKey) -> None:
        self._conn.execute("DELETE FROM otel_walk")
        self._visit(key)

    def _visit(self, key: _SpanKey) -> bool:
        """Mark ``key`` visited in the current walk; ``False`` when it was already."""
        return self._conn.execute("INSERT OR IGNORE INTO otel_walk VALUES (?, ?, ?)", key).rowcount == 1

    def _resolve_conversations(self) -> None:
        """Give every selected span the conversation of its nearest ancestor.

        A span's conversation is its own ``gen_ai.conversation.id`` or else
        its parent's, with the resource of the span that names it; the walk
        stops at a missing parent or a cycle. Each walk's path is kept in
        scratch and every node on it takes the walk's result, so no span is
        walked twice.
        """
        conn = self._conn
        conn.execute(
            "CREATE TABLE otel_walk (resource_id BLOB, trace_id BLOB, span_id BLOB, "
            "PRIMARY KEY (resource_id, trace_id, span_id)) WITHOUT ROWID"
        )
        for key in self._keys("resolved = 0"):
            if conn.execute(
                "SELECT resolved FROM otel_selected WHERE resource_id = ? AND trace_id = ? AND span_id = ?", key
            ).fetchone()[0]:
                continue
            conn.execute("DELETE FROM otel_walk")
            result: tuple[bytes | None, bytes | None] = (None, None)
            node: _SpanKey | None = key
            while node is not None and self._visit(node):
                conversation_id, resolved, conversation, conversation_resource = conn.execute(
                    "SELECT conversation_id, resolved, conversation, conversation_resource FROM otel_selected "
                    "WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
                    node,
                ).fetchone()
                if resolved:
                    result = (conversation, conversation_resource)
                    break
                if conversation_id:
                    result = (conversation_id, node[0])
                    break
                node = self._parent_of(node)
            conn.execute(
                "UPDATE otel_selected SET resolved = 1, conversation = ?, conversation_resource = ? "
                "WHERE (resource_id, trace_id, span_id) IN (SELECT resource_id, trace_id, span_id FROM otel_walk)",
                result,
            )

    def _stored_claim(self, key: _SpanKey) -> tuple[bool, _Claim | None]:
        """Whether ``key`` has no conversation of its own, and its adopted claim."""
        row = self._conn.execute(
            "SELECT conversation, claim_start, claim_span, claim_conversation, claim_resource FROM otel_selected "
            "WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
            key,
        ).fetchone()
        claim = None if row[1] is None else cast(_Claim, (row[1], row[2], row[3], row[4]))
        return row[0] is None, claim

    def _adopt_ancestors(self) -> None:
        """Let each conversation claim the conversation-less spans above it.

        A span with no conversation of its own or above it (the HTTP/root
        span over a GenAI child) is topology evidence of the conversation
        below it, not a separate trace session. When several conversations
        share the ancestor it joins the one whose descendant started first,
        an owner a later sibling conversation appended to the export cannot
        displace. A walk stops at an ancestor whose claim already wins: the
        walk that left that claim went on through every ancestor above it.
        """
        conn = self._conn
        for key in self._keys("conversation IS NOT NULL"):
            start_key, conversation, conversation_resource = conn.execute(
                "SELECT start_key, conversation, conversation_resource FROM otel_selected "
                "WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
                key,
            ).fetchone()
            claim: _Claim = (start_key, key[2], conversation, conversation_resource)
            self._start_walk(key)
            parent = self._parent_of(key)
            while parent is not None and self._visit(parent):
                unowned, stored = self._stored_claim(parent)
                if unowned:
                    if stored is not None and stored <= claim:
                        break
                    conn.execute(
                        "UPDATE otel_selected SET claim_start = ?, claim_span = ?, claim_conversation = ?, "
                        "claim_resource = ? WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
                        (*claim, *parent),
                    )
                parent = self._parent_of(parent)
        # A trace with exactly one conversation keeps every remaining span.
        conn.execute(
            "CREATE TABLE otel_trace_only (trace_id BLOB PRIMARY KEY, conversation BLOB NOT NULL, "
            "conversation_resource BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute(
            "INSERT INTO otel_trace_only SELECT trace_id, conversation, conversation_resource FROM ("
            "SELECT DISTINCT trace_id, conversation, conversation_resource FROM otel_selected "
            "WHERE conversation IS NOT NULL) GROUP BY trace_id HAVING COUNT(*) = 1"
        )

    def _group_conversation(self, key: _SpanKey) -> tuple[bytes, bytes] | None:
        """The conversation a conversation-less span joins: an adopter's, else its trace's only one."""
        self._start_walk(key)
        node: _SpanKey | None = key
        while node is not None:
            _unowned, claim = self._stored_claim(node)
            if claim is not None:
                return claim[2], claim[3]
            node = self._parent_of(node)
            if node is not None and not self._visit(node):
                break
        row = self._conn.execute(
            "SELECT conversation, conversation_resource FROM otel_trace_only WHERE trace_id = ?", (key[1],)
        ).fetchone()
        return (row[0], row[1]) if row is not None else None

    def _assign_groups(self) -> None:
        """Key each span's session: its conversation's, or its own trace's."""
        conn = self._conn
        conn.execute(
            "UPDATE otel_selected SET group_resource = conversation_resource, group_kind = 'conversation', "
            "group_identity = conversation WHERE conversation IS NOT NULL"
        )
        for key in self._keys("conversation IS NULL"):
            conversation = self._group_conversation(key)
            conn.execute(
                "UPDATE otel_selected SET group_resource = ?, group_kind = ?, group_identity = ? "
                "WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
                (
                    (conversation[1], "conversation", conversation[0])
                    if conversation is not None
                    else (key[0], "trace", key[1])
                )
                + key,
            )
        conn.execute(
            "CREATE INDEX otel_selected_group ON otel_selected"
            "(group_resource, group_kind, group_identity, start_key, span_key, trace_id, resource_id, span_id)"
        )

    def sessions(
        self,
        *,
        new_messages: Callable[[], MutableSequence[ParsedMessage]],
        new_events: Callable[[], MutableSequence[ParsedSessionEvent]],
    ) -> Iterator[ParsedSession]:
        """Yield one session per conversation, or per trace, in key order."""
        self._select_variants()
        self._resolve_conversations()
        self._adopt_ancestors()
        self._assign_groups()
        conn = self._conn
        transcript = _Transcript(conn)
        for resource_key, kind, identity_key in conn.execute(
            "SELECT DISTINCT group_resource, group_kind, group_identity FROM otel_selected "
            "ORDER BY group_resource, group_kind, group_identity"
        ):
            messages = new_messages()
            events = new_events()
            transcript.clear()
            group = (resource_key, kind, identity_key)
            for span_seq, span_resource, trace_key, span_key in conn.execute(
                "SELECT span_seq, resource_id, trace_id, span_id FROM otel_selected "
                "WHERE group_resource = ? AND group_kind = ? AND group_identity = ? "
                "ORDER BY start_key, span_key, trace_id, resource_id, span_id",
                group,
            ):
                row = conn.execute("SELECT schema_url FROM otel_span WHERE seq = ?", (span_seq,)).fetchone()
                conflicts = (
                    (self._span(conflict_seq), cast("str | None", json.loads(schema_url_json)))
                    for conflict_seq, schema_url_json in conn.execute(
                        "SELECT c.span_seq, s.schema_url FROM otel_conflict c JOIN otel_span s ON s.seq = c.span_seq "
                        "WHERE c.resource_id = ? AND c.trace_id = ? AND c.span_id = ? ORDER BY c.seq",
                        (span_resource, trace_key, span_key),
                    )
                )
                _append_span(
                    self._span(span_seq),
                    json.loads(row[0]),
                    conflicts,
                    messages,
                    events,
                    transcript,
                    _from_text_key(span_resource) if span_resource != resource_key else None,
                )
            models = [
                _from_text_key(model)
                for (model,) in conn.execute(
                    "SELECT DISTINCT model FROM otel_selected WHERE group_resource = ? AND group_kind = ? "
                    "AND group_identity = ? AND model IS NOT NULL ORDER BY model",
                    group,
                )
            ]
            resource_id = _from_text_key(resource_key)
            group_identity = _from_text_key(identity_key)
            session = ParsedSession(
                source_name=Provider.OTEL_GENAI,
                provider_session_id=_session_identity(resource_id, kind, group_identity),
                title=f"OpenTelemetry GenAI {group_identity}",
                messages=messages if isinstance(messages, list) else [],
                session_events=events if isinstance(events, list) else [],
                models_used=models,
            )
            if not isinstance(messages, list) or not isinstance(events, list):
                session = session.model_copy(update={"messages": messages, "session_events": events})
            yield session

    def close(self) -> None:
        for table in self._TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")


def parse(payload: JSONDocument, fallback_id: str) -> list[ParsedSession]:
    """Normalize OTLP GenAI spans into resource/conversation or trace sessions."""
    del fallback_id  # stable source coordinates, never an import filename
    with closing(sqlite3.connect(":memory:")) as conn:
        index = OtelSpanIndex(conn)
        for resource_id, span, schema_url in _iter_spans(_mapping(payload)):
            index.add(resource_id, span, schema_url)
        return list(index.sessions(new_messages=list, new_events=list))


__all__ = [
    "OtelSpanIndex",
    "has_span_identity",
    "looks_like",
    "parse",
    "resource_id_for",
    "scope_schema_url",
]


def detection_projection() -> DetectorProjection:
    """Fold eligible spans completely under each enclosing scope's schema URL."""
    scalar = DetectorProjection()
    attribute = DetectorProjection(fields={"key": scalar, "value": None})
    attributes = DetectorProjection(
        item=attribute,
        array_fold="any",
        array_predicate=lambda item: (
            isinstance(item, dict)
            and isinstance(item.get("key"), str)
            and item["key"].startswith("gen_ai.")
            and "value" in item
        ),
        mapping_key_predicate=lambda key: key.startswith("gen_ai."),
        mapping_witness={"gen_ai.fold": None},
    )
    span = DetectorProjection(
        fields={
            **dict.fromkeys(("traceId", "trace_id", "spanId", "span_id"), scalar),
            "attributes": attributes,
        }
    )
    spans = DetectorProjection(
        item=span,
        array_fold="any",
        array_predicate=lambda item: (
            isinstance(item, dict)
            and has_span_identity(item)
            and any(key.startswith("gen_ai.") for key in _attributes(item.get("attributes")))
        ),
    )
    scope = DetectorProjection(fields={"schemaUrl": scalar, "schema_url": scalar, "spans": spans})
    scopes = DetectorProjection(
        item=scope,
        array_fold="any",
        array_predicate=lambda item: looks_like({"resourceSpans": [{"scopeSpans": [item]}]}),
    )
    resource = DetectorProjection(fields={"scopeSpans": scopes, "instrumentationLibrarySpans": scopes})
    resources = DetectorProjection(
        item=resource,
        array_fold="any",
        array_predicate=lambda item: looks_like({"resourceSpans": [item]}),
    )
    return DetectorProjection(fields={"resourceSpans": resources, "resource_spans": resources})
