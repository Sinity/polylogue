"""Parse configured OTLP-JSON files that contain GenAI span attributes.

This is an import adapter for retained OTLP trace exports.  It does not start
an OTel receiver or infer a complete conversation from a trace.  Unsupported
fields remain in a span-evidence session event rather than being discarded.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.json import JSONDocument
from polylogue.core.payload_coercion import optional_string
from polylogue.core.timestamps import iso_from_epoch_ms, to_epoch_ms
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent

SEMCONV_SCHEMA_URL = "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev"
OTLP_JSON_DIALECT = "OTLP-JSON ExportTraceServiceRequest (protobuf JSON mapping)"


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


def _schema_url(scope: dict[str, object]) -> str | None:
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
    schema_rank = 0 if schema_url == SEMCONV_SCHEMA_URL else 1 if schema_url is None else 2
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


def _resource_id(resource_attrs: dict[str, object]) -> str:
    """Name one OTLP resource by its service, or by its stable attributes.

    Two ``resourceSpans`` entries without ``service.name`` are still distinct
    resources when their configured attributes differ; collapsing both onto
    one shared fallback grouped their spans into a single provider session.
    Per-instance attributes (``process.pid``, ``service.instance.id`` ...)
    are left out, so a restarted process exports the same conversation under
    the same identity.
    """
    service_name = optional_string(resource_attrs.get("service.name"))
    if service_name:
        return service_name
    stable = {
        key: _json_value(value)
        for key, value in resource_attrs.items()
        if not key.startswith(_INSTANCE_RESOURCE_PREFIXES)
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return f"resource-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _iter_spans(payload: dict[str, object]) -> Iterable[tuple[str, dict[str, object], str | None]]:
    resource_spans = payload.get("resourceSpans", payload.get("resource_spans"))
    if not isinstance(resource_spans, list):
        return
    for resource_span in resource_spans:
        resource = _mapping(resource_span)
        resource_attrs = _attributes(_mapping(resource.get("resource")).get("attributes"))
        resource_id = _resource_id(resource_attrs)
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
                trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
                span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id"))
                if trace_id and span_id:
                    yield resource_id, span, _schema_url(scope)


def looks_like(payload: object) -> bool:
    """Recognize an OTLP JSON document with a normalizable GenAI span."""
    record = _mapping(payload)
    for _resource, span, schema_url in _iter_spans(record):
        if schema_url not in (None, SEMCONV_SCHEMA_URL):
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


def _history_overlap(inputs: list[_TranscriptEntry], transcript: list[_TranscriptEntry]) -> int:
    """Count leading ``inputs`` the conversation transcript already ends with.

    A GenAI request's ``gen_ai.input.messages`` is the history sent with that
    request, so the second turn of a chat carries ``[Q1, A1, Q2]`` after the
    first carried ``[Q1]`` and produced ``A1``. Only the unseen suffix is new
    material; re-emitting the prefix duplicated every earlier message once
    per later span.
    """
    # A request carries at least one new message: the current turn. Only a
    # trailing tool entry (a result the tool span already recorded) may be
    # replayed whole; an identical input-only turn repeated by the user is a
    # second occurrence, not history.
    largest = len(inputs) if inputs and inputs[-1][2] else len(inputs) - 1
    for size in range(min(largest, len(transcript)), 0, -1):
        if inputs[:size] == transcript[-size:]:
            return size
    return 0


def _messages_for_span(
    span: dict[str, object],
    attrs: dict[str, object],
    trace_id: str,
    transcript: list[_TranscriptEntry],
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
    if not any(entry[0] == Role.ASSISTANT.value and tool_id in entry[2] for entry in reversed(transcript)):
        transcript.append((Role.ASSISTANT.value, None, (tool_id,)))
    transcript.append((Role.TOOL.value, None, (tool_id,)))
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
                        text=tool_result if isinstance(tool_result, str) else None,
                        tool_outcome=outcome,
                        is_error=is_error,
                        outcome_unknown_reason=unknown_reason,
                    )
                ],
            ),
        )
    )
    return messages


def parse(payload: JSONDocument, fallback_id: str) -> list[ParsedSession]:
    """Normalize OTLP GenAI spans into resource/conversation or trace sessions."""
    del fallback_id  # stable source coordinates, never an import filename
    variants: dict[tuple[str, str, str], list[tuple[dict[str, object], str | None]]] = defaultdict(list)
    for resource_id, span, schema_url in _iter_spans(_mapping(payload)):
        variants[_span_coordinate(resource_id, span)].append((span, schema_url))
    spans: list[tuple[str, dict[str, object], str | None]] = []
    conflicts: dict[tuple[str, str, str], list[tuple[dict[str, object], str | None]]] = {}
    for coordinate, copies in sorted(variants.items()):
        ordered = sorted(copies, key=_span_variant_key)
        selected = ordered[0]
        spans.append((coordinate[0], *selected))
        selected_identity = (selected[1], _span_variant_key(selected)[2])
        seen = {selected_identity}
        alternatives = []
        for item in ordered[1:]:
            identity = (item[1], _span_variant_key(item)[2])
            if identity not in seen:
                alternatives.append(item)
                seen.add(identity)
        if alternatives:
            conflicts[coordinate] = alternatives
    # A GenAI export may carry ordinary HTTP/database traces beside the GenAI
    # one. Only traces that contain a GenAI span become sessions; their
    # non-GenAI spans stay as topology evidence inside that session. A trace
    # with no GenAI span at all is not this origin's material.
    # Membership reads every variant of a coordinate, so a trace whose GenAI
    # attributes survive only in a conflicting copy keeps its session and the
    # ``otel_conflicting_span_id`` evidence.
    genai_traces = {
        coordinate[:2]
        for coordinate, copies in variants.items()
        if any(any(key.startswith("gen_ai.") for key in _attributes(copy.get("attributes"))) for copy, _ in copies)
    }
    spans = [
        (resource_id, span, schema_url)
        for resource_id, span, schema_url in spans
        if _span_coordinate(resource_id, span)[:2] in genai_traces
    ]
    span_details: dict[tuple[str, str, str], tuple[str | None, str | None]] = {}
    for resource_id, span, _schema_url in spans:
        trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
        span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id"))
        if trace_id and span_id:
            attrs = _attributes(span.get("attributes"))
            span_details[(resource_id, trace_id, span_id)] = (
                optional_string(attrs.get("gen_ai.conversation.id")),
                optional_string(span.get("parentSpanId")) or optional_string(span.get("parent_span_id")),
            )

    def conversation_for(resource_id: str, trace_id: str, span_id: str) -> str | None:
        seen: set[str] = set()
        while span_id not in seen:
            seen.add(span_id)
            details = span_details.get((resource_id, trace_id, span_id))
            if details is None:
                break
            conversation_id, parent_id = details
            if conversation_id:
                return conversation_id
            if parent_id is None:
                break
            span_id = parent_id
        return None

    resolved: dict[tuple[str, str, str], str | None] = {key: conversation_for(*key) for key in span_details}
    # A span with no conversation of its own or above it (the HTTP/root span
    # over a GenAI child) is topology evidence of the conversation below it,
    # not a separate trace session: it joins the conversation of its
    # descendants, the lowest id when several share the ancestor, and spans
    # beneath it follow. A trace with exactly one conversation keeps every
    # remaining span there too.
    adopted: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (resource_id, trace_id, span_id), conversation_id in resolved.items():
        if conversation_id is None:
            continue
        seen_ids = {span_id}
        parent_id = span_details[(resource_id, trace_id, span_id)][1]
        while parent_id is not None and parent_id not in seen_ids:
            seen_ids.add(parent_id)
            parent_key = (resource_id, trace_id, parent_id)
            if parent_key not in span_details:
                break
            if resolved.get(parent_key) is None:
                adopted[parent_key].add(conversation_id)
            parent_id = span_details[parent_key][1]
    trace_conversations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (resource_id, trace_id, _span_id), conversation_id in resolved.items():
        if conversation_id is not None:
            trace_conversations[(resource_id, trace_id)].add(conversation_id)

    def group_conversation(resource_id: str, trace_id: str, span_id: str) -> str | None:
        key = (resource_id, trace_id, span_id)
        if resolved.get(key) is not None:
            return resolved[key]
        seen_ids: set[str] = set()
        current: str | None = span_id
        while current is not None and current not in seen_ids:
            seen_ids.add(current)
            current_key = (resource_id, trace_id, current)
            if adopted.get(current_key):
                return min(adopted[current_key])
            details = span_details.get(current_key)
            current = details[1] if details is not None else None
        only = trace_conversations.get((resource_id, trace_id), set())
        return next(iter(only)) if len(only) == 1 else None

    groups: dict[tuple[str, str, str], list[tuple[dict[str, object], str | None]]] = defaultdict(list)
    for resource_id, span, schema_url in spans:
        trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
        span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id"))
        if trace_id is None or span_id is None:
            continue
        conversation_id = group_conversation(resource_id, trace_id, span_id)
        kind, group_identity = ("conversation", conversation_id) if conversation_id else ("trace", trace_id)
        groups[(resource_id, kind, group_identity)].append((span, schema_url))

    sessions: list[ParsedSession] = []
    for (resource_id, kind, group_identity), scoped_spans in sorted(groups.items()):
        messages: list[ParsedMessage] = []
        events: list[ParsedSessionEvent] = []
        models: set[str] = set()
        transcript: list[_TranscriptEntry] = []
        for span, schema_url in sorted(
            scoped_spans,
            key=lambda item: (_span_key(item[0]), _span_coordinate(resource_id, item[0])[1]),
        ):
            attrs = _attributes(span.get("attributes"))
            trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
            if trace_id is None:
                continue
            timestamp, _occurred_at_ms = _timestamp(span)
            events.append(
                ParsedSessionEvent(
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
                        if schema_url == SEMCONV_SCHEMA_URL
                        else "unsupported",
                        "dialect": OTLP_JSON_DIALECT,
                        "message_fidelity": {
                            field: _message_fidelity(attrs, field)
                            for field in ("gen_ai.input.messages", "gen_ai.output.messages")
                        },
                        "usage_fidelity": _usage_fidelity(attrs),
                        "events": _json_value(span.get("events", [])),
                    },
                )
            )
            for conflicting_span, conflicting_schema_url in conflicts.get(_span_coordinate(resource_id, span), ()):
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
            if schema_url not in (None, SEMCONV_SCHEMA_URL):
                continue
            span_messages = _messages_for_span(span, attrs, trace_id, transcript)
            messages.extend(span_messages)
            model = optional_string(attrs.get("gen_ai.response.model")) or optional_string(
                attrs.get("gen_ai.request.model")
            )
            if model:
                models.add(model)
            usage = _usage_counts(attrs)
            # Any GenAI operation that reports usage counters keeps them:
            # ``text_completion`` and ``generate_content`` spans whose message
            # bodies were not exported still carry billable tokens.
            if any(count is not None for count in usage) and not any(
                message.input_tokens is not None
                or message.output_tokens is not None
                or message.cache_read_tokens is not None
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
                            "model": model,
                        },
                    )
                )
        if events:
            sessions.append(
                ParsedSession(
                    source_name=Provider.OTEL_GENAI,
                    provider_session_id=f"{resource_id}:{kind}:{group_identity}",
                    title=f"OpenTelemetry GenAI {group_identity}",
                    messages=messages,
                    session_events=events,
                    models_used=sorted(models),
                )
            )
    return sessions


__all__ = ["OTLP_JSON_DIALECT", "SEMCONV_SCHEMA_URL", "looks_like", "parse"]
