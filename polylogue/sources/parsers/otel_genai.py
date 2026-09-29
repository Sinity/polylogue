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
    schema_rank = 0 if schema_url == SEMCONV_SCHEMA_URL else 1 if schema_url is None else 2
    return (
        schema_rank,
        _span_key(span)[0],
        json.dumps(span, sort_keys=True, separators=(",", ":")),
        schema_url or "",
    )


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
        if schema_url not in (None, SEMCONV_SCHEMA_URL):
            continue
        if any(key.startswith("gen_ai.") for key in _attributes(span.get("attributes"))):
            return True
    return False


def _messages_for_span(span: dict[str, object], attrs: dict[str, object], trace_id: str) -> list[ParsedMessage]:
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
        for index, raw_message in enumerate(_messages(attrs.get(field))[0]):
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
    resource_attrs = _attributes(_mapping(resource.get("resource")).get("attributes"))
    return optional_string(resource_attrs.get("service.name")) or "resource"


def has_span_identity(span: dict[str, object]) -> bool:
    trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
    span_id = optional_string(span.get("spanId")) or optional_string(span.get("span_id"))
    return bool(trace_id and span_id)


def _span_evidence_event(
    span: dict[str, object], attrs: dict[str, object], trace_id: str, schema_url: str | None
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
            if schema_url == SEMCONV_SCHEMA_URL
            else "unsupported",
            "dialect": OTLP_JSON_DIALECT,
            "message_fidelity": {
                field: _message_fidelity(attrs, field) for field in ("gen_ai.input.messages", "gen_ai.output.messages")
            },
            "usage_fidelity": _usage_fidelity(attrs),
            "events": _json_value(span.get("events", [])),
        },
    )


def _append_span(
    span: dict[str, object],
    schema_url: str | None,
    conflicts: Iterable[tuple[dict[str, object], str | None]],
    messages: MutableSequence[ParsedMessage],
    events: MutableSequence[ParsedSessionEvent],
) -> None:
    """Append one selected span's evidence, conflicts, messages and usage."""
    attrs = _attributes(span.get("attributes"))
    trace_id = optional_string(span.get("traceId")) or optional_string(span.get("trace_id"))
    if trace_id is None:
        return
    timestamp, _occurred_at_ms = _timestamp(span)
    events.append(_span_evidence_event(span, attrs, trace_id, schema_url))
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
    if schema_url not in (None, SEMCONV_SCHEMA_URL):
        return
    span_messages = _messages_for_span(span, attrs, trace_id)
    messages.extend(span_messages)
    model = optional_string(attrs.get("gen_ai.request.model"))
    usage = _usage_counts(attrs)
    if (
        optional_string(attrs.get("gen_ai.operation.name")) == "chat"
        and any(count is not None for count in usage)
        and not any(
            message.input_tokens is not None
            or message.output_tokens is not None
            or message.cache_read_tokens is not None
            for message in span_messages
        )
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


class OtelSpanIndex:
    """Select, deduplicate and group one OTLP document's spans in SQLite.

    Span copies, conflicting variants, the parent walk that finds each span's
    conversation, and the session grouping all live in ``conn``; Python holds
    one span at a time whatever the document's span count. The object parser
    runs this same index over an in-memory connection.
    """

    _TABLES = ("otel_span", "otel_seen", "otel_conflict", "otel_selected", "otel_walk")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._next_seq = 0
        #: Whether any span is one ``looks_like`` would accept.
        self.normalizable = False
        conn.execute(
            "CREATE TABLE otel_span (seq INTEGER PRIMARY KEY, resource_id BLOB NOT NULL, trace_id BLOB NOT NULL, "
            "span_id BLOB NOT NULL, schema_url TEXT NOT NULL, schema_rank INTEGER NOT NULL, "
            "start_key BLOB NOT NULL, canonical TEXT NOT NULL, schema_sort BLOB NOT NULL, span_json TEXT NOT NULL)"
        )

    def add(self, resource_id: str, span: dict[str, object], schema_url: str | None) -> None:
        """Record one span copy, in document order."""
        _resource, trace_id, span_id = _span_coordinate(resource_id, span)
        schema_rank, start, canonical, schema_sort = _span_variant_key((span, schema_url))
        if schema_url in (None, SEMCONV_SCHEMA_URL) and any(
            key.startswith("gen_ai.") for key in _attributes(span.get("attributes"))
        ):
            self.normalizable = True
        self._conn.execute(
            "INSERT INTO otel_span VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self._next_seq += 1

    def _span(self, seq: int) -> dict[str, object]:
        row = self._conn.execute("SELECT span_json FROM otel_span WHERE seq = ?", (seq,)).fetchone()
        span = json.loads(row[0])
        assert isinstance(span, dict)
        return span

    def _select_variants(self) -> None:
        """Keep the first variant of each coordinate; record distinct others."""
        conn = self._conn
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
            "span_key BLOB NOT NULL, resolved INTEGER NOT NULL DEFAULT 0, conversation BLOB, group_kind TEXT, "
            "group_identity BLOB, PRIMARY KEY (resource_id, trace_id, span_id)) WITHOUT ROWID"
        )
        current: tuple[bytes, bytes, bytes] | None = None
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
                conversation_id = optional_string(attrs.get("gen_ai.conversation.id"))
                parent_id = optional_string(span.get("parentSpanId")) or optional_string(span.get("parent_span_id"))
                model = (
                    optional_string(attrs.get("gen_ai.request.model"))
                    if schema_url in (None, SEMCONV_SCHEMA_URL)
                    else None
                )
                start, span_key = _span_key(span)
                conn.execute(
                    "INSERT INTO otel_selected (resource_id, trace_id, span_id, span_seq, conversation_id, "
                    "parent_id, model, start_key, span_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *coordinate,
                        seq,
                        _text_key(conversation_id) if conversation_id is not None else None,
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

    def _resolve_conversations(self) -> None:
        """Give every selected span the conversation of its nearest ancestor.

        A span's conversation is its own ``gen_ai.conversation.id`` or else
        its parent's; the walk stops at a missing parent or a cycle. Each
        walk's path is kept in scratch and every node on it takes the walk's
        result, so no span is walked twice.
        """
        conn = self._conn
        conn.execute("CREATE TABLE otel_walk (span_id BLOB PRIMARY KEY) WITHOUT ROWID")
        last: tuple[bytes, bytes, bytes] = (b"", b"", b"")
        while True:
            row = conn.execute(
                "SELECT resource_id, trace_id, span_id FROM otel_selected "
                "WHERE resolved = 0 AND (resource_id, trace_id, span_id) > (?, ?, ?) "
                "ORDER BY resource_id, trace_id, span_id LIMIT 1",
                last,
            ).fetchone()
            if row is None:
                break
            resource_id, trace_id, node = row
            last = (resource_id, trace_id, node)
            result: bytes | None = None
            while conn.execute("INSERT OR IGNORE INTO otel_walk VALUES (?)", (node,)).rowcount == 1:
                details = conn.execute(
                    "SELECT conversation_id, parent_id, resolved, conversation FROM otel_selected "
                    "WHERE resource_id = ? AND trace_id = ? AND span_id = ?",
                    (resource_id, trace_id, node),
                ).fetchone()
                if details is None:
                    break
                conversation_id, parent_id, resolved, conversation = details
                if resolved:
                    result = conversation
                    break
                if conversation_id:
                    result = conversation_id
                    break
                if parent_id is None:
                    break
                node = parent_id
            conn.execute(
                "UPDATE otel_selected SET resolved = 1, conversation = ? "
                "WHERE resource_id = ? AND trace_id = ? AND span_id IN (SELECT span_id FROM otel_walk)",
                (result, resource_id, trace_id),
            )
            conn.execute("DELETE FROM otel_walk")
        conn.execute(
            "UPDATE otel_selected SET "
            "group_kind = CASE WHEN conversation IS NULL THEN 'trace' ELSE 'conversation' END, "
            "group_identity = COALESCE(conversation, trace_id)"
        )
        conn.execute(
            "CREATE INDEX otel_selected_group ON otel_selected"
            "(resource_id, group_kind, group_identity, start_key, span_key, trace_id, span_id)"
        )

    def sessions(
        self,
        *,
        new_messages: Callable[[], MutableSequence[ParsedMessage]],
        new_events: Callable[[], MutableSequence[ParsedSessionEvent]],
    ) -> Iterator[ParsedSession]:
        """Yield one session per resource and conversation, or trace, in order."""
        self._select_variants()
        self._resolve_conversations()
        conn = self._conn
        for resource_key, kind, identity_key in conn.execute(
            "SELECT DISTINCT resource_id, group_kind, group_identity FROM otel_selected "
            "ORDER BY resource_id, group_kind, group_identity"
        ):
            messages = new_messages()
            events = new_events()
            group = (resource_key, kind, identity_key)
            for span_seq, trace_key, span_key in conn.execute(
                "SELECT span_seq, trace_id, span_id FROM otel_selected "
                "WHERE resource_id = ? AND group_kind = ? AND group_identity = ? "
                "ORDER BY start_key, span_key, trace_id, span_id",
                group,
            ):
                row = conn.execute("SELECT schema_url FROM otel_span WHERE seq = ?", (span_seq,)).fetchone()
                conflicts = (
                    (self._span(conflict_seq), cast("str | None", json.loads(schema_url_json)))
                    for conflict_seq, schema_url_json in conn.execute(
                        "SELECT c.span_seq, s.schema_url FROM otel_conflict c JOIN otel_span s ON s.seq = c.span_seq "
                        "WHERE c.resource_id = ? AND c.trace_id = ? AND c.span_id = ? ORDER BY c.seq",
                        (resource_key, trace_key, span_key),
                    )
                )
                _append_span(self._span(span_seq), json.loads(row[0]), conflicts, messages, events)
            models = [
                _from_text_key(model)
                for (model,) in conn.execute(
                    "SELECT DISTINCT model FROM otel_selected WHERE resource_id = ? AND group_kind = ? "
                    "AND group_identity = ? AND model IS NOT NULL ORDER BY model",
                    group,
                )
            ]
            resource_id = _from_text_key(resource_key)
            group_identity = _from_text_key(identity_key)
            session = ParsedSession(
                source_name=Provider.OTEL_GENAI,
                provider_session_id=f"{resource_id}:{kind}:{group_identity}",
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
    "OTLP_JSON_DIALECT",
    "SEMCONV_SCHEMA_URL",
    "OtelSpanIndex",
    "has_span_identity",
    "looks_like",
    "parse",
    "resource_id_for",
    "scope_schema_url",
]
