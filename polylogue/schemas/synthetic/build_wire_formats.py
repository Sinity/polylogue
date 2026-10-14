"""Provider wire-format shaping helpers for synthetic corpora."""

from __future__ import annotations

import base64
import random
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import chain
from typing import Protocol, TypeAlias

from polylogue.schemas.record_structure import codex_session_header
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.demo_themes import SessionTheme
from polylogue.schemas.synthetic.models import GeneratedRecord, SchemaRecord, SyntheticGenerationState
from polylogue.schemas.synthetic.models import GeneratedValue as JSONValue
from polylogue.schemas.synthetic.observed_values import existing_value_is_supported
from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
from polylogue.schemas.synthetic.semantic_values import _text_for_role
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText
from polylogue.sources.parsers.claude.common import CLAUDE_CONTENT_SEGMENT_TYPES

SyntheticRecord: TypeAlias = dict[str, JSONValue] | GeneratedRecord


def _as_record(value: object) -> SyntheticRecord:
    return value if isinstance(value, dict | GeneratedRecord) else {}


def _as_schema(value: object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


def normalize_browser_capture_attachments(data: JSONValue) -> None:
    """Encode generated attachment strings at the browser-capture wire boundary."""
    from polylogue.browser_capture.models import looks_like_browser_capture
    from polylogue.sources.parsers.base_support import decode_attachment_base64

    if not isinstance(data, dict | GeneratedRecord) or not looks_like_browser_capture(data):
        return
    session = _as_record(data.get("session"))
    turns = session.get("turns")
    turn_owners = turns if isinstance(turns, list | GeneratedArray) else ()
    for owner in chain((session,), (turn for turn in turn_owners if isinstance(turn, dict | GeneratedRecord))):
        attachments = owner.get("attachments")
        if not isinstance(attachments, list | GeneratedArray):
            continue
        for attachment in attachments:
            if not isinstance(attachment, dict | GeneratedRecord):
                continue
            carriers: list[tuple[SyntheticRecord, tuple[str, ...]]] = [
                (attachment, ("content_base64", "inline_base64", "data"))
            ]
            metadata = attachment.get("provider_meta")
            if isinstance(metadata, dict | GeneratedRecord):
                carriers.append((metadata, ("content_base64", "inline_base64")))
            for carrier, keys in carriers:
                for key in keys:
                    value = carrier.get(key)
                    if not isinstance(value, str):
                        continue
                    try:
                        decode_attachment_base64(value, field_name=key)
                    except ValueError:
                        carrier[key] = base64.b64encode(value.encode("utf-8")).decode("ascii")


class _WireFieldContext(Protocol):
    _coverage_witness_mode: bool
    _public_record_plan: PublicRecordPlan | None
    _generation_state: SyntheticGenerationState

    def _generate_from_schema(self, schema: SchemaRecord, rng: random.Random, *, path: str = ...) -> JSONValue: ...


class _WireFormatContext(_WireFieldContext, Protocol):
    provider: str
    _public_record_plan: PublicRecordPlan | None
    _coverage_witness_mode: bool
    _generation_state: SyntheticGenerationState

    def _generate_from_schema(self, schema: SchemaRecord, rng: random.Random, *, path: str = ...) -> JSONValue: ...

    def _ensure_wire_chatgpt(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None: ...

    def _ensure_wire_claude_ai(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None: ...

    def _ensure_wire_claude_code(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None: ...

    def _ensure_wire_codex(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None: ...

    def _ensure_wire_gemini(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None: ...


def _ensure_wire_format(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    index: int,
    base_ts: float = 1700000000.0,
    theme: SessionTheme | None = None,
) -> None:
    ts = base_ts + index * 60
    match self.provider:
        case "chatgpt":
            self._ensure_wire_chatgpt(data, role, rng, ts, index=index, theme=theme)
        case "claude-ai":
            self._ensure_wire_claude_ai(data, role, rng, ts, index=index, theme=theme)
        case "claude-code":
            self._ensure_wire_claude_code(data, role, rng, ts, index=index, theme=theme)
        case "codex":
            self._ensure_wire_codex(data, role, rng, ts, index=index, theme=theme)
        case "gemini":
            self._ensure_wire_gemini(data, role, rng, index=index, theme=theme)
        case _:
            raise ValueError(f"No executable synthetic wire adapter for provider {self.provider!r}")


def _retain_wire_proposal(self: _WireFieldContext, value: JSONValue) -> JSONValue:
    """Retain fallback containers before adapters mutate their borrowed fields.

    Scalars and existing native coordinates keep their original identity;
    only explicit provider proposals need container custody. The traversal
    retains active ancestors, never a payload-sized completed-node registry.
    """
    owner = self._generation_state.relation_solver.fk_graph
    frames: list[
        tuple[
            dict[str, JSONValue] | list[JSONValue],
            GeneratedRecord | GeneratedArray,
            Iterator[tuple[str | int, JSONValue]],
        ]
    ] = []
    active: set[int] = set()

    def retain(current: JSONValue) -> JSONValue:
        if isinstance(current, dict | list):
            if id(current) in active:
                raise ValueError("cyclic wire proposals have no finite JSON representation")
            active.add(id(current))
            target: GeneratedRecord | GeneratedArray
            if isinstance(current, dict):
                target = owner.new_record({})
                children: Iterator[tuple[str | int, JSONValue]] = iter(current.items())
            else:
                target = owner.new_array()
                children = iter(enumerate(current))
            frames.append((current, target, children))
            return target
        return current

    result = retain(value)
    while frames:
        original, target, children = frames[-1]
        try:
            key, child = next(children)
        except StopIteration:
            active.remove(id(original))
            frames.pop()
            continue
        retained = retain(child)
        if isinstance(target, GeneratedRecord):
            if not isinstance(key, str):
                raise TypeError("wire object keys must be strings")
            target[key] = retained
        else:
            target.append(retained)
    return result


def _wire_default(
    self: _WireFieldContext,
    record: SyntheticRecord,
    fields: tuple[str, ...],
    rng: random.Random,
    proposal: JSONValue,
    *,
    semantic_role: str | None = None,
) -> JSONValue:
    """Generate a missing wire field from its actual selected record schema.

    Runtime binds every generated object to its selected union branch for the
    session lifetime. Provider adapters therefore use the same scalar owner,
    rather than guessing a branch or overwriting its observed constraints.
    """
    schema = _wire_schema(self, record, fields)
    origin = record.original_path if isinstance(record, GeneratedRecord) else None
    coordinate = origin + "." + ".".join(fields) if origin is not None else None
    plan = None if self._coverage_witness_mode else self._public_record_plan
    if coordinate is not None and plan is not None and coordinate in plan.scalars:
        selected = plan.scalars[coordinate]
        if schema and not existing_value_is_supported(schema, selected):
            raise ValueError("public wire default conflicts with selected scalar support")
        return selected
    if not schema:
        return _retain_wire_proposal(self, proposal)
    if semantic_role is not None and "x-polylogue-semantic-role" not in schema:
        schema = {**schema, "x-polylogue-semantic-role": semantic_role}
    if origin is None:
        # An opaque provider proposal has no schema field coordinate. It may
        # borrow selected scalar support, but cannot acquire root constraints.
        origin = "<wire-proposal>"
    return self._generate_from_schema(schema, rng, path=origin + "." + ".".join(fields))


def _wire_schema(self: _WireFieldContext, record: SyntheticRecord, fields: tuple[str, ...]) -> SchemaRecord:
    schema = record.selected_schema if isinstance(record, GeneratedRecord) else {}
    cursor: object = record
    for field in fields:
        if isinstance(cursor, GeneratedRecord):
            schema = cursor.selected_schema
        if field == "*":
            schema = _as_schema(schema.get("items"))
            cursor = cursor[0] if isinstance(cursor, list | GeneratedArray) and cursor else None
        else:
            properties = _as_schema(schema.get("properties"))
            selected_field = properties.get(field, schema.get("additionalProperties"))
            if selected_field is False:
                raise ValueError("required wire field is forbidden by the selected schema")
            schema = _as_schema(selected_field)
            cursor = cursor.get(field) if isinstance(cursor, dict | GeneratedRecord) else None
    return schema


def _wire_field(
    self: _WireFieldContext,
    record: SyntheticRecord,
    field: str,
    rng: random.Random,
    proposal: JSONValue,
    *,
    semantic_role: str | None = None,
) -> JSONValue:
    if semantic_role is None and field in {"role", "sender"}:
        semantic_role = "message_role"
    if field not in record:
        record[field] = _wire_default(self, record, (field,), rng, proposal, semantic_role=semantic_role)
    selected = record[field]
    if field in {"role", "sender", "type", "content_type", "name"} and not isinstance(selected, str):
        raise ValueError("selected wire discriminator must be a string")
    return selected


def _wire_alias(self: _WireFieldContext, record: SyntheticRecord, field: str, value: JSONValue) -> None:
    if field in record:
        if record[field] != value:
            raise ValueError("independent wire content owners cannot both be conserved")
        return
    schema = _wire_schema(self, record, (field,))
    if schema and not existing_value_is_supported(schema, value):
        raise ValueError("existing wire content conflicts with the selected alias schema")
    record[field] = value


def _wire_record(
    self: _WireFieldContext,
    root: SyntheticRecord,
    fields: tuple[str, ...],
    rng: random.Random,
) -> SyntheticRecord:
    current = root
    for index, field in enumerate(fields):
        if field not in current:
            current[field] = _wire_default(self, root, fields[: index + 1], rng, {})
        selected = current[field]
        if not isinstance(selected, dict | GeneratedRecord):
            raise ValueError("selected wire carrier must be an object")
        current = selected
    return current


def validate_wire_payload(provider: str, data: JSONValue) -> None:
    """Reject provider-wire combinations the production parser does not own."""

    if provider != "codex" or not isinstance(data, list | GeneratedArray):
        return
    has_flat_message = any(
        isinstance(record, dict | GeneratedRecord)
        and (record.get("type") == "message" or isinstance(record.get("role"), str))
        for record in data
    )
    has_envelope_message = any(
        isinstance(record, dict | GeneratedRecord)
        and record.get("type") in {"response_item", "event_msg", "turn_context", "session_meta"}
        for record in data
    )
    if has_flat_message and has_envelope_message:
        raise ValueError("Codex synthetic payload cannot mix flat records with envelope records")


def _ensure_wire_chatgpt(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    ts: float,
    *,
    index: int,
    theme: SessionTheme | None,
) -> None:
    if self._public_record_plan is not None and self._public_record_plan.kind == "record:empty":
        return
    msg = _wire_record(self, data, ("message",), rng)
    _wire_field(self, msg, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))

    author = _wire_record(self, data, ("message", "author"), rng)
    _wire_field(self, author, "role", rng, role, semantic_role="message_role")

    content = _wire_record(self, data, ("message", "content"), rng)
    _wire_field(
        self,
        content,
        "parts",
        rng,
        [_text_for_role(rng, role, turn_index=index, theme=theme)],
    )
    _wire_field(self, content, "content_type", rng, "text")

    if "create_time" not in msg:
        msg["create_time"] = _wire_default(
            self, data, ("message", "create_time"), rng, ts, semantic_role="message_timestamp"
        )


def _ensure_wire_claude_ai(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    ts: float,
    *,
    index: int,
    theme: SessionTheme | None,
) -> None:
    # Preserve schema-generated observed fields. Wire defaults only fill the
    # parser-required carriers that the selected schema did not generate.
    uuid_val = _wire_field(self, data, "uuid", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    text = data.get("text")
    if "text" not in data:
        text = _wire_default(
            self,
            data,
            ("text",),
            rng,
            _text_for_role(rng, role, turn_index=index, theme=theme),
            semantic_role="message_body",
        )
    data["uuid"] = uuid_val
    _wire_field(self, data, "sender", rng, role)
    data["text"] = text
    if "created_at" not in data:
        data["created_at"] = _wire_default(
            self,
            data,
            ("created_at",),
            rng,
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            semantic_role="message_timestamp",
        )


def _ensure_wire_claude_code(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    ts: float,
    *,
    index: int,
    theme: SessionTheme | None,
) -> None:
    plan = self._public_record_plan
    if plan is not None and plan.form != "claude-message":
        # A system/progress/metadata record can retain an incidental message
        # property in its original schema. Its selected dispatcher is not a
        # chat-message branch, so preserve those fields without chat defaults.
        return
    if self._public_record_plan is None:
        _wire_field(self, data, "type", rng, "user" if role == "tool" else role)
    record_type = _wire_field(self, data, "type", rng, role)
    if self._public_record_plan is None or record_type in {"assistant", "user", "system"}:
        msg = _wire_record(self, data, ("message",), rng)
        if self._public_record_plan is None:
            _wire_field(self, msg, "role", rng, "user" if role == "tool" else role)
        if "content" not in msg and self._public_record_plan is None:
            proposal = _claude_code_content_fallback(rng, role, index, theme)
            msg["content"] = _wire_default(self, data, ("message", "content"), rng, proposal)
        else:
            _normalize_claude_code_content(
                self, msg, rng, role, index, theme, flatten_nested=self._coverage_witness_mode
            )
    if "timestamp" not in data:
        data["timestamp"] = _wire_default(
            self,
            data,
            ("timestamp",),
            rng,
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            semantic_role="message_timestamp",
        )


def _normalize_claude_code_content(
    self: _WireFormatContext,
    message: SyntheticRecord,
    rng: random.Random,
    role: str,
    index: int,
    theme: SessionTheme | None,
    *,
    flatten_nested: bool,
) -> None:
    """Keep coverage witnesses on block forms the Claude Code route emits."""
    content = message.get("content")
    if not isinstance(content, list | GeneratedArray):
        return
    for block in content:
        if not isinstance(block, dict | GeneratedRecord):
            continue
        block_type = _wire_field(self, block, "type", rng, "text")
        if block_type not in CLAUDE_CONTENT_SEGMENT_TYPES:
            raise ValueError("selected Claude Code block type is not parser-representable")
        if block_type in {"text", "thinking"}:
            _wire_field(
                self,
                block,
                str(block_type),
                rng,
                _text_for_role(rng, role, turn_index=index, theme=theme),
                semantic_role="message_body",
            )
        elif block_type == "tool_use" and self._public_record_plan is None:
            _wire_field(self, block, "name", rng, "Read")
            _wire_field(self, block, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
            _wire_record(self, block, ("input",), rng)
        elif "content" not in block and self._public_record_plan is None:
            _wire_field(
                self,
                block,
                "content",
                rng,
                _text_for_role(rng, role, turn_index=index, theme=theme),
                semantic_role="message_body",
            )
        if flatten_nested and isinstance(block.get("content"), list | GeneratedArray):
            raise ValueError("nested Claude Code content cannot be flattened without losing selected values")
        _alias_claude_code_block_content(self, block, rng)


def _alias_claude_code_block_content(self: _WireFieldContext, block: SyntheticRecord, rng: random.Random) -> None:
    """Make a text, thinking or tool-use block's ``content`` repeat its own body.

    The inferred block schema merges every form, so it offers the tool-result
    ``content`` body to the other forms too. Those forms carry no such field
    on the wire and their parser reads only the typed body, so a witness that
    keeps the field for construct coverage must not give it an independent
    value no parser of that form can conserve.
    """
    content = block.get("content")
    block_type = block.get("type")
    if not isinstance(content, str | SampledText | JoinedText) or block_type not in {"text", "thinking", "tool_use"}:
        return
    owner: SyntheticRecord = block
    field = str(block_type)
    if block_type == "tool_use":
        owner, field = _wire_record(self, block, ("input",), rng), "content"
    _wire_alias(self, owner, field, content)


def _claude_code_content_fallback(rng: random.Random, role: str, index: int, theme: SessionTheme | None) -> JSONValue:
    """Diverse content fallback matching production block-type distribution."""
    block_type = rng.choices(
        ["text", "tool_use", "tool_result", "thinking"],
        weights=[0.25, 0.14, 0.57, 0.04],
        k=1,
    )[0]
    if block_type == "text":
        return [{"type": "text", "text": _text_for_role(rng, role, turn_index=index, theme=theme)}]
    if block_type == "tool_use":
        return [
            {
                "type": "tool_use",
                "name": rng.choice(["Read", "Grep", "Bash", "Edit", "Write"]),
                "id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "input": {"query": _text_for_role(rng, role, turn_index=index, theme=theme)},
            }
        ]
    if block_type == "tool_result":
        return [
            {
                "type": "tool_result",
                "tool_use_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "content": _text_for_role(rng, "assistant", turn_index=index, theme=theme),
            }
        ]
    # thinking
    return [{"type": "thinking", "thinking": _text_for_role(rng, role, turn_index=index, theme=theme)}]


def _ensure_wire_codex(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    ts: float,
    *,
    index: int,
    theme: SessionTheme | None,
) -> None:
    if (
        self._public_record_plan is None
        and isinstance(data.get("payload"), dict | GeneratedRecord)
        and "type" not in data
    ):
        _wire_field(self, data, "type", rng, "response_item")
    if self._public_record_plan is not None and self._public_record_plan.form.startswith("codex-flat-header"):
        if codex_session_header(data) is None:
            raise ValueError("selected flat Codex header lacks its original identity/timestamp carrier")
        return
    if self._public_record_plan is not None and self._public_record_plan.form.startswith("codex-role-message"):
        if not isinstance(data.get("role"), str):
            raise ValueError("selected role-bearing Codex message lacks its original role carrier")
        return
    if (self._public_record_plan is not None and self._public_record_plan.form == "codex-envelope") or data.get(
        "type"
    ) in {
        "response_item",
        "event_msg",
        "turn_context",
        "session_meta",
    }:
        record_type = data.get("type")
        payload = _wire_record(self, data, ("payload",), rng)
        if record_type == "response_item":
            payload_type = _wire_field(self, payload, "type", rng, "message")
            if payload_type == "message":
                _wire_field(self, payload, "role", rng, role)
                if "content" not in payload:
                    if "content" in data:
                        _wire_alias(self, payload, "content", data["content"])
                    else:
                        payload["content"] = _wire_default(
                            self, data, ("payload", "content"), rng, _codex_content_fallback(rng, role, index, theme)
                        )
                if "content" in data and payload["content"] != data["content"]:
                    raise ValueError("Codex flat and payload content cannot both be independently conserved")
                _wire_field(self, payload, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        elif record_type == "event_msg":
            payload_type = _wire_field(
                self, payload, "type", rng, "user_message" if role == "user" else "agent_message"
            )
            if payload_type in {"user_message", "agent_message"} and "message" not in payload:
                payload["message"] = _wire_default(
                    self,
                    data,
                    ("payload", "message"),
                    rng,
                    _text_for_role(rng, role, turn_index=index, theme=theme),
                    semantic_role="message_body",
                )
        elif record_type == "turn_context":
            _wire_field(self, payload, "cwd", rng, "/workspace/project")
            _wire_field(self, payload, "model", rng, "synthetic-model")
        elif record_type == "session_meta":
            _wire_field(self, payload, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
            if "timestamp" not in payload:
                payload["timestamp"] = _wire_default(
                    self,
                    data,
                    ("payload", "timestamp"),
                    rng,
                    datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    semantic_role="message_timestamp",
                )
        if "timestamp" not in data:
            data["timestamp"] = _wire_default(
                self,
                data,
                ("timestamp",),
                rng,
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                semantic_role="message_timestamp",
            )
        return

    _wire_field(self, data, "type", rng, "message")
    _wire_field(self, data, "role", rng, role)
    if "content" not in data:
        data["content"] = _wire_default(
            self,
            data,
            ("content",),
            rng,
            _codex_content_fallback(rng, role, index, theme),
        )
    content = data.get("content")
    if isinstance(content, list | GeneratedArray):
        for block in content:
            if not isinstance(block, dict | GeneratedRecord):
                continue
            block_type = _wire_field(self, block, "type", rng, "input_text" if role == "user" else "output_text")
            if block_type not in {"input_text", "output_text", "thinking", "tool_use", "tool_result"}:
                raise ValueError("selected Codex block type is not parser-representable")
            if block_type in {"input_text", "output_text"}:
                _wire_field(
                    self,
                    block,
                    "text",
                    rng,
                    _text_for_role(rng, role, turn_index=index, theme=theme),
                    semantic_role="message_body",
                )
    _wire_field(self, data, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    flat_payload = data.get("payload")
    if isinstance(flat_payload, dict | GeneratedRecord):
        _wire_field(self, flat_payload, "role", rng, role, semantic_role="message_role")
        if "content" not in flat_payload:
            _wire_alias(self, flat_payload, "content", data["content"])
        if flat_payload["content"] != data["content"]:
            raise ValueError("Codex flat and payload content cannot both be independently conserved")


def _codex_content_fallback(rng: random.Random, role: str, index: int, theme: SessionTheme | None) -> JSONValue:
    """Diverse content fallback matching production block-type distribution."""
    block_type = rng.choices(
        ["text", "tool_use", "tool_result", "thinking"],
        weights=[0.45, 0.20, 0.30, 0.05],
        k=1,
    )[0]
    text = _text_for_role(rng, role, turn_index=index, theme=theme)
    if block_type == "tool_use":
        return [
            {
                "type": "tool_use",
                "name": rng.choice(["Bash", "Read", "Write", "Grep", "Search"]),
                "id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "input": {"query": text},
            }
        ]
    if block_type == "tool_result":
        return [
            {
                "type": "tool_result",
                "tool_use_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "content": text,
                "text": text,
            }
        ]
    if block_type == "thinking":
        return [{"type": "thinking", "thinking": text}]
    # text (including the old input_text/output_text distinction for backward compat)
    content_type = "input_text" if role == "user" else "output_text"
    return [{"type": content_type, "text": text}]


def _ensure_wire_gemini(
    self: _WireFormatContext,
    data: SyntheticRecord,
    role: str,
    rng: random.Random,
    *,
    index: int,
    theme: SessionTheme | None,
) -> None:
    # Preserve observed chunk fields; their parser-owned blocks are retained
    # by the same artifact projection used by coverage receipts.
    text = data.get("text")
    if "text" not in data:
        text = _wire_default(
            self,
            data,
            ("text",),
            rng,
            _text_for_role(rng, role, turn_index=index, theme=theme),
            semantic_role="message_body",
        )
    create_time = data.get("createTime")
    _wire_field(self, data, "role", rng, role)
    data["text"] = text
    if isinstance(create_time, str) and create_time:
        data["createTime"] = create_time


__all__ = [
    "_ensure_wire_chatgpt",
    "_ensure_wire_claude_ai",
    "_ensure_wire_claude_code",
    "_ensure_wire_codex",
    "_ensure_wire_format",
    "_ensure_wire_gemini",
    "validate_wire_payload",
]
