"""Provider wire-format shaping helpers for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Protocol, TypeAlias

from polylogue.archive.raw_payload.decode import JSONValue
from polylogue.schemas.synthetic.semantic_values import _text_for_role
from polylogue.schemas.synthetic.showcase import SessionTheme

SyntheticRecord: TypeAlias = dict[str, JSONValue]


def _as_record(value: object) -> SyntheticRecord:
    return value if isinstance(value, dict) else {}


def _record_field(record: SyntheticRecord, field_name: str) -> SyntheticRecord:
    nested = _as_record(record.get(field_name))
    record[field_name] = nested
    return nested


class _WireFormatContext(Protocol):
    provider: str

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
    msg = _record_field(data, "message")
    msg.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))

    author = _record_field(msg, "author")
    author.setdefault("role", role)

    content = _record_field(msg, "content")
    if "parts" not in content or not content["parts"]:
        content["parts"] = [_text_for_role(rng, role, turn_index=index, theme=theme)]
    content.setdefault("content_type", "text")

    msg.setdefault("create_time", ts)


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
    # A realistic claude.ai ``chat_messages`` entry is a compact turn: a sender
    # plus text. ``_generate_from_schema`` instead fills every optional field
    # (``content`` blocks, ``attachments``, ``files``), bloating each message so
    # the conversation's discriminating ``chat_messages`` key falls outside the
    # 8 KB acquisition detection prefix (``_DETECTION_PREFIX_SIZE``) and the
    # bundle mis-detects → zero sessions imported. Reset the message to the
    # realistic minimal shape, preserving a schema-generated uuid/created_at.
    uuid_val = data.get("uuid") or str(uuid.UUID(int=rng.getrandbits(128), version=4))
    text = data.get("text")
    if not isinstance(text, str) or not text:
        text = _text_for_role(rng, role, turn_index=index, theme=theme)
    created_at = data.get("created_at")
    data.clear()
    data["uuid"] = uuid_val
    data["sender"] = role
    data["text"] = text
    data["created_at"] = (
        created_at
        if isinstance(created_at, str) and created_at
        else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
    data.setdefault("type", role)
    msg = _record_field(data, "message")
    msg.setdefault("role", role)
    if "content" not in msg:
        msg["content"] = _claude_code_content_fallback(rng, role, index, theme)  # type: ignore[assignment]
    if "timestamp" not in data:
        data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _claude_code_content_fallback(rng: random.Random, role: str, index: int, theme: SessionTheme | None) -> object:
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
    data["type"] = "message"
    data.setdefault("role", role)
    if "content" not in data:
        data["content"] = _codex_content_fallback(rng, role, index, theme)  # type: ignore[assignment]
    data.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    # Strip schema-generated payload envelope so the parser reads the
    # top-level role/content directly. Without this, _effective_role finds
    # the payload dict but no role inside it, returns "unknown", and the
    # message is silently dropped.
    data.pop("payload", None)


def _codex_content_fallback(rng: random.Random, role: str, index: int, theme: SessionTheme | None) -> object:
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
    # A realistic AI Studio / Drive `chunkedPrompt.chunks` entry is a single
    # coherent turn: a role plus text. `_generate_from_schema` instead fills
    # every optional chunk field at once (`isThought`, `executableCode`,
    # `codeExecutionResult`, `errorMessage`, and the `drive*`/`inline*`
    # attachment fields). The drive parser then fragments one chunk into many
    # blocks (thinking + text + code + error), so the message's joined `.text`
    # never round-trips contiguously through the renderer. Reset the chunk to
    # the realistic minimal shape, preserving only a schema-generated
    # `createTime` for timestamp coverage.
    text = data.get("text")
    if not isinstance(text, str) or not text:
        text = _text_for_role(rng, role, turn_index=index, theme=theme)
    create_time = data.get("createTime")
    data.clear()
    data["role"] = role
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
]
