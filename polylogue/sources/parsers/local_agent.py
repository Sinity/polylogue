"""Parsers for local agent session JSON documents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from polylogue.archive.message.roles import Role
from polylogue.core.json import JSONDocument, json_document
from polylogue.types import ContentBlockType, Provider

from .base import ParsedContentBlock, ParsedConversation, ParsedMessage


def looks_like_gemini_cli(payload: JSONDocument) -> bool:
    return (
        isinstance(payload.get("sessionId"), str)
        and isinstance(payload.get("messages"), list)
        and ("startTime" in payload or "lastUpdated" in payload or payload.get("kind") == "chat")
    )


def looks_like_hermes(payload: JSONDocument) -> bool:
    return (
        isinstance(payload.get("session_id"), str)
        and isinstance(payload.get("messages"), list)
        and ("session_start" in payload or "last_updated" in payload or "platform" in payload)
    )


def parse_gemini_cli(payload: JSONDocument, fallback_id: str) -> ParsedConversation:
    session_id = _string(payload.get("sessionId")) or fallback_id
    messages = [
        parsed
        for index, item in enumerate(_list(payload.get("messages")), start=1)
        if (parsed := _parse_gemini_message(item, index=index)) is not None
    ]
    provider_meta = _conversation_meta(
        payload,
        keys=("kind", "projectHash", "summary"),
        source_family="gemini-cli",
    )
    return ParsedConversation(
        provider_name=Provider.GEMINI_CLI,
        provider_conversation_id=session_id,
        title=_string(payload.get("summary")) or session_id,
        created_at=_string(payload.get("startTime")),
        updated_at=_string(payload.get("lastUpdated")),
        messages=messages,
        provider_meta=provider_meta,
    )


def parse_hermes(payload: JSONDocument, fallback_id: str) -> ParsedConversation:
    session_id = _string(payload.get("session_id")) or fallback_id
    messages: list[ParsedMessage] = []
    system_prompt = _string(payload.get("system_prompt"))
    if system_prompt:
        messages.append(
            ParsedMessage(
                provider_message_id=f"{session_id}:system",
                role=Role.SYSTEM,
                text=system_prompt,
                content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=system_prompt)],
            )
        )
    messages.extend(
        parsed
        for index, item in enumerate(_list(payload.get("messages")), start=1)
        if (parsed := _parse_hermes_message(item, index=index)) is not None
    )
    provider_meta = _conversation_meta(
        payload,
        keys=("model", "base_url", "platform", "tools", "message_count"),
        source_family="hermes",
    )
    return ParsedConversation(
        provider_name=Provider.HERMES,
        provider_conversation_id=session_id,
        title=session_id,
        created_at=_string(payload.get("session_start")),
        updated_at=_string(payload.get("last_updated")),
        messages=messages,
        provider_meta=provider_meta,
    )


def _parse_gemini_message(item: object, *, index: int) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    thoughts = _list(record.get("thoughts"))
    for thought_index, thought in enumerate(thoughts, start=1):
        thought_text = _content_text(thought)
        if thought_text:
            content_blocks.append(
                ParsedContentBlock(
                    type=ContentBlockType.THINKING,
                    text=thought_text,
                    metadata={"index": thought_index},
                )
            )
    for tool_index, tool_call in enumerate(_list(record.get("toolCalls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        content_blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{index}-{tool_index}"))
    if not text and not content_blocks:
        return None
    provider_meta = _message_meta(record, keys=("model", "tokens"))
    return ParsedMessage(
        provider_message_id=_string(record.get("id")) or f"msg-{index}",
        role=_role(_string(record.get("type")) or "unknown", assistant_aliases={"gemini", "model"}),
        text=text,
        timestamp=_string(record.get("timestamp")),
        content_blocks=content_blocks or [ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
        provider_meta=provider_meta,
    )


def _parse_hermes_message(item: object, *, index: int) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    reasoning = _string(record.get("reasoning_content")) or _string(record.get("reasoning"))
    if reasoning:
        content_blocks.append(ParsedContentBlock(type=ContentBlockType.THINKING, text=reasoning))
    for tool_index, tool_call in enumerate(_list(record.get("tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        content_blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{index}-{tool_index}"))
    tool_call_id = _string(record.get("tool_call_id"))
    role = _role(_string(record.get("role")) or "unknown")
    if role is Role.TOOL and text:
        content_blocks.append(ParsedContentBlock(type=ContentBlockType.TOOL_RESULT, tool_id=tool_call_id, text=text))
    if not text and not content_blocks:
        return None
    provider_meta = _message_meta(record, keys=("finish_reason",))
    return ParsedMessage(
        provider_message_id=tool_call_id or f"msg-{index}",
        role=role,
        text=text,
        content_blocks=content_blocks or [ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
        provider_meta=provider_meta,
    )


def _conversation_meta(payload: JSONDocument, *, keys: Sequence[str], source_family: str) -> dict[str, object]:
    meta: dict[str, object] = {"source_family": source_family}
    for key in keys:
        value = payload.get(key)
        if value is not None:
            meta[key] = value
    return meta


def _message_meta(record: JSONDocument, *, keys: Sequence[str]) -> dict[str, object] | None:
    meta: dict[str, object] = {}
    for key in keys:
        value = record.get(key)
        if value is not None:
            meta[key] = value
    return meta or None


def _content_blocks_from_content(content: object) -> list[ParsedContentBlock]:
    if isinstance(content, str):
        return [ParsedContentBlock(type=ContentBlockType.TEXT, text=content)] if content else []
    if isinstance(content, list):
        blocks: list[ParsedContentBlock] = []
        for index, item in enumerate(content, start=1):
            text = _content_text(item)
            if text:
                blocks.append(
                    ParsedContentBlock(
                        type=ContentBlockType.TEXT,
                        text=text,
                        metadata={"index": index} if not isinstance(item, str) else None,
                    )
                )
        return blocks
    if isinstance(content, Mapping):
        text = _content_text(content)
        return [ParsedContentBlock(type=ContentBlockType.TEXT, text=text)] if text else []
    return []


def _content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content if content else None
    if isinstance(content, list):
        parts = [_content_text(item) for item in content]
        text = "\n".join(part for part in parts if part)
        return text or None
    if isinstance(content, Mapping):
        for key in ("text", "content", "message", "value"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        try:
            return json.dumps(content, sort_keys=True)
        except TypeError:
            return str(content)
    return None


def _tool_use_block(record: JSONDocument, *, fallback_id: str) -> ParsedContentBlock:
    function = json_document(record.get("function"))
    tool_name = _string(record.get("name")) or _string(function.get("name")) or _string(record.get("type")) or "tool"
    tool_id = _string(record.get("id")) or _string(record.get("call_id")) or fallback_id
    raw_input = record.get("arguments") if "arguments" in record else function.get("arguments")
    return ParsedContentBlock(
        type=ContentBlockType.TOOL_USE,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=_tool_input(raw_input),
    )


def _tool_input(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"arguments": value}
        return dict(parsed) if isinstance(parsed, dict) else {"arguments": value}
    return {}


def _role(raw: str, *, assistant_aliases: set[str] | None = None) -> Role:
    lowered = raw.strip().lower()
    if assistant_aliases and lowered in assistant_aliases:
        return Role.ASSISTANT
    return Role.normalize(lowered)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


__all__ = [
    "looks_like_gemini_cli",
    "looks_like_hermes",
    "parse_gemini_cli",
    "parse_hermes",
]
