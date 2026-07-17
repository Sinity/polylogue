"""Parsers for local agent session JSON documents."""

from __future__ import annotations

import json
from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, BranchType, Provider
from polylogue.core.json import JSONDocument, json_document

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent


def looks_like_gemini_cli(payload: JSONDocument) -> bool:
    return (
        isinstance(payload.get("sessionId"), str)
        and isinstance(payload.get("messages"), list)
        and ("startTime" in payload or "lastUpdated" in payload or payload.get("kind") in {"chat", "main", "subagent"})
    )


def looks_like_hermes(payload: JSONDocument) -> bool:
    return (
        isinstance(payload.get("session_id"), str)
        and isinstance(payload.get("messages"), list)
        and ("session_start" in payload or "last_updated" in payload or "platform" in payload)
    )


def parse_gemini_cli(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    session_id = _string(payload.get("sessionId")) or fallback_id
    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    models_used: set[str] = set()
    for index, item in enumerate(_list(payload.get("messages")), start=1):
        parsed = _parse_gemini_message(item, index=index, position=len(messages))
        if parsed is not None:
            messages.append(parsed)
            if parsed.model_name:
                models_used.add(parsed.model_name)
            if usage_event := _gemini_message_usage_event(item, parsed):
                session_events.append(usage_event)
    messages = _mark_active_leaf(messages)
    return ParsedSession(
        source_name=Provider.GEMINI_CLI,
        provider_session_id=session_id,
        title=_string(payload.get("summary")) or session_id,
        created_at=_string(payload.get("startTime")),
        updated_at=_string(payload.get("lastUpdated")),
        messages=messages,
        branch_type=BranchType.SUBAGENT if payload.get("kind") == "subagent" else None,
        session_events=session_events,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
        models_used=sorted(models_used),
        provider_project_ref=_string(payload.get("projectHash")),
        working_directories=[
            directory for directory in _list(payload.get("directories")) if isinstance(directory, str) and directory
        ],
    )


def parse_hermes(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    session_id = _string(payload.get("session_id")) or fallback_id
    messages: list[ParsedMessage] = []
    system_prompt = _string(payload.get("system_prompt"))
    if system_prompt:
        messages.append(
            ParsedMessage(
                provider_message_id=f"{session_id}:system",
                role=Role.SYSTEM,
                text=system_prompt,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=system_prompt)],
                position=0,
                variant_index=0,
                is_active_path=True,
                model_name=_string(payload.get("model")),
            )
        )
    for index, item in enumerate(_list(payload.get("messages")), start=1):
        parsed = _parse_hermes_message(
            item,
            index=index,
            position=len(messages),
            fallback_model=_string(payload.get("model")),
        )
        if parsed is not None:
            messages.append(parsed)
    messages = _mark_active_leaf(messages)
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=session_id,
        created_at=_string(payload.get("session_start")),
        updated_at=_string(payload.get("last_updated")),
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
    )


def _parse_gemini_message(item: object, *, index: int, position: int) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    thoughts = _list(record.get("thoughts"))
    for thought_index, thought in enumerate(thoughts, start=1):
        thought_record = json_document(thought)
        thought_text = (
            _string(thought_record.get("description"))
            or _content_text(thought)
            or _string(thought_record.get("subject"))
        )
        if thought_text:
            thought_metadata: dict[str, object] = {"index": thought_index}
            for key in ("subject", "timestamp"):
                value = thought_record.get(key)
                if isinstance(value, str) and value:
                    thought_metadata[key] = value
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=thought_text,
                    metadata=thought_metadata,
                )
            )
    for tool_index, tool_call in enumerate(_list(record.get("toolCalls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        fallback_tool_id = f"tool-{index}-{tool_index}"
        content_blocks.append(_tool_use_block(tool_record, fallback_id=fallback_tool_id))
        content_blocks.extend(_tool_result_blocks(tool_record, fallback_id=fallback_tool_id))
    if not text and not content_blocks:
        return None
    token_usage = _token_usage_fields(record)
    return ParsedMessage(
        provider_message_id=_string(record.get("id")) or f"msg-{index}",
        role=_role(_string(record.get("type")) or "unknown", assistant_aliases={"gemini", "model"}),
        text=text,
        timestamp=_string(record.get("timestamp")),
        blocks=content_blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)],
        position=position,
        variant_index=0,
        is_active_path=True,
        model_name=_string(record.get("model")),
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        cache_read_tokens=token_usage["cache_read_tokens"],
        cache_write_tokens=token_usage["cache_write_tokens"],
        duration_ms=_non_negative_int(
            record.get("durationMs") or record.get("duration_ms") or record.get("elapsed_ms")
        ),
    )


def _parse_hermes_message(
    item: object,
    *,
    index: int,
    position: int,
    fallback_model: str | None = None,
) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    reasoning = _string(record.get("reasoning_content")) or _string(record.get("reasoning"))
    if reasoning:
        content_blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning))
    for tool_index, tool_call in enumerate(_list(record.get("tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        content_blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{index}-{tool_index}"))
    tool_call_id = _string(record.get("tool_call_id"))
    role = _role(_string(record.get("role")) or "unknown")
    if role is Role.TOOL and text:
        content_blocks.append(ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=tool_call_id, text=text))
    if not text and not content_blocks:
        return None
    token_usage = _token_usage_fields(record)
    return ParsedMessage(
        provider_message_id=tool_call_id or f"msg-{index}",
        role=role,
        text=text,
        timestamp=_string(record.get("timestamp")) or _string(record.get("created_at")),
        blocks=content_blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)],
        position=position,
        variant_index=0,
        is_active_path=True,
        model_name=_string(record.get("model")) or fallback_model,
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        cache_read_tokens=token_usage["cache_read_tokens"],
        cache_write_tokens=token_usage["cache_write_tokens"],
        duration_ms=_non_negative_int(
            record.get("durationMs") or record.get("duration_ms") or record.get("elapsed_ms")
        ),
    )


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    if not messages:
        return messages
    active_leaf_message_provider_id = messages[-1].provider_message_id
    return [
        message.model_copy(update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id})
        for message in messages
    ]


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(float(value))
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _token_usage_fields(record: JSONDocument) -> dict[str, int]:
    usage = json_document(record.get("usage")) or json_document(record.get("tokens")) or record
    gemini_wire_fields = {"input", "output", "cached", "thoughts", "tool"}
    if any(key in usage for key in gemini_wire_fields):
        input_with_cached = _first_non_negative_int(usage, "input") or 0
        cache_read_tokens = _first_non_negative_int(usage, "cached") or 0
        return {
            "input_tokens": max(input_with_cached - cache_read_tokens, 0),
            "output_tokens": _first_non_negative_int(usage, "output") or 0,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": 0,
            "reasoning_output_tokens": _first_non_negative_int(usage, "thoughts") or 0,
            "tool_output_tokens": _first_non_negative_int(usage, "tool") or 0,
            "total_tokens": _first_non_negative_int(usage, "total") or 0,
        }
    input_tokens = _first_non_negative_int(usage, "input_tokens", "prompt_tokens") or 0
    explicit_output = _first_non_negative_int(
        usage,
        "output_tokens",
        "completion_tokens",
        "generated_tokens",
        "total_tokens",
        "total",
    )
    output_tokens = explicit_output or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": _first_non_negative_int(usage, "cache_read_tokens", "cache_read_input_tokens") or 0,
        "cache_write_tokens": _first_non_negative_int(
            usage,
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cache_write_input_tokens",
        )
        or 0,
        "reasoning_output_tokens": 0,
        "tool_output_tokens": 0,
        "total_tokens": _first_non_negative_int(usage, "total_tokens", "total") or 0,
    }


def _first_non_negative_int(payload: JSONDocument, *keys: str) -> int | None:
    for key in keys:
        if key in payload:
            value = _non_negative_int(payload.get(key))
            if value is not None:
                return value
    return None


def _gemini_message_usage_event(item: object, message: ParsedMessage) -> ParsedSessionEvent | None:
    record = json_document(item)
    raw_usage = json_document(record.get("usage")) or json_document(record.get("tokens"))
    if not raw_usage:
        return None
    usage = _token_usage_fields(record)
    last_usage = {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cached_input_tokens": usage["cache_read_tokens"],
        "cache_write_tokens": usage["cache_write_tokens"],
        "reasoning_output_tokens": usage["reasoning_output_tokens"],
        "total_tokens": usage["total_tokens"],
    }
    payload: dict[str, object] = {
        "type": "message_usage",
        "semantics": "per_message",
        "last_token_usage": last_usage,
        "wire_tokens": dict(raw_usage),
    }
    if usage["tool_output_tokens"]:
        payload["tool_output_tokens"] = usage["tool_output_tokens"]
    if message.model_name:
        payload["model"] = message.model_name
    return ParsedSessionEvent(
        event_type="message_usage",
        timestamp=message.timestamp,
        source_message_provider_id=message.provider_message_id,
        payload=payload,
    )


def _content_blocks_from_content(content: object) -> list[ParsedContentBlock]:
    if isinstance(content, str):
        return [ParsedContentBlock(type=BlockType.TEXT, text=content)] if content else []
    if isinstance(content, list):
        blocks: list[ParsedContentBlock] = []
        for index, item in enumerate(content, start=1):
            text = _content_text(item)
            if text:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=text,
                        metadata={"index": index} if not isinstance(item, str) else None,
                    )
                )
        return blocks
    if isinstance(content, Mapping):
        text = _content_text(content)
        return [ParsedContentBlock(type=BlockType.TEXT, text=text)] if text else []
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
    if "args" in record:
        raw_input = record.get("args")
    elif "arguments" in record:
        raw_input = record.get("arguments")
    else:
        raw_input = function.get("arguments")
    metadata = _tool_metadata(record)
    return ParsedContentBlock(
        type=BlockType.TOOL_USE,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=_tool_input(raw_input),
        metadata=metadata or None,
    )


def _tool_result_blocks(record: JSONDocument, *, fallback_id: str) -> list[ParsedContentBlock]:
    tool_id = _string(record.get("id")) or _string(record.get("call_id")) or fallback_id
    status = _string(record.get("status"))
    status_is_error = _status_is_error(status)
    metadata = _tool_metadata(record)
    blocks: list[ParsedContentBlock] = []
    for result_item in _list(record.get("result")):
        result_record = json_document(result_item)
        function_response = json_document(result_record.get("functionResponse"))
        if not function_response:
            continue
        response = json_document(function_response.get("response"))
        output = _string(response.get("output"))
        error = _string(response.get("error"))
        text = output or error or _content_text(record.get("resultDisplay"))
        if text is None and status is None:
            continue
        result_metadata = dict(metadata)
        function_name = _string(function_response.get("name"))
        if function_name:
            result_metadata["function_name"] = function_name
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=_string(function_response.get("id")) or tool_id,
                text=text or f"[{status}]",
                metadata=result_metadata or None,
                is_error=True if error else status_is_error,
            )
        )
    if blocks:
        return blocks
    display_text = _content_text(record.get("resultDisplay"))
    if display_text is None and status is None:
        return []
    return [
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            tool_id=tool_id,
            text=display_text or f"[{status or 'error'}]",
            metadata=metadata or None,
            is_error=status_is_error,
        )
    ]


def _tool_metadata(record: JSONDocument) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("status", "timestamp", "description", "displayName", "renderOutputAsMarkdown"):
        value = record.get(key)
        if isinstance(value, (str, bool)):
            metadata[key] = value
    return metadata


def _status_is_error(status: str | None) -> bool | None:
    if status is None:
        return None
    normalized = status.strip().lower()
    if normalized in {"success", "succeeded", "ok", "completed"}:
        return False
    if any(marker in normalized for marker in ("error", "fail", "timeout", "cancel", "blocked")):
        return True
    return None


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
