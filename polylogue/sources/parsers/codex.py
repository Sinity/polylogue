"""Codex JSONL session parser."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime

from pydantic import ValidationError

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.message.artifacts import classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.provider.semantics import extract_codex_text
from polylogue.core.timestamps import format_timestamp, parse_timestamp
from polylogue.logging import get_logger
from polylogue.sources.providers.codex import CodexRecord
from polylogue.types import ContentBlockType, Provider

from .base import (
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    ParsedProviderEvent,
    content_blocks_from_segments,
)

logger = get_logger(__name__)


def _normalize_timestamp(value: str | int | float | None) -> str | None:
    if isinstance(value, str):
        return value if parse_timestamp(value) is not None else None
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return format_timestamp(parsed)


def _latest_timestamp(*values: str | None) -> str | None:
    candidates: list[tuple[datetime, str]] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        parsed = parse_timestamp(value)
        if parsed is None:
            continue
        candidates.append((parsed, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _validate_record(item: object, *, index: int, context: str = "record") -> CodexRecord | None:
    if not isinstance(item, dict):
        return None
    try:
        return CodexRecord.model_validate(item)
    except ValidationError as exc:
        logger.debug("Skipping invalid %s at index %d: %s", context, index, exc)
        return None


def _dict_record(item: object) -> dict[str, object] | None:
    return item if isinstance(item, dict) else None


def _is_plausibly_codex_record(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("record_type") == "state":
        return True

    record_type = item.get("type")
    payload = item.get("payload")
    if record_type in {"session_meta", "response_item", "compacted", "turn_context"}:
        return isinstance(payload, dict)
    if isinstance(payload, dict):
        return True

    role = item.get("role")
    content = item.get("content")
    if record_type == "message" or isinstance(role, str):
        return "content" not in item or isinstance(content, list)

    return bool(item.get("id") and item.get("timestamp") and "message" not in item)


def _payload_record(record: dict[str, object]) -> dict[str, object] | None:
    return _dict_record(record.get("payload"))


def _record_type(record: dict[str, object]) -> str | None:
    value = record.get("type")
    return value if isinstance(value, str) else None


def _record_id(record: dict[str, object]) -> str | None:
    value = record.get("id")
    return value if isinstance(value, str) else None


def _record_timestamp(record: dict[str, object]) -> str | int | float | None:
    value = record.get("timestamp")
    return value if isinstance(value, (str, int, float)) else None


def _record_instructions(record: dict[str, object]) -> str | None:
    value = record.get("instructions")
    return value if isinstance(value, str) else None


def _session_meta_record(record: dict[str, object]) -> dict[str, object] | None:
    if _record_type(record) == "session_meta":
        return _payload_record(record)
    if _record_id(record) and _record_timestamp(record) and not _record_type(record):
        return record
    return None


def _is_envelope(record: dict[str, object]) -> bool:
    return isinstance(record.get("payload"), dict)


def _is_state(record: dict[str, object]) -> bool:
    return record.get("record_type") == "state"


def _is_direct_message(record: dict[str, object]) -> bool:
    return _record_type(record) == "message" or isinstance(record.get("role"), str)


def _is_message(record: dict[str, object]) -> bool:
    if _is_envelope(record):
        return _record_type(record) == "response_item"
    return _is_direct_message(record)


def _message_record(record: dict[str, object]) -> dict[str, object] | None:
    if _is_state(record):
        return None
    if _record_type(record) == "response_item":
        inner = _payload_record(record)
        return inner if inner is not None and _is_message(inner) else None
    return record if _is_message(record) else None


def _git_context(record: dict[str, object]) -> dict[str, object] | None:
    git = _dict_record(record.get("git"))
    if git is None:
        return None
    payload = {str(key): value for key, value in git.items() if value is not None}
    return payload or None


def _record_payload(record: dict[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in record.items() if value is not None}


def _extract_cwd(payload: dict[str, object] | None) -> str | None:
    if not payload:
        return None
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return cwd.strip()
    turn_context = payload.get("turn_context")
    if isinstance(turn_context, dict):
        nested = turn_context.get("cwd")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _tool_input_from_arguments(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"arguments": value}
        return dict(parsed) if isinstance(parsed, dict) else {"arguments": value}
    return {}


def _codex_tool_message(record: dict[str, object], *, index: int) -> ParsedMessage | None:
    payload = _record_payload(record)
    record_type = _record_type(record)
    timestamp = _normalize_timestamp(_record_timestamp(record))
    if record_type == "function_call":
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return None
        tool_id = payload.get("call_id") or payload.get("id")
        return ParsedMessage(
            provider_message_id=str(payload.get("id") or tool_id or f"function-call-{index}"),
            role=Role.ASSISTANT,
            text=tool_name,
            timestamp=timestamp,
            content_blocks=[
                ParsedContentBlock(
                    type=ContentBlockType.TOOL_USE,
                    tool_name=tool_name,
                    tool_id=str(tool_id) if tool_id else None,
                    tool_input=_tool_input_from_arguments(payload.get("arguments")),
                )
            ],
        )
    if record_type == "function_call_output":
        tool_id = payload.get("call_id") or payload.get("id")
        output = payload.get("output")
        output_text = output if isinstance(output, str) else json.dumps(output, sort_keys=True) if output else None
        if not tool_id and not output_text:
            return None
        return ParsedMessage(
            provider_message_id=str(payload.get("id") or tool_id or f"function-call-output-{index}"),
            role=Role.TOOL,
            text=output_text,
            timestamp=timestamp,
            content_blocks=[
                ParsedContentBlock(
                    type=ContentBlockType.TOOL_RESULT,
                    tool_id=str(tool_id) if tool_id else None,
                    text=output_text,
                )
            ],
        )
    return None


def _effective_role(record: dict[str, object]) -> str:
    payload = _payload_record(record)
    if payload is not None:
        value = payload.get("role")
        return value if isinstance(value, str) else "unknown"
    value = record.get("role")
    return value if isinstance(value, str) else "unknown"


def _effective_content(record: dict[str, object]) -> list[object]:
    payload = _payload_record(record)
    value = payload.get("content") if payload is not None else record.get("content")
    return value if isinstance(value, list) else []


def _message_type_from_codex_message(record: dict[str, object], text: str | None) -> MessageType:
    role = _effective_role(record).strip().lower()
    if role in {"system", "developer"}:
        return MessageType.CONTEXT
    artifact_type = classify_text_message_type(text)
    return artifact_type or MessageType.MESSAGE


def looks_like(payload: Sequence[object]) -> bool:
    """Detect Codex JSONL format using typed validation.

    Newest format (envelope with typed payloads):
        {"type":"session_meta","payload":{"id":"...","timestamp":"...","git":{...}}}
        {"type":"response_item","payload":{"type":"message","role":"user","content":[...]}}

    Intermediate format (JSONL with session metadata + messages):
        {"id":"...","timestamp":"...","git":{...}}
        {"record_type":"state"}
        {"type":"message","role":"user","content":[...]}
    """
    if not isinstance(payload, list):
        return False

    for idx, item in enumerate(payload, start=1):
        if not _is_plausibly_codex_record(item):
            continue
        record = _validate_record(item, index=idx)
        if record is None:
            continue
        if record.format_type in ("envelope", "direct", "state"):
            return True
        if record.id and record.timestamp:
            return True

    return False


def _parse_records(records: Iterable[object], fallback_id: str) -> ParsedConversation:
    """Parse Codex JSONL session file using typed CodexRecord model.

    Supports two format generations via CodexRecord.format_type:
    - "envelope": {"type":"session_meta"|"response_item", "payload":{...}}
    - "direct": {"type":"message", "role":"...", "content":[...]}
    - "state": {"record_type":"state"} (skip markers)

    The CodexRecord model handles format normalization via properties:
    - effective_role: Normalized role from any format
    - text_content: Extracted text from any format
    - format_type: Detected format generation
    """
    messages: list[ParsedMessage] = []
    provider_events: list[ParsedProviderEvent] = []
    session_id = fallback_id
    session_timestamp: str | None = None
    latest_message_timestamp: str | None = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata
    working_directories: set[str] = set()

    for idx, item in enumerate(records, start=1):
        record = _dict_record(item)
        if record is None:
            continue

        # Handle compaction events (before message check so they don't fall through)
        if _record_type(record) == "compacted":
            timestamp = _normalize_timestamp(_record_timestamp(record))
            payload = _payload_record(record) or {}
            history = payload.get("replacement_history")
            event_payload: dict[str, object] = {
                "summary": str(payload.get("message", "") or ""),
                "has_replacement_history": isinstance(history, list) and bool(history),
            }
            if payload:
                event_payload["raw"] = payload
            provider_events.append(
                ParsedProviderEvent(
                    event_type="compaction",
                    timestamp=timestamp,
                    payload=event_payload,
                )
            )
            continue

        # Handle turn-context events
        if _record_type(record) == "turn_context":
            timestamp = _normalize_timestamp(_record_timestamp(record))
            tc_payload: dict[str, object] = {}
            turn_payload = _payload_record(record)
            if turn_payload:
                tc_payload["raw"] = turn_payload
                cwd = _extract_cwd(turn_payload)
                if cwd:
                    tc_payload["cwd"] = cwd
                    working_directories.add(cwd)
            provider_events.append(
                ParsedProviderEvent(
                    event_type="turn_context",
                    timestamp=timestamp,
                    payload=tc_payload,
                )
            )
            continue

        if _record_type(record) == "response_item":
            inner = _payload_record(record)
            if inner is not None and not _is_message(inner):
                event_payload = _record_payload(inner)
                provider_events.append(
                    ParsedProviderEvent(
                        event_type=_record_type(inner) or "response_item",
                        timestamp=_normalize_timestamp(_record_timestamp(inner) or _record_timestamp(record)),
                        payload={"raw": event_payload},
                    )
                )
                tool_message = _codex_tool_message(inner, index=idx)
                if tool_message is not None:
                    messages.append(tool_message)
                    latest_message_timestamp = _latest_timestamp(latest_message_timestamp, tool_message.timestamp)
                cwd = _extract_cwd(event_payload)
                if cwd:
                    working_directories.add(cwd)
                continue

        session_meta = _session_meta_record(record)
        if session_meta is not None:
            meta_id = _record_id(session_meta)
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                if len(session_metas_seen) == 1:
                    session_id = meta_id
                    session_timestamp = _normalize_timestamp(_record_timestamp(session_meta))
            git_context = _git_context(session_meta)
            if git_context and not session_git:
                session_git = git_context
            instructions = _record_instructions(session_meta)
            if instructions and not session_instructions:
                session_instructions = instructions
            continue

        message_record = _message_record(record)
        if message_record is not None:
            raw_role = _effective_role(message_record)
            content = _effective_content(message_record)
            text = extract_codex_text(content)
            timestamp = _normalize_timestamp(_record_timestamp(message_record))

            if not raw_role or raw_role == "unknown" or not text:
                continue
            role = Role.normalize(raw_role)

            msg_id = _record_id(message_record) or f"msg-{idx}"
            content_blocks = content_blocks_from_segments(content)
            if not content_blocks and text:
                from .base import ParsedContentBlock

                content_blocks = [ParsedContentBlock(type=ContentBlockType.TEXT, text=text)]

            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    content_blocks=content_blocks,
                    message_type=_message_type_from_codex_message(message_record, text),
                )
            )
            latest_message_timestamp = _latest_timestamp(latest_message_timestamp, timestamp)

    # Second session_meta ID (if present) is the parent session
    parent_id = session_metas_seen[1] if len(session_metas_seen) > 1 else None
    branch_type = BranchType.CONTINUATION if parent_id else None

    # Build conversation-level provider_meta with session context
    conv_meta: dict[str, object] | None = None
    if session_git or session_instructions or working_directories:
        conv_meta = {}
        if session_git:
            conv_meta["git"] = session_git
        if session_instructions:
            conv_meta["instructions"] = session_instructions
        if working_directories:
            conv_meta["working_directories"] = sorted(working_directories)

    return ParsedConversation(
        provider_name=Provider.CODEX,
        provider_conversation_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=_latest_timestamp(latest_message_timestamp, session_timestamp),
        messages=messages,
        provider_meta=conv_meta,
        provider_events=provider_events,
        parent_conversation_provider_id=parent_id,
        branch_type=branch_type,
    )


def parse(payload: Sequence[object], fallback_id: str) -> ParsedConversation:
    return _parse_records(payload, fallback_id)


def parse_stream(records: Iterable[object], fallback_id: str) -> ParsedConversation:
    return _parse_records(records, fallback_id)
