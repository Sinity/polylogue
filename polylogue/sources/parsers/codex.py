"""Codex JSONL session parser using typed Pydantic models.

Uses CodexRecord from polylogue.sources.providers.codex for type-safe parsing
with automatic validation and normalization.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime

from pydantic import ValidationError

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.core.timestamps import format_timestamp, parse_timestamp
from polylogue.lib.roles import Role
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


def _payload_record(record: CodexRecord, *, index: int, context: str) -> CodexRecord | None:
    payload = record.payload
    if not isinstance(payload, dict):
        return None
    return _validate_record(payload, index=index, context=context)


def _session_meta_record(record: CodexRecord, *, index: int) -> CodexRecord | None:
    if record.type == "session_meta":
        return _payload_record(record, index=index, context="session_meta payload")
    if record.id and record.timestamp and not record.type:
        return record
    return None


def _message_record(record: CodexRecord, *, index: int) -> CodexRecord | None:
    if record.format_type == "state":
        return None
    if record.type == "response_item":
        inner = _payload_record(record, index=index, context="response_item payload")
        return inner if inner is not None and inner.is_message else None
    return record if record.is_message else None


def _git_context(record: CodexRecord) -> dict[str, object] | None:
    if record.git is None:
        return None
    return record.git.model_dump(exclude_none=True) or None


def _record_payload(record: CodexRecord) -> dict[str, object]:
    return record.model_dump(mode="json", exclude_none=True)


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


def _codex_tool_message(record: CodexRecord, *, index: int) -> ParsedMessage | None:
    payload = _record_payload(record)
    record_type = record.type
    timestamp = _normalize_timestamp(record.timestamp)
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
    context_compactions: list[dict[str, object]] = []
    session_id = fallback_id
    session_timestamp: str | None = None
    latest_message_timestamp: str | None = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata
    working_directories: set[str] = set()

    for idx, item in enumerate(records, start=1):
        record = _validate_record(item, index=idx)
        if record is None:
            continue

        # Handle compaction events (before message check so they don't fall through)
        if record.is_compaction:
            timestamp = _normalize_timestamp(record.timestamp)
            event_payload: dict[str, object] = {
                "summary": record.compacted_message,
                "has_replacement_history": record.has_replacement_history,
            }
            if record.payload:
                event_payload["raw"] = record.payload
            provider_events.append(
                ParsedProviderEvent(
                    event_type="compaction",
                    timestamp=timestamp,
                    payload=event_payload,
                )
            )
            context_compactions.append(event_payload)
            continue

        # Handle turn-context events
        if record.is_turn_context:
            timestamp = _normalize_timestamp(record.timestamp)
            tc_payload: dict[str, object] = {}
            if record.payload:
                tc_payload["raw"] = record.payload
                cwd = _extract_cwd(record.payload)
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

        if record.type == "response_item":
            inner = _payload_record(record, index=idx, context="response_item payload")
            if inner is not None and not inner.is_message:
                event_payload = _record_payload(inner)
                provider_events.append(
                    ParsedProviderEvent(
                        event_type=inner.type or "response_item",
                        timestamp=_normalize_timestamp(inner.timestamp or record.timestamp),
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

        session_meta = _session_meta_record(record, index=idx)
        if session_meta is not None:
            meta_id = session_meta.id
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                if len(session_metas_seen) == 1:
                    session_id = meta_id
                    session_timestamp = _normalize_timestamp(session_meta.timestamp)
            git_context = _git_context(session_meta)
            if git_context and not session_git:
                session_git = git_context
            if session_meta.instructions and not session_instructions:
                session_instructions = session_meta.instructions
            continue

        message_record = _message_record(record, index=idx)
        if message_record is not None:
            raw_role = message_record.effective_role
            text = message_record.text_content
            timestamp = _normalize_timestamp(message_record.timestamp)

            if not raw_role or raw_role == "unknown" or not text:
                continue
            role = Role.normalize(raw_role)

            msg_id = message_record.id or f"msg-{idx}"
            content_blocks = content_blocks_from_segments(message_record.effective_content)
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
                )
            )
            latest_message_timestamp = _latest_timestamp(latest_message_timestamp, timestamp)

    # Second session_meta ID (if present) is the parent session
    parent_id = session_metas_seen[1] if len(session_metas_seen) > 1 else None
    branch_type = BranchType.CONTINUATION if parent_id else None

    # Build conversation-level provider_meta with session context
    conv_meta: dict[str, object] | None = None
    if session_git or session_instructions or context_compactions or working_directories:
        conv_meta = {}
        if session_git:
            conv_meta["git"] = session_git
        if session_instructions:
            conv_meta["instructions"] = session_instructions
        if context_compactions:
            conv_meta["context_compactions"] = context_compactions
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
