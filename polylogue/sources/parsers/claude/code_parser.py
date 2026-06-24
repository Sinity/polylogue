"""Claude Code session parsing helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TypeAlias

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, PasteBoundary, Provider
from polylogue.logging import get_logger
from polylogue.pipeline.semantic_capture import detect_context_compaction

from ..base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
    ParsedSessionEvent,
    content_blocks_from_segments,
)
from .common import (
    _message_duration_ms,
    _message_model_effort,
    _message_model_name,
    extract_message_text,
    normalize_timestamp,
    reclassify_tool_result_envelope,
)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Claude Code elides pasted content in the persisted JSONL, leaving a
# ``[Pasted text #N]`` marker (optionally ``[Pasted text #N +M lines]``) in the
# user prompt text. The live UserPromptSubmit hook captures the same paste with
# a real content hash (boundary_state=hash_only); batch re-ingest can only
# recover the marker's exact location, so it stamps the span as PROJECTED.
_PASTE_MARKER_RE = re.compile(r"\[Pasted text #(\d+)[^\]]*\]")


def _detect_paste_spans(text: str | None) -> list[ParsedPasteEvidence]:
    """Detect ``[Pasted text #N]`` markers in a user prompt as paste evidence."""
    if not text:
        return []
    return [
        ParsedPasteEvidence(
            position=int(match.group(1)),
            start_offset=match.start(),
            end_offset=match.end(),
            boundary_state=PasteBoundary.PROJECTED.value,
            source_marker=match.group(0),
        )
        for match in _PASTE_MARKER_RE.finditer(text)
    ]


ClaudeCodeContextCompaction: TypeAlias = dict[str, object]


def _clean_title_text(text: str) -> str:
    """Strip protocol artifacts from user message text for title extraction."""
    if not text:
        return ""
    cleaned = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<task-notification>.*?</task-notification>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<local-command-stdout>.*?</local-command-stdout>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<command-name>.*?</command-name>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<command-message>.*?</command-message>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<command-args>.*?</command-args>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[Request interrupted by user\]", "", cleaned)
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    # Take first line of remaining text
    first_line = cleaned.split("\n")[0].strip()
    return first_line


logger = get_logger(__name__)


def _safe_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _content_blocks_from_record(message: object, text: str | None) -> list[ParsedContentBlock]:
    raw_msg_content = message.get("content") if isinstance(message, dict) else None
    content_blocks = content_blocks_from_segments(raw_msg_content) if raw_msg_content else []
    if not content_blocks and text:
        return [ParsedContentBlock(type=BlockType.TEXT, text=text)]
    return content_blocks


def _message_type_from_code_record(item: dict[str, object], text: str | None) -> MessageType:
    artifact_type = classify_text_message_type(text)
    if artifact_type is not None:
        return artifact_type
    if item.get("isMeta"):
        return MessageType.CONTEXT
    return MessageType.MESSAGE


def _message_usage_event_payload(
    usage: dict[object, object],
    *,
    model_name: str | None,
    model_effort: str | None,
) -> dict[str, object]:
    last_usage: dict[str, int] = {
        "input_tokens": _safe_int(usage.get("input_tokens")),
        "output_tokens": _safe_int(usage.get("output_tokens")),
        "cached_input_tokens": _safe_int(usage.get("cache_read_input_tokens")),
        "cache_write_tokens": _safe_int(usage.get("cache_creation_input_tokens")),
    }
    total_tokens = _safe_int(usage.get("total_tokens"))
    if total_tokens:
        last_usage["total_tokens"] = total_tokens
    payload: dict[str, object] = {
        "type": "message_usage",
        "semantics": "per_message",
        "last_token_usage": last_usage,
    }
    if model_name:
        payload["model"] = model_name
    if model_effort:
        payload["model_effort"] = model_effort
    return payload


def _record_role(item: dict[str, object], message: object) -> Role:
    if isinstance(message, dict):
        message_role = message.get("role")
        if isinstance(message_role, str) and message_role:
            normalized = Role.normalize(message_role)
            if normalized is not Role.UNKNOWN:
                return normalized

    record_role = item.get("role")
    if isinstance(record_role, str) and record_role:
        normalized = Role.normalize(record_role)
        if normalized is not Role.UNKNOWN:
            return normalized

    record_type = item.get("type")
    if record_type == "user":
        return Role.USER
    if record_type == "assistant":
        return Role.ASSISTANT
    if record_type in {"summary", "system", "file-history-snapshot", "queue-operation"}:
        return Role.SYSTEM
    if record_type in {"progress", "result"}:
        return Role.TOOL
    return Role.UNKNOWN


def _string_field(item: dict[str, object], key: str) -> str | None:
    value = item.get(key)
    return value if isinstance(value, str) and value else None


def _parse_code_records(records: Iterable[object], fallback_id: str) -> ParsedSession:
    """Parse Claude Code JSONL payloads into a canonical session model."""
    messages: list[ParsedMessage] = []
    created_at: str | None = None
    updated_at: str | None = None
    seen_record_uuids: set[str] = set()
    duplicate_uuid_count = 0
    first_duplicate_uuid: str | None = None
    first_duplicate_index: int | None = None
    session_id: str | None = None
    session_events: list[ParsedSessionEvent] = []
    total_cost = 0.0
    total_duration = 0
    saw_cost_field = False
    saw_duration_field = False
    has_sidechain = False
    cwds: set[str] = set()
    models: set[str] = set()
    message_position = 0

    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            continue

        compaction = detect_context_compaction(item)
        if compaction:
            raw_timestamp = compaction.get("timestamp")
            compaction_timestamp = normalize_timestamp(
                raw_timestamp if isinstance(raw_timestamp, str | int | float) else None
            )
            context_compaction = dict(compaction)
            session_events.append(
                ParsedSessionEvent(
                    event_type="compaction",
                    timestamp=compaction_timestamp,
                    payload=context_compaction,
                )
            )
            summary_text = str(context_compaction.get("summary") or "")
            messages.append(
                ParsedMessage(
                    provider_message_id=str(item.get("uuid") or f"summary-{index}"),
                    role=Role.SYSTEM,
                    text=summary_text,
                    timestamp=compaction_timestamp,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)] if summary_text else [],
                    message_type=MessageType.SUMMARY,
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                )
            )
            message_position += 1
            continue

        record_type = item.get("type")
        if not isinstance(record_type, str):
            logger.debug("Skipping invalid record at index %d: missing type", index)
            continue

        record_uuid = _string_field(item, "uuid")
        if record_uuid:
            if record_uuid in seen_record_uuids:
                duplicate_uuid_count += 1
                first_duplicate_uuid = first_duplicate_uuid or record_uuid
                first_duplicate_index = first_duplicate_index or index
                continue
            seen_record_uuids.add(record_uuid)

        # ``progress`` records are claude-code hook lifecycle events
        # (`hookEvent`, `hookName`, `command`) carried alongside the
        # tool they fired on — they are NOT message content. Persisting
        # them as messages produces empty rows under the ``tool_result``
        # message_type that dominate the ``role=unknown, text='', blocks=[]``
        # consumer surface and inflate every messages-table count by
        # ~23%. See #1617 for the full forensic. We drop them here at the
        # parser; the hook payload, if useful for analytics, belongs in
        # a future ``session_event`` capture, not in the messages table.
        if record_type in {"init", "file-history-snapshot", "queue-operation", "progress"}:
            continue

        if not session_id:
            session_id = _string_field(item, "sessionId")

        raw_timestamp = item.get("timestamp")
        timestamp = normalize_timestamp(raw_timestamp if isinstance(raw_timestamp, str | int | float) else None)
        if timestamp:
            created_at = timestamp if created_at is None or timestamp < created_at else created_at
            updated_at = timestamp if updated_at is None or timestamp > updated_at else updated_at

        message = item.get("message")
        raw_content = message.get("content") if isinstance(message, dict) else item.get("content")
        text = extract_message_text(raw_content)
        envelope_role = _record_role(item, message)
        content_blocks = _content_blocks_from_record(message, text)
        message_type = _message_type_from_code_record(item, text)
        # Claude Code records carry per-message token usage at
        # ``record.message.usage``; propagate so MaterializedMessage and the
        # downstream cost estimator see real numbers instead of zeros.
        msg_usage = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(msg_usage, dict):
            msg_usage = {}
        message_payload = message if isinstance(message, dict) else {}
        msg_model = _message_model_name(message_payload) or _message_model_name(item)
        msg_effort = _message_model_effort(message_payload) or _message_model_effort(item)
        msg_duration_ms = _message_duration_ms(item)
        resolved_role = reclassify_tool_result_envelope(envelope_role, content_blocks)
        material_origin = classify_material_origin(
            role=resolved_role,
            message_type=message_type,
            text=text,
            block_types=tuple(block.type for block in content_blocks),
        )
        # Paste markers only appear in user prompts; restricting detection to the
        # user role avoids false positives from assistant text that quotes a marker.
        paste_spans = _detect_paste_spans(text) if resolved_role == Role.USER else []
        provider_message_id = str(record_uuid or f"msg-{index}")
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=resolved_role,
                text=text or "",
                timestamp=timestamp,
                blocks=content_blocks,
                message_type=message_type,
                material_origin=material_origin,
                parent_message_provider_id=_string_field(item, "parentUuid"),
                position=message_position,
                variant_index=0,
                is_active_path=True,
                input_tokens=_safe_int(msg_usage.get("input_tokens")),
                output_tokens=_safe_int(msg_usage.get("output_tokens")),
                cache_read_tokens=_safe_int(msg_usage.get("cache_read_input_tokens")),
                cache_write_tokens=_safe_int(msg_usage.get("cache_creation_input_tokens")),
                model_name=msg_model,
                model_effort=msg_effort,
                duration_ms=msg_duration_ms,
                paste_spans=paste_spans,
            )
        )
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            session_events.append(
                ParsedSessionEvent(
                    event_type="message_usage",
                    timestamp=timestamp,
                    source_message_provider_id=provider_message_id,
                    payload=_message_usage_event_payload(
                        msg_usage,
                        model_name=msg_model,
                        model_effort=msg_effort,
                    ),
                )
            )
        message_position += 1

        if "costUSD" in item:
            saw_cost_field = True
            total_cost += _safe_float(item.get("costUSD"))
        if "durationMs" in item:
            saw_duration_field = True
            total_duration += _safe_int(item.get("durationMs"))
        if item.get("isSidechain"):
            has_sidechain = True
        cwd = item.get("cwd")
        if isinstance(cwd, str):
            cwds.add(cwd)
        model_name = message_payload.get("model")
        if isinstance(model_name, str):
            models.add(model_name)

    if duplicate_uuid_count:
        logger.debug(
            "Skipped repeated Claude Code record uuids: count=%d first_index=%s first_uuid=%s",
            duplicate_uuid_count,
            first_duplicate_index,
            first_duplicate_uuid,
        )

    is_subagent = fallback_id.startswith("agent-")
    parent_session_id: str | None = None
    if is_subagent and session_id:
        composed_session_id = f"{session_id}:{fallback_id}"
        parent_session_id = session_id
    else:
        composed_session_id = session_id or fallback_id

    if is_subagent:
        branch_type: BranchType | None = BranchType.SUBAGENT
    elif has_sidechain:
        branch_type = BranchType.SIDECHAIN
    else:
        branch_type = None

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]

    title = str(composed_session_id)
    for message in messages:
        if message.material_origin is MaterialOrigin.HUMAN_AUTHORED and message.text and len(message.text.strip()) > 3:
            # Strip protocol artifacts before extracting title
            cleaned = _clean_title_text(message.text)
            if cleaned and len(cleaned) > 3:
                title = cleaned[:80]
                if len(cleaned) > 80:
                    title += "..."
                break

    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=str(composed_session_id),
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        session_events=session_events,
        parent_session_provider_id=parent_session_id,
        branch_type=branch_type,
        reported_cost_usd=total_cost if saw_cost_field else None,
        reported_duration_ms=total_duration if saw_duration_field else None,
        models_used=sorted(models),
        working_directories=sorted(cwds),
    )


def parse_code(payload: Sequence[object], fallback_id: str) -> ParsedSession:
    return _parse_code_records(payload, fallback_id)


def parse_code_stream(records: Iterable[object], fallback_id: str) -> ParsedSession:
    return _parse_code_records(records, fallback_id)


__all__ = ["parse_code", "parse_code_stream"]
