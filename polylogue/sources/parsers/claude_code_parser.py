"""Claude Code conversation parsing helpers."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from polylogue.lib.branch_type import BranchType
from polylogue.logging import get_logger
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.types import Provider

from .base import ParsedContentBlock, ParsedConversation, ParsedMessage, content_blocks_from_segments
from .claude_common import extract_message_text, normalize_timestamp

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


def _content_blocks_from_record(record: ClaudeCodeRecord, text: str | None) -> list[ParsedContentBlock]:
    raw_msg_content = (
        record.message.get("content")
        if isinstance(record.message, dict)
        else getattr(record.message, "content", None)
    )
    content_blocks = content_blocks_from_segments(raw_msg_content) if raw_msg_content else []
    if not content_blocks and text:
        return [ParsedContentBlock(type="text", text=text)]
    return content_blocks


def parse_code(payload: list[object], fallback_id: str) -> ParsedConversation:
    """Parse Claude Code JSONL payloads into a canonical conversation model."""
    messages: list[ParsedMessage] = []
    timestamps: list[str] = []
    session_id: str | None = None
    context_compactions: list[dict[str, Any]] = []
    total_cost = 0.0
    total_duration = 0
    saw_cost_field = False
    saw_duration_field = False
    has_sidechain = False
    cwds: set[str] = set()
    models: set[str] = set()

    # Deferred import avoids circular dependency via pipeline/__init__.py.
    from polylogue.pipeline.semantic_capture import detect_context_compaction  # noqa: PLC0415

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        compaction = detect_context_compaction(item)
        if compaction:
            context_compactions.append(compaction)
            continue

        try:
            record = ClaudeCodeRecord.model_validate(item)
        except ValidationError as exc:
            logger.debug("Skipping invalid record at index %d: %s", index, exc)
            continue

        if record.type in {"init", "file-history-snapshot", "queue-operation"}:
            continue

        if not session_id:
            session_id = record.sessionId

        timestamp = normalize_timestamp(record.timestamp)
        if timestamp:
            timestamps.append(timestamp)

        text = record.text_content or extract_message_text(
            record.message.get("content") if isinstance(record.message, dict) else None
        )
        messages.append(
            ParsedMessage(
                provider_message_id=str(record.uuid or f"msg-{index}"),
                role=record.role,
                text=text or "",
                timestamp=timestamp,
                content_blocks=_content_blocks_from_record(record, text),
                parent_message_provider_id=record.parentUuid,
            )
        )

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
        message_payload = record.message if isinstance(record.message, dict) else {}
        model_name = message_payload.get("model")
        if isinstance(model_name, str):
            models.add(model_name)

    created_at = min(timestamps) if timestamps else None
    updated_at = max(timestamps) if timestamps else None

    is_subagent = fallback_id.startswith("agent-")
    parent_session_id: str | None = None
    if is_subagent and session_id:
        conversation_id = f"{session_id}:{fallback_id}"
        parent_session_id = session_id
    else:
        conversation_id = session_id or fallback_id

    provider_meta: dict[str, Any] = {}
    if context_compactions:
        provider_meta["context_compactions"] = context_compactions
    if saw_cost_field:
        provider_meta["total_cost_usd"] = total_cost
    if saw_duration_field:
        provider_meta["total_duration_ms"] = total_duration
    if cwds:
        provider_meta["working_directories"] = sorted(cwds)
    if models:
        provider_meta["models_used"] = sorted(models)

    if is_subagent:
        branch_type: BranchType | None = BranchType.SUBAGENT
    elif has_sidechain:
        branch_type = BranchType.SIDECHAIN
    else:
        branch_type = None

    title = str(conversation_id)
    for message in messages:
        if message.role == "user" and message.text and len(message.text.strip()) > 3:
            first_line = message.text.strip().split("\n")[0]
            title = first_line[:80]
            if len(first_line) > 80:
                title += "..."
            break

    return ParsedConversation(
        provider_name=Provider.CLAUDE_CODE,
        provider_conversation_id=str(conversation_id),
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        provider_meta=provider_meta if provider_meta else None,
        parent_conversation_provider_id=parent_session_id,
        branch_type=branch_type,
    )


__all__ = ["parse_code"]
