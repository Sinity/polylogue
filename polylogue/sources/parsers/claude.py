"""Claude parser using typed Pydantic models.

Uses ClaudeCodeRecord from polylogue.sources.providers.claude_code for type-safe parsing
with automatic validation and normalized property access.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from polylogue.lib.branch_type import BranchType
from polylogue.logging import get_logger
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.types import Provider

from .base import (
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    content_blocks_from_segments,
)
from .claude_ai_parser import looks_like_ai as _looks_like_ai
from .claude_ai_parser import parse_ai as _parse_ai
from .claude_common import extract_message_text as _extract_message_text
from .claude_common import (
    extract_messages_from_chat_messages as _extract_messages_from_chat_messages,
)
from .claude_common import extract_text_from_segments as _extract_text_from_segments
from .claude_common import normalize_timestamp
from .claude_index import (
    SessionIndexEntry,
)
from .claude_index import (
    enrich_conversation_from_index as _enrich_conversation_from_index,
)
from .claude_index import (
    find_sessions_index as _find_sessions_index,
)
from .claude_index import (
    parse_sessions_index as _parse_sessions_index,
)

logger = get_logger(__name__)


def looks_like_ai(payload: object) -> bool:
    return _looks_like_ai(payload)


def parse_sessions_index(index_path):
    return _parse_sessions_index(index_path)


def find_sessions_index(session_path):
    return _find_sessions_index(session_path)


def enrich_conversation_from_index(conv: ParsedConversation, index_entry: SessionIndexEntry) -> ParsedConversation:
    return _enrich_conversation_from_index(conv, index_entry)


def extract_text_from_segments(segments: list[object]) -> str | None:
    return _extract_text_from_segments(segments)


def extract_messages_from_chat_messages(chat_messages: list[object]):
    return _extract_messages_from_chat_messages(chat_messages)


def looks_like_code(payload: list[object]) -> bool:
    if not isinstance(payload, list):
        return False
    # Known Claude Code record types that don't carry chat-structural keys
    _code_only_types = {
        "file-history-snapshot", "queue-operation", "custom-title",
        "user", "assistant", "summary", "progress", "result",
    }
    for item in payload:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in ("parentUuid", "leafUuid", "sessionId", "session_id")):
            return True
        # Recognize metadata-only sessions by their unique type values
        item_type = item.get("type")
        if isinstance(item_type, str) and item_type in _code_only_types:
            return True
    return False


def parse_ai(payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    return _parse_ai(payload, fallback_id)


def parse_code(payload: list[object], fallback_id: str) -> ParsedConversation:
    """Parse claude-code JSONL format using typed ClaudeCodeRecord model.

    Extracts semantic data including:
    - Thinking traces (via ClaudeCodeRecord.extract_reasoning_traces())
    - Tool invocations (via ClaudeCodeRecord.extract_tool_calls())
    - Git operations (from Bash commands)
    - File changes (from Read/Write/Edit)
    - Subagent spawns (from Task tool)
    - Context compaction events

    The ClaudeCodeRecord model handles format normalization via properties:
    - role: Normalized role (user/assistant/system/unknown)
    - text_content: Extracted plain text from message
    - content_blocks_raw: Raw content blocks for semantic extraction
    """
    messages: list[ParsedMessage] = []
    timestamps: list[str] = []
    session_id: str | None = None
    context_compactions: list[dict[str, Any]] = []
    # Deferred import avoids circular dependency via pipeline/__init__.py
    from polylogue.pipeline.semantic import detect_context_compaction  # noqa: PLC0415

    # Conversation-level stats accumulated from raw items (not from message provider_meta)
    total_cost: float = 0.0
    total_duration: int = 0
    saw_cost_field = False
    saw_duration_field = False
    has_sidechain: bool = False
    cwds: set[str] = set()
    models: set[str] = set()

    def _safe_float(val: object) -> float:
        try:
            return float(str(val))
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(val: object) -> int:
        try:
            return int(float(str(val)))
        except (ValueError, TypeError):
            return 0

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        # Detect context compaction events first (before validation)
        compaction = detect_context_compaction(item)
        if compaction:
            context_compactions.append(compaction)
            continue

        # Parse using typed model
        try:
            record = ClaudeCodeRecord.model_validate(item)
        except ValidationError as exc:
            logger.debug("Skipping invalid record at index %d: %s", idx, exc)
            continue

        # Skip non-message record types (init, metadata snapshots, ops)
        if record.type in {"init", "file-history-snapshot", "queue-operation"}:
            continue

        # Extract session ID for conversation grouping
        if not session_id:
            session_id = record.sessionId

        # Get message UUID and role from typed model
        msg_id = str(record.uuid or f"msg-{idx}")
        role = record.role  # Uses typed property: user/assistant/system/unknown

        # Get timestamp
        timestamp = normalize_timestamp(record.timestamp)
        if timestamp:
            timestamps.append(timestamp)

        # Extract text using typed property
        text = record.text_content or _extract_message_text(
            record.message.get("content") if isinstance(record.message, dict) else None
        )

        # Build content blocks from the raw message content
        if isinstance(record.message, dict):
            raw_msg_content = record.message.get("content")
        else:
            # ClaudeCodeMessageContent typed model — access .content directly
            raw_msg_content = getattr(record.message, "content", None)
        content_blocks = content_blocks_from_segments(raw_msg_content) if raw_msg_content else []
        if not content_blocks and text:
            content_blocks = [ParsedContentBlock(type="text", text=text)]

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text or "",
                timestamp=timestamp,
                content_blocks=content_blocks,
                parent_message_provider_id=record.parentUuid,
            )
        )

        # Accumulate conversation-level stats from the raw item
        if "costUSD" in item:
            saw_cost_field = True
            cost_val = item.get("costUSD")
            total_cost += _safe_float(cost_val)
        if "durationMs" in item:
            saw_duration_field = True
            dur_val = item.get("durationMs")
            total_duration += _safe_int(dur_val)
        if item.get("isSidechain"):
            has_sidechain = True
        cwd_val = item.get("cwd")
        if isinstance(cwd_val, str):
            cwds.add(cwd_val)
        msg_content = record.message if isinstance(record.message, dict) else {}
        model_val = msg_content.get("model")
        if isinstance(model_val, str):
            models.add(model_val)

    # Derive conversation timestamps from messages
    created_at = min(timestamps) if timestamps else None
    updated_at = max(timestamps) if timestamps else None

    # Use session_id as conversation ID if available.
    # Subagent files (agent-<hash>.jsonl) share the parent's sessionId,
    # so we must make their conversation_id unique to prevent UPSERT collision.
    is_subagent = fallback_id.startswith("agent-")
    parent_session_id: str | None = None
    if is_subagent and session_id:
        # Subagent: use session_id:fallback_id as unique conversation ID
        conv_id = f"{session_id}:{fallback_id}"
        parent_session_id = session_id
    else:
        conv_id = session_id or fallback_id

    # Build conversation-level provider_meta
    conv_meta: dict[str, Any] = {}
    if context_compactions:
        conv_meta["context_compactions"] = context_compactions
    if saw_cost_field:
        conv_meta["total_cost_usd"] = total_cost
    if saw_duration_field:
        conv_meta["total_duration_ms"] = total_duration
    if cwds:
        conv_meta["working_directories"] = sorted(cwds)
    if models:
        conv_meta["models_used"] = sorted(models)

    # Detect branch type: subagent (from file path) or sidechain (from content)
    if is_subagent:
        branch_type: BranchType | None = BranchType.SUBAGENT
    elif has_sidechain:
        branch_type = BranchType.SIDECHAIN
    else:
        branch_type = None

    # Infer title from first user message (fallback to session ID)
    title = str(conv_id)
    for m in messages:
        if m.role == "user" and m.text and len(m.text.strip()) > 3:
            first_line = m.text.strip().split("\n")[0]
            title = first_line[:80]
            if len(first_line) > 80:
                title += "..."
            break

    return ParsedConversation(
        provider_name=Provider.CLAUDE_CODE,
        provider_conversation_id=str(conv_id),
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        provider_meta=conv_meta if conv_meta else None,
        parent_conversation_provider_id=parent_session_id,
        branch_type=branch_type,
    )


# Symmetric aliases — makes the claude module conform to the same interface
# as chatgpt.py and codex.py (parse + looks_like at module level).
# parse_code / looks_like_code remain for explicit dispatch in source.py.
parse = parse_code
looks_like = looks_like_code
