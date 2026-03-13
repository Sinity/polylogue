"""Claude parser using typed Pydantic models.

Uses ClaudeCodeRecord from polylogue.sources.providers.claude_code for type-safe parsing
with automatic validation and normalized property access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from polylogue.lib.branch_type import BranchType
from polylogue.logging import get_logger
from polylogue.lib.roles import Role
from polylogue.sources.providers.claude_ai import ClaudeAIConversation
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.types import Provider

from .base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    attachment_from_meta,
    content_blocks_from_segments,
)

logger = get_logger(__name__)


@dataclass
class SessionIndexEntry:
    """Parsed entry from Claude Code sessions-index.json."""

    session_id: str
    full_path: str
    first_prompt: str | None
    summary: str | None
    message_count: int
    created: str | None
    modified: str | None
    git_branch: str | None
    project_path: str | None
    is_sidechain: bool
    file_mtime: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionIndexEntry:
        return cls(
            session_id=data.get("sessionId", ""),
            full_path=data.get("fullPath", ""),
            first_prompt=data.get("firstPrompt"),
            summary=data.get("summary"),
            message_count=data.get("messageCount", 0),
            created=data.get("created"),
            modified=data.get("modified"),
            git_branch=data.get("gitBranch"),
            project_path=data.get("projectPath"),
            is_sidechain=data.get("isSidechain", False),
            file_mtime=data.get("fileMtime"),
        )


def parse_sessions_index(index_path: Path) -> dict[str, SessionIndexEntry]:
    """Parse Claude Code sessions-index.json file.

    Args:
        index_path: Path to sessions-index.json

    Returns:
        Dict mapping session_id to SessionIndexEntry
    """
    if not index_path.exists():
        return {}

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        return {
            entry["sessionId"]: SessionIndexEntry.from_dict(entry)
            for entry in entries
            if isinstance(entry, dict) and "sessionId" in entry
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # Log but don't fail - fall back to parsing JSONL files directly
        logger.debug("Failed to parse sessions-index.json: %s", exc)
        return {}


def find_sessions_index(session_path: Path) -> Path | None:
    """Find sessions-index.json for a Claude Code session path.

    Checks the directory containing the JSONL file for sessions-index.json.

    Args:
        session_path: Path to a session JSONL file

    Returns:
        Path to sessions-index.json if found, else None
    """
    index_path = session_path.parent / "sessions-index.json"
    return index_path if index_path.exists() else None


def enrich_conversation_from_index(
    conv: ParsedConversation,
    index_entry: SessionIndexEntry,
) -> ParsedConversation:
    """Enrich a parsed conversation with metadata from sessions-index.json.

    Args:
        conv: Parsed conversation from JSONL
        index_entry: Corresponding entry from sessions-index.json

    Returns:
        Enriched ParsedConversation with better title, summary, metadata
    """
    # Use summary or firstPrompt as title (more descriptive than session ID)
    title = conv.title
    if index_entry.summary and index_entry.summary != "User Exits CLI Session":
        title = index_entry.summary
    elif index_entry.first_prompt and index_entry.first_prompt != "No prompt":
        # Truncate long prompts
        title = index_entry.first_prompt[:80]
        if len(index_entry.first_prompt) > 80:
            title += "..."

    # Merge provider_meta with index metadata
    provider_meta = dict(conv.provider_meta) if conv.provider_meta else {}
    provider_meta.update({
        "gitBranch": index_entry.git_branch,
        "projectPath": index_entry.project_path,
        "isSidechain": index_entry.is_sidechain,
        "summary": index_entry.summary,
        "firstPrompt": index_entry.first_prompt,
    })

    return ParsedConversation(
        provider_name=conv.provider_name,
        provider_conversation_id=conv.provider_conversation_id,
        title=title,
        created_at=index_entry.created or conv.created_at,
        updated_at=index_entry.modified or conv.updated_at,
        messages=conv.messages,
        attachments=conv.attachments,
        provider_meta=provider_meta,
    )


def extract_text_from_segments(segments: list[object]) -> str | None:
    lines: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            if segment:
                lines.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        # Check type first - tool_use/tool_result should be serialized as JSON
        seg_type = segment.get("type")
        if seg_type in {"tool_use", "tool_result"}:
            lines.append(json.dumps(segment, sort_keys=True))
            continue
        # Handle thinking blocks - wrap in XML tags for semantic detection
        if seg_type == "thinking":
            seg_thinking = segment.get("thinking")
            if isinstance(seg_thinking, str):
                lines.append(f"<thinking>{seg_thinking}</thinking>")
                continue
        seg_text = segment.get("text")
        if isinstance(seg_text, str):
            lines.append(seg_text)
            continue
        seg_content = segment.get("content")
        if isinstance(seg_content, str):
            lines.append(seg_content)
            continue
    combined = "\n".join(line for line in lines if line)
    return combined or None


def normalize_timestamp(ts: int | float | str | None) -> str | None:
    """Normalize a timestamp to epoch seconds string.

    Handles numeric epochs (int/float/str), millisecond epochs, and ISO 8601 strings.
    """
    if ts is None:
        return None
    # Try numeric first
    try:
        val = float(ts)
        # If timestamp is > 1e11 (year 5138), assume milliseconds and convert to seconds
        if val > 1e11:
            val = val / 1000.0
        return str(val)
    except (ValueError, TypeError):
        pass
    # Try ISO 8601 string
    if isinstance(ts, str):
        from polylogue.lib.timestamps import parse_timestamp

        dt = parse_timestamp(ts)
        if dt is not None:
            return str(dt.timestamp())
    return None



def extract_messages_from_chat_messages(chat_messages: list[object]) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    for idx, item in enumerate(chat_messages, start=1):
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("uuid") or item.get("id") or item.get("message_id") or f"msg-{idx}")
        # Role is required - skip messages without one
        raw_role = item.get("sender") or item.get("role")
        if not raw_role:
            continue
        role = Role.normalize(raw_role)

        raw_ts = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        timestamp = normalize_timestamp(raw_ts)

        # Check for text field directly first (Claude AI format)
        text = item.get("text") if isinstance(item.get("text"), str) else None
        # Then check content field
        if text is None:
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = extract_text_from_segments(content)
            elif isinstance(content, dict):
                text = content.get("text") if isinstance(content.get("text"), str) else None
                if text is None and isinstance(content.get("parts"), list):
                    text = "\n".join(str(part) for part in content["parts"] if part)
        # Build content blocks from structured content if available
        raw_content = item.get("content")
        content_blocks = content_blocks_from_segments(raw_content) if isinstance(raw_content, list) else []
        # Fall back to single text block if no structured content
        if not content_blocks and text:
            content_blocks = [ParsedContentBlock(type="text", text=text)]

        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=message_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    content_blocks=content_blocks,
                )
            )
        for att_idx, meta in enumerate(item.get("attachments") or item.get("files") or [], start=1):
            attachment = attachment_from_meta(meta, message_id, att_idx)
            if attachment:
                attachments.append(attachment)
    return messages, attachments


def looks_like_ai(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("chat_messages"), list)


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


def _extract_message_text(message_content: object) -> str | None:
    """Extract text from claude-code message content structure."""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return extract_text_from_segments(message_content)
    if isinstance(message_content, dict):
        text = message_content.get("text")
        if isinstance(text, str):
            return text
        parts = message_content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p)
    return None


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
        cost_val = item.get("costUSD")
        if cost_val:
            total_cost += _safe_float(cost_val)
        dur_val = item.get("durationMs")
        if dur_val:
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
    if total_cost > 0:
        conv_meta["total_cost_usd"] = total_cost
    if total_duration > 0:
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


def parse_ai(payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    try:
        conv = ClaudeAIConversation.model_validate(payload)
    except ValidationError:
        # Fall back to untyped extraction for non-standard exports
        chat_msgs = payload.get("chat_messages") or []
        if not isinstance(chat_msgs, list):
            chat_msgs = []
        messages, attachments = extract_messages_from_chat_messages(chat_msgs)
        title = payload.get("title") or payload.get("name") or fallback_id
        conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
        return ParsedConversation(
            provider_name=Provider.CLAUDE,
            provider_conversation_id=str(conv_id or fallback_id),
            title=str(title),
            created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
            updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
            messages=messages,
            attachments=attachments,
        )

    # Typed path: extract from validated model
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    for msg in conv.chat_messages:
        timestamp = normalize_timestamp(msg.created_at)
        if msg.text:
            # Build content blocks from structured content if available
            raw_content = msg.model_dump().get("content")
            content_blocks = content_blocks_from_segments(raw_content) if isinstance(raw_content, list) else []
            if not content_blocks and msg.text:
                content_blocks = [ParsedContentBlock(type="text", text=msg.text)]
            messages.append(
                ParsedMessage(
                    provider_message_id=msg.uuid,
                    role=msg.role_normalized,
                    text=msg.text,
                    timestamp=timestamp,
                    content_blocks=content_blocks,
                )
            )
        for att_idx, meta in enumerate(msg.attachments + msg.files, start=1):
            attachment = attachment_from_meta(meta, msg.uuid, att_idx)
            if attachment:
                attachments.append(attachment)

    return ParsedConversation(
        provider_name="claude",
        provider_conversation_id=conv.uuid,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
        attachments=attachments,
    )


# Symmetric aliases — makes the claude module conform to the same interface
# as chatgpt.py and codex.py (parse + looks_like at module level).
# parse_code / looks_like_code remain for explicit dispatch in source.py.
parse = parse_code
looks_like = looks_like_code
