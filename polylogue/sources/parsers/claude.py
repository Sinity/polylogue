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

from polylogue.lib.log import get_logger
from polylogue.sources.providers.claude_ai import ClaudeAIConversation
from polylogue.sources.providers.claude_code import ClaudeCodeRecord

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, attachment_from_meta, normalize_role

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
        role = normalize_role(raw_role)

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
        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=message_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    provider_meta={"raw": item},
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


# =============================================================================
# Claude Code Semantic Extractors
# =============================================================================


def extract_thinking_traces(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured thinking traces from content blocks.

    Args:
        content_blocks: List of content block dicts from Claude Code message

    Returns:
        List of thinking trace dicts with text and optional metadata
    """
    traces = []
    for block in content_blocks:
        if block.get("type") == "thinking":
            text = block.get("thinking") or block.get("text") or ""
            if text:
                trace = {
                    "text": text,
                    "token_count": len(text.split()),  # Rough approximation
                }
                traces.append(trace)
    return traces


def extract_tool_invocations(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured tool invocations from content blocks.

    Args:
        content_blocks: List of content block dicts from Claude Code message

    Returns:
        List of tool invocation dicts with name, id, input, and derived semantics
    """
    invocations = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            invocation = {
                "tool_name": block.get("name"),
                "tool_id": block.get("id"),
                "input": block.get("input") or {},
            }
            # Add derived semantics
            tool_name = invocation["tool_name"]
            if tool_name:
                invocation["is_file_operation"] = tool_name in {"Read", "Write", "Edit", "NotebookEdit"}
                invocation["is_search_operation"] = tool_name in {"Glob", "Grep", "WebSearch"}
                invocation["is_subagent"] = tool_name == "Task"

                # Check for git operations in Bash
                if tool_name == "Bash":
                    cmd = (invocation.get("input") or {}).get("command", "")
                    invocation["is_git_operation"] = isinstance(cmd, str) and cmd.strip().startswith("git ")

            invocations.append(invocation)
    return invocations


def parse_git_operation(tool_invocation: dict[str, Any]) -> dict[str, Any] | None:
    """Parse git operation details from a Bash tool invocation.

    Args:
        tool_invocation: Tool invocation dict with command in input

    Returns:
        Git operation dict or None if not a git command
    """
    if tool_invocation.get("tool_name") != "Bash":
        return None

    command = tool_invocation.get("input", {}).get("command", "")
    if not isinstance(command, str) or not command.strip().startswith("git "):
        return None

    parts = command.strip().split()
    if len(parts) < 2:
        return None

    git_cmd = parts[1]  # git <subcommand>

    result: dict[str, Any] = {
        "command": git_cmd,
        "full_command": command,
    }

    # Parse specific git commands
    if git_cmd == "commit":
        # Look for -m "message"
        for i, part in enumerate(parts):
            if part == "-m" and i + 1 < len(parts):
                # Get message (may be quoted)
                msg_start = command.find('-m') + 2
                if msg_start > 2:
                    # Try to extract quoted message
                    remaining = command[msg_start:].strip()
                    if remaining.startswith('"'):
                        end_quote = remaining.find('"', 1)
                        if end_quote > 0:
                            result["message"] = remaining[1:end_quote]
                    elif remaining.startswith("'"):
                        end_quote = remaining.find("'", 1)
                        if end_quote > 0:
                            result["message"] = remaining[1:end_quote]

    elif git_cmd in ("checkout", "switch"):
        # Look for branch name
        for part in parts[2:]:
            if not part.startswith("-"):
                result["branch"] = part
                break

    elif git_cmd == "push":
        # Extract remote and branch
        non_flags = [p for p in parts[2:] if not p.startswith("-")]
        if len(non_flags) >= 2:
            result["remote"] = non_flags[0]
            result["branch"] = non_flags[1]
        elif len(non_flags) == 1:
            result["remote"] = non_flags[0]

    elif git_cmd in ("add", "rm"):
        # Extract files
        result["files"] = [p for p in parts[2:] if not p.startswith("-")]

    return result


def extract_file_changes(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract file change information from tool invocations.

    Args:
        tool_invocations: List of tool invocation dicts

    Returns:
        List of file change dicts with path, operation type, and content
    """
    changes = []
    for invocation in tool_invocations:
        tool_name = invocation.get("tool_name")
        input_data = invocation.get("input", {})

        if tool_name == "Read":
            path = input_data.get("file_path") or input_data.get("path")
            if path:
                changes.append({
                    "path": path,
                    "operation": "read",
                })

        elif tool_name == "Write":
            path = input_data.get("file_path") or input_data.get("path")
            content = input_data.get("content")
            if path:
                changes.append({
                    "path": path,
                    "operation": "write",
                    "new_content": content[:500] if isinstance(content, str) else None,  # Truncate
                })

        elif tool_name == "Edit":
            path = input_data.get("file_path") or input_data.get("path")
            old_string = input_data.get("old_string")
            new_string = input_data.get("new_string")
            if path:
                changes.append({
                    "path": path,
                    "operation": "edit",
                    "old_content": old_string[:200] if isinstance(old_string, str) else None,
                    "new_content": new_string[:200] if isinstance(new_string, str) else None,
                })

    return changes


def extract_subagent_spawns(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract subagent spawn information from Task tool invocations.

    Args:
        tool_invocations: List of tool invocation dicts

    Returns:
        List of subagent spawn dicts
    """
    spawns = []
    for invocation in tool_invocations:
        if invocation.get("tool_name") != "Task":
            continue

        input_data = invocation.get("input", {})
        spawn = {
            "agent_type": input_data.get("subagent_type", "general-purpose"),
            "prompt": input_data.get("prompt", ""),
            "description": input_data.get("description"),
            "run_in_background": input_data.get("run_in_background", False),
        }
        spawns.append(spawn)

    return spawns


def detect_context_compaction(item: dict[str, Any]) -> dict[str, Any] | None:
    """Detect if a message represents a context compaction event.

    Args:
        item: Raw message dict from Claude Code JSONL

    Returns:
        Context compaction dict or None if not a compaction
    """
    msg_type = item.get("type")

    # Summary messages indicate context compaction
    if msg_type == "summary":
        message = item.get("message", {})
        content = ""
        if isinstance(message, dict):
            content_raw = message.get("content")
            if isinstance(content_raw, str):
                content = content_raw
            elif isinstance(content_raw, list):
                for block in content_raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break

        return {
            "summary": content,
            "timestamp": item.get("timestamp"),
        }

    return None


def extract_git_operations(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all git operations from tool invocations.

    Args:
        tool_invocations: List of tool invocation dicts

    Returns:
        List of git operation dicts
    """
    operations = []
    for invocation in tool_invocations:
        git_op = parse_git_operation(invocation)
        if git_op:
            operations.append(git_op)
    return operations


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

        # Build provider_meta with useful fields from typed record
        meta: dict[str, object] = {"raw": item}
        if record.costUSD:
            meta["costUSD"] = record.costUSD
        if record.durationMs:
            meta["durationMs"] = record.durationMs
        if record.isSidechain:
            meta["isSidechain"] = True
        if record.isMeta:
            meta["isMeta"] = True
        # Extract record-level metadata (cwd, gitBranch, version, model, token usage)
        if record.cwd:
            meta["cwd"] = record.cwd
        if record.gitBranch:
            meta["gitBranch"] = record.gitBranch
        if record.version:
            meta["version"] = record.version
        # Extract model and token usage from message content
        if isinstance(record.message, dict):
            model = record.message.get("model")
            if isinstance(model, str) and model:
                meta["model"] = model
            usage = record.message.get("usage")
            if isinstance(usage, dict) and usage:
                token_info: dict[str, object] = {}
                for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                    v = usage.get(k)
                    if isinstance(v, int) and v > 0:
                        token_info[k] = v
                if token_info:
                    meta["token_usage"] = token_info
            stop_reason = record.message.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                meta["stop_reason"] = stop_reason
        elif hasattr(record.message, "model") and record.message is not None:
            if hasattr(record.message, "model") and record.message.model:
                meta["model"] = record.message.model
            if hasattr(record.message, "usage") and record.message.usage:
                usage_obj = record.message.usage
                token_info = {}
                if hasattr(usage_obj, "input_tokens") and usage_obj.input_tokens:
                    token_info["input_tokens"] = usage_obj.input_tokens
                if hasattr(usage_obj, "output_tokens") and usage_obj.output_tokens:
                    token_info["output_tokens"] = usage_obj.output_tokens
                if hasattr(usage_obj, "cache_creation_input_tokens") and usage_obj.cache_creation_input_tokens:
                    token_info["cache_creation_input_tokens"] = usage_obj.cache_creation_input_tokens
                if hasattr(usage_obj, "cache_read_input_tokens") and usage_obj.cache_read_input_tokens:
                    token_info["cache_read_input_tokens"] = usage_obj.cache_read_input_tokens
                if token_info:
                    meta["token_usage"] = token_info
            if hasattr(record.message, "stop_reason") and record.message.stop_reason:
                meta["stop_reason"] = record.message.stop_reason

        # Extract content blocks using typed model
        content_blocks_raw = record.content_blocks_raw
        if content_blocks_raw:
            # Build serializable content blocks for storage
            content_blocks = []
            for seg in content_blocks_raw:
                if isinstance(seg, dict):
                    block_type = seg.get("type")
                    if block_type == "thinking":
                        content_blocks.append({
                            "type": "thinking",
                            "text": seg.get("thinking"),
                        })
                    elif block_type == "tool_use":
                        content_blocks.append({
                            "type": "tool_use",
                            "name": seg.get("name"),
                            "id": seg.get("id"),
                            "input": seg.get("input"),
                        })
                    elif block_type == "tool_result":
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": seg.get("tool_use_id"),
                            "content": seg.get("content"),
                            "is_error": seg.get("is_error", False),
                        })
                    elif block_type == "text":
                        content_blocks.append({
                            "type": "text",
                            "text": seg.get("text"),
                        })
                    else:
                        text_content = seg.get("text") or seg.get("content")
                        if text_content:
                            content_blocks.append({
                                "type": "text",
                                "text": text_content,
                            })
                elif isinstance(seg, str):
                    content_blocks.append({
                        "type": "text",
                        "text": seg,
                    })

            if content_blocks:
                meta["content_blocks"] = content_blocks

                # Extract semantic data from content blocks
                thinking_traces = extract_thinking_traces(content_blocks)
                if thinking_traces:
                    meta["thinking_traces"] = thinking_traces

                tool_invocations = extract_tool_invocations(content_blocks)
                if tool_invocations:
                    meta["tool_invocations"] = tool_invocations

                    # Extract derived semantics from tool invocations
                    git_operations = extract_git_operations(tool_invocations)
                    if git_operations:
                        meta["git_operations"] = git_operations

                    file_changes = extract_file_changes(tool_invocations)
                    if file_changes:
                        meta["file_changes"] = file_changes

                    subagent_spawns = extract_subagent_spawns(tool_invocations)
                    if subagent_spawns:
                        meta["subagent_spawns"] = subagent_spawns

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text or "",
                timestamp=timestamp,
                provider_meta=meta,
                parent_message_provider_id=record.parentUuid,
            )
        )

    # Derive conversation timestamps from messages
    created_at = min(timestamps) if timestamps else None
    updated_at = max(timestamps) if timestamps else None

    # Use session_id as conversation ID if available
    conv_id = session_id or fallback_id

    # Build conversation-level provider_meta
    conv_meta: dict[str, Any] = {}
    if context_compactions:
        conv_meta["context_compactions"] = context_compactions

    # Aggregate statistics from messages
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

    total_cost = sum(
        _safe_float(m.provider_meta.get("costUSD", 0))
        for m in messages
        if m.provider_meta and m.provider_meta.get("costUSD")
    )
    if total_cost > 0:
        conv_meta["total_cost_usd"] = total_cost

    total_duration = sum(
        _safe_int(m.provider_meta.get("durationMs", 0))
        for m in messages
        if m.provider_meta and m.provider_meta.get("durationMs")
    )
    if total_duration > 0:
        conv_meta["total_duration_ms"] = total_duration

    # Detect if any message has isSidechain flag
    has_sidechain = any(
        m.provider_meta and m.provider_meta.get("isSidechain")
        for m in messages
    )
    branch_type = "sidechain" if has_sidechain else None

    # Collect unique cwd paths and models across messages
    cwds = set()
    models = set()
    for m in messages:
        if m.provider_meta:
            cwd_val = m.provider_meta.get("cwd")
            if isinstance(cwd_val, str):
                cwds.add(cwd_val)
            model_val = m.provider_meta.get("model")
            if isinstance(model_val, str):
                models.add(model_val)
    if cwds:
        conv_meta["working_directories"] = sorted(cwds)
    if models:
        conv_meta["models_used"] = sorted(models)

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
        provider_name="claude-code",
        provider_conversation_id=str(conv_id),
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        provider_meta=conv_meta if conv_meta else None,
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
            provider_name="claude",
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
            messages.append(
                ParsedMessage(
                    provider_message_id=msg.uuid,
                    role=msg.role_normalized,
                    text=msg.text,
                    timestamp=timestamp,
                    provider_meta={"raw": msg.model_dump()},
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
