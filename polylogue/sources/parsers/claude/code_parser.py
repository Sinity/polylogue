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
from polylogue.sources.providers.claude_code_models import ClaudeCodeBackgroundTaskNotification

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
_BACKGROUND_TASK_ID_METADATA_KEY = "claude_background_task_id"
_BACKGROUND_COMPLETION_STATUS_METADATA_KEY = "claude_background_completion_status"
_BACKGROUND_OUTPUT_FILE_METADATA_KEY = "claude_background_output_file"


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
_SKIPPED_SIDECAR_RECORD_TYPES = frozenset(
    {
        "init",
        "file-history-snapshot",
        "queue-operation",
        "progress",
        "agent-name",
        "ai-title",
        "attachment",
        "bridge-session",
        "last-prompt",
        "mode",
        "permission-mode",
        "pr-link",
    }
)


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
    origin_kind = _record_origin_kind(item)
    if origin_kind not in (None, "human"):
        return MessageType.PROTOCOL
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
    # Anthropic bills web_search / web_fetch separately from token usage and
    # reports the request counts under usage.server_tool_use. The token lanes
    # above never carry them, so they would otherwise be lost. Record only when a
    # count is positive: most CLI sessions never call web tools, and an all-zero
    # sub-dict on every message would bloat the persisted payload_json for no
    # signal. The bytes ride into payload_json via the existing writer, so this
    # needs no schema change. (server_tool_use research, agent #03.)
    server_tool_use = usage.get("server_tool_use")
    if isinstance(server_tool_use, dict):
        web_search_requests = _safe_int(server_tool_use.get("web_search_requests"))
        web_fetch_requests = _safe_int(server_tool_use.get("web_fetch_requests"))
        if web_search_requests or web_fetch_requests:
            payload["server_tool_use"] = {
                "web_search_requests": web_search_requests,
                "web_fetch_requests": web_fetch_requests,
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


def _record_origin_kind(item: dict[str, object]) -> str | None:
    origin = item.get("origin")
    if isinstance(origin, dict):
        kind = origin.get("kind")
        return kind if isinstance(kind, str) and kind else None
    return None


def _claude_code_user_turn_origin(
    item: dict[str, object],
    *,
    role: Role,
    message_type: MessageType,
    content_blocks: Sequence[ParsedContentBlock],
    is_agent_session: bool,
) -> MaterialOrigin | None:
    """Return provider-evidenced origin for one plain Claude Code user turn.

    Claude Code's top-level transcript gives a plain ``type=user`` row positive
    interactive-prompt provenance once meta/context/tool-result shapes have been
    excluded. Subagent JSONL uses the same wire shape for generated task
    instructions, however. The ``agent-*`` artifact identity is therefore
    required to keep an origin-less worker instruction out of authored-user
    projections. An explicit ``origin.kind=human`` remains authoritative in
    either artifact family.
    """
    if item.get("type") != "user" or role is not Role.USER or message_type is not MessageType.MESSAGE:
        return None
    if item.get("isMeta") or item.get("isCompactSummary") or item.get("isVisibleInTranscriptOnly"):
        return None
    if item.get("toolUseResult") is not None:
        return None
    if any(block.type is BlockType.TOOL_RESULT for block in content_blocks):
        return None
    origin_kind = _record_origin_kind(item)
    if origin_kind == "human":
        return MaterialOrigin.HUMAN_AUTHORED
    if origin_kind is None:
        return MaterialOrigin.GENERATED_CONTEXT_PACK if is_agent_session else MaterialOrigin.HUMAN_AUTHORED
    return None


def _string_field(item: dict[str, object], key: str) -> str | None:
    value = item.get(key)
    return value if isinstance(value, str) and value else None


def _is_fresh_task_prompt_head(item: dict[str, object]) -> bool:
    """Return whether ``item`` is a structurally fresh Task-agent prompt.

    Claude Code can emit ``agent-acompact-*`` artifacts for two different
    shapes: a main-session compactor replaying the parent transcript, and a
    Task subagent compacting its own fresh transcript.  The parser normally
    sees only this one sidecar, not the main-session file, so it cannot always
    perform the authoritative content-membership comparison here.  A root
    sidechain user prompt with explicit Task prompt/agent identifiers is the
    provider's positive fresh-spawn marker.  Keep the predicate deliberately
    narrow; ambiguous artifacts remain continuations until the archive writer
    can compare them against the resolved parent content.
    """
    if item.get("type") != "user":
        return False
    if item.get("parentUuid") is not None or item.get("isSidechain") is not True:
        return False
    if item.get("isMeta") or item.get("isCompactSummary") or item.get("toolUseResult") is not None:
        return False
    if _record_origin_kind(item) == "human":
        return False
    message = item.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    return _string_field(item, "agentId") is not None and _string_field(item, "promptId") is not None


def _background_task_id(item: dict[str, object]) -> str | None:
    tool_result = item.get("toolUseResult")
    if not isinstance(tool_result, dict):
        return None
    task_id = tool_result.get("backgroundTaskId")
    return task_id if isinstance(task_id, str) and task_id else None


def _task_notification_from_record(
    item: dict[str, object], message: object
) -> ClaudeCodeBackgroundTaskNotification | None:
    """Read task protocol from message, queue-operation, or queued-command attachment."""
    candidates: list[object] = []
    if isinstance(message, dict):
        candidates.append(message.get("content"))
    candidates.append(item.get("content"))
    attachment = item.get("attachment")
    if isinstance(attachment, dict):
        candidates.append(attachment.get("prompt"))
    for candidate in candidates:
        if isinstance(candidate, str):
            notification = ClaudeCodeBackgroundTaskNotification.from_protocol_text(candidate)
            if notification is not None:
                return notification
    return None


def _mark_background_task_start(
    content_blocks: list[ParsedContentBlock], task_id: str | None
) -> list[ParsedContentBlock]:
    """Mark the immediate background acknowledgement as outcome-unknown.

    Claude's initial Bash result acknowledges only that a task started. Its
    ``is_error=false`` must not be projected as a completed-command success.
    """
    if task_id is None:
        return content_blocks
    marked: list[ParsedContentBlock] = []
    for block in content_blocks:
        if block.type is not BlockType.TOOL_RESULT:
            marked.append(block)
            continue
        metadata = dict(block.metadata or {})
        metadata[_BACKGROUND_TASK_ID_METADATA_KEY] = task_id
        marked.append(block.model_copy(update={"metadata": metadata, "is_error": None, "exit_code": None}))
    return marked


def _project_background_task_completions(
    messages: list[ParsedMessage], notifications: Sequence[ClaudeCodeBackgroundTaskNotification]
) -> list[ParsedMessage]:
    """Apply the final structured completion outcome to its start result.

    The exact ``(task-id, tool-use-id)`` pair is the provider protocol join
    key. Later notifications deliberately replace earlier ones for the same
    pair, making duplicate delivery and provider updates deterministic.
    """
    starts: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for message_index, message in enumerate(messages):
        for block_index, block in enumerate(message.blocks):
            if block.type is not BlockType.TOOL_RESULT or not block.tool_id:
                continue
            metadata = block.metadata or {}
            task_id = metadata.get(_BACKGROUND_TASK_ID_METADATA_KEY)
            if isinstance(task_id, str) and task_id:
                starts.setdefault((task_id, block.tool_id), []).append((message_index, block_index))

    starts_by_task: dict[str, list[tuple[int, int]]] = {}
    for (task_id, _), locations in starts.items():
        starts_by_task.setdefault(task_id, []).extend(locations)

    terminal_by_start: dict[tuple[int, int], ClaudeCodeBackgroundTaskNotification] = {}
    for notification in notifications:
        matched_location: tuple[int, int] | None = (
            _unique_background_start(starts.get((notification.task_id, notification.tool_use_id), []))
            if notification.tool_use_id is not None
            else _unique_background_start(starts_by_task.get(notification.task_id, []))
        )
        if matched_location is not None:
            terminal_by_start[matched_location] = notification
    projected = list(messages)
    for location, notification in terminal_by_start.items():
        message_index, block_index = location
        message = projected[message_index]
        block = message.blocks[block_index]
        metadata = dict(block.metadata or {})
        metadata[_BACKGROUND_COMPLETION_STATUS_METADATA_KEY] = notification.status
        metadata[_BACKGROUND_OUTPUT_FILE_METADATA_KEY] = notification.output_file
        updated_block = block.model_copy(
            update={
                "metadata": metadata,
                "is_error": None if notification.exit_code is None else notification.exit_code != 0,
                "exit_code": notification.exit_code,
            }
        )
        blocks = list(message.blocks)
        blocks[block_index] = updated_block
        projected[message_index] = message.model_copy(update={"blocks": blocks})
    return projected


def _unique_background_start(locations: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    """Return a task-only match only when provider evidence identifies one start."""
    return locations[0] if len(locations) == 1 else None


def _workflow_invocation_events(
    content_blocks: Sequence[ParsedContentBlock],
    *,
    source_message_provider_id: str,
    timestamp: str | None,
) -> list[ParsedSessionEvent]:
    """Project provider-native Workflow calls without promoting child sessions.

    Only the coordinator tool-use record establishes this invocation edge.  A
    parent/child session relation remains insufficient evidence for Workflow
    membership, so no agent transcript is inferred here.
    """

    events: list[ParsedSessionEvent] = []
    for block in content_blocks:
        if block.type is not BlockType.TOOL_USE or block.tool_name != "Workflow":
            continue
        tool_input = block.tool_input or {}
        payload = {
            key: value
            for key, value in tool_input.items()
            if key
            in {
                "runId",
                "run_id",
                "taskId",
                "task_id",
                "resumeFromRunId",
                "resume_from_run_id",
                "workflow",
                "workflowName",
                "scriptPath",
                "scriptHash",
                "labels",
                "phases",
            }
        }
        events.append(
            ParsedSessionEvent(
                event_type="claude_workflow_invocation",
                timestamp=timestamp,
                source_message_provider_id=source_message_provider_id,
                payload=payload,
            )
        )
    return events


def _parse_code_records(
    records: Iterable[object],
    fallback_id: str,
    *,
    record_index_start: int = 0,
    seen_record_uuids: set[str] | None = None,
) -> ParsedSession:
    """Parse Claude Code JSONL payloads into a canonical session model.

    ``record_index_start`` and ``seen_record_uuids`` are compact streaming
    continuation state used when one provider-native session is split by
    interleaved JSONL rows. They preserve eager-path fallback identifiers and
    first-record-wins UUID semantics without retaining raw records.
    """
    messages: list[ParsedMessage] = []
    created_at: str | None = None
    updated_at: str | None = None
    seen_uuids = seen_record_uuids if seen_record_uuids is not None else set()
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
    fresh_task_prompt_head = False
    saw_plain_user_head = False
    cwds: set[str] = set()
    models: set[str] = set()
    message_position = 0
    background_notifications: list[tuple[ClaudeCodeBackgroundTaskNotification, str | None, str | None]] = []
    is_agent = fallback_id.startswith("agent-")
    is_acompact = fallback_id.startswith("agent-acompact-")

    for index, item in enumerate(records, start=record_index_start + 1):
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
            if record_uuid in seen_uuids:
                duplicate_uuid_count += 1
                first_duplicate_uuid = first_duplicate_uuid or record_uuid
                first_duplicate_index = first_duplicate_index or index
                continue
            seen_uuids.add(record_uuid)

        if not session_id:
            session_id = _string_field(item, "sessionId")
        raw_timestamp = item.get("timestamp")
        timestamp = normalize_timestamp(raw_timestamp if isinstance(raw_timestamp, str | int | float) else None)
        message = item.get("message")
        notification = _task_notification_from_record(item, message)

        # ``progress`` records are claude-code hook lifecycle events
        # (`hookEvent`, `hookName`, `command`) carried alongside the
        # tool they fired on — they are NOT message content. Persisting
        # them as messages produces empty rows under the ``tool_result``
        # message_type that dominate the ``role=unknown, text='', blocks=[]``
        # consumer surface and inflate every messages-table count by
        # ~23%. See #1617 for the full forensic. We drop them here at the
        # parser; the hook payload, if useful for analytics, belongs in
        # a future ``session_event`` capture, not in the messages table.
        if record_type in _SKIPPED_SIDECAR_RECORD_TYPES:
            if notification is not None:
                background_notifications.append((notification, record_uuid, timestamp))
            continue
        if timestamp:
            created_at = timestamp if created_at is None or timestamp < created_at else created_at
            updated_at = timestamp if updated_at is None or timestamp > updated_at else updated_at

        raw_content = message.get("content") if isinstance(message, dict) else item.get("content")
        text = extract_message_text(raw_content)
        envelope_role = _record_role(item, message)
        content_blocks = _content_blocks_from_record(message, text)
        content_blocks = _mark_background_task_start(content_blocks, _background_task_id(item))
        message_type = _message_type_from_code_record(item, text)
        if envelope_role is Role.SYSTEM and message_type is MessageType.MESSAGE:
            message_type = MessageType.CONTEXT
        if not saw_plain_user_head and envelope_role is Role.USER and message_type is MessageType.MESSAGE:
            saw_plain_user_head = True
            fresh_task_prompt_head = _is_fresh_task_prompt_head(item)
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
        if material_origin is MaterialOrigin.UNKNOWN:
            evidenced_origin = _claude_code_user_turn_origin(
                item,
                role=resolved_role,
                message_type=message_type,
                content_blocks=content_blocks,
                is_agent_session=is_agent,
            )
            if evidenced_origin is not None:
                material_origin = evidenced_origin
        if not text and not content_blocks and record_type != "summary":
            keep_empty_human_turn = (
                resolved_role is Role.USER
                and message_type is MessageType.MESSAGE
                and material_origin is MaterialOrigin.HUMAN_AUTHORED
            )
            if not keep_empty_human_turn:
                continue
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
        session_events.extend(
            _workflow_invocation_events(
                content_blocks,
                source_message_provider_id=provider_message_id,
                timestamp=timestamp,
            )
        )
        if notification is not None:
            background_notifications.append((notification, provider_message_id, timestamp))
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

    messages = _project_background_task_completions(
        messages, [notification for notification, _, _ in background_notifications]
    )
    final_background_notifications = {
        (notification.task_id, notification.tool_use_id): (notification, source_message_provider_id, timestamp)
        for notification, source_message_provider_id, timestamp in background_notifications
    }
    for notification, source_message_provider_id, timestamp in final_background_notifications.values():
        session_events.append(
            ParsedSessionEvent(
                event_type="background_task_completion",
                timestamp=timestamp,
                source_message_provider_id=source_message_provider_id,
                payload={
                    "task_id": notification.task_id,
                    "tool_use_id": notification.tool_use_id,
                    "output_file": notification.output_file,
                    "status": notification.status,
                    "summary": notification.summary,
                    "exit_code": notification.exit_code,
                },
            )
        )

    if duplicate_uuid_count:
        logger.debug(
            "Skipped repeated Claude Code record uuids: count=%d first_index=%s first_uuid=%s",
            duplicate_uuid_count,
            first_duplicate_index,
            first_duplicate_uuid,
        )

    # `agent-acompact-*` is overloaded by Claude Code. A main-session compactor
    # replays the parent transcript and is a continuation; a Task subagent can
    # also self-compact under the same filename prefix. A structurally fresh Task
    # prompt is positive evidence for sidechain topology. Ambiguous cases stay
    # continuations here and are reclassified by the archive writer after its
    # bounded content-membership check against the resolved parent transcript.
    parent_session_id: str | None = None
    if is_agent and session_id:
        composed_session_id = f"{session_id}:{fallback_id}"
        parent_session_id = session_id
    else:
        composed_session_id = session_id or fallback_id

    if is_acompact and fresh_task_prompt_head:
        branch_type: BranchType | None = BranchType.SIDECHAIN
    elif is_acompact:
        branch_type = BranchType.CONTINUATION
    elif is_agent:
        branch_type = BranchType.SUBAGENT
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
        # Title heuristic: the first plain human-authored user turn. Claude Code
        # has enough structural provenance (`isMeta`, `toolUseResult`, `origin`)
        # to avoid the old "unknown but title-worthy" compromise.
        if (
            message.role is Role.USER
            and message.message_type is MessageType.MESSAGE
            and message.material_origin is MaterialOrigin.HUMAN_AUTHORED
            and message.text
            and len(message.text.strip()) > 3
        ):
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


def _background_notification_from_event(
    event: ParsedSessionEvent,
) -> ClaudeCodeBackgroundTaskNotification | None:
    if event.event_type != "background_task_completion":
        return None
    payload = event.payload
    task_id = payload.get("task_id")
    output_file = payload.get("output_file")
    status = payload.get("status")
    summary = payload.get("summary")
    tool_use_id = payload.get("tool_use_id")
    exit_code = payload.get("exit_code")
    if not isinstance(task_id, str) or not task_id:
        return None
    if not isinstance(output_file, str) or not output_file:
        return None
    if not isinstance(status, str) or not status:
        return None
    if not isinstance(summary, str) or not summary:
        return None
    if tool_use_id is not None and not isinstance(tool_use_id, str):
        return None
    if exit_code is not None and not isinstance(exit_code, int):
        return None
    return ClaudeCodeBackgroundTaskNotification(
        task_id=task_id,
        tool_use_id=tool_use_id,
        output_file=output_file,
        status=status,
        summary=summary,
        exit_code=exit_code,
    )


def reconcile_code_session_chunks(session: ParsedSession) -> ParsedSession:
    """Finalize one Claude Code session after streaming chunk merge.

    Eager parsing sees every background completion before projecting outcomes
    and collapses duplicate/update notifications by provider join key. Streaming
    chunks must perform the same final pass after non-contiguous chunks have
    been merged, otherwise a late completion cannot update an earlier start and
    event order/count diverges from ordinary parsing.
    """
    ordinary_events: list[ParsedSessionEvent] = []
    completion_events: dict[tuple[str, str | None], ParsedSessionEvent] = {}
    for event in session.session_events:
        notification = _background_notification_from_event(event)
        if notification is None:
            ordinary_events.append(event)
            continue
        completion_events[(notification.task_id, notification.tool_use_id)] = event

    final_events = list(completion_events.values())
    notifications = [
        notification
        for event in final_events
        if (notification := _background_notification_from_event(event)) is not None
    ]
    messages = _project_background_task_completions(session.messages, notifications)
    return session.model_copy(update={"messages": messages, "session_events": [*ordinary_events, *final_events]})


def parse_code_stream(
    records: Iterable[object],
    fallback_id: str,
    *,
    record_index_start: int = 0,
    seen_record_uuids: set[str] | None = None,
) -> ParsedSession:
    return _parse_code_records(
        records,
        fallback_id,
        record_index_start=record_index_start,
        seen_record_uuids=seen_record_uuids,
    )


__all__ = ["parse_code", "parse_code_stream", "reconcile_code_session_chunks"]
