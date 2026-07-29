"""Claude Code session parsing helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, PasteBoundary, Provider, TitleSource
from polylogue.logging import get_logger
from polylogue.pipeline.semantic_capture import detect_context_compaction, detect_micro_compaction
from polylogue.sources.providers.claude_code_models import ClaudeCodeBackgroundTaskNotification

if TYPE_CHECKING:
    # ``polylogue.sources.live`` is a heavy package (pipeline/ingest_batch,
    # which imports back into ``sources.dispatch`` -> ``sources.parsers.claude``)
    # -- a module-level runtime import here is a circular import. This type is
    # only used for an optional parameter annotation; the value is never
    # constructed or introspected by name at runtime in this module.
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult

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
# ``_SKIPPED_SIDECAR_RECORD_TYPES`` marks record types that never become a
# ``ParsedMessage`` row (they are not chat content). That is still correct for
# all twelve types below. What changed (polylogue-pbuh, audited against the
# live corpus 2026-07-29, ~1.17M records total) is that most of them ARE
# evidence-bearing and are no longer silently discarded: they persist through
# ``_sidecar_evidence_payload``/the ``progress`` delegation accumulator below
# as typed ``session_events`` (index.db). ``session_events.event_type`` has no
# CHECK-constrained vocabulary (see ``storage/sqlite/archive_tiers/index.py``),
# so adding new event types here is additive data, not a schema change.
#
# Per-type disposition (counts = live corpus, rg single pass, 2026-07-29):
#   ai-title (18,561)              EVIDENCE, wins session title (TitleSource.ORIGIN)
#   agent-name (5,001)             EVIDENCE -> claude_agent_name event
#   pr-link (20,889)               EVIDENCE -> claude_pr_link event
#   bridge-session (13,594)        EVIDENCE -> claude_bridge_session event
#   file-history-snapshot (34,182) EVIDENCE -> claude_file_history_snapshot event
#   permission-mode (25,882)       EVIDENCE -> claude_permission_mode event (values
#                                    vary: auto/default/acceptEdits/plan/
#                                    bypassPermissions -- real operational signal)
#   last-prompt (37,799)           EVIDENCE -> claude_last_prompt event
#   queue-operation (60,709)       EVIDENCE -> claude_queue_operation event
#   attachment (86,237)            EVIDENCE -> claude_attachment event (20 distinct
#                                    attachment.type payloads incl. real files,
#                                    edited-file records, diagnostics; per-subtype
#                                    fidelity split is a follow-up, not this pass)
#   progress (850,678)             MIXED, not uniformly evidence-bearing -- an
#                                    earlier draft of this bead's tracking issue
#                                    guessed the whole type was noise, then
#                                    corrected to "it's all a delegation graph".
#                                    Neither guess survived reading the payload:
#                                    only ``data.type == "agent_progress"``
#                                    (118,135 of 850,678) carries a genuine
#                                    dispatcher->subagent edge (parentToolUseID
#                                    != toolUseID naming the *dispatching* Task
#                                    tool_use, not a synthetic per-tick id); it
#                                    is persisted as one deduplicated
#                                    claude_delegation_progress event per
#                                    (session, parentToolUseID). The other six
#                                    subtypes (bash_progress 449,363,
#                                    hook_progress 279,894, waiting_for_task
#                                    1,350, query_update 834,
#                                    search_results_received 834, mcp_progress
#                                    268) all stamp a synthetic per-tick
#                                    toolUseID (e.g. "bash-progress-3",
#                                    "search-progress-1") against the SAME
#                                    already-captured originating tool_use --
#                                    they are streaming ticks superseded by that
#                                    tool's own tool_result block, not new
#                                    facts. Persisting them would 4-8x the
#                                    session_events row count for zero
#                                    incremental evidence. Kept transient.
#   init (0 live occurrences)      TRANSIENT: every corpus occurrence on record
#                                    is a bare {"type": "init"} marker with no
#                                    field beyond ``type``/``sessionId`` --
#                                    nothing to lose.
#   mode (20,779)                  TRANSIENT: ``mode`` was the literal string
#                                    "normal" in 100% of live records -- zero
#                                    information content in the corpus as
#                                    observed. Re-audit if a non-"normal" value
#                                    is ever seen.
#
# Two more sidecar record types (parser-diff triage, 2026-07-29, ~600-file
# sample of the live corpus) were not in the original twelve and fell through
# to ordinary message parsing, where they carry no ``message``/text and are
# silently skipped:
#   custom-title (1,056)           EVIDENCE, an EXPLICIT user rename -> wins
#                                    over both the heuristic title and the
#                                    provider-suggested ai-title (higher
#                                    intent signal than a provider guess).
#   file-history-delta (304)       EVIDENCE -> claude_file_history_delta event
#                                    (per-file incremental backup, the
#                                    fine-grained sibling of the
#                                    whole-snapshot file-history-snapshot type)
_SKIPPED_SIDECAR_RECORD_TYPES = frozenset(
    {
        "init",
        "file-history-snapshot",
        "file-history-delta",
        "queue-operation",
        "progress",
        "agent-name",
        "ai-title",
        "custom-title",
        "attachment",
        "bridge-session",
        "last-prompt",
        "mode",
        "permission-mode",
        "pr-link",
    }
)

# record_type -> session_events.event_type for the sidecar types persisted
# generically via ``_sidecar_evidence_payload``. ``progress``, ``ai-title``,
# and ``custom-title`` are handled separately in ``_parse_code_records``
# (progress needs whole-session deduplication; ai-title/custom-title feed
# title resolution as well as an audit-trail event). ``init``/``mode`` map to
# nothing (transient, see above).
_SIDECAR_EVENT_TYPES: dict[str, str] = {
    "agent-name": "claude_agent_name",
    "pr-link": "claude_pr_link",
    "bridge-session": "claude_bridge_session",
    "file-history-snapshot": "claude_file_history_snapshot",
    "file-history-delta": "claude_file_history_delta",
    "permission-mode": "claude_permission_mode",
    "last-prompt": "claude_last_prompt",
    "queue-operation": "claude_queue_operation",
    "attachment": "claude_attachment",
    "ai-title": "claude_ai_title",
    "custom-title": "claude_custom_title",
}


def _sidecar_evidence_payload(record_type: str, item: dict[str, object]) -> dict[str, object] | None:
    """Return a typed evidence payload for a skipped-as-message sidecar record.

    Returns ``None`` when the specific record instance carries no usable
    signal (e.g. an ``attachment`` record whose ``attachment`` field is
    missing) so the caller can skip emitting an empty event.
    """
    if record_type == "agent-name":
        name = _string_field(item, "agentName")
        return {"agent_name": name, "summary": name} if name else None
    if record_type == "pr-link":
        pr_number = item.get("prNumber")
        pr_url = _string_field(item, "prUrl")
        return {
            "pr_number": pr_number,
            "pr_url": pr_url,
            "pr_repository": _string_field(item, "prRepository"),
            "summary": f"PR #{pr_number}: {pr_url}" if pr_number is not None and pr_url else pr_url,
        }
    if record_type == "bridge-session":
        bridge_session_id = _string_field(item, "bridgeSessionId")
        if not bridge_session_id:
            return None
        return {
            "bridge_session_id": bridge_session_id,
            "last_sequence_num": item.get("lastSequenceNum"),
            "summary": bridge_session_id,
        }
    if record_type == "file-history-snapshot":
        snapshot = item.get("snapshot")
        backups = snapshot.get("trackedFileBackups") if isinstance(snapshot, dict) else None
        files = sorted(backups) if isinstance(backups, dict) else []
        return {
            "message_id": _string_field(item, "messageId"),
            "is_snapshot_update": bool(item.get("isSnapshotUpdate")),
            "file_count": len(files),
            "files": files,
            "summary": f"{len(files)} tracked file backup(s)",
        }
    if record_type == "permission-mode":
        mode = _string_field(item, "permissionMode")
        return {"permission_mode": mode, "summary": mode} if mode else None
    if record_type == "last-prompt":
        prompt = _string_field(item, "lastPrompt")
        return {"last_prompt": prompt, "summary": prompt} if prompt else None
    if record_type == "queue-operation":
        operation = _string_field(item, "operation")
        if not operation:
            return None
        payload: dict[str, object] = {"operation": operation}
        content = _string_field(item, "content")
        if content:
            payload["content"] = content
            payload["summary"] = f"{operation}: {content}"
        else:
            payload["summary"] = operation
        return payload
    if record_type == "attachment":
        attachment = item.get("attachment")
        if not isinstance(attachment, dict):
            return None
        payload = dict(attachment)
        attachment_kind = attachment.get("type")
        payload["summary"] = str(attachment_kind) if attachment_kind is not None else "attachment"
        return payload
    if record_type == "ai-title":
        ai_title = _string_field(item, "aiTitle")
        return {"ai_title": ai_title, "summary": ai_title} if ai_title else None
    if record_type == "custom-title":
        custom_title = _string_field(item, "customTitle")
        return {"custom_title": custom_title, "summary": custom_title} if custom_title else None
    if record_type == "file-history-delta":
        backup = item.get("backup")
        backup_mapping = backup if isinstance(backup, dict) else {}
        tracking_path = _string_field(item, "trackingPath")
        return {
            "message_id": _string_field(item, "messageId"),
            "snapshot_message_id": _string_field(item, "snapshotMessageId"),
            "tracking_path": tracking_path,
            "backup_file_name": _string_field(backup_mapping, "backupFileName"),
            "backup_version": backup_mapping.get("version"),
            "summary": tracking_path,
        }
    return None


@dataclass
class _DelegationProgressStats:
    count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None


def _accumulate_delegation_progress(
    item: dict[str, object],
    timestamp: str | None,
    accumulator: dict[str, _DelegationProgressStats],
) -> None:
    """Fold one ``progress``/``agent_progress`` record into its dispatch edge.

    Only ``data.type == "agent_progress"`` carries a genuine dispatcher edge
    (see the classification comment above ``_SKIPPED_SIDECAR_RECORD_TYPES``);
    every occurrence under the same ``parentToolUseID`` is one streaming tick
    of the same subagent dispatch, so ticks are counted and time-bounded
    rather than persisted as one row apiece.
    """
    data = item.get("data")
    if not isinstance(data, dict) or data.get("type") != "agent_progress":
        return
    parent_tool_use_id = _string_field(item, "parentToolUseID")
    if not parent_tool_use_id:
        return
    entry = accumulator.setdefault(parent_tool_use_id, _DelegationProgressStats())
    entry.count += 1
    if timestamp:
        if entry.first_seen is None or timestamp < entry.first_seen:
            entry.first_seen = timestamp
        if entry.last_seen is None or timestamp > entry.last_seen:
            entry.last_seen = timestamp


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
    message: Mapping[str, object] | None = None,
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
    # 1h/5m cache-creation split, service tier, and inference geo are billing
    # and infra evidence Anthropic began reporting on ``usage`` alongside the
    # existing aggregate cache_creation_input_tokens -- parser-diff triage
    # (2026-07-29) found these read nowhere despite carrying real values on
    # the live corpus (e.g. ephemeral_1h_input_tokens, service_tier=standard,
    # inference_geo). Recorded only when present so a provider that never
    # emits them (older CLI versions) does not bloat every event.
    cache_creation = usage.get("cache_creation")
    if isinstance(cache_creation, dict):
        ephemeral_5m = cache_creation.get("ephemeral_5m_input_tokens")
        ephemeral_1h = cache_creation.get("ephemeral_1h_input_tokens")
        if ephemeral_5m is not None or ephemeral_1h is not None:
            payload["cache_creation_by_ttl"] = {
                "ephemeral_5m_input_tokens": _safe_int(ephemeral_5m),
                "ephemeral_1h_input_tokens": _safe_int(ephemeral_1h),
            }
    service_tier = usage.get("service_tier")
    if isinstance(service_tier, str) and service_tier:
        payload["service_tier"] = service_tier
    inference_geo = usage.get("inference_geo")
    if isinstance(inference_geo, str) and inference_geo:
        payload["inference_geo"] = inference_geo
    iterations = usage.get("iterations")
    if isinstance(iterations, int) and not isinstance(iterations, bool):
        payload["iterations"] = iterations
    speed = usage.get("speed")
    if isinstance(speed, (int, float)) and not isinstance(speed, bool):
        payload["speed"] = speed
    if model_name:
        payload["model"] = model_name
    if model_effort:
        payload["model_effort"] = model_effort
    if message is not None:
        ttft_ms = message.get("ttftMs")
        if isinstance(ttft_ms, int) and not isinstance(ttft_ms, bool):
            payload["ttft_ms"] = ttft_ms
        stop_reason = message.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason:
            payload["stop_reason"] = stop_reason
        stop_sequence = message.get("stop_sequence")
        if isinstance(stop_sequence, str) and stop_sequence:
            payload["stop_sequence"] = stop_sequence
        diagnostics = message.get("diagnostics")
        if isinstance(diagnostics, dict):
            cache_miss_reason = diagnostics.get("cache_miss_reason")
            if isinstance(cache_miss_reason, dict):
                reason_type = cache_miss_reason.get("type")
                if isinstance(reason_type, str) and reason_type:
                    payload["cache_miss_reason"] = reason_type
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


def _task_output_outcome(item: dict[str, object]) -> tuple[int | None, bool] | None:
    """Read a polled background task's own verdict from ``toolUseResult.task``.

    Claude Code's ``TaskOutput`` tool (used to poll a backgrounded command or
    agent) reports retrieval success/failure at the Anthropic tool_result
    envelope level (``content[].is_error``) *separately* from the underlying
    task's own outcome: a successful *poll* of a *failed* command still
    surfaces envelope ``is_error=false`` -- the poll itself succeeded -- which
    would otherwise silently mask the command's real exit code from
    ``blocks.tool_result_is_error``/``tool_result_exit_code``. The sibling
    ``<task-notification>`` protocol (``_task_notification_from_record``)
    only fires as an injected reminder on a *later* turn and is not always
    present in a transcript that instead polled ``TaskOutput`` directly, so
    this is an independent structured-evidence path, not a duplicate of it.

    ``toolUseResult.task`` is Claude Code's own structured verdict for this
    poll: ``exitCode`` for ``local_bash`` tasks, else ``status`` for
    ``local_agent`` tasks that never carry a process exit code. Only trusted
    when ``retrieval_status == "success"`` -- i.e. the poll actually returned
    a terminal result, as opposed to ``not_ready``/``timeout`` (task still
    running, no verdict yet).
    """
    tool_result = item.get("toolUseResult")
    if not isinstance(tool_result, dict) or tool_result.get("retrieval_status") != "success":
        return None
    task = tool_result.get("task")
    if not isinstance(task, dict):
        return None
    exit_code = task.get("exitCode")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code, exit_code != 0
    status = task.get("status")
    if status == "completed":
        return None, False
    if status in ("failed", "killed"):
        return None, True
    return None


_TOOL_RESULT_STRUCTURAL_KEYS: tuple[str, ...] = (
    "sandbox",
    "interrupted",
    "isImage",
    "userModified",
    "numFiles",
    "numLines",
    "totalLines",
    "durationSeconds",
    "backgroundedByUser",
    "assistantAutoBackgrounded",
    "isAsync",
)


def _tool_execution_result_payload(item: dict[str, object]) -> dict[str, object] | None:
    """Project Claude Code's own structured tool-result sidecar (``toolUseResult``).

    This is Claude-Code-specific enrichment parallel to (not a duplicate of)
    the Anthropic-protocol ``tool_result`` content block: the block carries
    what the model *saw*; ``toolUseResult`` carries facts Claude Code itself
    recorded about the call (e.g. whether Bash ran sandboxed, a Read's
    file/line extents, a structured diff). Parser-diff triage (2026-07-29)
    found only two of its ~60 observed subfields read anywhere
    (``backgroundTaskId``, ``retrieval_status``/``task`` -- see
    ``_background_task_id``/``_task_output_outcome`` above); this covers the
    remaining structurally bounded facts. Deliberately excluded: free-text
    output fields (``stdout``/``stderr``/``output``/``fullOutput``) that
    duplicate content already visible in the message's own ``tool_result``
    block and could be unbounded in size; ``filenames``/``file.filePath`` are
    kept because they are bounded path lists, not command output.
    """
    tool_result = item.get("toolUseResult")
    if not isinstance(tool_result, dict):
        return None
    payload: dict[str, object] = {}
    for key in _TOOL_RESULT_STRUCTURAL_KEYS:
        value = tool_result.get(key)
        if isinstance(value, (bool, int, float)):
            payload[key] = value
    file_info = tool_result.get("file")
    if isinstance(file_info, dict):
        file_path = file_info.get("filePath")
        if isinstance(file_path, str) and file_path:
            payload["file_path"] = file_path
        for key in ("numLines", "totalLines", "startLine"):
            value = file_info.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                payload[f"file_{key}"] = value
    elif isinstance(tool_result.get("filePath"), str):
        payload["file_path"] = tool_result["filePath"]
    filenames = tool_result.get("filenames")
    if isinstance(filenames, list):
        bounded_filenames = [name for name in filenames if isinstance(name, str)]
        if bounded_filenames:
            payload["filenames"] = bounded_filenames[:50]
    structured_patch = tool_result.get("structuredPatch")
    if isinstance(structured_patch, list) and structured_patch:
        hunks = [hunk for hunk in structured_patch if isinstance(hunk, dict)]
        if hunks:
            payload["structured_patch_hunk_count"] = len(hunks)
            payload["structured_patch_lines_changed"] = sum(
                len(hunk.get("lines", [])) if isinstance(hunk.get("lines"), list) else 0 for hunk in hunks
            )
    return payload or None


def _todo_state_payload(item: dict[str, object]) -> dict[str, object] | None:
    """Project TodoWrite's before/after task-list state from ``toolUseResult``.

    ``newTodos``/``oldTodos`` are TodoWrite's own structured result (each todo
    carries ``content``/``status``/``activeForm``/``priority``); the tool
    *call* input (``content[].input.todos``) is already captured wholesale via
    the generic ``tool_input`` dict copy, but the *result* -- the actual
    accepted state transition, including ``priority`` -- was read nowhere.
    """
    tool_result = item.get("toolUseResult")
    if not isinstance(tool_result, dict):
        return None
    new_todos = tool_result.get("newTodos")
    if not isinstance(new_todos, list):
        return None

    def _clean_todos(raw: object) -> list[dict[str, object]]:
        if not isinstance(raw, list):
            return []
        cleaned: list[dict[str, object]] = []
        for todo in raw:
            if not isinstance(todo, dict):
                continue
            cleaned.append(
                {
                    key: todo[key]
                    for key in ("content", "status", "activeForm", "priority")
                    if isinstance(todo.get(key), str)
                }
            )
        return cleaned

    return {
        "new_todos": _clean_todos(new_todos),
        "old_todos": _clean_todos(tool_result.get("oldTodos")),
    }


def _mark_task_output_outcome(
    content_blocks: list[ParsedContentBlock], outcome: tuple[int | None, bool] | None
) -> list[ParsedContentBlock]:
    """Project a polled background task's real verdict onto its TaskOutput result block."""
    if outcome is None:
        return content_blocks
    exit_code, is_error = outcome
    return [
        block.model_copy(update={"is_error": is_error, "exit_code": exit_code})
        if block.type is BlockType.TOOL_RESULT
        else block
        for block in content_blocks
    ]


def _mark_background_task_start(
    content_blocks: list[ParsedContentBlock], task_id: str | None
) -> list[ParsedContentBlock]:
    """Mark the immediate background acknowledgement as outcome-unknown.

    Claude's initial Bash result acknowledges only that a task started. Its
    ``is_error=false`` must not be projected as a completed-command success.

    DISPOSITION (bd polylogue-9x22, "dropped, and correctly so"): the
    ``_BACKGROUND_TASK_ID_METADATA_KEY``/``_BACKGROUND_COMPLETION_STATUS_
    METADATA_KEY``/``_BACKGROUND_OUTPUT_FILE_METADATA_KEY`` entries this
    function and ``_project_background_task_completions`` below write into
    ``block.metadata`` never reach storage (the ``blocks`` table has no
    metadata column; the write path only reads ``language`` back out of it).
    Unlike the other polylogue-9x22 sites, that is not a data-loss gap here:
    ``task_id`` is a same-pass join key with no meaning after
    ``_project_background_task_completions`` resolves it, and
    ``status``/``output_file`` are already durably captured -- with more
    fields (``summary``, ``exit_code``, ``tool_use_id``) and a real
    ``source_message_provider_id`` join key -- by the independently emitted
    ``background_task_completion`` session_event (see the
    ``final_background_notifications`` loop in ``parse_code``, and
    ``test_parse_code_projects_background_completion_outcomes_through_actions``
    which pins both the block-metadata carrier and the session_event
    verbatim). Left as an in-process carrier rather than removed, matching
    the ``claude_ai_web_tool_evidence`` precedent's use of block.metadata as
    scratch space.
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
    # polylogue-pbuh: provider-supplied session title (``ai-title`` sidecar
    # record) and the deduplicated agent-dispatch delegation edges extracted
    # from ``progress``/``agent_progress`` records -- both need whole-session
    # state, so they are accumulated here and applied/flushed after the loop.
    latest_ai_title: str | None = None
    latest_custom_title: str | None = None
    session_kind_value: str | None = None
    git_branch_value: str | None = None
    delegation_progress: dict[str, _DelegationProgressStats] = {}

    for index, item in enumerate(records, start=record_index_start + 1):
        if not isinstance(item, dict):
            continue

        if session_kind_value is None:
            raw_session_kind = item.get("sessionKind")
            if isinstance(raw_session_kind, str) and raw_session_kind:
                session_kind_value = raw_session_kind
        if git_branch_value is None:
            # ``item.gitBranch`` (stamped on every record) was previously read
            # only via the separate legacy sessions-index.json enrichment path
            # (index.py, ``enrich_session_from_index``), which is absent for
            # many sessions -- read it directly from the record itself so
            # every session with a real branch gets one, not just those with
            # a surviving sidecar index file. A blank string (no branch
            # checked out / detached-ish states some CLI versions emit) is
            # deliberately not treated as a value.
            raw_git_branch = item.get("gitBranch")
            if isinstance(raw_git_branch, str) and raw_git_branch:
                git_branch_value = raw_git_branch

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

        micro_compaction = detect_micro_compaction(item)
        if micro_compaction is not None:
            raw_micro_timestamp = micro_compaction["timestamp"]
            micro_compaction_timestamp = normalize_timestamp(
                raw_micro_timestamp if isinstance(raw_micro_timestamp, (int, float, str)) else None
            )
            session_events.append(
                ParsedSessionEvent(
                    event_type="micro_compaction",
                    timestamp=micro_compaction_timestamp,
                    payload={
                        "trigger": micro_compaction["trigger"],
                        "pre_tokens": micro_compaction["pre_tokens"],
                        "tokens_saved": micro_compaction["tokens_saved"],
                        "compacted_tool_use_ids": micro_compaction["compacted_tool_use_ids"],
                        "cleared_attachment_count": micro_compaction["cleared_attachment_count"],
                    },
                )
            )
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

        # These twelve record types are never chat content -- see the
        # classification comment above ``_SKIPPED_SIDECAR_RECORD_TYPES`` for
        # why each one either persists as typed ``session_events`` evidence
        # (polylogue-pbuh) or stays genuinely transient. ``progress`` (hook
        # lifecycle pings, streaming tool-progress ticks, and the one
        # evidence-bearing subtype ``agent_progress``) previously also
        # produced empty ``tool_result``-shaped message rows before the
        # record-type check below existed; see #1617 for that forensic.
        if record_type in _SKIPPED_SIDECAR_RECORD_TYPES:
            if notification is not None:
                background_notifications.append((notification, record_uuid, timestamp))
            if record_type == "progress":
                _accumulate_delegation_progress(item, timestamp, delegation_progress)
            else:
                if record_type == "ai-title":
                    ai_title_text = _string_field(item, "aiTitle")
                    if ai_title_text:
                        latest_ai_title = ai_title_text
                elif record_type == "custom-title":
                    custom_title_text = _string_field(item, "customTitle")
                    if custom_title_text:
                        latest_custom_title = custom_title_text
                event_type = _SIDECAR_EVENT_TYPES.get(record_type)
                if event_type is not None:
                    evidence_payload = _sidecar_evidence_payload(record_type, item)
                    if evidence_payload is not None:
                        # Message linkage belongs in the typed field, not buried
                        # in the payload dict: source_message_provider_id is what
                        # the archive joins on, and every other claude_* emitter
                        # here already uses it. Two sidecar payloads carried a
                        # bare "message_id" key instead, leaving the typed field
                        # NULL and the linkage unqueryable.
                        source_message_id = evidence_payload.pop("message_id", None)
                        session_events.append(
                            ParsedSessionEvent(
                                event_type=event_type,
                                timestamp=timestamp,
                                payload=evidence_payload,
                                source_message_provider_id=(
                                    str(source_message_id) if source_message_id is not None else None
                                ),
                            )
                        )
            continue
        if timestamp:
            created_at = timestamp if created_at is None or timestamp < created_at else created_at
            updated_at = timestamp if updated_at is None or timestamp > updated_at else updated_at

        raw_content = message.get("content") if isinstance(message, dict) else item.get("content")
        text = extract_message_text(raw_content)
        envelope_role = _record_role(item, message)
        content_blocks = _content_blocks_from_record(message, text)
        content_blocks = _mark_background_task_start(content_blocks, _background_task_id(item))
        content_blocks = _mark_task_output_outcome(content_blocks, _task_output_outcome(item))
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
        tool_execution_payload = _tool_execution_result_payload(item)
        if tool_execution_payload is not None:
            session_events.append(
                ParsedSessionEvent(
                    event_type="claude_tool_execution_result",
                    timestamp=timestamp,
                    source_message_provider_id=provider_message_id,
                    payload=tool_execution_payload,
                )
            )
        todo_state_payload = _todo_state_payload(item)
        if todo_state_payload is not None:
            session_events.append(
                ParsedSessionEvent(
                    event_type="claude_todo_state",
                    timestamp=timestamp,
                    source_message_provider_id=provider_message_id,
                    payload=todo_state_payload,
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
                        message=message_payload,
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

    # polylogue-pbuh: flush deduplicated agent-dispatch delegation edges
    # gathered from ``progress``/``agent_progress`` records (see
    # ``_accumulate_delegation_progress``) -- one event per distinct
    # dispatching tool_use, not one per streaming tick.
    for parent_tool_use_id in sorted(delegation_progress):
        stats = delegation_progress[parent_tool_use_id]
        session_events.append(
            ParsedSessionEvent(
                event_type="claude_delegation_progress",
                timestamp=stats.last_seen,
                source_message_provider_id=parent_tool_use_id,
                payload={
                    "parent_tool_use_id": parent_tool_use_id,
                    "progress_tick_count": stats.count,
                    "first_seen": stats.first_seen,
                    "last_seen": stats.last_seen,
                    "summary": f"delegated work under tool_use {parent_tool_use_id} ({stats.count} progress ticks)",
                },
            )
        )

    # ``sessionKind`` classifies the whole file (e.g. "bg" for a
    # run_in_background Bash task's own transcript) -- a session-wide fact,
    # not a per-message one, so it is recorded once rather than per record.
    if session_kind_value is not None:
        session_events.append(
            ParsedSessionEvent(
                event_type="claude_session_kind",
                timestamp=created_at,
                payload={"session_kind": session_kind_value},
            )
        )

    title = str(composed_session_id)
    title_source: TitleSource | None = None
    title_ref: str | None = None
    title_confidence: float | None = None
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
                title_source = TitleSource.HEURISTIC
                title_ref = f"message:{message.provider_message_id}"
                title_confidence = 0.5
                break

    # polylogue-pbuh: Claude Code's own ``ai-title`` sidecar record is a
    # provider-computed session title (Codex's equivalent-tier evidence is its
    # "thread name" -- both get TitleSource.ORIGIN + confidence 1.0). It wins
    # over the first-human-message heuristic above and the raw UUID fallback:
    # it is the reason 84.6% of Claude Code sessions were titled with a raw
    # UUID (bd polylogue-pbuh) even though the provider supplies a real title.
    if latest_ai_title:
        cleaned_ai_title = latest_ai_title.strip()
        if cleaned_ai_title:
            title = cleaned_ai_title[:80] + ("..." if len(cleaned_ai_title) > 80 else "")
            title_source = TitleSource.ORIGIN
            title_ref = f"claude-ai-title:{composed_session_id}"
            title_confidence = 1.0

    # An explicit user rename (``custom-title``) is a stronger intent signal
    # than the provider-suggested ``ai-title`` and wins over it when both are
    # present -- the user deliberately renamed the session, not the provider
    # guessing at one.
    if latest_custom_title:
        cleaned_custom_title = latest_custom_title.strip()
        if cleaned_custom_title:
            title = cleaned_custom_title[:80] + ("..." if len(cleaned_custom_title) > 80 else "")
            title_source = TitleSource.ORIGIN
            title_ref = f"claude-custom-title:{composed_session_id}"
            title_confidence = 1.0
    if title_source is None:
        title_source = TitleSource.UNKNOWN

    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=str(composed_session_id),
        title=title,
        title_source=title_source,
        title_ref=title_ref,
        title_confidence=title_confidence,
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
        git_branch=git_branch_value,
    )


def apply_tool_result_sidecars(session: ParsedSession, join_result: SidecarJoinResult) -> ParsedSession:
    """Attach acquired ``tool-results/`` sidecar content to its owning blocks.

    Never adds a message, never touches session identity/count (polylogue-rujy
    AC1): a genuinely-truncated sidecar's full text replaces its
    ``tool_result`` block's preview text in place; every sidecar (matched or
    debt) is recorded as a bounded ``claude_tool_result_sidecar`` session
    event -- never the raw bytes, which live in the (already unbounded) block
    text field once replaced, not in this structured fact.
    """
    if not join_result.matched and not join_result.debt:
        return session

    replacements = {match.tool_use_id: match for match in join_result.matched if match.was_truncated}
    messages = session.messages
    if replacements:
        updated_messages: list[ParsedMessage] = []
        for message in session.messages:
            if not any(
                block.type is BlockType.TOOL_RESULT and block.tool_id in replacements for block in message.blocks
            ):
                updated_messages.append(message)
                continue
            new_blocks = [
                block.model_copy(update={"text": replacements[block.tool_id].full_text})
                if block.type is BlockType.TOOL_RESULT and block.tool_id in replacements
                else block
                for block in message.blocks
            ]
            updated_messages.append(message.model_copy(update={"blocks": new_blocks}))
        messages = updated_messages

    events = list(session.session_events)
    for match in join_result.matched:
        events.append(
            ParsedSessionEvent(
                event_type="claude_tool_result_sidecar",
                payload={
                    "acquisition_status": "matched",
                    "tool_use_id": match.tool_use_id,
                    "filename": match.filename,
                    "byte_size": match.byte_size,
                    "content_hash": match.content_hash,
                    "content_replaced": match.was_truncated,
                },
            )
        )
    for debt in join_result.debt:
        events.append(
            ParsedSessionEvent(
                event_type="claude_tool_result_sidecar",
                payload={
                    "acquisition_status": "debt",
                    "filename": debt.filename,
                    "byte_size": debt.byte_size,
                    "reason": debt.reason,
                },
            )
        )
    return session.model_copy(update={"messages": messages, "session_events": events})


def parse_code(
    payload: Sequence[object],
    fallback_id: str,
    *,
    tool_result_sidecars: SidecarJoinResult | None = None,
) -> ParsedSession:
    session = _parse_code_records(payload, fallback_id)
    if tool_result_sidecars is not None:
        session = apply_tool_result_sidecars(session, tool_result_sidecars)
    return session


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
    tool_result_sidecars: SidecarJoinResult | None = None,
) -> ParsedSession:
    session = _parse_code_records(
        records,
        fallback_id,
        record_index_start=record_index_start,
        seen_record_uuids=seen_record_uuids,
    )
    if tool_result_sidecars is not None:
        session = apply_tool_result_sidecars(session, tool_result_sidecars)
    return session


__all__ = [
    "apply_tool_result_sidecars",
    "parse_code",
    "parse_code_stream",
    "reconcile_code_session_chunks",
]
