"""Claude Code session parsing helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    PasteBoundary,
    Provider,
    SessionRefKind,
    TitleSource,
    ToolResultUnknownReason,
)
from polylogue.core.timestamps import format_timestamp
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
    ParsedFileEdit,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
    ParsedSessionEvent,
    ParsedSessionRef,
    content_blocks_from_segments,
    text_blocks_prose,
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
# ``_NON_MESSAGE_SIDECAR_RECORD_TYPES`` marks record types that never become a
# ``ParsedMessage`` row (they are not chat content). That is still correct for
# all twelve types below -- the name used to be
# ``_SKIPPED_SIDECAR_RECORD_TYPES``, which stopped being accurate the day most
# of these started persisting as ``session_events`` (polylogue-pbuh); 13 of
# the 15 members below ARE persisted today, so "skipped" described the
# pre-polylogue-pbuh behavior, not the current one. Renamed (polylogue lane,
# audited against the live corpus 2026-07-31) with no compat alias -- this
# repo does not carry old spellings forward. What changed originally
# (polylogue-pbuh, audited 2026-07-29, ~1.17M records total) is that most of
# these ARE evidence-bearing and are no longer silently discarded: they
# persist through ``_sidecar_evidence_payload``/the ``progress`` delegation
# accumulator/the attachment-subtype dispatch below as typed ``session_events``
# (index.db). ``session_events.event_type`` has no CHECK-constrained
# vocabulary (see ``storage/sqlite/archive_tiers/index.py``), so adding new
# event types here is additive data, not a schema change.
#
# Per-type disposition (counts = live corpus, rg single pass, 2026-07-29
# unless noted):
#   ai-title (18,561)              EVIDENCE, wins session title (TitleSource.ORIGIN)
#   agent-name (5,001)             EVIDENCE -> claude_agent_name event, ALSO wins
#                                    session title (TitleSource.ORIGIN, below
#                                    ai-title/custom-title in precedence) --
#                                    corpus-sampled values are readable task
#                                    names ("polylogue-history-rebuild"), not
#                                    the raw UUID this loop otherwise leaves
#   pr-link (20,889)               EVIDENCE -> claude_pr_link event
#   bridge-session (13,594)        EVIDENCE -> claude_bridge_session event
#   file-history-snapshot (34,182) EVIDENCE -> claude_file_history_snapshot event
#   permission-mode (25,882)       EVIDENCE -> claude_permission_mode event (values
#                                    vary: auto/default/acceptEdits/plan/
#                                    bypassPermissions -- real operational signal)
#   last-prompt (37,799)           EVIDENCE -> claude_last_prompt event
#   queue-operation (60,709)       EVIDENCE -> claude_queue_operation event
#   attachment (87,539 as of 2026-07-31 re-audit) EVIDENCE, subtype-dispatched --
#                                    see ``_ATTACHMENT_SUBTYPE_EVENT_TYPES``
#                                    below. The type used to collapse all 20+
#                                    ``attachment.type`` payloads into one
#                                    ``claude_attachment`` event regardless of
#                                    whether the record was a real referenced
#                                    file or a hook success ping; a 2026-07-31
#                                    full-corpus enumeration found 38 distinct
#                                    subtypes (corpus growth/CLI evolution
#                                    since the 20-subtype estimate above), now
#                                    routed to ~22 semantically grouped event
#                                    types plus 4 confirmed-transient subtypes.
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
#                                    OUT OF SCOPE for the 2026-07-31 lane that
#                                    added the attachment-subtype dispatch and
#                                    re-verified init/mode below -- this
#                                    disposition is unchanged and was
#                                    deliberately not revisited.
#   init (0 live occurrences, re-confirmed 2026-07-31 against the full
#         ~11.7K-session-file corpus) TRANSIENT: every corpus occurrence on
#                                    record is a bare {"type": "init"} marker
#                                    with no field beyond ``type``/
#                                    ``sessionId`` -- nothing to lose. Zero
#                                    occurrences both times measured.
#   mode (21,545 as of 2026-07-31 re-audit, up from 20,779 on 2026-07-29 --
#         corpus growth) TRANSIENT: ``mode`` was the literal string "normal"
#                                    in 100% of live records both times
#                                    measured (21,545/21,545 on the 2026-07-31
#                                    pass) -- zero information content in the
#                                    corpus as observed, in contrast to the
#                                    sibling ``permission-mode`` type above
#                                    (kept: 5 distinct values observed). Kept
#                                    transient on repeated evidence, not
#                                    assumption; re-audit if a non-"normal"
#                                    value is ever seen.
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
_NON_MESSAGE_SIDECAR_RECORD_TYPES = frozenset(
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
# ``custom-title``, and ``attachment`` are handled separately in
# ``_parse_code_records`` (progress needs whole-session deduplication;
# ai-title/custom-title feed title resolution as well as an audit-trail
# event; attachment needs subtype-dependent dispatch -- see
# ``_ATTACHMENT_SUBTYPE_EVENT_TYPES`` below). ``init``/``mode`` map to nothing
# (transient, see above).
_SIDECAR_EVENT_TYPES: dict[str, str] = {
    "agent-name": "claude_agent_name",
    "pr-link": "claude_pr_link",
    "bridge-session": "claude_bridge_session",
    "file-history-snapshot": "claude_file_history_snapshot",
    "file-history-delta": "claude_file_history_delta",
    "permission-mode": "claude_permission_mode",
    "last-prompt": "claude_last_prompt",
    "queue-operation": "claude_queue_operation",
    "ai-title": "claude_ai_title",
    "custom-title": "claude_custom_title",
}

# ``attachment.type`` subtype -> session_events.event_type (polylogue lane,
# 2026-07-31). Replaces the single collapsed ``claude_attachment`` bucket: a
# full-corpus enumeration (~11.7K Claude Code session files,
# ``~/.claude/projects``) found 38 distinct ``attachment.type`` payloads
# spanning genuinely different kinds of fact -- a real referenced file's
# content is not the same *kind of thing* as a hook success ping, and both
# were previously indistinguishable at the ``event_type`` level (payload had
# to be JSON-inspected to tell them apart). Grouping rule: subtypes that
# report the same real-world entity (a hook firing, a mode transition, a
# capability/tool-surface delta) share one event_type distinguished by
# payload fields, rather than one event_type per subtype -- that would have
# produced ~38 near-empty event types for what are structurally the same
# few kinds of fact. Real file/reference content gets its own dedicated
# event_type per subtype since each is a structurally distinct artifact
# (a full file vs. an edit snippet vs. a bare path reference).
#
# Counts below are live-corpus occurrences of the *subtype*, 2026-07-31,
# same corpus as the 87,539 total above:
_ATTACHMENT_SUBTYPE_EVENT_TYPES: dict[str, str] = {
    # Real file/reference content -- each subtype is a structurally distinct
    # artifact, so each gets its own event_type (no collapsing these into
    # each other, let alone into the hook/capability buckets below).
    "file": "claude_attachment_file",  # 711: a real referenced file's full content
    "edited_text_file": "claude_attachment_edited_file",  # 2,446: editor-buffer snippet at edit time
    "nested_memory": "claude_attachment_nested_memory",  # 487: a nested CLAUDE.md/memory file's content
    "plan_file_reference": "claude_attachment_plan_reference",  # 43: a plan file's full content
    "compact_file_reference": "claude_attachment_file_reference",  # 460: a bare path reference, no content
    "directory": "claude_attachment_directory_listing",  # 16: a directory listing (path + entry names)
    # Hook lifecycle -- six subtypes report the same real-world entity (a
    # hook firing) distinguished only by outcome; one event_type with the
    # subtype riding in payload["type"] (already present via dict(attachment))
    # avoids six near-identical event types for one entity.
    "hook_success": "claude_hook_event",  # 27,678
    "hook_non_blocking_error": "claude_hook_event",  # 558
    "hook_blocking_error": "claude_hook_event",  # 60
    "hook_cancelled": "claude_hook_event",  # 303
    "hook_system_message": "claude_hook_event",  # 129
    "hook_additional_context": "claude_hook_event",  # 1,003
    # Agent-mode transitions (auto-mode / plan-mode enter-exit-reentry) -- one
    # session-state-transition entity, five lifecycle phases.
    "auto_mode": "claude_agent_mode_event",  # 644
    "auto_mode_exit": "claude_agent_mode_event",  # 130
    "plan_mode": "claude_agent_mode_event",  # 84
    "plan_mode_exit": "claude_agent_mode_event",  # 99
    "plan_mode_reentry": "claude_agent_mode_event",  # 15
    # Capability/tool-surface deltas presented to the model mid-session --
    # bounded to names/counts, not the full injected instruction blocks
    # (which can be large and duplicate what's already visible on the
    # transcript), matching the existing bounded-summary precedent in
    # ``_tool_execution_result_payload``.
    "deferred_tools_delta": "claude_capability_delta",  # 2,856
    "mcp_instructions_delta": "claude_capability_delta",  # 1,050
    "agent_listing_delta": "claude_capability_delta",  # 200
    "skill_listing": "claude_capability_snapshot",  # 2,690
    "invoked_skills": "claude_capability_snapshot",  # 117
    # Small, clean, high-signal single-field events -- real operational
    # state, same tier of evidence as the existing ``permission-mode`` type.
    "output_style": "claude_output_style",  # 10,055
    "command_permissions": "claude_command_permissions",  # 570
    # Queue activity -- semantically the same entity as the top-level
    # ``queue-operation`` record type (see ``_SIDECAR_EVENT_TYPES`` above),
    # just reached via a different code path in the provider; reuses the
    # same event_type with ``operation="queued_command"`` rather than
    # inventing a sibling type for the identical concept.
    "queued_command": "claude_queue_operation",  # 7,387
    # Task/todo evidence.
    "task_status": "claude_task_status",  # 80: one polled background task's status
    "task_reminder": "claude_task_reminder",  # 19,751: 37% carry real todo-list
    #                                            state (id/subject/status/blocks/
    #                                            blockedBy) when non-empty --
    #                                            NOT uniformly noise despite the
    #                                            majority being an empty-list
    #                                            reminder; see the todo_reminder
    #                                            sibling below for the contrast
    #                                            that justifies keeping this one.
    # Misc small, real, evidence-bearing signals.
    "diagnostics": "claude_diagnostics",  # 1,583: bounded per-file diagnostic counts
    "goal_status": "claude_agent_goal_status",  # 367: sentinel goal condition evaluation
    "date_change": "claude_date_change",  # 163: session crossed a calendar day
    "read_truncation_notice": "claude_read_truncation_notice",  # 97: a Read tool output was truncated
    "ultrathink_effort": "claude_agent_effort",  # 11: explicit reasoning-effort signal
    "structured_output": "claude_structured_output",  # 1: rare but potentially meaningful
    "max_turns_reached": "claude_max_turns_reached",  # 1: session hit a turn budget
}

# Subtypes audited and found to carry zero information content in the live
# corpus -- dropped (no event emitted), the same evidentiary bar applied to
# ``init``/``mode`` above, not an assumption:
#   total_tokens_reminder (5,677) -- the literal string
#     "<total_tokens>Infinite tokens left</total_tokens>" in every sample
#     checked; zero variance observed.
#   todo_reminder (5) -- always {"content": [], "itemCount": 0} in every
#     occurrence observed (contrast with the sibling task_reminder above,
#     which is 37% non-empty and kept); volume too small to rule out a
#     future non-empty value, but zero signal in every occurrence measured.
#   context_tip (11) -- CLI feature-adoption UI hints ("try /compact",
#     "you have background agents stopped") aimed at the human operator's
#     product experience, not evidence about the session's actual work.
#   companion_intro (1) -- a novelty/branding record (pet companion name);
#     not operational evidence by any reading.
_ATTACHMENT_TRANSIENT_SUBTYPES = frozenset(
    {
        "total_tokens_reminder",
        "todo_reminder",
        "context_tip",
        "companion_intro",
    }
)

# Fallback for an attachment subtype not in the table above -- e.g. a new
# Claude Code CLI version introducing a 39th subtype. FAIL LOUD: still
# persisted (never silently dropped into the generic bucket, which is
# exactly the collapse this dispatch replaces), tagged with a event_type
# that is greppable/triageable on its own, distinct from every classified
# bucket above.
_ATTACHMENT_UNCLASSIFIED_EVENT_TYPE = "claude_attachment_unclassified"


_SKILL_LISTING_NAME_RE = re.compile(r"^-\s*([\w-]+):", re.MULTILINE)


def _bounded_delta_payload(attachment: Mapping[str, object]) -> dict[str, object]:
    """Bound a capability-delta attachment to names/counts.

    ``deferred_tools_delta``/``mcp_instructions_delta``/``agent_listing_delta``
    carry an ``added*`` name list alongside an ``added*`` body-text list (full
    tool/skill/MCP-server instruction blocks, potentially large and already
    duplicated on disk or in the provider's own capability registry). Keeping
    the names but dropping the body text matches the existing bounded-summary
    precedent in ``_tool_execution_result_payload`` (hunk counts, not full
    diffs) and ``_file_history_snapshot`` handling (file list, not file
    contents).
    """
    added_names: list[str] = []
    added_body_count = 0
    for key, value in attachment.items():
        if not isinstance(value, list):
            continue
        if key.endswith("Names") or key.endswith("Types"):
            added_names.extend(str(v) for v in value if isinstance(v, str))
        elif key.endswith("Lines") or key.endswith("Blocks"):
            added_body_count += len(value)
    return {"added_names": added_names, "added_body_count": added_body_count}


def _bounded_capability_snapshot_payload(attachment: Mapping[str, object]) -> dict[str, object]:
    """Bound a capability-snapshot attachment to names/counts.

    ``skill_listing`` carries one large concatenated-markdown ``content``
    string (all available skills' full descriptions); ``invoked_skills``
    carries a ``skills`` list whose ``content`` field is each skill's full
    body. Both duplicate content that already exists as a skill file on
    disk -- extract just the names (bounded, queryable) rather than persist
    the full text verbatim into every session that loads the skill roster.
    """
    skills = attachment.get("skills")
    if isinstance(skills, list):
        names = [str(skill.get("name")) for skill in skills if isinstance(skill, dict) and skill.get("name")]
        return {"skill_names": names, "skill_count": len(skills)}
    content = attachment.get("content")
    if isinstance(content, str):
        names = _SKILL_LISTING_NAME_RE.findall(content)
        return {"skill_names": names, "skill_count": len(names)}
    return {"skill_names": [], "skill_count": 0}


def _bounded_diagnostics_payload(attachment: Mapping[str, object]) -> dict[str, object]:
    """Bound a diagnostics attachment to per-file counts, not full messages.

    ``diagnostics.files[].diagnostics[]`` carries full Pyright/LSP message
    text, source ranges, and codes per finding -- unbounded in principle (as
    many findings as the language server reports). Persist file-level counts
    (queryable: "how many diagnostics did this session generate, on which
    files") rather than the full message text, matching the
    ``structured_patch_hunk_count`` precedent in ``_tool_execution_result_payload``.
    """
    files = attachment.get("files")
    if not isinstance(files, list):
        return {"file_count": 0, "diagnostic_count": 0, "files": []}
    file_summaries: list[dict[str, object]] = []
    total = 0
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        diagnostics = file_entry.get("diagnostics")
        count = len(diagnostics) if isinstance(diagnostics, list) else 0
        total += count
        uri = file_entry.get("uri")
        file_summaries.append({"uri": uri if isinstance(uri, str) else None, "diagnostic_count": count})
    return {"file_count": len(file_summaries), "diagnostic_count": total, "files": file_summaries}


# Subtypes whose generic ``dict(attachment)`` payload carries unbounded free
# text (full injected instruction blocks, full skill bodies, full diagnostic
# messages) -- these get a dedicated bounded builder instead of the raw
# pass-through every other subtype uses. Real file/reference content
# (file/edited_text_file/nested_memory/plan_file_reference/task_reminder) is
# deliberately NOT in this set: the full text there IS the evidence, not
# duplicated decoration around it.
_ATTACHMENT_BOUNDED_PAYLOAD_BUILDERS: dict[str, Callable[[Mapping[str, object]], dict[str, object]]] = {
    "deferred_tools_delta": _bounded_delta_payload,
    "mcp_instructions_delta": _bounded_delta_payload,
    "agent_listing_delta": _bounded_delta_payload,
    "skill_listing": _bounded_capability_snapshot_payload,
    "invoked_skills": _bounded_capability_snapshot_payload,
    "diagnostics": _bounded_diagnostics_payload,
}


def _attachment_sidecar_event(item: dict[str, object], timestamp: str | None) -> ParsedSessionEvent | None:
    """Build the typed session_event for one ``attachment`` sidecar record.

    Subtype-dispatched (see ``_ATTACHMENT_SUBTYPE_EVENT_TYPES`` above) instead
    of the collapsed single ``claude_attachment`` event_type this replaces.
    Returns ``None`` only when the record carries no usable ``attachment``
    dict, or the subtype is confirmed-transient (``_ATTACHMENT_TRANSIENT_SUBTYPES``).
    """
    attachment = item.get("attachment")
    if not isinstance(attachment, dict):
        return None
    raw_subtype = attachment.get("type")
    subtype = str(raw_subtype) if raw_subtype is not None else None
    if subtype in _ATTACHMENT_TRANSIENT_SUBTYPES:
        return None
    event_type = (
        _ATTACHMENT_SUBTYPE_EVENT_TYPES.get(subtype, _ATTACHMENT_UNCLASSIFIED_EVENT_TYPE)
        if subtype
        else _ATTACHMENT_UNCLASSIFIED_EVENT_TYPE
    )
    if subtype == "queued_command":
        # Fold into the shared claude_queue_operation shape (operation/content)
        # instead of the raw commandMode/prompt field names.
        payload: dict[str, object] = {
            "operation": "queued_command",
            "content": attachment.get("prompt"),
            "command_mode": attachment.get("commandMode"),
        }
    elif subtype is not None and subtype in _ATTACHMENT_BOUNDED_PAYLOAD_BUILDERS:
        payload = _ATTACHMENT_BOUNDED_PAYLOAD_BUILDERS[subtype](attachment)
    else:
        payload = dict(attachment)
    payload["summary"] = subtype or "attachment"
    return ParsedSessionEvent(event_type=event_type, timestamp=timestamp, payload=payload)


def _sidecar_evidence_payload(record_type: str, item: dict[str, object]) -> dict[str, object] | None:
    """Return a typed evidence payload for a non-message sidecar record.

    Returns ``None`` when the specific record instance carries no usable
    signal (e.g. a ``last-prompt`` record whose ``lastPrompt`` field is
    empty) so the caller can skip emitting an empty event. ``attachment`` is
    handled separately by ``_attachment_sidecar_event`` (subtype-dependent
    dispatch), not here.
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
) -> bool:
    """Fold one ``progress``/``agent_progress`` record into its dispatch edge.

    Only ``data.type == "agent_progress"`` carries a genuine dispatcher edge
    (see the classification comment above ``_NON_MESSAGE_SIDECAR_RECORD_TYPES``);
    every occurrence under the same ``parentToolUseID`` is one streaming tick
    of the same subagent dispatch, so ticks are counted and time-bounded
    rather than persisted as one row apiece. Returns whether this record
    instance folded into a dispatch edge (polylogue-pbuh AC5 coverage
    counter) -- the other six ``progress`` subtypes return ``False`` and stay
    genuinely transient, see the classification comment.
    """
    data = item.get("data")
    if not isinstance(data, dict) or data.get("type") != "agent_progress":
        return False
    parent_tool_use_id = _string_field(item, "parentToolUseID")
    if not parent_tool_use_id:
        return False
    entry = accumulator.setdefault(parent_tool_use_id, _DelegationProgressStats())
    entry.count += 1
    if timestamp:
        if entry.first_seen is None or timestamp < entry.first_seen:
            entry.first_seen = timestamp
        if entry.last_seen is None or timestamp > entry.last_seen:
            entry.last_seen = timestamp
    return True


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
    """Classify a record's ``message_type``.

    ``text`` must be the message's TEXT-block-only prose (see
    ``text_blocks_prose``), not the combined multi-segment string
    ``extract_message_text``/``extract_text_from_segments`` produce --
    those fold THINKING/TOOL_USE/TOOL_RESULT segments into the same
    string, which is a different input than
    ``storage.message_type_backfill`` reconstructs from the persisted
    ``blocks`` table (TEXT-type rows only). Passing the combined text
    here silently diverges from the backfill's re-classification of the
    same row (bd polylogue-c831).
    """
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
    record: Mapping[str, object] | None = None,
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
    if record is not None:
        # requestId is the Anthropic API request identifier for this specific
        # call -- a real correlation key for cross-referencing a message
        # against provider-side support/billing records, distinct from any
        # id already captured (uuid is Claude Code's own record id, tool_id
        # is per tool-call). parser-diff triage (2026-07-29 / cgfy) found it
        # unread despite 1,171 occurrences in the sample corpus.
        request_id = record.get("requestId")
        if isinstance(request_id, str) and request_id:
            payload["request_id"] = request_id
        # thinkingMetadata.maxThinkingTokens is the extended-thinking token
        # budget Claude Code configured for this turn -- a real reasoning-
        # effort signal distinct from the actual token counts already in
        # last_token_usage. Low corpus frequency (34 in the cgfy sample) but
        # unambiguous and cheap to carry once this payload is already built.
        thinking_metadata = record.get("thinkingMetadata")
        if isinstance(thinking_metadata, dict):
            max_thinking_tokens = thinking_metadata.get("maxThinkingTokens")
            if (
                isinstance(max_thinking_tokens, int)
                and not isinstance(max_thinking_tokens, bool)
                and max_thinking_tokens >= 0
            ):
                payload["max_thinking_tokens"] = max_thinking_tokens
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


def _file_edit_from_tool_result(item: dict[str, object]) -> ParsedFileEdit | None:
    """Build the real file-edit evidence from ``toolUseResult`` (polylogue-2qx.4 / polylogue-cgfy).

    ``_tool_execution_result_payload`` above only projects a bounded summary
    (hunk count, lines-changed) into ``claude_tool_execution_result`` events;
    the full ``structuredPatch``/``originalFile``/``oldString``/``newString``
    evidence -- what raises a file trajectory from "observed" to
    "checkpointed" -- was captured nowhere. Returns ``None`` when the record's
    ``toolUseResult`` carries no edit-shaped fields at all (e.g. a Bash or Read
    result), so a non-edit tool call never gets a spurious empty row.
    """
    tool_result = item.get("toolUseResult")
    if not isinstance(tool_result, dict):
        return None
    structured_patch = tool_result.get("structuredPatch")
    original_file = tool_result.get("originalFile")
    old_string = tool_result.get("oldString")
    new_string = tool_result.get("newString")
    if structured_patch is None and original_file is None and old_string is None and new_string is None:
        return None
    file_path: str | None = None
    file_info = tool_result.get("file")
    if isinstance(file_info, dict) and isinstance(file_info.get("filePath"), str):
        file_path = file_info["filePath"]
    elif isinstance(tool_result.get("filePath"), str):
        file_path = tool_result["filePath"]
    hunks: list[Mapping[str, object]] | None = None
    if isinstance(structured_patch, list):
        hunks = [hunk for hunk in structured_patch if isinstance(hunk, dict)] or None
    replace_all = tool_result.get("replaceAll")
    user_modified = tool_result.get("userModified")
    return ParsedFileEdit(
        file_path=file_path,
        structured_patch=hunks,
        original_file=original_file if isinstance(original_file, str) else None,
        old_string=old_string if isinstance(old_string, str) else None,
        new_string=new_string if isinstance(new_string, str) else None,
        replace_all=replace_all if isinstance(replace_all, bool) else None,
        user_modified=user_modified if isinstance(user_modified, bool) else None,
    )


def _attach_file_edit(
    content_blocks: list[ParsedContentBlock], file_edit: ParsedFileEdit | None
) -> list[ParsedContentBlock]:
    """Attach file-edit evidence to the record's own TOOL_RESULT block.

    ``toolUseResult`` is a property of the ``user`` record that reports a
    tool's outcome, which is exactly the record whose content carries the
    Anthropic-protocol ``tool_result`` block -- the writer resolves the
    paired TOOL_USE via that block's ``tool_id`` (see
    ``ParsedFileEdit`` docstring / ``_write_file_edits``).
    """
    if file_edit is None:
        return content_blocks
    return [
        block.model_copy(update={"file_edit": file_edit}) if block.type is BlockType.TOOL_RESULT else block
        for block in content_blocks
    ]


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
        block.model_copy(update={"is_error": is_error, "exit_code": exit_code, "outcome_unknown_reason": None})
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
        marked.append(
            block.model_copy(
                update={
                    "metadata": metadata,
                    "is_error": None,
                    "exit_code": None,
                    # polylogue-2qx.4 / polylogue-cuxz.8: the wire's own
                    # is_error=false is positively distrusted here (it only
                    # confirms the background task *started*), not merely
                    # absent -- record the distrust, not a bare unknown.
                    "outcome_unknown_reason": ToolResultUnknownReason.DISTRUSTED.value,
                }
            )
        )
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


@dataclass
class _SessionAccumulator:
    """Per-session accumulator state for one Claude Code record stream.

    Extracted from ``_parse_code_records``'s locals (bd polylogue-taj0o,
    "unify eager and streaming parsers into one incremental multi-way
    merge", Stage 1) so a future multi-way merge can key one instance of
    this per session id instead of assuming exactly one open session per
    call. This stage is a pure mechanical extraction -- ``_fold_code_record``
    and ``_finalize_code_session`` below carry the exact same per-record and
    post-loop logic ``_parse_code_records`` used to hold as ~30 loose local
    variables; only the storage location moved, onto this dataclass's
    fields. ``fallback_id``/``trust_fallback_id``/``is_agent``/
    ``is_acompact`` are per-call identity inputs (constant for the whole
    fold), not accumulated facts, but live here too so ``_finalize_code_session``
    needs only the accumulator to produce a ``ParsedSession``.
    """

    fallback_id: str
    trust_fallback_id: bool = False
    is_agent: bool = False
    is_acompact: bool = False

    messages: list[ParsedMessage] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    seen_uuids: set[str] = field(default_factory=set)
    duplicate_uuid_count: int = 0
    first_duplicate_uuid: str | None = None
    first_duplicate_index: int | None = None
    session_id: str | None = None
    session_events: list[ParsedSessionEvent] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: int = 0
    saw_cost_field: bool = False
    saw_duration_field: bool = False
    has_sidechain: bool = False
    fresh_task_prompt_head: bool = False
    saw_plain_user_head: bool = False
    cwds: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    message_position: int = 0
    background_notifications: list[tuple[ClaudeCodeBackgroundTaskNotification, str | None, str | None]] = field(
        default_factory=list
    )
    # polylogue-pbuh: provider-supplied session title (``ai-title`` sidecar
    # record) and the deduplicated agent-dispatch delegation edges extracted
    # from ``progress``/``agent_progress`` records -- both need whole-session
    # state, so they are accumulated here and applied/flushed after the loop.
    latest_ai_title: str | None = None
    latest_custom_title: str | None = None
    latest_agent_name: str | None = None
    session_kind_value: str | None = None
    git_branch_value: str | None = None
    delegation_progress: dict[str, _DelegationProgressStats] = field(default_factory=dict)
    # polylogue-2qx.4 / polylogue-cgfy: ``slug`` (the human-readable session
    # name Claude Code assigns, e.g. "greedy-squishing-hamming") is stamped on
    # every record of a session file -- main or subagent alike -- once the
    # CLI version emits it at all. Read like ``git_branch_value`` above: first
    # non-empty value wins, since it is constant within one file.
    session_slug_value: str | None = None
    session_refs: list[ParsedSessionRef] = field(default_factory=list)
    # polylogue-pbuh AC5: per-record-type seen/persisted counts for the
    # sidecar types this parser used to drop wholesale, plus a bounded
    # sample of record types that fell all the way through ordinary message
    # parsing to the empty-content drop below with no text/blocks -- so a
    # *future* silently-dropped type is visible in the archive (one
    # ``claude_parse_coverage`` session_event) instead of requiring another
    # rg-the-corpus audit to notice.
    sidecar_seen_counts: dict[str, int] = field(default_factory=dict)
    sidecar_persisted_counts: dict[str, int] = field(default_factory=dict)
    empty_drop_counts: dict[str, int] = field(default_factory=dict)


def _fold_code_record(acc: _SessionAccumulator, index: int, item: dict[str, object]) -> None:
    """Fold one already-dict-typed Claude Code record into ``acc``.

    Exactly the per-record body ``_parse_code_records``'s main loop used to
    run inline; every early ``continue`` in the original loop is a ``return``
    here since each call handles exactly one record.
    """
    if acc.session_kind_value is None:
        raw_session_kind = item.get("sessionKind")
        if isinstance(raw_session_kind, str) and raw_session_kind:
            acc.session_kind_value = raw_session_kind
    if acc.git_branch_value is None:
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
            acc.git_branch_value = raw_git_branch
    if acc.session_slug_value is None:
        raw_slug = item.get("slug")
        if isinstance(raw_slug, str) and raw_slug:
            acc.session_slug_value = raw_slug

    compaction = detect_context_compaction(item)
    if compaction:
        raw_timestamp = compaction.get("timestamp")
        compaction_timestamp = normalize_timestamp(
            raw_timestamp if isinstance(raw_timestamp, str | int | float) else None
        )
        context_compaction = dict(compaction)
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="compaction",
                timestamp=compaction_timestamp,
                payload=context_compaction,
            )
        )
        summary_text = str(context_compaction.get("summary") or "")
        acc.messages.append(
            ParsedMessage(
                # polylogue-slshy: no positional fallback -- empty id lets
                # _message_comparison_id's content-anchor fallback run.
                provider_message_id=str(item.get("uuid") or ""),
                role=Role.SYSTEM,
                text=summary_text,
                timestamp=compaction_timestamp,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)] if summary_text else [],
                message_type=MessageType.SUMMARY,
                position=acc.message_position,
                variant_index=0,
                is_active_path=True,
            )
        )
        acc.message_position += 1
        return

    micro_compaction = detect_micro_compaction(item)
    if micro_compaction is not None:
        raw_micro_timestamp = micro_compaction["timestamp"]
        micro_compaction_timestamp = normalize_timestamp(
            raw_micro_timestamp if isinstance(raw_micro_timestamp, (int, float, str)) else None
        )
        acc.session_events.append(
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
        return

    record_type = item.get("type")
    if not isinstance(record_type, str):
        logger.debug("Skipping invalid record at index %d: missing type", index)
        return

    record_uuid = _string_field(item, "uuid")
    if record_uuid:
        if record_uuid in acc.seen_uuids:
            acc.duplicate_uuid_count += 1
            acc.first_duplicate_uuid = acc.first_duplicate_uuid or record_uuid
            acc.first_duplicate_index = acc.first_duplicate_index or index
            return
        acc.seen_uuids.add(record_uuid)

    if not acc.session_id:
        acc.session_id = _string_field(item, "sessionId")
    raw_record_timestamp = item.get("timestamp")
    timestamp = normalize_timestamp(
        raw_record_timestamp if isinstance(raw_record_timestamp, str | int | float) else None
    )
    message = item.get("message")
    notification = _task_notification_from_record(item, message)

    # These twelve record types are never chat content -- see the
    # classification comment above ``_NON_MESSAGE_SIDECAR_RECORD_TYPES``
    # for why each one either persists as typed ``session_events``
    # evidence (polylogue-pbuh) or stays genuinely transient. ``progress``
    # (hook lifecycle pings, streaming tool-progress ticks, and the one
    # evidence-bearing subtype ``agent_progress``) previously also
    # produced empty ``tool_result``-shaped message rows before the
    # record-type check below existed; see #1617 for that forensic.
    if record_type in _NON_MESSAGE_SIDECAR_RECORD_TYPES:
        acc.sidecar_seen_counts[record_type] = acc.sidecar_seen_counts.get(record_type, 0) + 1
        persisted_this_record = False
        if notification is not None:
            acc.background_notifications.append((notification, record_uuid, timestamp))
        if record_type == "progress":
            persisted_this_record = _accumulate_delegation_progress(item, timestamp, acc.delegation_progress)
        else:
            if record_type == "ai-title":
                ai_title_text = _string_field(item, "aiTitle")
                if ai_title_text:
                    acc.latest_ai_title = ai_title_text
                    persisted_this_record = True
            elif record_type == "custom-title":
                custom_title_text = _string_field(item, "customTitle")
                if custom_title_text:
                    acc.latest_custom_title = custom_title_text
                    persisted_this_record = True
            elif record_type == "agent-name":
                agent_name_text = _string_field(item, "agentName")
                if agent_name_text:
                    acc.latest_agent_name = agent_name_text
                    persisted_this_record = True
            elif record_type == "pr-link":
                # polylogue-2qx.4 / polylogue-cgfy: generalized,
                # tracker-agnostic evidence alongside the existing
                # claude_pr_link session_event (kept for the raw-payload
                # audit trail) -- session_refs is the structured relation
                # cijx.1 and its consumers read instead of regex/time-
                # window PR reconstruction.
                pr_url = _string_field(item, "prUrl")
                if pr_url:
                    raw_pr_number = item.get("prNumber")
                    acc.session_refs.append(
                        ParsedSessionRef(
                            kind=SessionRefKind.PULL_REQUEST.value,
                            url=pr_url,
                            repo=_string_field(item, "prRepository"),
                            number=raw_pr_number
                            if isinstance(raw_pr_number, int) and not isinstance(raw_pr_number, bool)
                            else None,
                        )
                    )
                    persisted_this_record = True
            elif record_type == "attachment":
                # Subtype-dispatched -- see ``_attachment_sidecar_event``
                # and ``_ATTACHMENT_SUBTYPE_EVENT_TYPES`` above. Handled
                # here rather than through the generic
                # ``_SIDECAR_EVENT_TYPES``/``_sidecar_evidence_payload``
                # path below because the event_type itself depends on the
                # nested ``attachment.type``, not just the outer
                # record_type.
                attachment_event = _attachment_sidecar_event(item, timestamp)
                if attachment_event is not None:
                    acc.session_events.append(attachment_event)
                    persisted_this_record = True
            event_type = _SIDECAR_EVENT_TYPES.get(record_type)
            if event_type is not None:
                evidence_payload = _sidecar_evidence_payload(record_type, item)
                if evidence_payload is not None:
                    persisted_this_record = True
                    # Message linkage belongs in the typed field, not buried
                    # in the payload dict: source_message_provider_id is what
                    # the archive joins on, and every other claude_* emitter
                    # here already uses it. Two sidecar payloads carried a
                    # bare "message_id" key instead, leaving the typed field
                    # NULL and the linkage unqueryable.
                    source_message_id = evidence_payload.pop("message_id", None)
                    acc.session_events.append(
                        ParsedSessionEvent(
                            event_type=event_type,
                            timestamp=timestamp,
                            payload=evidence_payload,
                            source_message_provider_id=(
                                str(source_message_id) if source_message_id is not None else None
                            ),
                        )
                    )
        if persisted_this_record:
            acc.sidecar_persisted_counts[record_type] = acc.sidecar_persisted_counts.get(record_type, 0) + 1
        return
    if timestamp:
        acc.created_at = timestamp if acc.created_at is None or timestamp < acc.created_at else acc.created_at
        acc.updated_at = timestamp if acc.updated_at is None or timestamp > acc.updated_at else acc.updated_at

    raw_content = message.get("content") if isinstance(message, dict) else item.get("content")
    text = extract_message_text(raw_content)
    envelope_role = _record_role(item, message)
    content_blocks = _content_blocks_from_record(message, text)
    content_blocks = _mark_background_task_start(content_blocks, _background_task_id(item))
    content_blocks = _mark_task_output_outcome(content_blocks, _task_output_outcome(item))
    content_blocks = _attach_file_edit(content_blocks, _file_edit_from_tool_result(item))
    # bd polylogue-c831: classify from the message's own TEXT-block-only
    # prose, not the combined `text` (which folds in THINKING/TOOL_USE/
    # TOOL_RESULT segments via extract_message_text/
    # extract_text_from_segments) -- the persisted `blocks` table only
    # ever carries TEXT-type rows for classify_text_message_type to see
    # again later (storage.message_type_backfill), so classifying off the
    # combined string here silently drifts from that re-classification.
    message_type = _message_type_from_code_record(item, text_blocks_prose(content_blocks))
    if envelope_role is Role.SYSTEM and message_type is MessageType.MESSAGE:
        message_type = MessageType.CONTEXT
    if not acc.saw_plain_user_head and envelope_role is Role.USER and message_type is MessageType.MESSAGE:
        acc.saw_plain_user_head = True
        acc.fresh_task_prompt_head = _is_fresh_task_prompt_head(item)
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
    # polylogue-2qx.4 / polylogue-cuxz.8: the provider's own terminal-state
    # signal (608,608 occurrences on the wire) -- already read into the
    # ``message_usage`` event payload below; also land it on the message
    # row itself so it feeds ``terminal_state`` directly instead of only
    # riding along in an event's JSON payload.
    raw_stop_reason = message_payload.get("stop_reason")
    msg_stop_reason = raw_stop_reason if isinstance(raw_stop_reason, str) and raw_stop_reason else None
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
            is_agent_session=acc.is_agent,
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
            acc.empty_drop_counts[record_type] = acc.empty_drop_counts.get(record_type, 0) + 1
            return
    # Paste markers only appear in user prompts; restricting detection to the
    # user role avoids false positives from assistant text that quotes a marker.
    paste_spans = _detect_paste_spans(text) if resolved_role == Role.USER else []
    # polylogue-slshy: no positional fallback -- empty id lets
    # _message_comparison_id's content-anchor fallback run instead of a
    # position-derived string that would change identity when array
    # order shifts across re-acquisitions.
    provider_message_id = str(record_uuid or "")
    acc.messages.append(
        ParsedMessage(
            provider_message_id=provider_message_id,
            role=resolved_role,
            text=text or "",
            timestamp=timestamp,
            blocks=content_blocks,
            message_type=message_type,
            material_origin=material_origin,
            parent_message_provider_id=_string_field(item, "parentUuid"),
            position=acc.message_position,
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
            stop_reason=msg_stop_reason,
        )
    )
    acc.session_events.extend(
        _workflow_invocation_events(
            content_blocks,
            source_message_provider_id=provider_message_id,
            timestamp=timestamp,
        )
    )
    tool_execution_payload = _tool_execution_result_payload(item)
    if tool_execution_payload is not None:
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="claude_tool_execution_result",
                timestamp=timestamp,
                source_message_provider_id=provider_message_id,
                payload=tool_execution_payload,
            )
        )
    todo_state_payload = _todo_state_payload(item)
    if todo_state_payload is not None:
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="claude_todo_state",
                timestamp=timestamp,
                source_message_provider_id=provider_message_id,
                payload=todo_state_payload,
            )
        )
    if notification is not None:
        acc.background_notifications.append((notification, provider_message_id, timestamp))
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="message_usage",
                timestamp=timestamp,
                source_message_provider_id=provider_message_id,
                payload=_message_usage_event_payload(
                    msg_usage,
                    model_name=msg_model,
                    model_effort=msg_effort,
                    message=message_payload,
                    record=item,
                ),
            )
        )
    acc.message_position += 1

    if "costUSD" in item:
        acc.saw_cost_field = True
        acc.total_cost += _safe_float(item.get("costUSD"))
    if "durationMs" in item:
        acc.saw_duration_field = True
        acc.total_duration += _safe_int(item.get("durationMs"))
    if item.get("isSidechain"):
        acc.has_sidechain = True
    cwd = item.get("cwd")
    if isinstance(cwd, str):
        acc.cwds.add(cwd)
    model_name = message_payload.get("model")
    if isinstance(model_name, str):
        acc.models.add(model_name)


def _finalize_code_session(acc: _SessionAccumulator) -> ParsedSession:
    """Emit the ``ParsedSession`` for a fully-folded ``_SessionAccumulator``.

    Exactly the post-loop logic ``_parse_code_records`` used to run inline
    after its ``for`` loop: background-completion projection, coverage/
    delegation-progress flush, ``session_kind``, title resolution.
    """
    messages = _project_background_task_completions(
        acc.messages, [notification for notification, _, _ in acc.background_notifications]
    )
    final_background_notifications = {
        (notification.task_id, notification.tool_use_id): (notification, source_message_provider_id, timestamp)
        for notification, source_message_provider_id, timestamp in acc.background_notifications
    }
    for notification, source_message_provider_id, timestamp in final_background_notifications.values():
        acc.session_events.append(
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

    if acc.duplicate_uuid_count:
        logger.debug(
            "Skipped repeated Claude Code record uuids: count=%d first_index=%s first_uuid=%s",
            acc.duplicate_uuid_count,
            acc.first_duplicate_index,
            acc.first_duplicate_uuid,
        )

    # `agent-acompact-*` is overloaded by Claude Code. A main-session compactor
    # replays the parent transcript and is a continuation; a Task subagent can
    # also self-compact under the same filename prefix. A structurally fresh Task
    # prompt is positive evidence for sidechain topology. Ambiguous cases stay
    # continuations here and are reclassified by the archive writer after its
    # bounded content-membership check against the resolved parent transcript.
    parent_session_id: str | None = None
    if acc.is_agent and acc.session_id:
        composed_session_id = f"{acc.session_id}:{acc.fallback_id}"
        parent_session_id = acc.session_id
    elif acc.trust_fallback_id and acc.session_id and acc.session_id != acc.fallback_id:
        # bd polylogue-jc4q: dispatch.py has already proven this run is a
        # resume/fork/usage-limit boundary carryover from an ancestor and
        # anchored ``fallback_id`` to this file's own identity accordingly.
        # Trusting the in-record ``sessionId`` (the ancestor's own id)
        # instead would collide this fragment's revision membership onto
        # the ancestor's own `logical_source_key` -- the ancestor id is
        # recorded as ``parent_session_id`` for lineage (``session_links``)
        # instead.
        composed_session_id = acc.fallback_id
        parent_session_id = acc.session_id
    else:
        composed_session_id = acc.session_id or acc.fallback_id

    if acc.is_acompact and acc.fresh_task_prompt_head:
        branch_type: BranchType | None = BranchType.SIDECHAIN
    elif acc.is_acompact:
        branch_type = BranchType.CONTINUATION
    elif acc.is_agent:
        branch_type = BranchType.SUBAGENT
    elif acc.has_sidechain:
        branch_type = BranchType.SIDECHAIN
    else:
        branch_type = None

    # polylogue-slshy: flag the active leaf by POSITION (the true last
    # message), never by comparing provider_message_id -- with positional
    # id-fallback strings removed (see the fix above), more than one
    # id-less message can legitimately share the same empty
    # provider_message_id, and an equality comparison would flag every one
    # of them, not just the real leaf. Mirrors
    # dispatch.merge_parsed_session_chunks/mark_last_occurrence_as_active_leaf's
    # identical fix for the streaming path (bd polylogue-2hwl).
    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        leaf_index = len(messages) - 1
        messages = [
            message.model_copy(update={"is_active_leaf": index == leaf_index}) for index, message in enumerate(messages)
        ]

    # polylogue-pbuh: flush deduplicated agent-dispatch delegation edges
    # gathered from ``progress``/``agent_progress`` records (see
    # ``_accumulate_delegation_progress``) -- one event per distinct
    # dispatching tool_use, not one per streaming tick.
    for parent_tool_use_id in sorted(acc.delegation_progress):
        stats = acc.delegation_progress[parent_tool_use_id]
        acc.session_events.append(
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
    if acc.session_kind_value is not None:
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="claude_session_kind",
                timestamp=acc.created_at,
                payload={"session_kind": acc.session_kind_value},
            )
        )

    # polylogue-pbuh AC5: one bounded coverage event per session so a future
    # silently-dropped record type is visible without another corpus audit --
    # counts for the known sidecar types (seen vs. actually persisted as
    # evidence, distinguishing e.g. an ``attachment`` record whose payload
    # was empty from one that produced an event) plus a sample of record
    # types that reached ordinary message parsing but carried no text/blocks
    # and were dropped there (the pre-#1617 failure mode this bead's method
    # note warns future readers not to repeat by assumption).
    if acc.sidecar_seen_counts or acc.empty_drop_counts:
        acc.session_events.append(
            ParsedSessionEvent(
                event_type="claude_parse_coverage",
                timestamp=acc.updated_at,
                payload={
                    "sidecar_seen": dict(sorted(acc.sidecar_seen_counts.items())),
                    "sidecar_persisted": dict(sorted(acc.sidecar_persisted_counts.items())),
                    "empty_dropped_by_record_type": dict(sorted(acc.empty_drop_counts.items())),
                },
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

    # polylogue-pbuh: the ``agent-name`` sidecar record is a provider-assigned
    # label for a background/agent-mode session (e.g. "polylogue-history-
    # rebuild") -- corpus-sampled values are readable task names, a strictly
    # better fallback than the raw UUID/"UUID:agent-suffix" composed id this
    # loop otherwise leaves in place. It wins over the raw-id/heuristic title
    # but yields to the stronger explicit signals below (``ai-title`` is the
    # provider's own title computation; ``custom-title`` is an explicit user
    # rename) when either of those is also present for the same session.
    if acc.latest_agent_name:
        cleaned_agent_name = acc.latest_agent_name.strip()
        if cleaned_agent_name:
            title = cleaned_agent_name[:80] + ("..." if len(cleaned_agent_name) > 80 else "")
            title_source = TitleSource.ORIGIN
            title_ref = f"claude-agent-name:{composed_session_id}"
            title_confidence = 0.9

    # polylogue-pbuh: Claude Code's own ``ai-title`` sidecar record is a
    # provider-computed session title (Codex's equivalent-tier evidence is its
    # "thread name" -- both get TitleSource.ORIGIN + confidence 1.0). It wins
    # over the first-human-message heuristic above and the raw UUID fallback:
    # it is the reason 84.6% of Claude Code sessions were titled with a raw
    # UUID (bd polylogue-pbuh) even though the provider supplies a real title.
    if acc.latest_ai_title:
        cleaned_ai_title = acc.latest_ai_title.strip()
        if cleaned_ai_title:
            title = cleaned_ai_title[:80] + ("..." if len(cleaned_ai_title) > 80 else "")
            title_source = TitleSource.ORIGIN
            title_ref = f"claude-ai-title:{composed_session_id}"
            title_confidence = 1.0

    # An explicit user rename (``custom-title``) is a stronger intent signal
    # than the provider-suggested ``ai-title`` and wins over it when both are
    # present -- the user deliberately renamed the session, not the provider
    # guessing at one.
    if acc.latest_custom_title:
        cleaned_custom_title = acc.latest_custom_title.strip()
        if cleaned_custom_title:
            title = cleaned_custom_title[:80] + ("..." if len(cleaned_custom_title) > 80 else "")
            title_source = TitleSource.ORIGIN
            title_ref = f"claude-custom-title:{composed_session_id}"
            title_confidence = 1.0
    # polylogue-5dfu: leave title_source as None (not a TitleSource.UNKNOWN
    # sentinel) when no branch above resolved a title -- NULL already means
    # "no title evidence" for this nullable column, and TitleSource.UNKNOWN
    # was a redundant second spelling of the same fact.

    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=str(composed_session_id),
        title=title,
        title_source=title_source,
        title_ref=title_ref,
        title_confidence=title_confidence,
        created_at=acc.created_at,
        updated_at=acc.updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        session_events=order_session_events(acc.session_events),
        parent_session_provider_id=parent_session_id,
        branch_type=branch_type,
        reported_cost_usd=acc.total_cost if acc.saw_cost_field else None,
        reported_duration_ms=acc.total_duration if acc.saw_duration_field else None,
        models_used=sorted(acc.models),
        working_directories=sorted(acc.cwds),
        git_branch=acc.git_branch_value,
        display_name=acc.session_slug_value,
        session_refs=acc.session_refs,
    )


def _parse_code_records(
    records: Iterable[object],
    fallback_id: str,
    *,
    trust_fallback_id: bool = False,
) -> ParsedSession:
    """Parse Claude Code JSONL payloads into a canonical session model.

    ``trust_fallback_id`` (bd polylogue-jc4q): the caller (dispatch.py's
    ``_claude_code_multiway_parse``) sets this when ``fallback_id`` is a
    composite it has already anchored to THIS file's own identity for a
    resume/fork/usage-limit boundary carryover fragment -- a run of records
    stamped with an ANCESTOR session's id even though it is not that
    ancestor's own content. Ordinary calls leave it ``False`` and keep the
    long-standing default of trusting the record's own ``sessionId`` --
    correct whenever ``fallback_id`` is just a caller-supplied label rather
    than a proven identity (the overwhelmingly common case, including most
    test call sites).

    This is a thin walk-and-fold wrapper (bd polylogue-taj0o Stage 1) over
    ``_fold_code_record``/``_finalize_code_session``, which carry the actual
    per-record and post-loop logic against one ``_SessionAccumulator``. It
    always sees one session's whole record set in a single call -- routing
    the same raw ``records`` iterable to more than one logical session
    (a file with interleaved sessionIds) is dispatch.py's
    ``_claude_code_multiway_parse``'s job (bd polylogue-taj0o Stage 2), which
    keys one ``_SessionAccumulator`` per session id directly via
    ``_fold_code_record``/``_finalize_code_session`` instead of calling this
    function per group.
    """
    acc = _SessionAccumulator(
        fallback_id=fallback_id,
        trust_fallback_id=trust_fallback_id,
        is_agent=fallback_id.startswith("agent-"),
        is_acompact=fallback_id.startswith("agent-acompact-"),
    )
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            continue
        _fold_code_record(acc, index, item)
    return _finalize_code_session(acc)


def apply_tool_result_sidecars(session: ParsedSession, join_result: SidecarJoinResult) -> ParsedSession:
    """Attach acquired ``tool-results/`` sidecar content to its owning blocks.

    Never adds a message, never touches session identity/count (polylogue-rujy
    AC1): a genuinely-truncated sidecar's full text replaces its
    ``tool_result`` block's preview text in place; every sidecar (matched or
    debt) is recorded as a bounded ``claude_tool_result_sidecar`` session
    event -- never the raw bytes, which live in the (already unbounded) block
    text field once replaced, not in this structured fact.

    Each event's ``timestamp`` is set from the sidecar file's own mtime
    (``SidecarMatch``/``SidecarDebt.file_mtime_ms``) so ``occurred_at_ms`` is
    populated on write instead of permanently NULL -- the join has no better
    time source (sidecar files carry no embedded timestamp, and for debt the
    owning block is by definition unresolvable).
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
                timestamp=_sidecar_event_timestamp(match.file_mtime_ms),
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
                timestamp=_sidecar_event_timestamp(debt.file_mtime_ms),
                payload={
                    "acquisition_status": "debt",
                    "filename": debt.filename,
                    "byte_size": debt.byte_size,
                    "reason": debt.reason,
                },
            )
        )
    return session.model_copy(update={"messages": messages, "session_events": events})


def _sidecar_event_timestamp(file_mtime_ms: int | None) -> str | None:
    """Format a sidecar file's mtime (epoch ms) as the ISO timestamp for its session event."""
    if file_mtime_ms is None:
        return None
    return format_timestamp(file_mtime_ms / 1000.0)


# polylogue-4987i: a session's summary session_events -- background
# completions, delegation progress, session_kind, coverage -- are appended
# after ``_fold_code_record``'s main loop, in that fixed order, because
# ``_finalize_code_session`` only sees "no more records" once, at true end
# of input. Sort by (timestamp, event-type tier, encounter order) rather
# than relying on append order alone so the final order depends only on the
# parsed content.
_SESSION_EVENT_TYPE_ORDER_TIER: dict[str, int] = {
    "background_task_completion": 1,
    "claude_delegation_progress": 2,
    "claude_session_kind": 3,
    "claude_parse_coverage": 4,
}


def order_session_events(events: Sequence[ParsedSessionEvent]) -> list[ParsedSessionEvent]:
    """Deterministic session_events order (polylogue-4987i).

    Missing timestamps sort first (stable; matches prior append-order
    behavior for the rare untimestamped event). The trailing encounter index
    is an irreducible tiebreak: two same-type events genuinely stamped at
    the same instant have no other ordering evidence.
    """

    def sort_key(indexed: tuple[int, ParsedSessionEvent]) -> tuple[str, int, int]:
        index, event = indexed
        return (event.timestamp or "", _SESSION_EVENT_TYPE_ORDER_TIER.get(event.event_type, 0), index)

    return [event for _, event in sorted(enumerate(events), key=sort_key)]


def parse_code(
    payload: Iterable[object],
    fallback_id: str,
    *,
    tool_result_sidecars: SidecarJoinResult | None = None,
    trust_fallback_id: bool = False,
) -> ParsedSession:
    """Parse one Claude Code session's whole record set in a single pass.

    Works identically whether ``payload`` is a materialized list or a true
    one-pass iterator (bd polylogue-taj0o Stage 2: mirrors
    ``polylogue/sources/parsers/codex.py``'s ``parse``/``parse_stream`` --
    both names below are the same function). Splitting a raw record stream
    that mixes more than one logical session (interleaved ``sessionId``\\s)
    is dispatch.py's ``_claude_code_multiway_parse``'s job, not this one's.
    """
    session = _parse_code_records(payload, fallback_id, trust_fallback_id=trust_fallback_id)
    if tool_result_sidecars is not None:
        session = apply_tool_result_sidecars(session, tool_result_sidecars)
    return session


parse_code_stream = parse_code


__all__ = [
    "apply_tool_result_sidecars",
    "order_session_events",
    "parse_code",
    "parse_code_stream",
]
