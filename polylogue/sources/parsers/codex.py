"""Codex JSONL session parser."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.provider.semantics import extract_codex_text
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.timestamps import parse_timestamp_pair
from polylogue.logging import get_logger
from polylogue.sources.providers.codex import CodexRecord

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    content_blocks_from_segments,
    fill_linear_parent_chain,
    mark_last_occurrence_as_active_leaf,
    synthetic_message_id,
)

logger = get_logger(__name__)
_TimestampPair = tuple[datetime, str]
#: Tool names whose payload is a patch-format string rather than JSON.
#: Mirrors the ``apply_patch`` child-type aliases registered below, so the
#: standalone ``function_call`` path and the batched code-mode child path
#: recognise exactly the same set.
_PATCH_TOOL_NAMES = frozenset({"apply_patch", "patch"})

_EXECUTION_TOOL_NAMES = frozenset(
    {
        "bash",
        "exec",
        "exec_command",
        "functions.exec",
        "functions.exec_command",
        "local_shell_call",
        "run",
        "shell",
        "shell_command",
        "terminal",
    }
)
_CODE_MODE_EXEC_TOOL_NAMES = frozenset({"exec", "functions.exec"})
_CODE_MODE_CHILD_PROVENANCE_KEY = "_polylogue"
_CODE_MODE_CHILD_ID_MARKER = "::polylogue-child::"
_CODE_MODE_CHILD_COLLECTION_KEYS = (
    "calls",
    "tool_calls",
    "children",
    "operations",
    "actions",
    "invocations",
)
_CODE_MODE_RESULT_COLLECTION_KEYS = (
    "results",
    "tool_results",
    "children",
    "outputs",
    "responses",
)
_STRUCTURAL_PATH_KEYS = frozenset({"path", "file_path", "paths", "file_paths", "image_path"})
_STRUCTURAL_BYTE_KEYS = frozenset({"bytes", "byte_count", "bytes_written", "size_bytes", "written_bytes"})


@dataclass(frozen=True, slots=True)
class _CodexExecChildType:
    kind: str
    aliases: frozenset[str]


_CODE_MODE_CHILD_REGISTRY = (
    _CodexExecChildType(
        kind="exec_command",
        aliases=frozenset(
            {
                "bash",
                "exec_command",
                "local_shell_call",
                "shell",
                "shell_command",
                "terminal",
                "unified_exec",
            }
        ),
    ),
    _CodexExecChildType(kind="apply_patch", aliases=frozenset({"apply_patch", "patch"})),
    _CodexExecChildType(kind="write_stdin", aliases=frozenset({"send_input", "write_stdin"})),
    _CodexExecChildType(kind="update_plan", aliases=frozenset({"plan", "update_plan"})),
    _CodexExecChildType(kind="wait", aliases=frozenset({"wait", "wait_for_cell", "wait_for_process"})),
    _CodexExecChildType(
        kind="web",
        aliases=frozenset(
            {
                "open_url",
                "search",
                "tool_search",
                "web",
                "web_open",
                "web_search",
            }
        ),
    ),
    _CodexExecChildType(
        kind="image",
        aliases=frozenset(
            {
                "generated_image",
                "image",
                "image_generation",
                "image_query",
                "view_image",
            }
        ),
    ),
    _CodexExecChildType(
        kind="mcp",
        aliases=frozenset(
            {
                "list_mcp_resource_templates",
                "list_mcp_resources",
                "read_mcp_resource",
            }
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _CodexExecChildCall:
    tool_path: tuple[str, ...]
    tool_name: str
    registry_type: str
    argument: object
    raw_argument: str | None
    parse_state: str
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True, slots=True)
class _CodexExecChildResult:
    raw: object
    text: str | None
    is_error: bool | None
    exit_code: int | None
    paths: tuple[str, ...]
    byte_count: int | None


@dataclass(frozen=True, slots=True)
class _CodexExecEnvelope:
    transport_tool_name: str
    transport_tool_id: str | None
    transport_provider_message_id: str
    children: tuple[_CodexExecChildCall, ...]
    results: tuple[_CodexExecChildResult, ...] = ()


class _JsLiteralError(ValueError):
    pass


def _iso_or_none(value: str | int | float | None) -> str | None:
    pair = parse_timestamp_pair(value)
    return pair[1] if pair is not None else None


def _newer_timestamp(
    current: _TimestampPair | None,
    value: str | None,
) -> _TimestampPair | None:
    if not isinstance(value, str) or not value:
        return current
    return _newer_timestamp_pair(current, parse_timestamp_pair(value))


def _newer_timestamp_pair(
    current: _TimestampPair | None,
    candidate: _TimestampPair | None,
) -> _TimestampPair | None:
    if candidate is None:
        return current
    if current is None or candidate[0] > current[0]:
        return candidate
    return current


def _has_continuation_evidence(
    *,
    first_timestamp: _TimestampPair | None,
    second_timestamp: _TimestampPair | None,
    first_cwd: str | None,
    second_cwd: str | None,
    first_repo_url: str | None,
    second_repo_url: str | None,
) -> bool:
    """Structural test for the legacy (no `forked_from_id`) continuation fallback.

    A resumed Codex session physically replays the parent conversation's own
    original `session_meta` record as the file's second distinct `session_meta`
    (verified against real multi-meta rollout files: the replayed header's
    timestamp always *precedes* the new session's own start time, and reports
    the same `cwd`/git remote, because the resumed conversation continues in
    the same working tree). A bare count of session_meta records proves
    neither fact -- two structurally unrelated session_meta records
    concatenated into one payload would satisfy the count without satisfying
    this check, so the count alone is not sufficient evidence of a parent
    relationship.
    """
    if first_timestamp is None or second_timestamp is None:
        return False
    if second_timestamp[0] > first_timestamp[0]:
        # The candidate parent's own header postdates the child's -- not a
        # replayed prefix.
        return False
    cwd_match = bool(first_cwd) and bool(second_cwd) and first_cwd == second_cwd
    repo_match = bool(first_repo_url) and bool(second_repo_url) and first_repo_url == second_repo_url
    return cwd_match or repo_match


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
    if record_type in {"session_meta", "response_item", "event_msg", "compacted", "turn_context"}:
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
    return value if isinstance(value, str | int | float) else None


def _message_timestamp(record: dict[str, object], message_record: dict[str, object]) -> str | int | float | None:
    return _record_timestamp(message_record) or _record_timestamp(record)


def _record_instructions(record: dict[str, object]) -> str | None:
    value = record.get("instructions")
    return value if isinstance(value, str) else None


def _string_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_field(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        if value := _string_value(record.get(key)):
            return value
    return None


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip():
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return 0
    return 0


def _optional_int_field(record: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        if key in record:
            return _int_value(record.get(key))
    return None


def _codex_token_usage_payload(record: dict[str, object] | None) -> dict[str, int]:
    if not record:
        return {}
    usage: dict[str, int] = {}
    field_aliases = {
        "input_tokens": ("input_tokens", "inputTokenCount"),
        "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens", "cached_tokens"),
        "cache_write_tokens": ("cache_write_tokens", "cache_creation_input_tokens", "cache_write_input_tokens"),
        "uncached_input_tokens": ("uncached_input_tokens", "uncachedInputTokens"),
        "output_tokens": ("output_tokens", "outputTokenCount"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "reasoning_tokens"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
    }
    for public_key, aliases in field_aliases.items():
        value = _optional_int_field(record, *aliases)
        if value is not None:
            usage[public_key] = value
    return usage


def _turn_context_payload(payload: dict[str, object]) -> dict[str, object]:
    nested = payload.get("turn_context")
    if isinstance(nested, dict):
        merged = {str(key): value for key, value in nested.items()}
        merged.update({str(key): value for key, value in payload.items() if key != "turn_context"})
        return merged
    return payload


def _token_usage(record: dict[str, object]) -> dict[str, int]:
    """Extract per-message usage as disjoint additive pricing lanes.

    Codex input includes cache reads, while message pricing bills fresh input
    and cache reads separately. Normalize at the source; Claude already reports
    these lanes disjointly. The event rollup applies the same rule in
    ``_provider_usage_disjoint_lanes``.
    """
    usage = _dict_record(record.get("usage")) or _dict_record(record.get("tokens")) or record
    input_with_cached = _int_value(usage.get("input_tokens") or usage.get("inputTokenCount"))
    explicit_uncached_input = _optional_int_field(usage, "uncached_input_tokens", "uncachedInputTokens")
    cache_read_tokens = _int_value(
        usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cached_input_tokens")
        or usage.get("cached_tokens")
    )
    return {
        "input_tokens": (
            explicit_uncached_input
            if explicit_uncached_input is not None
            else max(input_with_cached - cache_read_tokens, 0)
        ),
        "output_tokens": _int_value(usage.get("output_tokens") or usage.get("outputTokenCount")),
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": _int_value(
            usage.get("cache_write_tokens")
            or usage.get("cache_creation_input_tokens")
            or usage.get("cache_write_input_tokens")
        ),
    }


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


def _compact_response_payload(
    payload: dict[str, object],
    *,
    index: int,
    current_model_name: str | None = None,
    current_model_effort: str | None = None,
) -> dict[str, object]:
    compact: dict[str, object] = {"source_index": index}
    for key in ("type", "id", "call_id", "name", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            compact[key] = value
    timestamp = _record_timestamp(payload)
    if timestamp is not None:
        compact["timestamp"] = timestamp
    output = payload.get("output")
    if isinstance(output, str):
        compact["output_chars"] = len(output)
    elif output is not None:
        compact["has_output"] = True
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        compact["argument_chars"] = len(arguments)
    elif arguments is not None:
        compact["has_arguments"] = True
    cwd = _extract_cwd(payload)
    if cwd:
        compact["cwd"] = cwd
    # `metadata.turn_id` correlates a response_item/event_msg record (most
    # commonly `reasoning`) back to the turn that produced it. Small, always
    # present or absent as a single scalar -- safe to carry on every event.
    metadata = _dict_record(payload.get("metadata"))
    if metadata:
        turn_id = metadata.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            compact["turn_id"] = turn_id
    if compact.get("type") == "token_count":
        if current_model_name and not _string_field(compact, "model", "model_name"):
            compact["model"] = current_model_name
        if current_model_effort:
            compact["model_effort"] = current_model_effort
        info = _dict_record(payload.get("info")) or {}
        last_usage = _codex_token_usage_payload(_dict_record(info.get("last_token_usage")))
        total_usage = _codex_token_usage_payload(_dict_record(info.get("total_token_usage")))
        if not last_usage:
            last_usage = _codex_token_usage_payload(_dict_record(payload.get("last_token_usage")) or payload)
        if not total_usage:
            total_usage = _codex_token_usage_payload(_dict_record(payload.get("total_token_usage")))
        if last_usage:
            compact["last_token_usage"] = last_usage
        if total_usage:
            compact["total_token_usage"] = total_usage
        # Rate-limit windows are quota telemetry Codex reports alongside each
        # token_count tick -- small, bounded, and otherwise invisible.
        rate_limits = _dict_record(payload.get("rate_limits"))
        if rate_limits:
            compact_rate_limits: dict[str, object] = {}
            for lane in ("primary", "secondary"):
                window = _dict_record(rate_limits.get(lane))
                if not window:
                    continue
                lane_payload: dict[str, object] = {}
                used_percent = window.get("used_percent")
                if isinstance(used_percent, int | float) and not isinstance(used_percent, bool):
                    lane_payload["used_percent"] = used_percent
                for int_key in ("window_minutes", "resets_in_seconds"):
                    int_value = _optional_int_field(window, int_key)
                    if int_value is not None:
                        lane_payload[int_key] = int_value
                if lane_payload:
                    compact_rate_limits[lane] = lane_payload
            if compact_rate_limits:
                compact["rate_limits"] = compact_rate_limits
    elif compact.get("type") == "ghost_snapshot":
        # Codex's shadow-git snapshot (undo/diff tracking): the commit id,
        # its parent, and pre-existing untracked paths at snapshot time.
        ghost_commit = _dict_record(payload.get("ghost_commit"))
        if ghost_commit:
            compact["ghost_commit"] = dict(ghost_commit)
    elif compact.get("type") in {"exec_command_begin", "exec_command_end"}:
        # `aggregated_output`/`formatted_output`/`stdout`/`stderr` duplicate
        # text already captured verbatim as the paired function_call_output's
        # tool_result text (same call_id, same command output, just wrapped
        # with different transport metadata) -- deliberately not re-stored
        # here. `process_id` and `parsed_cmd` (Codex's own command
        # classification: search/read/etc with extracted query/path) are new.
        process_id = payload.get("process_id")
        if isinstance(process_id, str) and process_id:
            compact["process_id"] = process_id
        parsed_cmd = payload.get("parsed_cmd")
        if isinstance(parsed_cmd, list) and parsed_cmd:
            compact["parsed_cmd"] = parsed_cmd
    elif compact.get("type") in {"patch_apply_begin", "patch_apply_end"}:
        # `success`/`changes` were completely unread (polylogue-cgfy triage,
        # codex lane): the tool_use block on the paired function_call already
        # stores the *requested* patch text verbatim
        # (`_codex_tool_input`/`_PATCH_TOOL_NAMES`), and `stdout` duplicates a
        # human-readable summary already captured as the function_call_output
        # tool_result text -- same dedup rule as exec_command above. `changes`
        # is materially different from both: it is codex's own post-apply
        # per-file classification (add/update/delete + `unified_diff` +
        # `move_path` for renames), the structural equivalent of Claude
        # Code's `structuredPatch` (polylogue-cgfy). Nothing upstream
        # decomposes it, so it is retained verbatim, keyed by path, alongside
        # the boolean apply outcome.
        success = payload.get("success")
        if isinstance(success, bool):
            compact["success"] = success
        changes = _dict_record(payload.get("changes"))
        if changes:
            compact_changes: dict[str, object] = {}
            for changed_path, change in changes.items():
                change_record = _dict_record(change)
                if change_record is None:
                    continue
                entry: dict[str, object] = {}
                change_type = change_record.get("type")
                if isinstance(change_type, str) and change_type:
                    entry["type"] = change_type
                unified_diff = change_record.get("unified_diff")
                if isinstance(unified_diff, str) and unified_diff:
                    entry["unified_diff"] = unified_diff
                move_path = change_record.get("move_path")
                if isinstance(move_path, str) and move_path:
                    entry["move_path"] = move_path
                if entry:
                    compact_changes[str(changed_path)] = entry
            if compact_changes:
                compact["changes"] = compact_changes
    return compact


# response_item/event_msg inner ``type`` values that reach the generic
# session_event dispatch below (the `else` branch: not a message, not
# `compacted`/`turn_context`/`world_state`, no dedicated handler elsewhere in
# this file) with an explicit classification, rather than being stored
# verbatim under Codex's own wire name with no code aware of what that name
# means. Producer/consumer audit, polylogue-fuky (2026-08-02): every type
# below was previously an unaudited passthrough -- zero literal reference
# anywhere in the repo -- discovered via a live-archive `session_events`
# scan finding large row counts (318K-3K) under Codex wire names no code
# branches on. Each entry here has been read against raw wire samples (not
# inferred from row shape); this table does not change the *stored*
# `event_type` string for any of them -- every one still passes through
# under its own wire name (see ``_codex_response_item_event_type`` below).
# What changes is that a type NOT in this set (a future, never-examined
# Codex wire addition) no longer silently joins the same vocabulary
# unclassified -- it is routed to ``_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE``
# instead (fail loud/greppable, matching the
# ``_ATTACHMENT_UNCLASSIFIED_EVENT_TYPE`` precedent in
# ``sources/parsers/claude/code_parser.py``).
#
# TRANSIENT -- confirmed zero information content beyond the bare
# ``{"type": ...}`` marker already captured generically, read directly off
# raw wire samples:
#   context_compacted -- literal ``{"type": "context_compacted"}`` in every
#     sample read (live archive + source JSONL). Distinct from the
#     ``compacted`` record type handled explicitly above (~line 2148),
#     which DOES carry ``replacement_history`` -- this is a bare completion
#     marker for a separate/newer compaction notification path with
#     nothing else on the wire to capture.
#   agent_reasoning -- confirmed DUPLICATE, not merely transient: live-wire
#     comparison across three Codex sessions found `agent_reasoning.text`
#     values are the same live-streamed reasoning-summary bullets already
#     carried in full by the `reasoning` record's `summary[].text` (one
#     session: 156/156 identical; a second: 262/262 identical set; a third:
#     1,846 reasoning bullets vs. 1,859 agent_reasoning ticks, >99% overlap,
#     the residual being minor text-normalization differences on the same
#     underlying bullets, not new content). `reasoning` records are already
#     materialized as a THINKING-block ``ParsedMessage`` via
#     ``_codex_reasoning_message`` (index v50) -- `agent_reasoning` is a
#     live per-tick echo of that same content, matching the
#     "streaming ticks superseded by the final record" pattern documented
#     for Claude Code's `progress` subtypes in `claude/code_parser.py`. The
#     session_event this file emits for it is filtered back out at write
#     time (`_SESSION_EVENTS_REDUNDANT_TYPES` in
#     `storage/sqlite/archive_tiers/write.py`) rather than at parse time, to
#     keep this file's classification table describing what the WIRE means
#     (still a real, known type) separately from the WRITER's
#     zero-evidence-loss dedup decision. ``reasoning`` itself is NOT
#     reclassified here -- it already has a confirmed message consumer and
#     is out of scope for this audit.
#
# EVIDENCE, currently under-captured by ``_compact_response_payload`` above
# (its generic lift only covers type/id/call_id/name/status/timestamp/
# output-or-argument-length/cwd/metadata.turn_id -- every field named below
# is real signal read off the raw wire that falls outside that allowlist
# and is silently dropped today, not merely "not yet interesting"). Kept
# passing through under their own wire name for now; a dedicated extraction
# pass for this cluster is out of scope for a classification-only audit
# (each needs its own bounded field-set decision plus an
# INDEX_SCHEMA_VERSION SEMANTIC_REPARSE bump) and is tracked as a follow-up
# (see the bead filed alongside this change):
#   thread_goal_updated (45,814 live rows) -- `goal.objective` (free text
#     session objective), `goal.status`/`tokensUsed`/`timeUsedSeconds`.
#   sub_agent_activity (41,769) -- `agent_thread_id`, `agent_path`, `kind`
#     (e.g. "interacted") -- subagent delegation evidence.
#   task_started (20,432) -- `turn_id`, `model_context_window`,
#     `collaboration_mode_kind`.
#   task_complete (17,055) -- `turn_id`, `last_agent_message`.
#   turn_aborted (3,394) -- `turn_id`, `reason` (e.g. "interrupted").
#   thread_settings_applied (3,548) -- full per-turn settings snapshot
#     (model, reasoning_effort, personality, collaboration_mode including
#     `developer_instructions` text) -- partially overlaps the
#     `turn_context` capture above (~line 2232) but is a distinct wire
#     record, not yet cross-checked for full redundancy.
#   collab_agent_spawn_end / collab_waiting_end / collab_close_end /
#     collab_agent_interaction_end (~1,000 combined) -- subagent-delegation
#     evidence (new_thread_id, new_agent_nickname, new_agent_role, prompt,
#     receiver_thread_id, receiver_agent_nickname/role, status text) beyond
#     the call_id/status the generic compactor already lifts.
#   item_completed (199) -- `item.text` (e.g. full plan content).
#   entered_review_mode / exited_review_mode (13 each) --
#     `target.instructions`/`user_facing_hint` and
#     `review_output.findings`/`overall_correctness`/`overall_explanation`/
#     `overall_confidence_score`.
#   view_image_tool_call (62) -- `path` (the referenced image file).
#   web_search_end (1,357) -- `query`, `action.queries` -- a distinct
#     completion-marker wire shape from the already-handled
#     `web_search_call`/`web_search_output` pair (different call_id
#     namespace, `ws_...`); real search-query evidence, currently dropped.
#   thread_rolled_back (92) -- `num_turns` (rollback extent).
#   error (42) -- `message` (user-facing text, e.g. a usage-limit message)
#     and `codex_error_info` (a real error code, e.g.
#     "usage_limit_exceeded") -- operationally significant and currently
#     dropped entirely.
# Pre-existing types already dispatched/consumed elsewhere in this file
# (``_compact_response_payload``'s own elif chain above, ``_codex_tool_message``,
# ``_codex_event_message``, ``_codex_reasoning_message``,
# ``_codex_mcp_tool_call_messages``) -- unaffected by this audit, listed here
# only so the classifier below has a complete allowlist and none of them are
# ever misrouted into the unclassified bucket:
_CODEX_PRIOR_AUDITED_RESPONSE_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "token_count",
        "message_usage",
        "ghost_snapshot",
        "exec_command_begin",
        "exec_command_end",
        "patch_apply_begin",
        "patch_apply_end",
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_output",
        "web_search_call",
        "web_search_output",
        "local_shell_call",
        "user_message",
        "agent_message",
        "mcp_tool_call_end",
    }
)

_CODEX_KNOWN_RESPONSE_ITEM_TYPES: frozenset[str] = _CODEX_PRIOR_AUDITED_RESPONSE_ITEM_TYPES | frozenset(
    {
        "context_compacted",
        "agent_reasoning",
        "thread_goal_updated",
        "sub_agent_activity",
        "task_started",
        "task_complete",
        "turn_aborted",
        "thread_settings_applied",
        "collab_agent_spawn_end",
        "collab_waiting_end",
        "collab_close_end",
        "collab_agent_interaction_end",
        "item_completed",
        "entered_review_mode",
        "exited_review_mode",
        "view_image_tool_call",
        "web_search_end",
        "thread_rolled_back",
        "error",
    }
)

# Fallback for a response_item/event_msg inner type not in the table above --
# e.g. a new Codex CLI version introducing a wire shape this repo has never
# read. FAIL LOUD: still persisted (never silently merged into the audited
# vocabulary above, which is exactly the "unaudited passthrough" defect this
# classification replaces), tagged with an event_type that is greppable/
# triageable on its own. The original wire type string is not lost -- it
# stays in the event payload's own ``type`` field (``_compact_response_payload``
# always lifts it when present).
_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE = "codex_unclassified_response_item"


def _codex_response_item_event_type(inner_type: str | None, record_type: str | None) -> str:
    """Classify a response_item/event_msg inner ``type`` for ``session_events``.

    Every type in ``_CODEX_KNOWN_RESPONSE_ITEM_TYPES`` has been read against
    raw wire samples and passes through under its own Codex wire name
    unchanged. A type this repo has never examined is routed to
    ``_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE`` instead of silently adopting
    its own wire name, so an unaudited type can never again commingle with
    the audited vocabulary above without a human noticing the greppable
    marker.
    """
    resolved = inner_type or record_type
    if resolved is None:
        return "response_item"
    if resolved in _CODEX_KNOWN_RESPONSE_ITEM_TYPES:
        return resolved
    return _CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE


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


def _is_js_identifier_start(char: str) -> bool:
    return char == "_" or char == "$" or char.isalpha()


def _is_js_identifier_part(char: str) -> bool:
    return _is_js_identifier_start(char) or char.isdigit()


class _JsLiteralParser:
    """Conservative parser for the JSON-like argument literals used by Code Mode.

    This deliberately accepts only literals. Expressions, interpolation, spreads,
    and references stay as raw evidence instead of being evaluated or guessed.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> object:
        self._skip_space_and_comments()
        value = self._parse_value()
        self._skip_space_and_comments()
        if self.position != len(self.text):
            raise _JsLiteralError("trailing JavaScript expression")
        return value

    def _peek(self) -> str | None:
        if self.position >= len(self.text):
            return None
        return self.text[self.position]

    def _skip_space_and_comments(self) -> None:
        while self.position < len(self.text):
            char = self.text[self.position]
            if char.isspace():
                self.position += 1
                continue
            if self.text.startswith("//", self.position):
                newline = self.text.find("\n", self.position + 2)
                self.position = len(self.text) if newline == -1 else newline + 1
                continue
            if self.text.startswith("/*", self.position):
                end = self.text.find("*/", self.position + 2)
                if end == -1:
                    raise _JsLiteralError("unterminated JavaScript comment")
                self.position = end + 2
                continue
            return

    def _parse_value(self) -> object:
        char = self._peek()
        if char is None:
            raise _JsLiteralError("missing JavaScript literal")
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char in {'"', "'", "`"}:
            return self._parse_string()
        if char == "-" or char.isdigit():
            return self._parse_number()
        if _is_js_identifier_start(char):
            identifier = self._parse_identifier()
            if identifier == "true":
                return True
            if identifier == "false":
                return False
            if identifier in {"null", "undefined"}:
                return None
            raise _JsLiteralError(f"non-literal JavaScript identifier: {identifier}")
        raise _JsLiteralError(f"unsupported JavaScript literal token: {char}")

    def _parse_object(self) -> dict[str, object]:
        self.position += 1
        result: dict[str, object] = {}
        self._skip_space_and_comments()
        if self._peek() == "}":
            self.position += 1
            return result
        while True:
            self._skip_space_and_comments()
            key_char = self._peek()
            if key_char in {'"', "'", "`"}:
                key_value = self._parse_string()
                if not isinstance(key_value, str):
                    raise _JsLiteralError("object key is not text")
                key = key_value
            elif key_char is not None and _is_js_identifier_start(key_char):
                key = self._parse_identifier()
            else:
                raise _JsLiteralError("unsupported JavaScript object key")
            self._skip_space_and_comments()
            if self._peek() != ":":
                raise _JsLiteralError("JavaScript object shorthand is not a literal")
            self.position += 1
            self._skip_space_and_comments()
            result[key] = self._parse_value()
            self._skip_space_and_comments()
            delimiter = self._peek()
            if delimiter == "}":
                self.position += 1
                return result
            if delimiter != ",":
                raise _JsLiteralError("missing JavaScript object delimiter")
            self.position += 1
            self._skip_space_and_comments()
            if self._peek() == "}":
                self.position += 1
                return result

    def _parse_array(self) -> list[object]:
        self.position += 1
        result: list[object] = []
        self._skip_space_and_comments()
        if self._peek() == "]":
            self.position += 1
            return result
        while True:
            self._skip_space_and_comments()
            result.append(self._parse_value())
            self._skip_space_and_comments()
            delimiter = self._peek()
            if delimiter == "]":
                self.position += 1
                return result
            if delimiter != ",":
                raise _JsLiteralError("missing JavaScript array delimiter")
            self.position += 1
            self._skip_space_and_comments()
            if self._peek() == "]":
                self.position += 1
                return result

    def _parse_identifier(self) -> str:
        start = self.position
        if self.position >= len(self.text) or not _is_js_identifier_start(self.text[self.position]):
            raise _JsLiteralError("missing JavaScript identifier")
        self.position += 1
        while self.position < len(self.text) and _is_js_identifier_part(self.text[self.position]):
            self.position += 1
        return self.text[start : self.position]

    def _parse_string(self) -> str:
        quote = self.text[self.position]
        self.position += 1
        parts: list[str] = []
        while self.position < len(self.text):
            char = self.text[self.position]
            self.position += 1
            if char == quote:
                return "".join(parts)
            if quote == "`" and char == "$" and self._peek() == "{":
                raise _JsLiteralError("template interpolation is not a literal")
            if char != "\\":
                parts.append(char)
                continue
            if self.position >= len(self.text):
                raise _JsLiteralError("unterminated JavaScript string escape")
            escaped = self.text[self.position]
            self.position += 1
            escapes = {
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "v": "\v",
                "0": "\0",
                "\\": "\\",
                "'": "'",
                '"': '"',
                "`": "`",
            }
            if escaped in escapes:
                parts.append(escapes[escaped])
                continue
            if escaped in {"\n", "\r"}:
                if escaped == "\r" and self._peek() == "\n":
                    self.position += 1
                continue
            if escaped == "x":
                parts.append(self._parse_hex_escape(2))
                continue
            if escaped == "u":
                if self._peek() == "{":
                    self.position += 1
                    end = self.text.find("}", self.position)
                    if end == -1:
                        raise _JsLiteralError("unterminated JavaScript Unicode escape")
                    token = self.text[self.position : end]
                    self.position = end + 1
                    try:
                        parts.append(chr(int(token, 16)))
                    except (ValueError, OverflowError) as exc:
                        raise _JsLiteralError("invalid JavaScript Unicode escape") from exc
                else:
                    parts.append(self._parse_hex_escape(4))
                continue
            # JavaScript treats an otherwise-unknown escaped character as the
            # character itself. Preserving it is safer than rejecting evidence.
            parts.append(escaped)
        raise _JsLiteralError("unterminated JavaScript string")

    def _parse_hex_escape(self, width: int) -> str:
        token = self.text[self.position : self.position + width]
        if len(token) != width or any(char not in "0123456789abcdefABCDEF" for char in token):
            raise _JsLiteralError("invalid JavaScript hexadecimal escape")
        self.position += width
        return chr(int(token, 16))

    def _parse_number(self) -> int | float:
        match = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", self.text[self.position :])
        if match is None:
            raise _JsLiteralError("invalid JavaScript number")
        token = match.group(0)
        self.position += len(token)
        return float(token) if any(char in token for char in ".eE") else int(token)


def _parse_js_literal(text: str) -> tuple[object, bool]:
    candidate = text.strip()
    if not candidate:
        return None, True
    try:
        return json.loads(candidate), True
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return _JsLiteralParser(candidate).parse(), True
    except _JsLiteralError:
        return candidate, False


def _skip_js_string_or_comment(source: str, position: int) -> int | None:
    if source.startswith("//", position):
        newline = source.find("\n", position + 2)
        return len(source) if newline == -1 else newline + 1
    if source.startswith("/*", position):
        end = source.find("*/", position + 2)
        return len(source) if end == -1 else end + 2
    if position >= len(source) or source[position] not in {'"', "'", "`"}:
        return None
    quote = source[position]
    position += 1
    while position < len(source):
        char = source[position]
        position += 1
        if char == "\\":
            position += 1
            continue
        if char == quote:
            return position
    return len(source)


def _skip_js_space_and_comments(source: str, position: int) -> int:
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        skipped = _skip_js_string_or_comment(source, position)
        if skipped is not None and source.startswith(("//", "/*"), position):
            position = skipped
            continue
        return position
    return position


def _parse_js_member_chain(source: str, position: int) -> tuple[tuple[str, ...], int] | None:
    if position >= len(source) or not _is_js_identifier_start(source[position]):
        return None
    start = position
    position += 1
    while position < len(source) and _is_js_identifier_part(source[position]):
        position += 1
    parts = [source[start:position]]
    while True:
        position = _skip_js_space_and_comments(source, position)
        if position < len(source) and source[position] == ".":
            position = _skip_js_space_and_comments(source, position + 1)
            if position >= len(source) or not _is_js_identifier_start(source[position]):
                return tuple(parts), position
            start = position
            position += 1
            while position < len(source) and _is_js_identifier_part(source[position]):
                position += 1
            parts.append(source[start:position])
            continue
        if position < len(source) and source[position] == "[":
            member_start = _skip_js_space_and_comments(source, position + 1)
            if member_start >= len(source) or source[member_start] not in {'"', "'", "`"}:
                return tuple(parts), position
            parser = _JsLiteralParser(source[member_start:])
            try:
                member = parser._parse_string()
            except _JsLiteralError:
                return tuple(parts), position
            member_end = member_start + parser.position
            member_end = _skip_js_space_and_comments(source, member_end)
            if member_end >= len(source) or source[member_end] != "]":
                return tuple(parts), position
            parts.append(member)
            position = member_end + 1
            continue
        return tuple(parts), position


def _balanced_js_call_argument(source: str, open_position: int) -> tuple[str, int, bool]:
    depth = 1
    position = open_position + 1
    argument_start = position
    while position < len(source):
        skipped = _skip_js_string_or_comment(source, position)
        if skipped is not None:
            position = skipped
            continue
        char = source[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[argument_start:position], position + 1, True
        position += 1
    return source[argument_start:], len(source), False


def _first_js_argument(arguments: str) -> str:
    depths = {"(": 0, "[": 0, "{": 0}
    closers = {")": "(", "]": "[", "}": "{"}
    position = 0
    while position < len(arguments):
        skipped = _skip_js_string_or_comment(arguments, position)
        if skipped is not None:
            position = skipped
            continue
        char = arguments[position]
        if char in depths:
            depths[char] += 1
        elif char in closers:
            opener = closers[char]
            depths[opener] = max(depths[opener] - 1, 0)
        elif char == "," and all(depth == 0 for depth in depths.values()):
            return arguments[:position]
        position += 1
    return arguments


def _classify_code_mode_child(tool_path: tuple[str, ...]) -> str:
    normalized = tuple(part.strip().lower().replace("-", "_") for part in tool_path if part.strip())
    child_parts = normalized[1:] if normalized and normalized[0] in {"tools", "functions"} else normalized
    if not child_parts:
        return "unknown"

    # Namespaced tools are MCP delegations unless the namespace itself is a
    # first-class Codex family. This check intentionally precedes leaf aliases:
    # tools.mcp.repo.search is MCP, not a generic web search.
    if child_parts[0] == "web":
        return "web"
    if child_parts[0] == "image":
        return "image"
    if any("mcp" in part for part in child_parts) or len(child_parts) > 1:
        return "mcp"

    leaf = child_parts[-1]
    for spec in _CODE_MODE_CHILD_REGISTRY:
        if leaf in spec.aliases:
            return spec.kind
    return "unknown"


def _code_mode_tool_name(tool_path: tuple[str, ...], registry_type: str) -> str:
    child_parts = tool_path[1:] if tool_path and tool_path[0].lower() in {"tools", "functions"} else tool_path
    raw_name = ".".join(child_parts) if child_parts else "unknown"
    # First-class registry entries use stable names so downstream semantic
    # normalization does not depend on provider spelling. MCP and unknown
    # calls keep their exact names; their registry type remains in provenance.
    return registry_type if registry_type not in {"mcp", "unknown"} else raw_name


def _scan_code_mode_child_calls(source: str) -> tuple[_CodexExecChildCall, ...]:
    calls: list[_CodexExecChildCall] = []
    position = 0
    while position < len(source):
        skipped = _skip_js_string_or_comment(source, position)
        if skipped is not None:
            position = skipped
            continue
        if not _is_js_identifier_start(source[position]):
            position += 1
            continue
        parsed_chain = _parse_js_member_chain(source, position)
        if parsed_chain is None:
            position += 1
            continue
        tool_path, after_chain = parsed_chain
        after_chain = _skip_js_space_and_comments(source, after_chain)
        if not tool_path or tool_path[0].lower() not in {"tools", "functions"}:
            position = max(after_chain, position + 1)
            continue
        if len(tool_path) < 2 or after_chain >= len(source) or source[after_chain] != "(":
            position = max(after_chain, position + 1)
            continue
        raw_arguments, call_end, balanced = _balanced_js_call_argument(source, after_chain)
        first_argument = _first_js_argument(raw_arguments)
        argument, parsed = _parse_js_literal(first_argument)
        registry_type = _classify_code_mode_child(tool_path)
        calls.append(
            _CodexExecChildCall(
                tool_path=tool_path,
                tool_name=_code_mode_tool_name(tool_path, registry_type),
                registry_type=registry_type,
                argument=argument,
                raw_argument=first_argument.strip() or None,
                parse_state="parsed" if parsed and balanced else "malformed",
                source_start=position,
                source_end=call_end,
            )
        )
        position = max(call_end, position + 1)
    return tuple(calls)


def _mapping_string(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _structured_code_mode_child_calls(value: object) -> tuple[_CodexExecChildCall, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    if not isinstance(value, dict):
        return ()
    child_items: list[object] | None = None
    for key in _CODE_MODE_CHILD_COLLECTION_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, list):
            child_items = candidate
            break
    if child_items is None:
        return ()
    calls: list[_CodexExecChildCall] = []
    for item in child_items:
        if not isinstance(item, dict):
            calls.append(
                _CodexExecChildCall(
                    tool_path=("tools", "unknown"),
                    tool_name="unknown",
                    registry_type="unknown",
                    argument=item,
                    raw_argument=_codex_tool_output_text(item),
                    parse_state="malformed",
                )
            )
            continue
        raw_name = _mapping_string(item, "name", "tool_name", "tool", "operation", "kind", "type")
        if raw_name:
            name_parts = tuple(part for part in raw_name.replace("::", ".").split(".") if part)
            tool_path = (
                name_parts if name_parts and name_parts[0].lower() in {"tools", "functions"} else ("tools", *name_parts)
            )
        else:
            tool_path = ("tools", "unknown")
        argument: object = {}
        for key in ("arguments", "input", "action", "params", "parameters"):
            if key in item:
                argument = item[key]
                break
        if isinstance(argument, str):
            parsed_argument, parsed = _parse_js_literal(argument)
        else:
            parsed_argument, parsed = argument, True
        registry_type = _classify_code_mode_child(tool_path)
        calls.append(
            _CodexExecChildCall(
                tool_path=tool_path,
                tool_name=_code_mode_tool_name(tool_path, registry_type),
                registry_type=registry_type,
                argument=parsed_argument,
                raw_argument=argument if isinstance(argument, str) else _codex_tool_output_text(argument),
                parse_state="parsed" if raw_name and parsed else "malformed",
            )
        )
    return tuple(calls)


def _code_mode_source(value: object) -> str | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, dict):
            value = parsed
        else:
            return value
    if not isinstance(value, dict):
        return None
    for key in ("source", "code", "script", "javascript", "js", "command", "arguments", "input"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _code_mode_children(value: object) -> tuple[_CodexExecChildCall, ...]:
    structured = _structured_code_mode_child_calls(value)
    if structured:
        return structured
    source = _code_mode_source(value)
    return _scan_code_mode_child_calls(source) if source else ()


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen or normalized == "/dev/null":
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _patch_touched_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    marker_prefixes = (
        "*** Add File:",
        "*** Update File:",
        "*** Delete File:",
        "*** Move to:",
        "*** Move File:",
    )
    for line in patch.splitlines():
        stripped = line.strip()
        for prefix in marker_prefixes:
            if stripped.startswith(prefix):
                paths.append(stripped.removeprefix(prefix).strip())
                break
        else:
            if stripped.startswith(("+++ ", "--- ")):
                candidate = stripped[4:].split("\t", 1)[0].strip()
                if candidate.startswith(("a/", "b/")):
                    candidate = candidate[2:]
                paths.append(candidate)
            elif stripped.startswith("diff --git "):
                pieces = stripped.removeprefix("diff --git ").split()
                if len(pieces) >= 2:
                    candidate = pieces[-1]
                    paths.append(candidate[2:] if candidate.startswith("b/") else candidate)
    return _dedupe_strings(paths)


def _structural_paths(value: object) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(item: object, *, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key).lower()
                if key in _STRUCTURAL_PATH_KEYS:
                    if isinstance(child, str):
                        paths.append(child)
                    elif isinstance(child, list):
                        paths.extend(value for value in child if isinstance(value, str))
                if isinstance(child, dict | list):
                    visit(child, depth=depth + 1)
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, dict | list):
                    visit(child, depth=depth + 1)

    visit(value, depth=0)
    return _dedupe_strings(paths)


def _structural_byte_count(value: object) -> int | None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            if str(raw_key).lower() in _STRUCTURAL_BYTE_KEYS and isinstance(child, int) and not isinstance(child, bool):
                return child if child >= 0 else None
        for child in value.values():
            if isinstance(child, dict | list):
                nested = _structural_byte_count(child)
                if nested is not None:
                    return nested
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict | list):
                nested = _structural_byte_count(child)
                if nested is not None:
                    return nested
    return None


_CODEX_CHUNK_ID_PREFIX_RE = re.compile(r"\AChunk ID: [0-9a-f]+\n")
_CODEX_EXEC_ENVELOPE_OUTCOME_RE = re.compile(
    r"\AWall time: [0-9.]+ seconds\n"
    r"(?:Process exited with code (?P<exit_code>-?\d+)"
    r"|Process completed with exit code (?P<exit_code2>-?\d+)"
    r"|Process running with session ID \d+)"
)


def _codex_exec_envelope_outcome(output: object) -> tuple[bool | None, int | None]:
    """Read the exit outcome Codex's own unified-exec tool ("exec_command"/
    "write_stdin"/code-mode exec children) always stamps on its result text.

    Unlike a shell transcript, this preamble is generated by the Codex CLI
    itself, never the model: ``[Chunk ID: <hex>\\n]Wall time: <float>
    seconds\\n`` followed by exactly ``Process exited with code <N>`` (or the
    older ``Process completed with exit code <N>``) once the process has
    exited, or ``Process running with session ID <N>`` while a
    long-lived/chunked session is still attached. Matching is anchored at the
    very start of the field, so it can never fire on an unrelated occurrence
    of similar wording deep inside captured subprocess output (e.g. a CI log
    that itself prints "Process completed with exit code 1") -- that text
    would only ever appear after this preamble, never as a substitute for it.
    A still-running session has no outcome yet and stays honestly unknown.
    """
    if not isinstance(output, str):
        return None, None
    text = _CODEX_CHUNK_ID_PREFIX_RE.sub("", output, count=1)
    match = _CODEX_EXEC_ENVELOPE_OUTCOME_RE.match(text)
    if match is None:
        return None, None
    exit_code_str = match.group("exit_code") or match.group("exit_code2")
    if exit_code_str is None:
        return None, None
    exit_code = int(exit_code_str)
    return exit_code != 0, exit_code


def _codex_tool_result_outcome(raw: object) -> tuple[bool | None, int | None]:
    """Resolve (is_error, exit_code) for a Codex tool-result payload.

    Tries the JSON-structural outcome first (``exit_code``/``is_error``
    fields nested in a decoded JSON object), then falls back to the
    unified-exec text envelope (see ``_codex_exec_envelope_outcome``) when the
    raw payload is a string that JSON-decoding did not resolve to a mapping
    carrying either field. Anything else remains unknown.
    """
    decoded = _decoded_json_value(raw) if isinstance(raw, str) else raw
    is_error, exit_code = _structural_outcome(decoded)
    if is_error is None and exit_code is None and isinstance(raw, str):
        return _codex_exec_envelope_outcome(raw)
    return is_error, exit_code


def _structural_outcome(value: object) -> tuple[bool | None, int | None]:
    wrappers: list[dict[str, object]] = []
    if isinstance(value, dict):
        wrappers.append(value)
        for key in ("metadata", "result", "output"):
            nested = value.get(key)
            if isinstance(nested, dict):
                wrappers.append(nested)
    exit_code: int | None = None
    is_error: bool | None = None
    for wrapper in wrappers:
        raw_exit = wrapper.get("exit_code")
        if isinstance(raw_exit, int) and not isinstance(raw_exit, bool):
            exit_code = raw_exit
            break
    for wrapper in wrappers:
        raw_error = wrapper.get("is_error")
        if isinstance(raw_error, bool):
            is_error = raw_error
            break
    if exit_code is not None:
        derived = exit_code != 0
        if is_error is None:
            is_error = derived
    if is_error is None:
        # A structural "timed_out": true (e.g. the `wait`/`write_stdin`
        # child tools' timeout envelope, `{"message": "Wait timed out.",
        # "timed_out": true}`) is itself the provider's own outcome signal --
        # the operation did not complete successfully -- even when no
        # exit_code/is_error field is present.
        for wrapper in wrappers:
            raw_timed_out = wrapper.get("timed_out")
            if isinstance(raw_timed_out, bool):
                if raw_timed_out:
                    is_error = True
                break
    return is_error, exit_code


def _decoded_json_value(value: object) -> object | None:
    if not isinstance(value, str):
        return value
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded


def _decoded_structural_text_item(item: object) -> tuple[object, ...]:
    if not isinstance(item, dict):
        return ()
    item_type = item.get("type")
    if item_type not in {"input_text", "output_text"}:
        return ()
    text = item.get("text")
    if not isinstance(text, str):
        return ()
    decoded = _decoded_json_value(text)
    if isinstance(decoded, dict):
        for key in _CODE_MODE_RESULT_COLLECTION_KEYS:
            candidate = decoded.get(key)
            if isinstance(candidate, list) and all(isinstance(value, dict | list) for value in candidate):
                return tuple(candidate)
        return (decoded,)
    if isinstance(decoded, list) and all(isinstance(value, dict | list) for value in decoded):
        # A JSON array emitted as one content item represents ordered child
        # results only when every element is itself a structured value.
        return tuple(decoded)
    return ()


def _code_mode_result_items(output: object, *, child_count: int) -> tuple[object, ...]:
    if child_count <= 0:
        return ()
    parsed = _decoded_json_value(output)
    if isinstance(parsed, dict):
        for key in _CODE_MODE_RESULT_COLLECTION_KEYS:
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                return tuple(candidate[:child_count])
        if parsed.get("type") in {"input_text", "output_text", "input_image", "image"}:
            return _decoded_structural_text_item(parsed)[:child_count]
        # A non-content-item mapping is one exact child result when the
        # envelope has one child. It may carry no promoted outcome fields; in
        # that case the paired result is retained with outcome=unknown.
        return (parsed,) if child_count == 1 else ()
    if parsed is None and child_count == 1 and isinstance(output, str):
        # The whole output failed to decode as JSON -- when there is exactly
        # one child, the raw text itself (possibly Codex's own unified-exec
        # envelope, see _codex_exec_envelope_outcome) is that child's result.
        return (output,)
    if isinstance(parsed, list):
        if all(
            isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "input_image", "image"}
            for item in parsed
        ):
            emitted: list[object] = []
            for item in parsed:
                emitted.extend(_decoded_structural_text_item(item))
                if len(emitted) >= child_count:
                    break
            return tuple(emitted[:child_count])
        return tuple(parsed[:child_count])
    return ()


def _code_mode_child_results(output: object, *, child_count: int) -> tuple[_CodexExecChildResult, ...]:
    results: list[_CodexExecChildResult] = []
    for item in _code_mode_result_items(output, child_count=child_count):
        is_error, exit_code = _codex_tool_result_outcome(item)
        results.append(
            _CodexExecChildResult(
                raw=item,
                text=_codex_tool_output_text(item),
                is_error=is_error,
                exit_code=exit_code,
                paths=_structural_paths(item),
                byte_count=_structural_byte_count(item),
            )
        )
    return tuple(results)


def _response_inner_record(item: object) -> dict[str, object] | None:
    record = _dict_record(item)
    if record is None or _record_type(record) not in {"response_item", "event_msg"}:
        return None
    inner = _payload_record(record)
    return inner if inner is not None and not _is_message(inner) else None


def _code_mode_exec_envelopes(records: Sequence[object]) -> dict[int, _CodexExecEnvelope]:
    call_occurrences: dict[str, list[tuple[int, _CodexExecEnvelope]]] = defaultdict(list)
    output_occurrences: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    envelopes_by_record: dict[int, _CodexExecEnvelope] = {}
    for index, item in enumerate(records, start=1):
        inner = _response_inner_record(item)
        if inner is None:
            continue
        payload = _record_payload(inner)
        record_type = _record_type(inner)
        if record_type in {
            "function_call",
            "custom_tool_call",
            "tool_search_call",
            "web_search_call",
            "local_shell_call",
        }:
            tool_name = payload.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                tool_name = payload.get("execution")
            if not isinstance(tool_name, str) or tool_name.lower() not in _CODE_MODE_EXEC_TOOL_NAMES:
                continue
            raw_arguments = payload.get("arguments")
            if raw_arguments is None:
                raw_arguments = payload.get("input")
            if raw_arguments is None:
                raw_arguments = payload.get("action")
            children = _code_mode_children(raw_arguments)
            if not children:
                continue
            raw_tool_id = payload.get("call_id") or payload.get("id")
            tool_id = str(raw_tool_id) if raw_tool_id else None
            # polylogue-slshy: no positional fallback -- an empty id lets
            # _message_comparison_id's content-anchor (role + timestamp)
            # fallback run instead of a position-derived string that would
            # change identity when array order shifts across re-acquisitions.
            provider_message_id = str(payload.get("id") or raw_tool_id or "")
            envelope = _CodexExecEnvelope(
                transport_tool_name=tool_name,
                transport_tool_id=tool_id,
                transport_provider_message_id=provider_message_id,
                children=children,
            )
            envelopes_by_record[index] = envelope
            if tool_id:
                call_occurrences[tool_id].append((index, envelope))
        elif record_type in {
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
            "web_search_output",
        }:
            raw_tool_id = payload.get("call_id") or payload.get("id")
            if raw_tool_id:
                output_occurrences[str(raw_tool_id)].append((index, inner))

    for tool_id, calls in call_occurrences.items():
        outputs = output_occurrences.get(tool_id, [])
        for occurrence, (call_index, envelope) in enumerate(calls):
            if occurrence >= len(outputs):
                continue
            output_index, output_record = outputs[occurrence]
            output = output_record.get("output")
            if output is None:
                output = output_record.get("tools")
            if output is None:
                output = output_record.get("result")
            enriched = replace(
                envelope,
                results=_code_mode_child_results(output, child_count=len(envelope.children)),
            )
            envelopes_by_record[call_index] = enriched
            envelopes_by_record[output_index] = enriched
    return envelopes_by_record


def _normalized_command(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    return None


def _child_tool_input(
    child: _CodexExecChildCall,
    *,
    child_index: int,
    envelope: _CodexExecEnvelope,
    result: _CodexExecChildResult | None,
) -> dict[str, object]:
    if isinstance(child.argument, dict):
        tool_input: dict[str, object] = {str(key): value for key, value in child.argument.items()}
    elif child.argument is None:
        tool_input = {}
    else:
        tool_input = {"input": child.argument}

    command: str | None = None
    if child.parse_state == "parsed" and child.registry_type == "exec_command":
        command = _normalized_command(tool_input.get("command")) or _normalized_command(tool_input.get("cmd"))
        if command is None:
            command = _normalized_command(child.argument)
    elif child.parse_state == "parsed" and child.registry_type == "apply_patch":
        patch = (
            _normalized_command(tool_input.get("patch"))
            or _normalized_command(tool_input.get("input"))
            or _normalized_command(tool_input.get("command"))
            or _normalized_command(child.argument)
        )
        if patch is not None:
            tool_input.setdefault("patch", patch)
            command = patch
            patch_paths = _patch_touched_paths(patch)
            if patch_paths:
                tool_input["paths"] = list(patch_paths)
                tool_input["path"] = patch_paths[0]
    if command is not None:
        tool_input["command"] = command

    paths = list(_structural_paths(tool_input))
    byte_count = _structural_byte_count(tool_input)
    if result is not None:
        paths.extend(result.paths)
        if byte_count is None:
            byte_count = result.byte_count
    normalized_paths = _dedupe_strings(paths)
    if normalized_paths:
        tool_input["paths"] = list(normalized_paths)
        tool_input.setdefault("path", normalized_paths[0])
    if byte_count is not None:
        tool_input["byte_count"] = byte_count
    if (child.parse_state != "parsed" or child.registry_type == "unknown") and child.raw_argument is not None:
        tool_input["raw_arguments"] = child.raw_argument

    provenance: dict[str, object] = {
        "kind": "codex.functions_exec_child",
        "registry_type": child.registry_type,
        "parse_state": child.parse_state,
        "raw_tool_path": ".".join(child.tool_path),
        "transport_child_index": child_index,
        "transport": {
            "provider_message_id": envelope.transport_provider_message_id,
            "tool_id": envelope.transport_tool_id,
            "tool_name": envelope.transport_tool_name,
            "block_position": 0,
        },
    }
    if child.source_start is not None and child.source_end is not None:
        provenance["source_span"] = [child.source_start, child.source_end]
    if result is not None and (result.paths or result.byte_count is not None):
        result_fields: dict[str, object] = {}
        if result.paths:
            result_fields["paths"] = list(result.paths)
        if result.byte_count is not None:
            result_fields["byte_count"] = result.byte_count
        provenance["structural_result_fields"] = result_fields
    tool_input[_CODE_MODE_CHILD_PROVENANCE_KEY] = provenance
    return tool_input


def _child_tool_id(envelope: _CodexExecEnvelope, child_index: int) -> str | None:
    if not envelope.transport_tool_id:
        return None
    return f"{envelope.transport_tool_id}{_CODE_MODE_CHILD_ID_MARKER}{child_index}"


def _code_mode_child_use_blocks(envelope: _CodexExecEnvelope) -> list[ParsedContentBlock]:
    blocks: list[ParsedContentBlock] = []
    for child_index, child in enumerate(envelope.children):
        result = envelope.results[child_index] if child_index < len(envelope.results) else None
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=child.tool_name,
                tool_id=_child_tool_id(envelope, child_index),
                tool_input=_child_tool_input(
                    child,
                    child_index=child_index,
                    envelope=envelope,
                    result=result,
                ),
            )
        )
    return blocks


# polylogue-9x22: ``ParsedContentBlock.metadata`` is never persisted -- the
# ``blocks`` table has no metadata column and the only key the write path
# reads back out of it is ``language`` (``storage/sqlite/archive_tiers/
# write.py:_block_language``). ``_code_mode_child_result_blocks`` below still
# attaches ``codex_functions_exec_*``/``paths``/``byte_count`` to
# ``metadata`` as an in-process carrier; ``_code_mode_child_result_evidence_
# events`` projects that same dict into ``session_events`` (same precedent
# as ``claude/common.py``'s ``claude_ai_web_tool_evidence``), keyed to the
# tool-result message that owns the child blocks.
def _code_mode_child_result_evidence_events(
    envelope: _CodexExecEnvelope,
    *,
    source_message_provider_id: str,
    timestamp: str | None,
) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    for child_index in range(len(envelope.results)):
        result = envelope.results[child_index]
        metadata: dict[str, object] = {
            "codex_functions_exec_child_index": child_index,
            "codex_functions_exec_registry_type": envelope.children[child_index].registry_type,
        }
        if result.paths:
            metadata["paths"] = list(result.paths)
        if result.byte_count is not None:
            metadata["byte_count"] = result.byte_count
        events.append(
            ParsedSessionEvent(
                event_type="codex_functions_exec_child_result_evidence",
                timestamp=timestamp,
                source_message_provider_id=source_message_provider_id,
                payload=metadata,
            )
        )
    return events


def _code_mode_child_result_blocks(envelope: _CodexExecEnvelope) -> list[ParsedContentBlock]:
    blocks: list[ParsedContentBlock] = []
    for child_index, result in enumerate(envelope.results):
        metadata: dict[str, object] = {
            "codex_functions_exec_child_index": child_index,
            "codex_functions_exec_registry_type": envelope.children[child_index].registry_type,
        }
        if result.paths:
            metadata["paths"] = list(result.paths)
        if result.byte_count is not None:
            metadata["byte_count"] = result.byte_count
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=_child_tool_id(envelope, child_index),
                text=result.text,
                metadata=metadata,
                is_error=result.is_error,
                exit_code=result.exit_code,
            )
        )
    return blocks


def _tool_input_from_arguments(value: object, *, tool_name: str) -> dict[str, object]:
    if isinstance(value, dict):
        tool_input = dict(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            tool_input = {"arguments": value}
        else:
            tool_input = dict(parsed) if isinstance(parsed, dict) else {"arguments": value}
    else:
        return {}

    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        return tool_input

    cmd = tool_input.get("cmd")
    if isinstance(cmd, str) and cmd.strip():
        return {**tool_input, "command": cmd}

    arguments = tool_input.get("arguments")
    if tool_name.lower() in _EXECUTION_TOOL_NAMES and isinstance(arguments, str) and arguments.strip():
        return {**tool_input, "command": arguments}

    # apply_patch carries its whole payload as a PATCH-FORMAT STRING under
    # ``arguments`` -- not JSON -- so the operated-on path lives in
    # ``*** Update File: <path>`` / ``*** Add File:`` / ``*** Delete File:``
    # header lines, where no ``json_extract`` can reach it. blocks.tool_path
    # and blocks.search_text are both generated from
    # ``$.file_path``/``$.path`` (archive_tiers/index.py:307,310), so every
    # Codex file edit was invisible to structured path queries AND to FTS:
    # tool_path coverage measured 0.07% for codex-session against 44% for
    # claude-code-session, and apply_patch is 95% of Codex tool calls
    # (18,984 of a 20,000 sample). It also left Codex sessions unable to earn
    # a path-bearing structural label (polylogue-a9hx).
    #
    # The batched code-mode child path already extracts these via
    # ``_patch_touched_paths``; this is the standalone ``function_call``
    # branch, which did not. Same helper, so both paths agree.
    if tool_name.lower() in _PATCH_TOOL_NAMES and isinstance(arguments, str) and arguments.strip():
        patch_paths = _patch_touched_paths(arguments)
        if patch_paths:
            # ``path`` is what the generated column reads; ``paths`` keeps the
            # full set, since 11% of patches touch more than one file
            # (554 of 5,000 sampled) and a single column cannot hold them.
            return {**tool_input, "patch": arguments, "path": patch_paths[0], "paths": list(patch_paths)}
    return tool_input


def _codex_material_origin(role: Role, message_type: MessageType, text: str | None) -> MaterialOrigin:
    material_origin = classify_material_origin(role=role, message_type=message_type, text=text)
    if material_origin is MaterialOrigin.UNKNOWN and role is Role.USER and message_type is MessageType.MESSAGE:
        return MaterialOrigin.HUMAN_AUTHORED
    return material_origin


def _codex_tool_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
    exec_envelope: _CodexExecEnvelope | None = None,
) -> ParsedMessage | None:
    payload = _record_payload(record)
    record_type = _record_type(record)
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    if record_type in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "local_shell_call"}:
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = payload.get("execution")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = record_type
        tool_id = payload.get("call_id") or payload.get("id")
        raw_arguments = payload.get("arguments")
        if raw_arguments is None:
            raw_arguments = payload.get("input")
        if raw_arguments is None:
            raw_arguments = payload.get("action")
        blocks = [
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=tool_name,
                tool_id=str(tool_id) if tool_id else None,
                tool_input=_tool_input_from_arguments(raw_arguments, tool_name=tool_name),
            )
        ]
        if exec_envelope is not None:
            blocks.extend(_code_mode_child_use_blocks(exec_envelope))
        return ParsedMessage(
            # polylogue-slshy: see the sibling comment above; no positional fallback.
            provider_message_id=str(payload.get("id") or tool_id or ""),
            role=Role.ASSISTANT,
            text=tool_name,
            timestamp=timestamp,
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=blocks,
        )
    if record_type in {"function_call_output", "custom_tool_call_output", "tool_search_output", "web_search_output"}:
        tool_id = payload.get("call_id") or payload.get("id")
        output = payload.get("output")
        if output is None:
            output = payload.get("tools")
        if output is None:
            output = payload.get("result")
        output_text = _codex_tool_output_text(output)
        if not tool_id and not output_text:
            return None
        # Only exact structured fields (or Codex's own generated exec-tool
        # envelope, see _codex_exec_envelope_outcome) affect the outcome.
        # Arbitrary prose containing exit-code-like wording remains evidence
        # text with an unknown outcome.
        is_error, exit_code = _codex_tool_result_outcome(output)
        blocks = [
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=str(tool_id) if tool_id else None,
                text=output_text,
                is_error=is_error,
                exit_code=exit_code,
            )
        ]
        if exec_envelope is not None:
            blocks.extend(_code_mode_child_result_blocks(exec_envelope))
        return ParsedMessage(
            # polylogue-slshy: no positional fallback (see above).
            provider_message_id=str(payload.get("id") or tool_id or ""),
            role=Role.TOOL,
            text=output_text,
            timestamp=timestamp,
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=blocks,
        )
    return None


def _codex_reasoning_joined_text(value: object) -> str | None:
    """Join recoverable text out of a Codex ``reasoning`` record's `summary`/`content`.

    Both fields share the same OpenAI Responses-API shape: either a bare
    string, or a list of ``{"type": "summary_text"|"reasoning_text", "text": ...}``
    (or equivalent) dicts. Anything else (encrypted ciphertext, missing
    fields) yields no text -- that is a genuine absence, handled by the
    caller, not an extraction bug here.
    """
    if isinstance(value, str):
        return value or None
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in value:
        text: object = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def _codex_reasoning_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
) -> ParsedMessage | None:
    """Materialize a Codex ``reasoning`` response_item as a THINKING-block message.

    polylogue-vf9x: previously this record type was read only by
    ``_compact_response_payload``'s generic session_event compactor, which
    has no `reasoning`-specific branch -- neither `summary` nor `content` was
    read at all (not merely char-counted: the emitted session_event carries
    only ``{source_index, type}``), so every one of the measured 1,182,071
    Codex reasoning records in this operator's raw corpus contributed zero
    words to the archive, unreachable from FTS/search even in principle.

    `summary` (OpenAI's human-readable condensation) is present with
    recoverable text on ~24% of records measured; `content` (the full trace)
    is essentially always null on the wire -- Codex encrypts it into
    `encrypted_content` instead, which this archive cannot decrypt and does
    not attempt to store. Even when neither carries text, the message is
    still recorded (block text=None) so the FACT that the model reasoned
    here survives -- the same rationale as Claude Code's empty-body thinking
    blocks (base_support.py).

    Routed into `messages`/`blocks` (not left as a session_event only) so
    reasoning joins the normal content tree: FTS coverage, `thinking_count`,
    and the `material_origin`/`BlockType.THINKING` vocabulary every other
    origin's reasoning content already uses.
    """
    if _record_type(record) != "reasoning":
        return None
    payload = _record_payload(record)
    summary_text = _codex_reasoning_joined_text(payload.get("summary"))
    content_text = _codex_reasoning_joined_text(payload.get("content"))
    blocks: list[ParsedContentBlock] = []
    if summary_text:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=summary_text))
    if content_text and content_text != summary_text:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=content_text))
    if not blocks:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=None))
    combined_text = "\n\n".join(t for t in (summary_text, content_text) if t) or None
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    return ParsedMessage(
        provider_message_id=synthetic_message_id(
            role=Role.ASSISTANT,
            text=combined_text,
            timestamp=timestamp,
            kind="codex-reasoning",
        ),
        role=Role.ASSISTANT,
        text=combined_text,
        timestamp=timestamp,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=blocks,
        message_type=MessageType.THINKING,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
    )


def _mcp_invocation_tool_name(invocation: dict[str, object]) -> str:
    server = _string_value(invocation.get("server"))
    tool = _string_value(invocation.get("tool"))
    if server and tool:
        return f"mcp__{server}__{tool}"
    return tool or server or "mcp_tool_call"


def _mcp_result_outcome(result: object) -> tuple[bool | None, str | None]:
    """Extract (is_error, text) from an ``mcp_tool_call_end`` ``result``.

    Codex wraps MCP results as a Rust-style ``{"Ok": ...}`` / ``{"Err": "..."}``
    tagged union rather than the ``is_error``/``exit_code`` shape other Codex
    tool records use.
    """
    if not isinstance(result, dict):
        return None, _codex_tool_output_text(result)
    if "Err" in result:
        return True, _codex_tool_output_text(result.get("Err"))
    if "Ok" in result:
        return False, _codex_tool_output_text(result.get("Ok"))
    return None, _codex_tool_output_text(result)


def _codex_mcp_tool_call_messages(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
) -> tuple[ParsedMessage, ParsedMessage] | None:
    """Parse a Codex ``mcp_tool_call_end`` record into a tool_use/tool_result pair.

    Unlike ``function_call``/``function_call_output``, Codex emits MCP tool
    invocations as a single self-contained record carrying both the request
    (``invocation.server``/``invocation.tool``/``invocation.arguments``) and
    the response (``result``) -- there is no paired ``mcp_tool_call_begin``.
    Previously this whole record fell through to the generic event-summary
    path, which drops the invocation and result entirely; this was the
    largest single unread surface in the corpus (arbitrary downstream MCP
    server responses, e.g. github/serena/sinex tool calls made from Codex).
    """
    payload = _record_payload(record)
    if payload.get("type") != "mcp_tool_call_end":
        return None
    invocation = _dict_record(payload.get("invocation"))
    if invocation is None:
        return None
    tool_name = _mcp_invocation_tool_name(invocation)
    call_id = payload.get("call_id")
    tool_id = str(call_id) if isinstance(call_id, str) and call_id else f"mcp-call-{index}"
    arguments = invocation.get("arguments")
    if isinstance(arguments, dict):
        tool_input: dict[str, object] = dict(arguments)
    elif arguments is not None:
        tool_input = {"arguments": arguments}
    else:
        tool_input = {}
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    use_message = ParsedMessage(
        provider_message_id=f"{tool_id}::mcp-call",
        role=Role.ASSISTANT,
        text=tool_name,
        timestamp=timestamp,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=tool_name,
                tool_id=tool_id,
                tool_input=tool_input,
            )
        ],
    )
    is_error, result_text = _mcp_result_outcome(payload.get("result"))
    result_message = ParsedMessage(
        provider_message_id=f"{tool_id}::mcp-output",
        role=Role.TOOL,
        text=result_text,
        timestamp=timestamp,
        position=position + 1,
        variant_index=0,
        is_active_path=True,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text=result_text,
                is_error=is_error,
            )
        ],
    )
    return use_message, result_message


def _codex_tool_output_text(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        sanitized = _sanitize_codex_large_inline_payloads(output)
        if sanitized != output:
            return str(sanitized)
        try:
            parsed = json.loads(output)
        except (ValueError, TypeError):
            return output
        sanitized_parsed = _sanitize_codex_large_inline_payloads(parsed)
        if sanitized_parsed != parsed:
            return json.dumps(sanitized_parsed, sort_keys=True)
        return output
    sanitized = _sanitize_codex_large_inline_payloads(output)
    return json.dumps(sanitized, sort_keys=True) if sanitized else None


def _sanitize_codex_large_inline_payloads(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_codex_data_url(value)
    if isinstance(value, list):
        return [_sanitize_codex_large_inline_payloads(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_codex_large_inline_payloads(item) for key, item in value.items()}
    return value


def _sanitize_codex_data_url(value: str) -> str:
    if not value.startswith("data:image/") or ";base64," not in value:
        return value
    header, encoded = value.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0] or "image/unknown"
    digest = hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest()
    approx_bytes = (len(encoded.rstrip("=")) * 3) // 4
    return f"<inline image omitted; mime={mime}; approx_bytes={approx_bytes}; sha256_base64={digest}>"


def _codex_inline_image_summaries(content: object) -> tuple[str, ...]:
    """Return bounded evidence for inline image segments in ordinary messages."""
    if not isinstance(content, list):
        return ()
    summaries: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in {"input_image", "image"}:
            continue
        image_url = item.get("image_url")
        if not isinstance(image_url, str):
            continue
        summary = _sanitize_codex_data_url(image_url)
        if summary != image_url:
            summaries.append(summary)
    return tuple(summaries)


def _message_signature(role: Role | str, text: str | None) -> tuple[str, str]:
    role_value = role.value if isinstance(role, Role) else str(role)
    return (role_value, " ".join((text or "").split()))


def _response_message_signatures(records: Iterable[object]) -> set[tuple[str, str]]:
    signatures: set[tuple[str, str]] = set()
    for item in records:
        record = _dict_record(item)
        if record is None:
            continue
        message_record = _message_record(record)
        if message_record is None:
            continue
        raw_role = _effective_role(message_record)
        if not raw_role or raw_role == "unknown":
            continue
        text = extract_codex_text(_effective_content(message_record))
        signatures.add(_message_signature(Role.normalize(raw_role), text))
    return signatures


def _codex_event_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    response_signatures: set[tuple[str, str]],
    timestamp_fallback: str | int | float | None = None,
) -> ParsedMessage | None:
    record_type = _record_type(record)
    if record_type not in {"user_message", "agent_message"}:
        return None
    text = record.get("message")
    if not isinstance(text, str) or not text.strip():
        return None
    role = Role.USER if record_type == "user_message" else Role.ASSISTANT
    if _message_signature(role, text) in response_signatures:
        return None
    message_type = classify_text_message_type(text) or MessageType.MESSAGE
    return ParsedMessage(
        # polylogue-slshy: no positional fallback (see above).
        provider_message_id=str(record.get("client_id") or record.get("id") or ""),
        role=role,
        text=text,
        timestamp=_iso_or_none(_record_timestamp(record) or timestamp_fallback),
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        message_type=message_type,
        material_origin=_codex_material_origin(role, message_type, text),
    )


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


def is_supported_session_stream(payload: Sequence[object]) -> bool:
    """Return whether every record forms a materializable Codex session stream.

    This is stricter than :func:`looks_like`, which only needs one record to
    identify the parser. Artifact classification uses this full-stream
    contract before it lets a Codex JSONL payload reach schema inference.
    """
    has_session_header = False
    has_message = False
    has_envelope_record = False
    has_direct_record = False
    supported_envelope_types = {
        "session_meta",
        "response_item",
        "event_msg",
        "compacted",
        "turn_context",
        "world_state",
        "reasoning",
    }

    for index, item in enumerate(payload, start=1):
        if not _is_plausibly_codex_record(item):
            return False
        record = _dict_record(item)
        if record is None or _validate_record(record, index=index, context="session stream") is None:
            return False
        record_type = _record_type(record)
        if record_type in supported_envelope_types:
            # ``session_meta`` is the shared header for both the envelope
            # stream and the legacy direct-message stream. It must not make a
            # valid header-plus-direct stream look like mixed generations.
            has_envelope_record = has_envelope_record or record_type != "session_meta"
            if not _is_envelope(record):
                return False
            session_meta = _session_meta_record(record)
            if session_meta is not None:
                has_session_header = has_session_header or _record_id(session_meta) is not None
            if _message_record(record) is not None:
                has_message = True
            continue
        if _is_state(record):
            continue
        if _is_direct_message(record):
            has_direct_record = True
            has_message = True
            continue
        session_meta = _session_meta_record(record)
        if session_meta is not None:
            has_session_header = has_session_header or _record_id(session_meta) is not None
            continue
        return False

    if not has_message:
        return False
    if has_direct_record and has_envelope_record:
        # The parser supports these wire formats independently, but their
        # records cannot be combined into one trustworthy session stream.
        return False
    # Legacy direct-message streams and headerless envelope append deltas use
    # the acquisition fallback id. A bare header remains ineligible because
    # ``has_message`` is false above.
    return has_session_header or has_direct_record or has_envelope_record


def _parse_records(records: Iterable[object], fallback_id: str) -> ParsedSession:
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
    record_list = list(records)
    code_mode_envelopes = _code_mode_exec_envelopes(record_list)
    response_signatures = _response_message_signatures(record_list)
    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    session_id = fallback_id
    session_timestamp: str | None = None
    session_timestamp_pair: _TimestampPair | None = None
    latest_message_timestamp: _TimestampPair | None = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking
    # Explicit lineage markers from the child's own (first) session_meta. Codex
    # records `forked_from_id` for forks/resumes and a `source.subagent.thread_spawn`
    # block for spawned subagents; both inherit the parent's context as a copied
    # prefix in this rollout. See docs/design/session-lineage-model.md.
    forked_from_id: str | None = None
    is_subagent_spawn = False
    # Structural evidence for the legacy (no forked_from_id) continuation
    # fallback below: the child's own cwd/git, and the same facts read off
    # the second distinct session_meta encountered (a resumed session
    # physically replays the parent's original session_meta as the next
    # record). Captured independently of `session_git`, which prefers the
    # *first* meta's git and must not be overwritten by the second's.
    first_meta_cwd: str | None = None
    first_meta_repo_url: str | None = None
    second_meta_timestamp_pair: _TimestampPair | None = None
    second_meta_cwd: str | None = None
    second_meta_repo_url: str | None = None
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata
    working_directories: set[str] = set()
    current_model_name: str | None = None
    current_model_effort: str | None = None
    message_position = 0
    # Subagent/session identity facts that recur on every session_meta or
    # turn_context record for a given session (same value repeated per turn).
    # Captured once at first occurrence -- like session_instructions/session_git
    # above -- and emitted as a single one-time session_event after the loop,
    # rather than duplicated onto every turn_context event.
    session_agent_role: str | None = None
    session_agent_nickname: str | None = None
    session_model_provider: str | None = None
    session_developer_instructions: str | None = None

    for idx, item in enumerate(record_list, start=1):
        record = _dict_record(item)
        if record is None:
            continue

        # Handle compaction events (before message check so they don't fall through)
        if _record_type(record) == "compacted":
            timestamp = _iso_or_none(_record_timestamp(record))
            payload = _payload_record(record) or {}
            history = payload.get("replacement_history")
            history_list = history if isinstance(history, list) else []
            event_payload: dict[str, object] = {
                "source_index": idx,
                "summary": str(payload.get("message", "") or ""),
                "replacement_history_count": len(history_list),
            }
            # replacement_history re-embeds the exact pre-compaction records
            # (message/reasoning/ghost_snapshot) already parsed once from the
            # live stream earlier in this file -- storing them again here
            # would duplicate full message content. What it adds beyond the
            # count is per-item annotation Codex doesn't emit on the live
            # stream: an internal generation `phase` tag on content items, a
            # `ghost_commit` on some entries, and inline images. Those are
            # captured as bounded aggregates, not raw duplication.
            phase_counts: dict[str, int] = {}
            ghost_commit_count = 0
            image_count = 0
            for entry in history_list:
                if not isinstance(entry, dict):
                    continue
                if isinstance(entry.get("ghost_commit"), dict):
                    ghost_commit_count += 1
                entry_content = entry.get("content")
                if isinstance(entry_content, list):
                    for content_item in entry_content:
                        if not isinstance(content_item, dict):
                            continue
                        phase = content_item.get("phase")
                        if isinstance(phase, str) and phase:
                            phase_counts[phase] = phase_counts.get(phase, 0) + 1
                        if isinstance(content_item.get("image_url"), str | dict):
                            image_count += 1
            if phase_counts:
                event_payload["replacement_history_phase_counts"] = dict(sorted(phase_counts.items()))
            if ghost_commit_count:
                event_payload["replacement_history_ghost_commit_count"] = ghost_commit_count
            if image_count:
                event_payload["replacement_history_image_count"] = image_count
            session_events.append(
                ParsedSessionEvent(
                    event_type="compaction",
                    timestamp=timestamp,
                    payload=event_payload,
                )
            )
            # Materialize the compaction summary as a real message at the
            # boundary, mirroring Claude Code, so both providers present a uniform
            # summary message that replaces the prior context (#2467). The
            # pre-compaction messages stay stored once; the boundary marks where
            # context discontinues.
            summary_text = str(event_payload["summary"])
            if summary_text:
                messages.append(
                    ParsedMessage(
                        provider_message_id=synthetic_message_id(
                            role=Role.SYSTEM,
                            text=summary_text,
                            timestamp=timestamp,
                            kind="codex-compaction-summary",
                        ),
                        role=Role.SYSTEM,
                        text=summary_text,
                        timestamp=timestamp,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)],
                        message_type=MessageType.SUMMARY,
                        position=message_position,
                        variant_index=0,
                        is_active_path=True,
                    )
                )
                message_position += 1
            continue

        # Handle turn-context events
        if _record_type(record) == "turn_context":
            timestamp = _iso_or_none(_record_timestamp(record))
            tc_payload: dict[str, object] = {}
            turn_payload = _payload_record(record)
            if turn_payload:
                tc_payload["source_index"] = idx
                normalized_turn_context = _turn_context_payload(turn_payload)
                cwd = _extract_cwd(turn_payload)
                if cwd:
                    tc_payload["cwd"] = cwd
                    working_directories.add(cwd)
                if model_name := _string_field(normalized_turn_context, "model", "model_name"):
                    current_model_name = model_name
                    tc_payload["model"] = model_name
                if model_effort := _string_field(normalized_turn_context, "effort", "model_effort"):
                    current_model_effort = model_effort
                    tc_payload["effort"] = model_effort
                # `personality`/`summary`/`collaboration_mode` were unread
                # (polylogue-cgfy triage, codex lane): agent-persona,
                # reasoning-summary-verbosity, and collaboration-mode knobs
                # reported on every turn_context alongside model/effort, but
                # never carried through. `collaboration_mode.settings`
                # duplicates model/effort/developer_instructions already
                # captured from the top-level turn_context -- only the mode
                # name itself is new.
                if personality := _string_field(normalized_turn_context, "personality"):
                    tc_payload["personality"] = personality
                if reasoning_summary := _string_field(normalized_turn_context, "summary"):
                    tc_payload["reasoning_summary"] = reasoning_summary
                collaboration_mode_raw = _dict_record(normalized_turn_context.get("collaboration_mode"))
                if collaboration_mode_raw is not None:
                    collaboration_mode_name = _string_field(collaboration_mode_raw, "mode")
                    if collaboration_mode_name:
                        tc_payload["collaboration_mode"] = collaboration_mode_name
                # Emit agent_policy event when policy fields are present.
                # Payload keys match what _write_session_events expects via
                # _payload_string(event.payload, "approval_policy") etc.
                approval_policy = _string_field(normalized_turn_context, "approval_policy")
                sandbox_raw = normalized_turn_context.get("sandbox_policy")
                if approval_policy or sandbox_raw is not None:
                    policy_payload: dict[str, object] = {}
                    if approval_policy:
                        policy_payload["approval_policy"] = approval_policy
                    if isinstance(sandbox_raw, dict):
                        # Older Codex CLI builds key the sandbox kind as
                        # "mode" (workspace-write, read-only); newer builds
                        # use "type" (e.g. danger-full-access). Both are the
                        # same fact under different wire spellings.
                        mode = _string_field(sandbox_raw, "mode", "type")
                        if mode:
                            policy_payload["sandbox_policy"] = mode
                        network_val = sandbox_raw.get("network_access")
                        if network_val is not None:
                            policy_payload["network_policy"] = str(network_val).lower()
                        for flag_key in ("exclude_slash_tmp", "exclude_tmpdir_env_var"):
                            flag_val = sandbox_raw.get(flag_key)
                            if isinstance(flag_val, bool):
                                policy_payload[flag_key] = flag_val
                    elif isinstance(sandbox_raw, str) and sandbox_raw:
                        policy_payload["sandbox_policy"] = sandbox_raw
                    if policy_payload:
                        session_events.append(
                            ParsedSessionEvent(
                                event_type="agent_policy",
                                timestamp=timestamp,
                                payload=policy_payload,
                            )
                        )
                # Truncation policy and structured-output schema are small,
                # bounded turn-scoped config -- safe to carry on every
                # turn_context event (unlike the large instruction texts
                # below, they legitimately vary turn to turn).
                truncation_policy = _dict_record(normalized_turn_context.get("truncation_policy"))
                if truncation_policy:
                    tc_payload["truncation_policy"] = dict(truncation_policy)
                final_output_schema = _dict_record(normalized_turn_context.get("final_output_json_schema"))
                if final_output_schema:
                    tc_payload["final_output_json_schema"] = dict(final_output_schema)
                # `user_instructions` is the session-level system prompt
                # (CLAUDE.md/AGENTS.md-style content), re-declared on every
                # turn in this record generation. Fold it into the same
                # dedup slot the legacy per-session `instructions` field
                # uses instead of duplicating the full text per turn.
                if not session_instructions:
                    session_instructions = _string_value(normalized_turn_context.get("user_instructions"))
                # `developer_instructions` is a distinct, usually
                # subagent-role-specific prompt (e.g. "You are an awaiter.").
                # Captured once per session via the same one-time identity
                # event as agent_role/agent_nickname below.
                if not session_developer_instructions:
                    session_developer_instructions = _string_value(
                        normalized_turn_context.get("developer_instructions")
                    )
            session_events.append(
                ParsedSessionEvent(
                    event_type="turn_context",
                    timestamp=timestamp,
                    payload=tc_payload,
                )
            )
            continue

        if _record_type(record) in {"response_item", "event_msg"}:
            inner = _payload_record(record)
            if inner is not None and not _is_message(inner):
                event_payload = _compact_response_payload(
                    inner,
                    index=idx,
                    current_model_name=current_model_name,
                    current_model_effort=current_model_effort,
                )
                session_events.append(
                    ParsedSessionEvent(
                        event_type=_codex_response_item_event_type(_record_type(inner), _record_type(record)),
                        timestamp=_iso_or_none(_record_timestamp(inner) or _record_timestamp(record)),
                        payload=event_payload,
                    )
                )
                timestamp_fallback = _record_timestamp(record)
                tool_message = _codex_tool_message(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                    exec_envelope=code_mode_envelopes.get(idx),
                )
                if tool_message is not None:
                    messages.append(tool_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, tool_message.timestamp)
                    # code_mode_envelopes maps BOTH the call record's index
                    # and the matching output record's index to the same
                    # (results-enriched) envelope object -- only emit the
                    # child-result evidence once, on the output/tool_result
                    # message, not again on the call/tool_use message.
                    if _record_type(inner) in {
                        "function_call_output",
                        "custom_tool_call_output",
                        "tool_search_output",
                        "web_search_output",
                    }:
                        exec_envelope_for_message = code_mode_envelopes.get(idx)
                        if exec_envelope_for_message is not None and exec_envelope_for_message.results:
                            session_events.extend(
                                _code_mode_child_result_evidence_events(
                                    exec_envelope_for_message,
                                    source_message_provider_id=tool_message.provider_message_id,
                                    timestamp=tool_message.timestamp,
                                )
                            )
                event_message = _codex_event_message(
                    inner,
                    index=idx,
                    position=message_position,
                    response_signatures=response_signatures,
                    timestamp_fallback=timestamp_fallback,
                )
                if event_message is not None:
                    messages.append(event_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, event_message.timestamp)
                reasoning_message = _codex_reasoning_message(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                )
                if reasoning_message is not None:
                    messages.append(reasoning_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, reasoning_message.timestamp)
                mcp_messages = _codex_mcp_tool_call_messages(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                )
                if mcp_messages is not None:
                    messages.extend(mcp_messages)
                    message_position += len(mcp_messages)
                    for mcp_message in mcp_messages:
                        latest_message_timestamp = _newer_timestamp(latest_message_timestamp, mcp_message.timestamp)
                cwd = _extract_cwd(event_payload)
                if cwd:
                    working_directories.add(cwd)
                continue

        # World-state snapshots (full or delta) report ambient runtime
        # context -- most notably the live subagent roster
        # (`state.environments.subagents`) -- outside the
        # session_meta/turn_context/response_item shapes handled above, so
        # they previously fell through the whole dispatch chain unrecorded.
        # Only `environments` is carried: the other `state` keys on a full
        # snapshot (agents_md/apps_instructions/skills) are large repeated
        # context-file text with no ranked evidence of parser blindness yet.
        if _record_type(record) == "world_state":
            world_payload = _payload_record(record) or {}
            state = _dict_record(world_payload.get("state"))
            environments = _dict_record(state.get("environments")) if state else None
            if environments:
                session_events.append(
                    ParsedSessionEvent(
                        event_type="world_state",
                        timestamp=_iso_or_none(_record_timestamp(record)),
                        payload={"source_index": idx, "environments": dict(environments)},
                    )
                )
            continue

        session_meta = _session_meta_record(record)
        if session_meta is not None:
            meta_id = _record_id(session_meta)
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                if len(session_metas_seen) == 1:
                    session_id = meta_id
                    session_timestamp_pair = parse_timestamp_pair(_record_timestamp(session_meta))
                    session_timestamp = session_timestamp_pair[1] if session_timestamp_pair is not None else None
                    # Lineage markers live on the child's own (first) meta only.
                    forked_val = session_meta.get("forked_from_id")
                    if isinstance(forked_val, str) and forked_val.strip():
                        forked_from_id = forked_val.strip()
                    source_val = session_meta.get("source")
                    if isinstance(source_val, dict) and isinstance(source_val.get("subagent"), dict):
                        is_subagent_spawn = True
                    cwd_val = session_meta.get("cwd")
                    if isinstance(cwd_val, str) and cwd_val.strip():
                        first_meta_cwd = cwd_val.strip()
                    first_meta_git = _git_context(session_meta)
                    if first_meta_git is not None:
                        repo_val = first_meta_git.get("repository_url")
                        if isinstance(repo_val, str) and repo_val.strip():
                            first_meta_repo_url = repo_val.strip()
                elif len(session_metas_seen) == 2:
                    # The second distinct session_meta is the legacy-fallback
                    # candidate parent (see the CONTINUATION classification
                    # below) -- capture its own facts independently of
                    # `session_git`/`session_timestamp`, which track the
                    # first (child) meta only.
                    second_meta_timestamp_pair = parse_timestamp_pair(_record_timestamp(session_meta))
                    cwd_val = session_meta.get("cwd")
                    if isinstance(cwd_val, str) and cwd_val.strip():
                        second_meta_cwd = cwd_val.strip()
                    second_meta_git = _git_context(session_meta)
                    if second_meta_git is not None:
                        repo_val = second_meta_git.get("repository_url")
                        if isinstance(repo_val, str) and repo_val.strip():
                            second_meta_repo_url = repo_val.strip()
            git_context = _git_context(session_meta)
            if git_context and not session_git:
                session_git = git_context
            instructions = _record_instructions(session_meta)
            if not instructions:
                # Newer session_meta records carry `base_instructions` as a
                # {"text": ...} wrapper instead of the legacy flat string.
                base_instructions = _dict_record(session_meta.get("base_instructions"))
                if base_instructions:
                    instructions = _string_value(base_instructions.get("text"))
            if instructions and not session_instructions:
                session_instructions = instructions
            if not session_agent_role:
                session_agent_role = _string_field(session_meta, "agent_role")
            if not session_agent_nickname:
                session_agent_nickname = _string_field(session_meta, "agent_nickname")
            if not session_model_provider:
                session_model_provider = _string_field(session_meta, "model_provider")
            continue

        message_record = _message_record(record)
        if message_record is not None:
            raw_role = _effective_role(message_record)
            content = _effective_content(message_record)
            text = extract_codex_text(content)
            inline_image_summaries = _codex_inline_image_summaries(content)
            if inline_image_summaries:
                text = "\n".join((text, *inline_image_summaries)) if text else "\n".join(inline_image_summaries)
            timestamp_pair = parse_timestamp_pair(_message_timestamp(record, message_record))
            timestamp = timestamp_pair[1] if timestamp_pair is not None else None

            content_blocks = content_blocks_from_segments(content)
            content_blocks.extend(
                ParsedContentBlock(type=BlockType.TEXT, text=summary) for summary in inline_image_summaries
            )
            has_structured = any(
                cb.type in (BlockType.TOOL_USE, BlockType.TOOL_RESULT, BlockType.THINKING) for cb in content_blocks
            )
            if not raw_role or raw_role == "unknown":
                continue
            if not text and not has_structured:
                continue
            role = Role.normalize(raw_role)

            msg_id = _record_id(message_record) or ""
            if not content_blocks and text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            token_usage = _token_usage(message_record)
            model_name = _string_field(message_record, "model", "model_name") or current_model_name
            model_effort = _string_field(message_record, "effort", "model_effort") or current_model_effort
            duration_ms = _optional_int_field(message_record, "duration_ms", "durationMs", "elapsed_ms")

            message_type = _message_type_from_codex_message(message_record, text)
            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    blocks=content_blocks,
                    message_type=message_type,
                    material_origin=_codex_material_origin(role, message_type, text),
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                    input_tokens=token_usage["input_tokens"],
                    output_tokens=token_usage["output_tokens"],
                    cache_read_tokens=token_usage["cache_read_tokens"],
                    cache_write_tokens=token_usage["cache_write_tokens"],
                    model_name=model_name,
                    model_effort=model_effort,
                    duration_ms=duration_ms,
                )
            )
            message_position += 1
            latest_message_timestamp = _newer_timestamp_pair(latest_message_timestamp, timestamp_pair)

    # Emit the deduped subagent/session identity facts (agent_role,
    # agent_nickname, model_provider from session_meta; developer_instructions
    # from turn_context) once, if any were observed, instead of once per
    # session_meta/turn_context occurrence.
    identity_payload: dict[str, object] = {}
    if session_agent_role:
        identity_payload["agent_role"] = session_agent_role
    if session_agent_nickname:
        identity_payload["agent_nickname"] = session_agent_nickname
    if session_model_provider:
        identity_payload["model_provider"] = session_model_provider
    if session_developer_instructions:
        identity_payload["developer_instructions"] = session_developer_instructions
    if identity_payload:
        session_events.append(
            ParsedSessionEvent(
                event_type="codex_agent_identity",
                timestamp=session_timestamp,
                payload=identity_payload,
            )
        )

    # Lineage: prefer the explicit markers on the child's own session_meta.
    #   - `source.subagent.thread_spawn` → spawned subagent (positive evidence
    #     of a subagent relationship): assign SUBAGENT.
    #   - `forked_from_id` (no subagent block) → the child shares the parent's
    #     leading context prefix, but Codex sets this field for BOTH a divergent
    #     user fork AND a plain resume of the same thread. The marker proves a
    #     parent, not the relationship *type*. Assigning FORK here over-claimed:
    #     a resume was recorded as a fork. Leave the type unclassified
    #     (`None` → generic topology link, no `sessions.branch_type`) rather
    #     than fabricate FORK from absent evidence. The prefix-sharing
    #     normalization still records the branch point + `inheritance`, so the
    #     shared-prefix fact is preserved.
    # Fall back to the legacy heuristic (older exports with no `forked_from_id`
    # field at all) when no explicit marker is present: a second distinct
    # session_meta *can* be the replayed parent header of a plain resume, but
    # only when it carries the structural evidence of that -- see
    # `_has_continuation_evidence`. A second session_meta id with none of that
    # evidence is not proof of any relationship (e.g. two structurally
    # unrelated session_metas concatenated in one payload), so it stays fully
    # unclassified rather than fabricating CONTINUATION from a bare count.
    if forked_from_id is not None:
        parent_id: str | None = forked_from_id
        branch_type = BranchType.SUBAGENT if is_subagent_spawn else None
    elif len(session_metas_seen) > 1 and _has_continuation_evidence(
        first_timestamp=session_timestamp_pair,
        second_timestamp=second_meta_timestamp_pair,
        first_cwd=first_meta_cwd,
        second_cwd=second_meta_cwd,
        first_repo_url=first_meta_repo_url,
        second_repo_url=second_meta_repo_url,
    ):
        parent_id = session_metas_seen[1]
        branch_type = BranchType.CONTINUATION
    else:
        parent_id = None
        branch_type = None

    updated_at_pair = _newer_timestamp_pair(session_timestamp_pair, latest_message_timestamp)

    git_branch_typed: str | None = None
    git_repo_url_typed: str | None = None
    git_commit_hash_typed: str | None = None
    if session_git is not None:
        branch_val = session_git.get("branch")
        if isinstance(branch_val, str) and branch_val.strip():
            git_branch_typed = branch_val.strip()
        repo_val = session_git.get("repository_url")
        if isinstance(repo_val, str) and repo_val.strip():
            git_repo_url_typed = repo_val.strip()
        # commit_hash pins the session to an exact commit — the strongest
        # attribution signal codex provides. Previously kept only inside
        # provider_meta.git where downstream readers had to JSON-extract;
        # now graduated to a typed top-level field.
        commit_val = session_git.get("commit_hash")
        if isinstance(commit_val, str) and commit_val.strip():
            git_commit_hash_typed = commit_val.strip()
    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    messages = mark_last_occurrence_as_active_leaf(messages)
    # bd polylogue-ksgg: Codex rollout messages carry no parent-message
    # evidence at all (0% parented, 0 variant_index>0 rows) -- a strictly
    # linear turn sequence. Chain each message to the previous one so
    # readers of `parent_message_id` don't need origin-specific fallback to
    # position order.
    messages = fill_linear_parent_chain(messages)

    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=updated_at_pair[1] if updated_at_pair is not None else None,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        session_events=session_events,
        parent_session_provider_id=parent_id,
        branch_type=branch_type,
        instructions_text=session_instructions,
        working_directories=sorted(working_directories),
        git_branch=git_branch_typed,
        git_repository_url=git_repo_url_typed,
        git_commit_hash=git_commit_hash_typed,
    )


def parse(payload: Sequence[object], fallback_id: str) -> ParsedSession:
    return _parse_records(payload, fallback_id)


def parse_stream(records: Iterable[object], fallback_id: str) -> ParsedSession:
    return _parse_records(records, fallback_id)
