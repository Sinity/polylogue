"""Codex JSONL session parser."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.provider.semantics import extract_codex_text
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.timestamps import parse_timestamp_pair
from polylogue.logging import get_logger
from polylogue.sources.providers.codex import CodexRecord

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    content_blocks_from_segments,
)

logger = get_logger(__name__)
_TimestampPair = tuple[datetime, str]


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
    usage = _dict_record(record.get("usage")) or _dict_record(record.get("tokens")) or record
    return {
        "input_tokens": _int_value(usage.get("input_tokens")),
        "output_tokens": _int_value(usage.get("output_tokens")),
        "cache_read_tokens": _int_value(usage.get("cache_read_tokens") or usage.get("cache_read_input_tokens")),
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
        context_window = _optional_int_field(info, "model_context_window") or _optional_int_field(
            payload, "model_context_window"
        )
        if context_window is not None:
            compact["model_context_window"] = context_window
    return compact


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


def _codex_tool_message(record: dict[str, object], *, index: int, position: int) -> ParsedMessage | None:
    payload = _record_payload(record)
    record_type = _record_type(record)
    timestamp = _iso_or_none(_record_timestamp(record))
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
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
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
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id=str(tool_id) if tool_id else None,
                    text=output_text,
                )
            ],
        )
    return None


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
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata
    working_directories: set[str] = set()
    current_model_name: str | None = None
    current_model_effort: str | None = None
    message_position = 0

    for idx, item in enumerate(records, start=1):
        record = _dict_record(item)
        if record is None:
            continue

        # Handle compaction events (before message check so they don't fall through)
        if _record_type(record) == "compacted":
            timestamp = _iso_or_none(_record_timestamp(record))
            payload = _payload_record(record) or {}
            history = payload.get("replacement_history")
            event_payload: dict[str, object] = {
                "source_index": idx,
                "summary": str(payload.get("message", "") or ""),
                "replacement_history_count": len(history) if isinstance(history, list) else 0,
            }
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
                        provider_message_id=f"compaction-summary-{idx}",
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
                        mode = _string_field(sandbox_raw, "mode")
                        if mode:
                            policy_payload["sandbox_policy"] = mode
                        network_val = sandbox_raw.get("network_access")
                        if network_val is not None:
                            policy_payload["network_policy"] = str(network_val).lower()
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
                        event_type=_record_type(inner) or _record_type(record) or "response_item",
                        timestamp=_iso_or_none(_record_timestamp(inner) or _record_timestamp(record)),
                        payload=event_payload,
                    )
                )
                tool_message = _codex_tool_message(inner, index=idx, position=message_position)
                if tool_message is not None:
                    messages.append(tool_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, tool_message.timestamp)
                cwd = _extract_cwd(event_payload)
                if cwd:
                    working_directories.add(cwd)
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
            git_context = _git_context(session_meta)
            if git_context and not session_git:
                session_git = git_context
            instructions = _record_instructions(session_meta)
            if instructions and not session_instructions:
                session_instructions = instructions
            continue

        message_record = _message_record(record)
        if message_record is not None:
            raw_role = _effective_role(message_record)
            content = _effective_content(message_record)
            text = extract_codex_text(content)
            timestamp_pair = parse_timestamp_pair(_record_timestamp(message_record))
            timestamp = timestamp_pair[1] if timestamp_pair is not None else None

            content_blocks = content_blocks_from_segments(content)
            has_structured = any(
                cb.type in (BlockType.TOOL_USE, BlockType.TOOL_RESULT, BlockType.THINKING) for cb in content_blocks
            )
            if not raw_role or raw_role == "unknown":
                continue
            if not text and not has_structured:
                continue
            role = Role.normalize(raw_role)

            msg_id = _record_id(message_record) or f"msg-{idx}"
            if not content_blocks and text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            token_usage = _token_usage(message_record)
            model_name = _string_field(message_record, "model", "model_name") or current_model_name
            model_effort = _string_field(message_record, "effort", "model_effort") or current_model_effort
            duration_ms = _optional_int_field(message_record, "duration_ms", "durationMs", "elapsed_ms")

            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    blocks=content_blocks,
                    message_type=_message_type_from_codex_message(message_record, text),
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

    # Lineage: prefer the explicit markers on the child's own session_meta.
    #   - `source.subagent.thread_spawn` → spawned subagent (inherits a prefix)
    #   - `forked_from_id` (no subagent block) → user fork / resume of the parent
    # Both copy the parent's context as a leading prefix into this rollout; the
    # archive write path normalizes that copy into a lineage edge + branch point.
    # Fall back to the legacy heuristic (a second embedded session_meta id) when
    # no explicit marker is present.
    if forked_from_id is not None:
        parent_id: str | None = forked_from_id
        branch_type = BranchType.SUBAGENT if is_subagent_spawn else BranchType.FORK
    elif len(session_metas_seen) > 1:
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
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]

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
