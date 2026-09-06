"""Parsers for local agent session JSON documents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, BranchType, Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.timestamps import format_timestamp
from polylogue.sources.live.gemini_tool_output_sidecars import (
    is_masked_tool_output,
    join_gemini_tool_output_sidecars,
    resolve_tool_outputs_dir,
)
from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult
from polylogue.sources.parsers.hermes_tool_outcome import tool_result_outcome as hermes_tool_result_outcome
from polylogue.sources.tool_result_reasons import unknown_reason

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    fill_linear_parent_chain,
    human_authored_override,
    mark_last_occurrence_as_active_leaf,
    parser_admission,
)
from .hermes_finish_reason import end_turn_from_finish_reason as _end_turn_from_finish_reason
from .hermes_finish_reason import stop_reason_from_finish_reason as _stop_reason_from_finish_reason
from .hermes_identity import profile_key as _profile_key
from .hermes_identity import profile_root_for_session_snapshot as _profile_root_for_session_snapshot
from .hermes_identity import qualified_session_id as _qualified_session_id


# polylogue-9x22: ``ParsedContentBlock.metadata`` is never persisted -- the
# ``blocks`` table has no metadata column and the write path only reads a
# ``language`` key back out of it (``storage/sqlite/archive_tiers/write.py:
# _block_language``). ``_tool_metadata`` (shared by both gemini-cli and
# hermes tool_use/tool_result blocks) and ``_parse_gemini_message``'s
# "thought" blocks (subject/timestamp/index) still attach data to
# ``metadata`` as an in-process carrier; project it into ``session_events``
# instead -- same precedent as ``claude/common.py``'s
# ``claude_ai_web_tool_evidence`` and ``chatgpt.py``'s
# ``chatgpt_block_metadata``. One event per block carrying non-empty
# metadata, whole dict verbatim (no fixed cross-provider vocabulary here).
def _block_metadata_evidence_events(messages: list[ParsedMessage]) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    for message in messages:
        for block_index, block in enumerate(message.blocks):
            if not block.metadata:
                continue
            events.append(
                ParsedSessionEvent(
                    event_type="local_agent_block_metadata",
                    timestamp=message.timestamp,
                    source_message_provider_id=message.provider_message_id,
                    payload={"block_index": block_index, **dict(block.metadata)},
                )
            )
    return events


#: Gemini CLI's own "kind" enum for a chat/session checkpoint (present on
#: both wire shapes below).
_GEMINI_CLI_KIND_VALUES = frozenset({"chat", "main", "subagent"})


def gemini_cli_chat_identity(payload: JSONDocument, session_id: str) -> str:
    """Compose the identity of one Gemini CLI chat from its wire coordinates.

    ``sessionId`` names the CLI *process*, not a chat. One process writes a
    separate complete checkpoint for its main chat, for every subagent it
    spawns, and for every chat opened after a reset -- all under that one
    ``sessionId``, with disjoint message sets. Keyed on ``sessionId`` alone,
    those distinct chats full-replace each other.

    ``kind`` and ``startTime`` are the wire's own coordinates for which chat a
    checkpoint holds, and both are fixed when the chat opens: a checkpoint
    rewritten days later still carries its opening ``startTime``. Composing
    them separates sibling chats while keeping every save of one chat on one
    identity, which ``lastUpdated`` would not.

    Path coordinates stay unused -- ``Provider.GEMINI_CLI`` is declared
    path-independent for revision dedup
    (``revision_backfill._PATH_INDEPENDENT_PARSE_PROVIDERS``).
    """
    kind = _string(payload.get("kind"))
    start_time = _string(payload.get("startTime"))
    return ":".join(
        (
            session_id,
            kind if kind in _GEMINI_CLI_KIND_VALUES else "",
            start_time or "",
        )
    )


def looks_like_gemini_cli(payload: JSONDocument) -> bool:
    """Detect a Gemini CLI checkpoint document in either of its two shapes.

    The common "one JSON object per session, ``messages`` embedded" shape
    (``sessionId`` + a ``messages`` list + ``startTime``/``lastUpdated``/
    ``kind``) is the original detector. Gemini CLI also has a genuinely
    different on-disk shape for its ``.jsonl`` chat-log checkpoints: a
    session-*open* stub record (``sessionId`` + ``projectHash`` + ``kind``,
    written the instant a session starts, before any turn exists) followed
    by one JSON object per turn/event on subsequent lines -- the stub itself
    carries no ``messages`` key at all. Without a positive check for that
    stub shape, its only strong-looking field is a bare ``sessionId``, which
    also happens to be one of Claude Code's own
    ``code_detection._STRONG_SESSION_KEYS`` -- so a freshly-opened Gemini CLI
    session (no turns yet, or read mid-write) silently misclassified as
    ``claude-code-session`` (polylogue-hs3y, 4 confirmed live archive rows
    under ``~/.gemini/tmp/*/chats/*.jsonl``). ``projectHash`` is unique to
    Gemini CLI's own checkpoint envelope, so requiring it alongside the
    ``kind`` enum keeps this branch as tight as the ``messages``-bearing one.
    """
    if not isinstance(payload.get("sessionId"), str):
        return False
    if isinstance(payload.get("messages"), list):
        return "startTime" in payload or "lastUpdated" in payload or payload.get("kind") in _GEMINI_CLI_KIND_VALUES
    return isinstance(payload.get("projectHash"), str) and payload.get("kind") in _GEMINI_CLI_KIND_VALUES


def looks_like_hermes(payload: JSONDocument) -> bool:
    return (
        isinstance(payload.get("session_id"), str)
        and isinstance(payload.get("messages"), list)
        and ("session_start" in payload or "last_updated" in payload or "platform" in payload)
    )


@parser_admission("gemini_cli")
def parse_gemini_cli(
    payload: JSONDocument,
    fallback_id: str,
    *,
    source_path: str | Path | None = None,
) -> ParsedSession:
    session_id = _string(payload.get("sessionId")) or fallback_id
    chat_id = gemini_cli_chat_identity(payload, session_id)
    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    models_used: set[str] = set()
    for index, item in enumerate(_list(payload.get("messages")), start=1):
        parsed = _parse_gemini_message(item, index=index, position=len(messages))
        if parsed is not None:
            messages.append(parsed)
            if parsed.model_name:
                models_used.add(parsed.model_name)
            if usage_event := _gemini_message_usage_event(item, parsed):
                session_events.append(usage_event)
    # bd polylogue-ksgg: Gemini CLI sessions carry no parent-message evidence
    # (0% parented, 0 variant_index>0 rows) -- a linear turn sequence. Chain
    # each message to the previous one on the active path.
    messages = fill_linear_parent_chain(messages)
    messages = _mark_active_leaf(messages)
    if metadata_event := _gemini_cli_session_metadata_event(payload, message_count=len(messages)):
        session_events.append(metadata_event)
    if scratchpad_event := _gemini_cli_memory_scratchpad_event(payload):
        session_events.append(scratchpad_event)
    session_events.extend(_block_metadata_evidence_events(messages))
    session = ParsedSession(
        source_name=Provider.GEMINI_CLI,
        provider_session_id=chat_id,
        title=_string(payload.get("summary")) or chat_id,
        created_at=_string(payload.get("startTime")),
        updated_at=_string(payload.get("lastUpdated")),
        messages=messages,
        branch_type=BranchType.SUBAGENT if payload.get("kind") == "subagent" else None,
        session_events=session_events,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
        models_used=sorted(models_used),
        provider_project_ref=_string(payload.get("projectHash")),
        working_directories=[
            directory for directory in _list(payload.get("directories")) if isinstance(directory, str) and directory
        ],
    )
    # The sidecar directory on disk is named for the wire ``sessionId``
    # (``tool-outputs/session-<sessionId>/``), which all of a process's chats
    # share -- not for the composed chat identity.
    tool_outputs_dir = resolve_tool_outputs_dir(source_path, session_id)
    if tool_outputs_dir is not None:
        session = apply_gemini_tool_output_sidecars(
            session,
            join_gemini_tool_output_sidecars(payload, tool_outputs_dir),
        )
    return session


def apply_gemini_tool_output_sidecars(session: ParsedSession, join_result: SidecarJoinResult) -> ParsedSession:
    """Attach acquired ``tool-outputs/`` sidecar content to its owning blocks.

    Never adds a message and never touches session identity: a truncated
    sidecar's full text replaces its ``tool_result`` block's masked text in
    place (the envelope is a both-ends truncation, so the inline text is not a
    prefix and nothing is appended to it), and every sidecar -- matched or debt
    -- is recorded as a bounded ``gemini_cli_tool_output_sidecar`` session
    event carrying the file's identity and size, never its bytes.
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
            updated_messages.append(
                message.model_copy(
                    update={
                        "blocks": [
                            block.model_copy(update={"text": replacements[block.tool_id].full_text})
                            if block.type is BlockType.TOOL_RESULT and block.tool_id in replacements
                            else block
                            for block in message.blocks
                        ]
                    }
                )
            )
        messages = updated_messages

    events = list(session.session_events)
    for match in join_result.matched:
        events.append(
            ParsedSessionEvent(
                event_type="gemini_cli_tool_output_sidecar",
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
                event_type="gemini_cli_tool_output_sidecar",
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
    """A sidecar file's own mtime as the ISO timestamp for its session event.

    The join has no better time source: these files carry no embedded
    timestamp, and for debt the owning tool call is by definition unresolved.
    """
    if file_mtime_ms is None:
        return None
    return format_timestamp(file_mtime_ms / 1000.0)


@parser_admission("hermes")
def parse_hermes(
    payload: JSONDocument,
    fallback_id: str,
    *,
    source_path: str | Path | None = None,
) -> ParsedSession:
    """Parse one ``<hermes_root>/sessions/session_*.json`` snapshot.

    ``source_path`` carries the profile qualifier: this snapshot family and
    ``state.db`` (``hermes_state.py``) describe the same logical Hermes
    sessions, and both must build identity from
    ``hermes_identity.qualified_session_id`` off the same install root or one
    conversation lands as two archive sessions. Without a path no profile is
    assertable, so identity stays unqualified rather than inventing a key.
    """
    raw_session_id = _string(payload.get("session_id")) or fallback_id
    session_id = _hermes_qualified_session_id(raw_session_id, source_path)
    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    system_prompt = _string(payload.get("system_prompt"))
    if system_prompt:
        messages.append(
            ParsedMessage(
                provider_message_id=f"{session_id}:system",
                role=Role.SYSTEM,
                text=system_prompt,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=system_prompt)],
                position=0,
                variant_index=0,
                is_active_path=True,
                model_name=_string(payload.get("model")),
            )
        )
    for index, item in enumerate(_list(payload.get("messages")), start=1):
        parsed = _parse_hermes_message(
            item,
            index=index,
            position=len(messages),
            fallback_model=_string(payload.get("model")),
        )
        if parsed is not None:
            messages.append(parsed)
            if extras_event := _hermes_message_wire_extras_event(item, parsed):
                session_events.append(extras_event)
    messages = _mark_active_leaf(messages)
    if metadata_event := _hermes_session_metadata_event(payload, message_count=len(messages)):
        session_events.append(metadata_event)
    if tool_event := _hermes_tool_availability_event(payload):
        session_events.append(tool_event)
    session_events.extend(_block_metadata_evidence_events(messages))
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=raw_session_id,
        created_at=_string(payload.get("session_start")),
        updated_at=_string(payload.get("last_updated")),
        messages=messages,
        session_events=session_events,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
    )


def _hermes_qualified_session_id(raw_session_id: str, source_path: str | Path | None) -> str:
    if source_path is None:
        return raw_session_id
    return _qualified_session_id(
        raw_session_id,
        _profile_key(_profile_root_for_session_snapshot(Path(source_path))),
    )


def _parse_gemini_message(item: object, *, index: int, position: int) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    thoughts = _list(record.get("thoughts"))
    for thought_index, thought in enumerate(thoughts, start=1):
        thought_record = json_document(thought)
        thought_text = (
            _string(thought_record.get("description"))
            or _content_text(thought)
            or _string(thought_record.get("subject"))
        )
        if thought_text:
            thought_metadata: dict[str, object] = {"index": thought_index}
            for key in ("subject", "timestamp"):
                value = thought_record.get(key)
                if isinstance(value, str) and value:
                    thought_metadata[key] = value
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=thought_text,
                    metadata=thought_metadata,
                )
            )
    for tool_index, tool_call in enumerate(_list(record.get("toolCalls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        fallback_tool_id = f"tool-{index}-{tool_index}"
        content_blocks.append(_tool_use_block(tool_record, fallback_id=fallback_tool_id))
        content_blocks.extend(_tool_result_blocks(tool_record, fallback_id=fallback_tool_id))
    # polylogue-auy4z: a turn with no content is still billed, and the
    # checkpoint file is the only place its counts exist -- ``tokens`` is the
    # evidence that the turn happened, so the message is kept without blocks
    # and carries the counts into the cost rollup.
    if not text and not content_blocks and not _reports_wire_tokens(record):
        return None
    token_usage = _token_usage_fields(record)
    gemini_role = _role(_string(record.get("type")) or "unknown", assistant_aliases={"gemini", "model"})
    gemini_blocks = content_blocks or ([ParsedContentBlock(type=BlockType.TEXT, text=text)] if text else [])
    # A block-derived type (tool_use/tool_result from toolCalls above) must be
    # resolved BEFORE classify_material_origin runs, or a genuine tool turn
    # gets misclassified against an assumed plain MESSAGE type.
    gemini_message_type = (
        classify_block_message_type(tuple(block.type for block in gemini_blocks)) or MessageType.MESSAGE
    )
    # ``displayContent`` is the form the user was actually shown: the prompt
    # as typed, before ``@path`` references were expanded into the referenced
    # files' contents. ``content`` is the model-facing expansion, so keeping
    # only it loses the user's own words inside a payload that can be three
    # orders of magnitude larger. Appended after the message type is resolved
    # so a rendered form never reclassifies a tool turn.
    display_text = _content_text(record.get("displayContent"))
    if display_text and display_text != text:
        matching_index = next(
            (
                block_index
                for block_index, block in enumerate(gemini_blocks)
                if block.type is BlockType.TEXT and block.text == display_text
            ),
            None,
        )
        if matching_index is None:
            gemini_blocks = [
                *gemini_blocks,
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    text=display_text,
                    metadata={"gemini_display_content": True},
                ),
            ]
        else:
            matching = gemini_blocks[matching_index]
            gemini_blocks[matching_index] = matching.model_copy(
                update={"metadata": {**(matching.metadata or {}), "gemini_display_content": True}},
            )
    return ParsedMessage(
        # polylogue-slshy: no positional fallback -- empty id lets
        # _message_revision_match_id's content-anchor fallback run instead.
        provider_message_id=_string(record.get("id")) or "",
        role=gemini_role,
        text=text,
        timestamp=_string(record.get("timestamp")),
        blocks=gemini_blocks,
        message_type=gemini_message_type,
        position=position,
        variant_index=0,
        is_active_path=True,
        model_name=_string(record.get("model")),
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        cache_read_tokens=token_usage["cache_read_tokens"],
        cache_write_tokens=token_usage["cache_write_tokens"],
        duration_ms=_non_negative_int(
            record.get("durationMs") or record.get("duration_ms") or record.get("elapsed_ms")
        ),
        # polylogue-gzgyl: Gemini CLI has no agent/subagent artifact ambiguity
        # for a plain user turn -- positive-evidence override for the shared
        # classify_material_origin no-fallthrough (#2502).
        material_origin=human_authored_override(
            gemini_role,
            gemini_message_type,
            classify_material_origin(
                role=gemini_role,
                message_type=gemini_message_type,
                text=text,
                block_types=tuple(block.type for block in gemini_blocks),
            ),
        ),
    )


def _parse_hermes_message(
    item: object,
    *,
    index: int,
    position: int,
    fallback_model: str | None = None,
) -> ParsedMessage | None:
    record = json_document(item)
    if not record:
        return None
    text = _content_text(record.get("content"))
    content_blocks = _content_blocks_from_content(record.get("content"))
    output_text_blocks = _codex_output_text_blocks(record.get("codex_message_items"), covered_text=text)
    content_blocks.extend(output_text_blocks)
    if text is None and output_text_blocks:
        text = "\n".join(block.text for block in output_text_blocks if block.text)
    reasoning = _string(record.get("reasoning_content")) or _string(record.get("reasoning"))
    if reasoning:
        content_blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning))
    content_blocks.extend(_codex_reasoning_blocks(record.get("codex_reasoning_items"), covered_text=reasoning))
    for tool_index, tool_call in enumerate(_list(record.get("tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        content_blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{index}-{tool_index}"))
    tool_call_id = _string(record.get("tool_call_id"))
    role = _role(_string(record.get("role")) or "unknown")
    if role is Role.TOOL and text:
        hermes_is_error, hermes_exit_code, hermes_reason = hermes_tool_result_outcome(record.get("content"))
        content_blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_call_id,
                text=text,
                is_error=hermes_is_error,
                exit_code=hermes_exit_code,
                outcome_unknown_reason=hermes_reason,
            )
        )
    if not text and not content_blocks:
        return None
    token_usage = _token_usage_fields(record)
    hermes_blocks = content_blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)]
    # A block-derived type (tool_use/tool_result above) must be resolved
    # BEFORE classify_material_origin runs, or a genuine tool turn gets
    # misclassified against an assumed plain MESSAGE type.
    hermes_message_type = (
        classify_block_message_type(tuple(block.type for block in hermes_blocks)) or MessageType.MESSAGE
    )
    return ParsedMessage(
        # polylogue-slshy: no positional fallback (see above).
        provider_message_id=tool_call_id or "",
        role=role,
        text=text,
        timestamp=_string(record.get("timestamp")) or _string(record.get("created_at")),
        blocks=hermes_blocks,
        message_type=hermes_message_type,
        position=position,
        variant_index=0,
        is_active_path=True,
        model_name=_string(record.get("model")) or fallback_model,
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        cache_read_tokens=token_usage["cache_read_tokens"],
        cache_write_tokens=token_usage["cache_write_tokens"],
        duration_ms=_non_negative_int(
            record.get("durationMs") or record.get("duration_ms") or record.get("elapsed_ms")
        ),
        end_turn=_end_turn_from_finish_reason(record.get("finish_reason")),
        stop_reason=_stop_reason_from_finish_reason(record.get("finish_reason")),
        # polylogue-gzgyl: this JSON-sidecar Hermes wire path has no
        # agent/subagent artifact ambiguity for a plain user turn --
        # positive-evidence override for the shared classify_material_origin
        # no-fallthrough (#2502). (The separate hermes_state.py state-db path
        # already carries its own correct override.)
        material_origin=human_authored_override(
            role,
            hermes_message_type,
            classify_material_origin(
                role=role,
                message_type=hermes_message_type,
                text=text,
                block_types=tuple(block.type for block in hermes_blocks),
            ),
        ),
    )


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    return mark_last_occurrence_as_active_leaf(messages)


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(float(value))
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _token_usage_fields(record: JSONDocument) -> dict[str, int]:
    usage = json_document(record.get("usage")) or json_document(record.get("tokens")) or record
    gemini_wire_fields = {"input", "output", "cached", "thoughts", "tool"}
    if any(key in usage for key in gemini_wire_fields):
        input_with_cached = _first_non_negative_int(usage, "input") or 0
        cache_read_tokens = _first_non_negative_int(usage, "cached") or 0
        return {
            "input_tokens": max(input_with_cached - cache_read_tokens, 0),
            "output_tokens": _first_non_negative_int(usage, "output") or 0,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": 0,
            "reasoning_output_tokens": _first_non_negative_int(usage, "thoughts") or 0,
            "tool_output_tokens": _first_non_negative_int(usage, "tool") or 0,
            "total_tokens": _first_non_negative_int(usage, "total") or 0,
        }
    input_tokens = _first_non_negative_int(usage, "input_tokens", "prompt_tokens") or 0
    explicit_output = _first_non_negative_int(
        usage,
        "output_tokens",
        "completion_tokens",
        "generated_tokens",
        "total_tokens",
        "total",
    )
    output_tokens = explicit_output or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": _first_non_negative_int(usage, "cache_read_tokens", "cache_read_input_tokens") or 0,
        "cache_write_tokens": _first_non_negative_int(
            usage,
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cache_write_input_tokens",
        )
        or 0,
        "reasoning_output_tokens": 0,
        "tool_output_tokens": 0,
        "total_tokens": _first_non_negative_int(usage, "total_tokens", "total") or 0,
    }


def _first_non_negative_int(payload: JSONDocument, *keys: str) -> int | None:
    for key in keys:
        if key in payload:
            value = _non_negative_int(payload.get(key))
            if value is not None:
                return value
    return None


def _reports_wire_tokens(record: JSONDocument) -> bool:
    """Whether the record carries token counts of its own."""
    if not (json_document(record.get("usage")) or json_document(record.get("tokens"))):
        return False
    return any(_token_usage_fields(record).values())


def _gemini_message_usage_event(item: object, message: ParsedMessage) -> ParsedSessionEvent | None:
    record = json_document(item)
    raw_usage = json_document(record.get("usage")) or json_document(record.get("tokens"))
    if not raw_usage:
        return None
    usage = _token_usage_fields(record)
    last_usage = {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cached_input_tokens": usage["cache_read_tokens"],
        "cache_write_tokens": usage["cache_write_tokens"],
        "reasoning_output_tokens": usage["reasoning_output_tokens"],
        "total_tokens": usage["total_tokens"],
    }
    payload: dict[str, object] = {
        "type": "message_usage",
        "semantics": "per_message",
        "last_token_usage": last_usage,
        "wire_tokens": dict(raw_usage),
    }
    if usage["tool_output_tokens"]:
        payload["tool_output_tokens"] = usage["tool_output_tokens"]
    if message.model_name:
        payload["model"] = message.model_name
    return ParsedSessionEvent(
        event_type="message_usage",
        timestamp=message.timestamp,
        source_message_provider_id=message.provider_message_id,
        payload=payload,
    )


def _gemini_cli_session_metadata_event(payload: JSONDocument, *, message_count: int) -> ParsedSessionEvent | None:
    """Producer-reported session counters (polylogue-5o05): ``userMessageCount``
    and ``hasUserOrAssistantMessage`` were previously dropped entirely. Cheap
    completeness cross-check against the messages actually parsed.
    """
    has_user_or_assistant = payload.get("hasUserOrAssistantMessage")
    reported_count = _non_negative_int(payload.get("userMessageCount"))
    if not isinstance(has_user_or_assistant, bool) and reported_count is None:
        return None
    event_payload: dict[str, object] = {"parsed_message_count": message_count}
    if isinstance(has_user_or_assistant, bool):
        event_payload["has_user_or_assistant_message"] = has_user_or_assistant
    if reported_count is not None:
        event_payload["reported_user_message_count"] = reported_count
    return ParsedSessionEvent(
        event_type="gemini_cli_session_metadata",
        timestamp=_string(payload.get("lastUpdated")),
        payload=event_payload,
    )


def _gemini_cli_memory_scratchpad_event(payload: JSONDocument) -> ParsedSessionEvent | None:
    """Subagent memory-scratchpad summary (``version``/``workflowSummary``/
    ``toolSequence``/``touchedPaths``/``validationStatus``), captured verbatim
    (polylogue-5o05). Distinct from conversation content -- a producer-side
    working-memory snapshot, not something a user or assistant said.
    """
    scratchpad = json_document(payload.get("memoryScratchpad"))
    if not scratchpad:
        return None
    return ParsedSessionEvent(
        event_type="gemini_cli_memory_scratchpad",
        payload={"memory_scratchpad": dict(scratchpad)},
    )


def _hermes_session_metadata_event(payload: JSONDocument, *, message_count: int) -> ParsedSessionEvent | None:
    """Session-level routing/deployment metadata (polylogue-5o05): ``base_url``,
    ``platform``, ``message_count`` were present in 100% of 167 sampled
    JSON-snapshot documents and read by nothing. ``message_count`` is the
    producer's own count -- kept alongside the count we actually parsed as a
    parse-completeness cross-check, not a duplicate of the same fact.
    """
    base_url = _string(payload.get("base_url"))
    platform = _string(payload.get("platform"))
    reported_count = _non_negative_int(payload.get("message_count"))
    if base_url is None and platform is None and reported_count is None:
        return None
    event_payload: dict[str, object] = {"parsed_message_count": message_count}
    if base_url is not None:
        event_payload["base_url"] = base_url
    if platform is not None:
        event_payload["platform"] = platform
    if reported_count is not None:
        event_payload["reported_message_count"] = reported_count
    return ParsedSessionEvent(
        event_type="hermes_session_metadata",
        timestamp=_string(payload.get("last_updated")),
        payload=event_payload,
    )


def _hermes_tool_availability_event(payload: JSONDocument) -> ParsedSessionEvent | None:
    """The full tool-definition schema offered to the model (``tools``,
    ``tools[].function.{name,description,parameters}`` including the nested
    JSON-Schema ``$schema``/``additionalProperties``/``properties``/
    ``required`` keys) -- 100% of 167 sampled documents, read by nothing
    (polylogue-5o05). Materially different signal from tool CALLS (already
    captured on TOOL_USE blocks): which tools were AVAILABLE, not which were
    invoked. Captured verbatim -- this is the wire tool-definition schema, not
    conversation content.
    """
    tools = _list(payload.get("tools"))
    if not tools:
        return None
    return ParsedSessionEvent(
        event_type="hermes_tool_availability",
        payload={"tools": tools, "tool_count": len(tools)},
    )


def _hermes_message_wire_extras_event(item: object, message: ParsedMessage) -> ParsedSessionEvent | None:
    """Message-scoped Hermes wire fields with no home on ``ParsedContentBlock``
    (polylogue-5o05): ``ParsedContentBlock.metadata`` is parse-time-only and is
    never persisted (no ``metadata`` column on ``blocks``; every block read
    path selects a literal ``NULL AS metadata``) -- storing these there would
    silently reproduce the exact defect this triage is fixing. ``session_events``
    is real, durable, and already supports per-message attribution via
    ``source_message_provider_id``, so that is where these land instead.

    - ``codex_reasoning_items``/``codex_message_items`` (~59% of documents):
      reasoning/message-item blobs from a Codex-compatible backend. Stored
      verbatim, same shape ``hermes_state.py``'s SQLite path already captures
      for the equivalent state-db fields.
    - ``_empty_recovery_synthetic``/``_db_persisted`` (low-volume, informational
      producer markers) captured as booleans.
    - ``tool_calls[].extra_content`` (e.g. Google ``thought_signature`` on a
      Gemini-compatible backend's tool call) captured per tool call verbatim.
    """
    record = json_document(item)
    if not record:
        return None
    event_payload: dict[str, object] = {}
    reasoning_items = record.get("codex_reasoning_items")
    if reasoning_items is not None:
        event_payload["codex_reasoning_items"] = reasoning_items
    message_items = record.get("codex_message_items")
    if message_items is not None:
        event_payload["codex_message_items"] = message_items
    for marker in ("_empty_recovery_synthetic", "_db_persisted"):
        value = record.get(marker)
        if isinstance(value, bool):
            event_payload[marker] = value
    tool_extras: list[dict[str, object]] = []
    for tool_index, tool_call in enumerate(_list(record.get("tool_calls")), start=1):
        tool_record = json_document(tool_call)
        extra_content = tool_record.get("extra_content") if tool_record else None
        if isinstance(extra_content, Mapping):
            tool_extras.append(
                {
                    "tool_id": (
                        _string(tool_record.get("id")) or _string(tool_record.get("call_id")) or f"tool-{tool_index}"
                    ),
                    "extra_content": dict(extra_content),
                }
            )
    if tool_extras:
        event_payload["tool_calls_extra_content"] = tool_extras
    if not event_payload:
        return None
    return ParsedSessionEvent(
        event_type="hermes_message_wire_extras",
        timestamp=message.timestamp,
        source_message_provider_id=message.provider_message_id,
        payload=event_payload,
    )


def _content_blocks_from_content(content: object) -> list[ParsedContentBlock]:
    if isinstance(content, str):
        return [ParsedContentBlock(type=BlockType.TEXT, text=content)] if content else []
    if isinstance(content, list):
        blocks: list[ParsedContentBlock] = []
        for index, item in enumerate(content, start=1):
            text = _content_text(item)
            if text:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=text,
                        metadata={"index": index} if not isinstance(item, str) else None,
                    )
                )
        return blocks
    if isinstance(content, Mapping):
        text = _content_text(content)
        return [ParsedContentBlock(type=BlockType.TEXT, text=text)] if text else []
    return []


def _codex_output_text_blocks(items: object, *, covered_text: str | None) -> list[ParsedContentBlock]:
    """Project ``codex_message_items`` assistant prose into TEXT blocks.

    A Codex-compatible Hermes backend emits the assistant turn as a structured
    response item alongside the plain ``content`` field, and when ``content``
    is empty the item holds the only copy of the turn's prose. Without a block
    it reaches neither the block-derived display text nor FTS, both of which
    read ``blocks`` alone.

    Segments whose text ``covered_text`` already carries are skipped: the two
    fields usually hold the same prose, and projecting it again would double
    every such turn in the display text and in the search index.
    """
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return []
    covered = covered_text or ""
    blocks: list[ParsedContentBlock] = []
    for item in _list(items):
        record = json_document(item)
        for segment in _list(record.get("content")):
            segment_record = json_document(segment)
            if segment_record.get("type") != "output_text":
                continue
            text = _string(segment_record.get("text"))
            if text is None or text.strip() in covered:
                continue
            blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=text))
            covered = f"{covered}\n{text}"
    return blocks


def _codex_reasoning_blocks(items: object, *, covered_text: str | None) -> list[ParsedContentBlock]:
    """Project Codex reasoning summaries and text into durable THINKING blocks."""
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return []
    covered = covered_text or ""
    blocks: list[ParsedContentBlock] = []
    for item in _list(items):
        record = json_document(item)
        segments = [*_list(record.get("summary")), *_list(record.get("content"))]
        if not segments and _string(record.get("text")):
            segments = [record]
        for segment in segments:
            segment_record = json_document(segment)
            if segment_record.get("type") not in {
                "reasoning_text",
                "summary_text",
                "text",
            }:
                continue
            text = _string(segment_record.get("text"))
            if text is None or text.strip() in covered:
                continue
            blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=text))
            covered = f"{covered}\n{text}"
    return blocks


def _content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content if content else None
    if isinstance(content, list):
        parts = [_content_text(item) for item in content]
        text = "\n".join(part for part in parts if part)
        return text or None
    if isinstance(content, Mapping):
        for key in ("text", "content", "message", "value"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        try:
            return json.dumps(content, sort_keys=True)
        except TypeError:
            return str(content)
    return None


def _tool_use_block(record: JSONDocument, *, fallback_id: str) -> ParsedContentBlock:
    function = json_document(record.get("function"))
    tool_name = _string(record.get("name")) or _string(function.get("name")) or _string(record.get("type")) or "tool"
    tool_id = _string(record.get("id")) or _string(record.get("call_id")) or fallback_id
    if "args" in record:
        raw_input = record.get("args")
    elif "arguments" in record:
        raw_input = record.get("arguments")
    else:
        raw_input = function.get("arguments")
    metadata = _tool_metadata(record)
    return ParsedContentBlock(
        type=BlockType.TOOL_USE,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=_tool_input(raw_input),
        metadata=metadata or None,
    )


def _fullest_tool_result_text(output: str | None, error: str | None, result_display: object) -> str | None:
    """Return the tool result text that is not a truncation of the other field.

    Gemini CLI's ``functionResponse.response.output`` is sometimes the
    provider's masking envelope -- ``<tool_output_masked>`` / ``Output too
    large. Showing first N and last M characters`` -- announcing content it
    then discards, while the record's own ``resultDisplay`` sibling still
    carries the untruncated text. Reading ``output`` unconditionally stores the
    truncation notice and drops what it was announcing (polylogue-7yji2:
    66,676,100 characters across the measured corpus).

    Truncation is the only condition that overrides ``output``: the two fields
    are different renderings, neither a substring of the other, so preferring
    the longer one wholesale would replace the model-facing text with a
    display rendering wherever it merely happens to be wordier.
    """
    display = _content_text(result_display)
    if output and is_masked_tool_output(output) and display and len(display) > len(output):
        return display
    return output or error or display


def _tool_result_blocks(record: JSONDocument, *, fallback_id: str) -> list[ParsedContentBlock]:
    tool_id = _string(record.get("id")) or _string(record.get("call_id")) or fallback_id
    status = _string(record.get("status"))
    status_is_error = _status_is_error(status)
    metadata = _tool_metadata(record)
    blocks: list[ParsedContentBlock] = []
    for result_item in _list(record.get("result")):
        result_record = json_document(result_item)
        function_response = json_document(result_record.get("functionResponse"))
        if not function_response:
            continue
        response = json_document(function_response.get("response"))
        output = _string(response.get("output"))
        error = _string(response.get("error"))
        text = _fullest_tool_result_text(output, error, record.get("resultDisplay"))
        if text is None and status is None:
            continue
        result_metadata = dict(metadata)
        function_name = _string(function_response.get("name"))
        if function_name:
            result_metadata["function_name"] = function_name
        response_is_error, response_reason = _status_outcome(status, is_error=True if error else status_is_error)
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=_string(function_response.get("id")) or tool_id,
                text=text or f"[{status}]",
                metadata=result_metadata or None,
                is_error=response_is_error,
                outcome_unknown_reason=response_reason,
            )
        )
    if blocks:
        return blocks
    display_text = _content_text(record.get("resultDisplay"))
    if display_text is None and status is None:
        return []
    display_is_error, display_reason = _status_outcome(status, is_error=status_is_error)
    return [
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            tool_id=tool_id,
            text=display_text or f"[{status or 'error'}]",
            metadata=metadata or None,
            is_error=display_is_error,
            outcome_unknown_reason=display_reason,
        )
    ]


def _tool_metadata(record: JSONDocument) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("status", "timestamp", "description", "displayName", "renderOutputAsMarkdown"):
        value = record.get(key)
        if isinstance(value, (str, bool)):
            metadata[key] = value
    return metadata


#: Gemini CLI's failure-status vocabulary. Matched against the ``status``
#: field only -- never against result text.
_STATUS_ERROR_MARKERS = ("error", "fail", "timeout", "cancel", "blocked")


def _status_is_error(status: str | None) -> bool | None:
    if status is None:
        return None
    normalized = status.strip().lower()
    if normalized in {"success", "succeeded", "ok", "completed"}:
        return False
    if any(marker in normalized for marker in _STATUS_ERROR_MARKERS):
        return True
    return None


def _status_outcome(status: str | None, *, is_error: bool | None) -> tuple[bool | None, str | None]:
    """Map a Gemini CLI tool record's ``status`` field to (is_error, unknown reason).

    ``status`` is this origin's only structural verdict -- the record carries
    no exit code. A token outside the declared success/failure vocabulary is a
    verdict the mapping does not read, not an absent one.
    """
    unmapped_status = is_error is None and status is not None
    return is_error, unknown_reason(is_error=is_error, outcome_field_present=unmapped_status)


def _tool_input(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"arguments": value}
        return dict(parsed) if isinstance(parsed, dict) else {"arguments": value}
    return {}


def _role(raw: str, *, assistant_aliases: set[str] | None = None) -> Role:
    lowered = raw.strip().lower()
    if assistant_aliases and lowered in assistant_aliases:
        return Role.ASSISTANT
    return Role.normalize(lowered)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


__all__ = [
    "apply_gemini_tool_output_sidecars",
    "looks_like_gemini_cli",
    "looks_like_hermes",
    "parse_gemini_cli",
    "parse_hermes",
]
