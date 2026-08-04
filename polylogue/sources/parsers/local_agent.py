"""Parsers for local agent session JSON documents."""

from __future__ import annotations

import json
from collections.abc import Mapping

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, BranchType, Provider
from polylogue.core.json import JSONDocument, json_document

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    fill_linear_parent_chain,
    human_authored_override,
)


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


def parse_gemini_cli(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    session_id = _string(payload.get("sessionId")) or fallback_id
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
    return ParsedSession(
        source_name=Provider.GEMINI_CLI,
        provider_session_id=session_id,
        title=_string(payload.get("summary")) or session_id,
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


def parse_hermes(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    session_id = _string(payload.get("session_id")) or fallback_id
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
    # bd polylogue-ksgg: this JSON-sidecar Hermes shape has no parent-message
    # evidence either -- chain each message to the previous one on the
    # active path, same as the state-db conversational parser
    # (hermes_state.py::_parse_session_row).
    messages = fill_linear_parent_chain(messages)
    messages = _mark_active_leaf(messages)
    if metadata_event := _hermes_session_metadata_event(payload, message_count=len(messages)):
        session_events.append(metadata_event)
    if tool_event := _hermes_tool_availability_event(payload):
        session_events.append(tool_event)
    session_events.extend(_block_metadata_evidence_events(messages))
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=session_id,
        created_at=_string(payload.get("session_start")),
        updated_at=_string(payload.get("last_updated")),
        messages=messages,
        session_events=session_events,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
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
    if not text and not content_blocks:
        return None
    token_usage = _token_usage_fields(record)
    gemini_role = _role(_string(record.get("type")) or "unknown", assistant_aliases={"gemini", "model"})
    gemini_blocks = content_blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)]
    # A block-derived type (tool_use/tool_result from toolCalls above) must be
    # resolved BEFORE classify_material_origin runs, or a genuine tool turn
    # gets misclassified against an assumed plain MESSAGE type.
    gemini_message_type = (
        classify_block_message_type(tuple(block.type for block in gemini_blocks)) or MessageType.MESSAGE
    )
    return ParsedMessage(
        # polylogue-slshy: no positional fallback -- empty id lets
        # _message_comparison_id's content-anchor fallback run instead.
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
    reasoning = _string(record.get("reasoning_content")) or _string(record.get("reasoning"))
    if reasoning:
        content_blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning))
    for tool_index, tool_call in enumerate(_list(record.get("tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if not tool_record:
            continue
        content_blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{index}-{tool_index}"))
    tool_call_id = _string(record.get("tool_call_id"))
    role = _role(_string(record.get("role")) or "unknown")
    if role is Role.TOOL and text:
        content_blocks.append(ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=tool_call_id, text=text))
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
    if not messages:
        return messages
    active_leaf_message_provider_id = messages[-1].provider_message_id
    return [
        message.model_copy(update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id})
        for message in messages
    ]


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
        text = output or error or _content_text(record.get("resultDisplay"))
        if text is None and status is None:
            continue
        result_metadata = dict(metadata)
        function_name = _string(function_response.get("name"))
        if function_name:
            result_metadata["function_name"] = function_name
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=_string(function_response.get("id")) or tool_id,
                text=text or f"[{status}]",
                metadata=result_metadata or None,
                is_error=True if error else status_is_error,
            )
        )
    if blocks:
        return blocks
    display_text = _content_text(record.get("resultDisplay"))
    if display_text is None and status is None:
        return []
    return [
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            tool_id=tool_id,
            text=display_text or f"[{status or 'error'}]",
            metadata=metadata or None,
            is_error=status_is_error,
        )
    ]


def _tool_metadata(record: JSONDocument) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("status", "timestamp", "description", "displayName", "renderOutputAsMarkdown"):
        value = record.get(key)
        if isinstance(value, (str, bool)):
            metadata[key] = value
    return metadata


def _status_is_error(status: str | None) -> bool | None:
    if status is None:
        return None
    normalized = status.strip().lower()
    if normalized in {"success", "succeeded", "ok", "completed"}:
        return False
    if any(marker in normalized for marker in ("error", "fail", "timeout", "cancel", "blocked")):
        return True
    return None


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
    "looks_like_gemini_cli",
    "looks_like_hermes",
    "parse_gemini_cli",
    "parse_hermes",
]
