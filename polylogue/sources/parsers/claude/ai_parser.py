"""Claude AI session parser helpers.

Three distinct wire shapes live under the ``claude-ai`` acquisition family:

- ordinary claude.ai conversation exports (``chat_messages``) -- ``parse_ai``.
- Claude Design agentic chats (``design_chats/*.json``, bd polylogue-tbun) --
  a genuinely different product/backend (camelCase ``contentBlocks``/
  ``authorAccountUuid``/``turnChanges``, ``content`` is a dict not a list).
  Admitted as its own ``Origin.CLAUDE_DESIGN_SESSION`` / ``Provider.CLAUDE_DESIGN``
  rather than folded into claude.ai -- see ``parse_design``.
- the account-level ``memories.json`` GDPR sidecar (bd polylogue-zng9), which
  carries no session identity at all (no ``chat_messages``/``messages`` key) --
  represented as a synthetic ``generated_context_pack`` session under the
  existing claude.ai origin -- see ``parse_memories``.
"""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, SessionKind, TitleSource
from polylogue.logging import get_logger

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    attachment_from_meta,
    human_authored_override,
)
from .common import (
    _first_identity_field,
    _first_string_field,
    _message_model_effort,
    _message_model_name,
    _thinking_configuration,
    normalize_chat_messages,
    normalize_timestamp,
)

CLAUDE_TEMPORARY_CHAT_INGEST_FLAG = "capture:temporary-chat"
CLAUDE_ACCOUNT_MEMORY_INGEST_FLAG = "capture:claude-account-memory"

logger = get_logger(__name__)


#: Role-shaped keys ``normalize_chat_messages``/``_raw_role`` themselves read
#: (``common.py:_raw_role``): ``sender``/``role`` directly, or ``author.role``.
_CLAUDE_AI_MESSAGE_ROLE_KEYS = ("sender", "role", "author")
#: Content-shaped keys a genuine chat turn carries one of.
_CLAUDE_AI_MESSAGE_CONTENT_KEYS = ("text", "content")


def _chat_message_node_shape_is_plausible(item: object) -> bool:
    """Positive structural evidence for one claude.ai ``chat_messages`` entry.

    Mirrors the Claude Code envelope-marker fix (#3428,
    ``code_detection.py:looks_like_code``): a bare ``chat_messages`` key is
    not sufficient evidence on its own that a payload is a claude.ai export --
    require at least one entry shaped like a real chat turn (a role/sender
    field alongside a text/content field), matching what
    ``normalize_chat_messages``/``_raw_role`` themselves read.
    """
    if not isinstance(item, Mapping):
        return False
    has_role = any(key in item for key in _CLAUDE_AI_MESSAGE_ROLE_KEYS)
    has_content = any(key in item for key in _CLAUDE_AI_MESSAGE_CONTENT_KEYS)
    return has_role and has_content


def looks_like_ai(payload: object) -> bool:
    """Detect the claude.ai conversation-export shape.

    Tightened (fail-closed acquisition sweep, sibling fix to #3428): a bare
    ``chat_messages: []`` -- or a list whose entries don't resemble a real
    chat turn -- used to be accepted on the strength of the key's mere
    presence, the same "guess and proceed" shape #3428 closed for Claude
    Code's bare ``type`` check. This now requires at least one entry with
    positive structural evidence (see ``_chat_message_node_shape_is_plausible``).
    """
    if not isinstance(payload, dict):
        return False
    chat_messages = payload.get("chat_messages")
    if not isinstance(chat_messages, list) or not chat_messages:
        return False
    return any(_chat_message_node_shape_is_plausible(item) for item in chat_messages)


def looks_like_claude_design(payload: object) -> bool:
    """Detect the Claude Design chat shape (bd polylogue-tbun, measured).

    Distinguishing signature vs a claude.ai conversation export: a Design
    document carries ``messages`` (not ``chat_messages``) plus a ``project``
    key identifying which Design project owns the chat. Message content
    inside is a camelCase dict (``contentBlocks``/``authorAccountUuid``/
    ``turnChanges``), never claude.ai's list-of-segments shape.
    """
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("messages"), list)
        and "project" in payload
        and not isinstance(payload.get("chat_messages"), list)
    )


def looks_like_claude_memories(payload: object) -> bool:
    """Detect the claude.ai account-memory export shape (bd polylogue-zng9).

    ``memories.json`` in a GDPR export batch is a top-level JSON array with
    one record per account, each carrying ``account_uuid`` plus
    ``conversations_memory`` and/or ``project_memories`` -- no session
    identity, no message list. Distinctive enough (account_uuid combined with
    either memory field) that no other admitted shape collides with it.
    """
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("account_uuid"), str):
        return False
    return "conversations_memory" in payload or "project_memories" in payload


def _session_ingest_flags(payload: Mapping[str, object]) -> list[str]:
    flags: list[str] = []
    if payload.get("is_temporary") is True:
        flags.append(CLAUDE_TEMPORARY_CHAT_INGEST_FLAG)
    return flags


def _session_kind(payload: Mapping[str, object]) -> SessionKind:
    return SessionKind.TEMPORARY if payload.get("is_temporary") is True else SessionKind.STANDARD


def _resolve_claude_ai_title(
    payload: Mapping[str, object], resolved_session_id: str, *, ref_prefix: str
) -> tuple[str, TitleSource | None, str | None, float | None]:
    """Claude AI's web UI auto-titles every conversation with a short
    generated summary of the exchange (distinct from Codex's raw
    first-prompt echoes, bd polylogue-6e7m -- see assembly_codex.py's
    ``_is_prompt_echo``). ``title``/``name`` is genuine provider curation
    when present, not a parser guess, so it is worth marking ORIGIN rather
    than leaving ``title_source`` unset (as this parser did before).

    polylogue-5dfu: the no-evidence branch below now returns ``None`` rather
    than ``TitleSource.UNKNOWN`` -- NULL already means "no title evidence"
    on this nullable column, so UNKNOWN was a redundant second spelling of
    the same fact and has been deleted from the enum.
    """
    raw_title = payload.get("title") or payload.get("name")
    if isinstance(raw_title, str) and raw_title.strip():
        return raw_title, TitleSource.ORIGIN, f"{ref_prefix}:{resolved_session_id}", 1.0
    return str(resolved_session_id), None, None, None


# ---------------------------------------------------------------------------
# Claude Design (bd polylogue-tbun)
# ---------------------------------------------------------------------------

#: contentBlocks types this parser recognizes. Anything else is dropped with
#: a warning rather than guessed -- the corpus is 11 chats and the product is
#: still moving (bd polylogue-tbun scope note).
_DESIGN_KNOWN_BLOCK_TYPES = frozenset({"text", "thinking", "tool_call", "error", "user_interjection"})


def _design_attachment_from_meta(meta: object, message_id: str) -> ParsedAttachment | None:
    """Adapt a Design attachment record onto the shared attachment builder.

    Design attachments carry inline text under ``content`` (not the shared
    builder's ``extracted_content`` key) and use ``type`` for a coarse
    taxonomy (file/skill/text/image/folder) rather than a MIME type. ``skill``
    and ``folder`` have no equivalent in any other provider's attachment
    shape -- ``ParsedAttachment.attachment_kind`` is an open string field
    (see its docstring), so tagging them needs no schema change.
    """
    if not isinstance(meta, Mapping):
        return None
    shimmed: dict[str, object] = dict(meta)
    content = shimmed.get("content")
    if isinstance(content, str) and "extracted_content" not in shimmed:
        shimmed["extracted_content"] = content
    attachment = attachment_from_meta(shimmed, message_id)
    if attachment is None:
        return None
    attachment_type = meta.get("type")
    if attachment_type in ("skill", "folder"):
        attachment = attachment.model_copy(update={"attachment_kind": attachment_type})
    return attachment


def _design_tool_call_blocks(tool_call: Mapping[str, object]) -> list[ParsedContentBlock]:
    """toolCall -> TOOL_USE (+TOOL_RESULT when output is present).

    id/name/input/output share the ``toolu_*`` id space with the Claude API
    (bd polylogue-tbun), so ``tool_id`` joins cleanly across providers.
    """
    tool_id = tool_call.get("id")
    tool_id_str = str(tool_id) if isinstance(tool_id, str) and tool_id else None
    tool_name = tool_call.get("name")
    metadata: dict[str, object] = {}
    if tool_call.get("serverSide") is True:
        metadata["server_side"] = True
    tool_call_type = tool_call.get("type")
    if isinstance(tool_call_type, str) and tool_call_type:
        metadata["tool_call_type"] = tool_call_type
    tool_input = tool_call.get("input")
    blocks = [
        ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name=str(tool_name) if isinstance(tool_name, str) and tool_name else None,
            tool_id=tool_id_str,
            tool_input=tool_input if isinstance(tool_input, Mapping) else None,
            metadata=metadata or None,
        )
    ]
    output = tool_call.get("output")
    if output is not None:
        blocks.append(ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=tool_id_str, text=str(output)))
    return blocks


def _design_assistant_messages(
    raw_message: Mapping[str, object],
    content_payload: Mapping[str, object],
    *,
    start_position: int,
    resolved_session_id: str,
) -> tuple[list[ParsedMessage], ParsedSessionEvent | None, int]:
    """Parse one assistant turn's ``contentBlocks`` into ParsedMessage(s).

    ``user_interjection`` (a user message nested inside the assistant turn)
    is NOT flattened into an ordinary same-position user message -- that
    would destroy the interruption semantics and physical ordering (bd
    polylogue-tbun AC4). Instead the turn's blocks are split at the
    interjection boundary into separate ParsedMessage segments with
    incrementing positions, so the interjection lands as a real ``role=user``
    message physically between the two half-turns.
    """
    message_uuid = str(raw_message.get("uuid") or content_payload.get("id") or f"design-{start_position}")
    raw_blocks_value = content_payload.get("contentBlocks")
    raw_blocks = raw_blocks_value if isinstance(raw_blocks_value, list) else []
    has_interjection = any(
        isinstance(block, Mapping)
        and block.get("type") == "user_interjection"
        and isinstance(block.get("message"), Mapping)
        for block in raw_blocks
    )
    timestamp = content_payload.get("timestamp")
    timestamp_str = str(timestamp) if isinstance(timestamp, str) and timestamp else None
    turn_input_tokens = content_payload.get("turnInputTokens")

    messages: list[ParsedMessage] = []
    position = start_position
    current_blocks: list[ParsedContentBlock] = []
    segment_index = 0
    # turn-level facts (turnInputTokens) are attributed to the FIRST emitted
    # segment of the turn -- an arbitrary but documented single anchor point,
    # since the count describes the whole turn, not any one segment.
    first_segment_emitted = False

    def flush() -> None:
        nonlocal current_blocks, segment_index, position, first_segment_emitted
        if not current_blocks:
            return
        provider_message_id = f"{message_uuid}#{segment_index}" if has_interjection else message_uuid
        text_parts = [
            block.text for block in current_blocks if block.type is BlockType.TEXT and not block.is_error and block.text
        ]
        input_tokens = 0
        if not first_segment_emitted and isinstance(turn_input_tokens, int):
            input_tokens = turn_input_tokens
        first_segment_emitted = True
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=Role.ASSISTANT,
                text="\n".join(text_parts) if text_parts else None,
                timestamp=timestamp_str,
                blocks=list(current_blocks),
                position=position,
                variant_index=0,
                is_active_path=True,
                input_tokens=input_tokens,
            )
        )
        position += 1
        segment_index += 1
        current_blocks = []

    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            continue
        block_type = raw_block.get("type")
        if block_type not in _DESIGN_KNOWN_BLOCK_TYPES:
            logger.warning(
                "claude-design %s: unrecognized contentBlocks type %r, dropping block",
                resolved_session_id,
                block_type,
            )
            continue
        if block_type == "text":
            text = raw_block.get("text")
            if isinstance(text, str) and text:
                current_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=text))
        elif block_type == "thinking":
            text = raw_block.get("text")
            if isinstance(text, str) and text:
                current_blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=text))
        elif block_type == "tool_call":
            tool_call = raw_block.get("toolCall")
            if isinstance(tool_call, Mapping):
                current_blocks.extend(_design_tool_call_blocks(tool_call))
        elif block_type == "error":
            # The model refused to respond -- first-class error content (bd
            # polylogue-tbun), not silently dropped. No dedicated ERROR
            # BlockType exists; TEXT + is_error mirrors the established
            # unrecognized-block-shape-to-text mapping in drive_support_blocks.py.
            message_text = raw_block.get("message")
            current_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    text=str(message_text) if message_text is not None else "",
                    is_error=True,
                    metadata={"kind": "model_refusal"},
                )
            )
        elif block_type == "user_interjection":
            interjection_message = raw_block.get("message")
            if not isinstance(interjection_message, Mapping):
                logger.warning(
                    "claude-design %s: user_interjection block with no nested message, dropping",
                    resolved_session_id,
                )
                continue
            flush()
            interjection_id = str(interjection_message.get("id") or f"{message_uuid}-interjection-{position}")
            interjection_text = interjection_message.get("content")
            interjection_timestamp = interjection_message.get("timestamp")
            interjection_role = Role.normalize(str(interjection_message.get("role") or "user"))
            messages.append(
                ParsedMessage(
                    provider_message_id=interjection_id,
                    role=interjection_role,
                    text=str(interjection_text) if isinstance(interjection_text, str) and interjection_text else None,
                    timestamp=str(interjection_timestamp) if isinstance(interjection_timestamp, str) else None,
                    position=position,
                    is_active_path=True,
                    # polylogue-gzgyl: a user_interjection is unambiguously a
                    # real human interjection during an assistant turn -- not
                    # a generated/agent artifact -- so the shared
                    # classify_material_origin no-fallthrough (#2502) needs
                    # the same positive-evidence override as Codex/ChatGPT.
                    material_origin=human_authored_override(
                        interjection_role,
                        MessageType.MESSAGE,
                        classify_material_origin(
                            role=interjection_role,
                            message_type=MessageType.MESSAGE,
                            text=str(interjection_text) if isinstance(interjection_text, str) else None,
                        ),
                    ),
                )
            )
            position += 1
    flush()

    turn_changes = content_payload.get("turnChanges")
    session_event: ParsedSessionEvent | None = None
    if isinstance(turn_changes, Mapping) and messages:
        # Per-turn materialized filesystem diff -- nothing else in the
        # archive records what a turn changed on disk (bd polylogue-tbun).
        # Modeled as a session_event (payload_json), the same mechanism this
        # module already uses for claude_ai_conversation_summary/
        # claude_ai_web_tool_evidence -- turn-scoped structured evidence is
        # exactly what session_events already models; no new construct type
        # or table is needed.
        session_event = ParsedSessionEvent(
            event_type="claude_design_turn_changes",
            timestamp=timestamp_str,
            payload={
                "created": turn_changes.get("created") if isinstance(turn_changes.get("created"), list) else [],
                "edited": turn_changes.get("edited") if isinstance(turn_changes.get("edited"), list) else [],
                "deleted": turn_changes.get("deleted") if isinstance(turn_changes.get("deleted"), list) else [],
                "moved": turn_changes.get("moved") if isinstance(turn_changes.get("moved"), list) else [],
            },
            source_message_provider_id=messages[0].provider_message_id,
        )

    return messages, session_event, position


def _design_user_message(
    raw_message: Mapping[str, object],
    content_payload: Mapping[str, object],
    *,
    position: int,
) -> tuple[ParsedMessage, list[ParsedAttachment], ParsedSessionEvent | None]:
    message_uuid = str(raw_message.get("uuid") or content_payload.get("id") or f"design-{position}")
    text = content_payload.get("content")
    timestamp = content_payload.get("timestamp")
    author_name = content_payload.get("authorName")
    author_account_uuid = content_payload.get("authorAccountUuid")

    attachments: list[ParsedAttachment] = []
    raw_attachments = content_payload.get("attachments")
    for meta in raw_attachments if isinstance(raw_attachments, list) else []:
        attachment = _design_attachment_from_meta(meta, message_uuid)
        if attachment is not None:
            attachments.append(attachment)

    sender_name = str(author_name) if isinstance(author_name, str) and author_name else None
    timestamp_str = str(timestamp) if isinstance(timestamp, str) and timestamp else None
    design_message_text = str(text) if isinstance(text, str) and text else None
    message = ParsedMessage(
        provider_message_id=message_uuid,
        role=Role.USER,
        text=design_message_text,
        timestamp=timestamp_str,
        sender_name=sender_name,
        position=position,
        is_active_path=True,
        # polylogue-gzgyl: a top-level Claude Design user turn is
        # unambiguously human-authored (no agent/subagent artifact shape to
        # exclude here, unlike Claude Code) -- positive-evidence override for
        # the shared classify_material_origin no-fallthrough (#2502).
        material_origin=human_authored_override(
            Role.USER,
            MessageType.MESSAGE,
            classify_material_origin(role=Role.USER, message_type=MessageType.MESSAGE, text=design_message_text),
        ),
    )

    session_event: ParsedSessionEvent | None = None
    if isinstance(author_account_uuid, str) and author_account_uuid:
        # Named multi-account authorship -- absent from ordinary claude.ai
        # exports (bd polylogue-tbun). sender_name alone loses the account
        # uuid, the actual disambiguator across accounts.
        session_event = ParsedSessionEvent(
            event_type="claude_design_message_author",
            timestamp=timestamp_str,
            payload={"author_account_uuid": author_account_uuid, "author_name": sender_name},
            source_message_provider_id=message_uuid,
        )

    return message, attachments, session_event


def parse_design(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    resolved_session_id = str(payload.get("uuid") or payload.get("id") or fallback_id)
    raw_messages = payload.get("messages")
    design_messages = raw_messages if isinstance(raw_messages, list) else []

    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    session_events: list[ParsedSessionEvent] = []
    position = 0
    for raw_message in design_messages:
        if not isinstance(raw_message, Mapping):
            continue
        content_payload = raw_message.get("content")
        if not isinstance(content_payload, Mapping):
            logger.warning("claude-design %s: message with non-dict content, dropping", resolved_session_id)
            continue
        role = raw_message.get("role") or content_payload.get("role")
        if role == "user":
            message, message_attachments, author_event = _design_user_message(
                raw_message, content_payload, position=position
            )
            messages.append(message)
            attachments.extend(message_attachments)
            if author_event is not None:
                session_events.append(author_event)
            position += 1
        elif role == "assistant":
            segment_messages, turn_event, position = _design_assistant_messages(
                raw_message,
                content_payload,
                start_position=position,
                resolved_session_id=resolved_session_id,
            )
            messages.extend(segment_messages)
            if turn_event is not None:
                session_events.append(turn_event)
        else:
            logger.warning(
                "claude-design %s: unrecognized message role %r, dropping message", resolved_session_id, role
            )

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]

    title, title_source, title_ref, title_confidence = _resolve_claude_ai_title(
        payload, resolved_session_id, ref_prefix="claude-design-title"
    )
    return ParsedSession(
        source_name=Provider.CLAUDE_DESIGN,
        provider_session_id=resolved_session_id,
        title=str(title),
        title_source=title_source,
        title_ref=title_ref,
        title_confidence=title_confidence,
        session_kind=_session_kind(payload),
        created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        session_events=session_events,
    )


# ---------------------------------------------------------------------------
# claude.ai account memory (bd polylogue-zng9)
# ---------------------------------------------------------------------------


def parse_memories(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    """Parse one account's ``memories.json`` record into a synthetic session.

    ``memories.json`` carries no session identity (no message list, no
    conversation uuid) -- it is Claude's own standing summary of the user
    across every conversation, refreshed on each export. Represented as a
    session-scoped construct rather than a durable ``user.db`` assertion:
    it is provider-reported evidence re-derived on every export (like any
    other raw session content), not a user-authored annotation, so it
    belongs in the same rebuildable tier as everything else this parser
    produces. ``provider_session_id`` is deterministic
    (``account-memory:<account_uuid>``) so re-import updates the same
    session in place -- idempotent via the existing content-hash mechanism,
    the same as every other origin.
    """
    account_uuid = str(payload.get("account_uuid") or fallback_id)
    messages: list[ParsedMessage] = []
    position = 0

    conversations_memory = payload.get("conversations_memory")
    if isinstance(conversations_memory, str) and conversations_memory.strip():
        messages.append(
            ParsedMessage(
                provider_message_id="conversations",
                role=Role.SYSTEM,
                text=conversations_memory,
                material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
                position=position,
                is_active_path=True,
            )
        )
        position += 1

    project_memories = payload.get("project_memories")
    if isinstance(project_memories, Mapping):
        for project_uuid in sorted(str(key) for key in project_memories):
            memory_text = project_memories.get(project_uuid)
            if not isinstance(memory_text, str) or not memory_text.strip():
                continue
            messages.append(
                ParsedMessage(
                    provider_message_id=f"project:{project_uuid}",
                    role=Role.SYSTEM,
                    text=memory_text,
                    material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
                    position=position,
                    is_active_path=True,
                )
            )
            position += 1

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages[-1] = messages[-1].model_copy(update={"is_active_leaf": True})

    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=f"account-memory:{account_uuid}",
        title="Claude AI account memory",
        title_source=TitleSource.HEURISTIC,
        session_kind=SessionKind.STANDARD,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        ingest_flags=[CLAUDE_ACCOUNT_MEMORY_INGEST_FLAG],
    )


# ---------------------------------------------------------------------------
# claude.ai conversation export
# ---------------------------------------------------------------------------


def _session_timestamp(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float, str)):
            normalized = normalize_timestamp(value)
            if normalized is not None:
                return normalized
    return None


def _merge_session_attachments(
    message_attachments: list[ParsedAttachment],
    payload: Mapping[str, object],
) -> list[ParsedAttachment]:
    attachments = list(message_attachments)
    top_level: list[object] = []
    for key in ("attachments", "files"):
        value = payload.get(key)
        if isinstance(value, list):
            top_level.extend(value)
    for meta in top_level:
        attachment = attachment_from_meta(meta, None)
        if attachment is not None:
            attachments.append(attachment)

    merged: dict[str, ParsedAttachment] = {}
    for candidate in attachments:
        existing = merged.get(candidate.provider_attachment_id)
        if existing is None:
            merged[candidate.provider_attachment_id] = candidate
            continue
        preferred = candidate if candidate.inline_bytes is not None and existing.inline_bytes is None else existing
        other = existing if preferred is candidate else candidate
        merged[candidate.provider_attachment_id] = preferred.model_copy(
            update={
                "message_provider_id": preferred.message_provider_id or other.message_provider_id,
                "name": preferred.name or other.name,
                "mime_type": preferred.mime_type or other.mime_type,
                "size_bytes": preferred.size_bytes if preferred.size_bytes is not None else other.size_bytes,
                "provider_file_id": preferred.provider_file_id or other.provider_file_id,
                "provider_drive_id": preferred.provider_drive_id or other.provider_drive_id,
            }
        )
    return list(merged.values())


def parse_ai(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    # memories.json records arrive tagged Provider.CLAUDE_AI too (bd
    # polylogue-zng9 deliberately reuses the existing claude.ai bundle
    # plumbing rather than adding a second Provider) -- dispatch internally
    # on shape, the same pattern this module used for Claude Design chats
    # before they were split into their own Origin/Provider.
    if looks_like_claude_memories(payload):
        return parse_memories(payload, fallback_id)

    raw_messages = payload.get("chat_messages")
    chat_messages = raw_messages if isinstance(raw_messages, list) else []
    created_at = _session_timestamp(payload, "created_at", "create_time", "timestamp")
    updated_at = _session_timestamp(payload, "updated_at", "update_time")
    session_model = _message_model_name(payload)
    session_effort = _message_model_effort(payload)
    active_leaf_message_provider_id = _first_identity_field(
        payload,
        "current_leaf_message_uuid",
        "current_leaf_message_id",
        "active_leaf_message_uuid",
        "active_leaf_message_id",
        "current_message_uuid",
        "current_message_id",
        "current_node",
    )
    normalized = normalize_chat_messages(
        chat_messages,
        session_model=session_model,
        session_effort=session_effort,
        session_thinking_configuration=_thinking_configuration(payload),
        session_created_at=created_at,
        session_updated_at=updated_at,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
    )

    session_events = list(normalized.session_events)
    provider_status = _first_string_field(payload, "status", "conversation_status")
    if provider_status is not None:
        session_events.append(
            ParsedSessionEvent(
                event_type="provider_session_status",
                timestamp=updated_at or created_at,
                payload={"status": provider_status},
            )
        )
    # Claude AI's own generated conversation summary (top-level ``summary``,
    # distinct from ``name``/``title``) was read nowhere -- parser-diff triage
    # (2026-07-29) found it populated with real multi-paragraph provider
    # summaries on the live corpus.
    provider_summary = _first_string_field(payload, "summary")
    if provider_summary:
        session_events.append(
            ParsedSessionEvent(
                event_type="claude_ai_conversation_summary",
                timestamp=updated_at or created_at,
                payload={"summary": provider_summary},
            )
        )

    conversation_id = _first_identity_field(payload, "uuid", "id", "conversation_id", "conversationId")
    resolved_session_id = conversation_id or fallback_id
    title, title_source, title_ref, title_confidence = _resolve_claude_ai_title(
        payload, resolved_session_id, ref_prefix="claude-ai-title"
    )
    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=resolved_session_id,
        title=str(title),
        title_source=title_source,
        title_ref=title_ref,
        title_confidence=title_confidence,
        session_kind=_session_kind(payload),
        created_at=created_at,
        updated_at=updated_at,
        messages=normalized.messages,
        active_leaf_message_provider_id=normalized.active_leaf_message_provider_id,
        attachments=_merge_session_attachments(normalized.attachments, payload),
        session_events=session_events,
        reported_duration_ms=normalized.reported_duration_ms,
        models_used=normalized.models_used,
        ingest_flags=list(
            dict.fromkeys(
                [
                    *_session_ingest_flags(payload),
                    *normalized.ingest_flags,
                ]
            )
        ),
    )


__all__ = [
    "CLAUDE_ACCOUNT_MEMORY_INGEST_FLAG",
    "CLAUDE_TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like_ai",
    "looks_like_claude_design",
    "looks_like_claude_memories",
    "parse_ai",
    "parse_design",
    "parse_memories",
]
