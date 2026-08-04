from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.sources.providers.gemini import GeminiMessage

from .base import (
    ParsedAttachment,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    fill_linear_parent_chain,
    human_authored_override,
)
from .drive_support import (
    _attachment_from_doc as _attachment_from_doc_impl,
)
from .drive_support import (
    _collect_drive_docs as _collect_drive_docs_impl,
)
from .drive_support import (
    attachment_block_payloads as _attachment_block_payloads,
)
from .drive_support import (
    chunk_timestamp as _chunk_timestamp,
)
from .drive_support import (
    collect_chunk_attachments as _collect_chunk_attachments,
)
from .drive_support import (
    extract_text_from_chunk,
)
from .drive_support import (
    parsed_blocks_from_meta as _parsed_blocks_from_meta,
)
from .drive_support import (
    select_timestamp as _select_timestamp,
)
from .drive_support import (
    session_events_from_meta_blocks as _session_events_from_meta_blocks,
)
from .drive_support import (
    viewport_block_payload as _viewport_block_payload,
)

_logger = get_logger(__name__)

_CHUNK_CONTENT_KEYS = frozenset(
    {
        "text",
        "parts",
        "executableCode",
        "codeExecutionResult",
        "driveDocument",
        "driveImage",
        "driveAudio",
        "driveVideo",
        "inlineFile",
        "inlineImage",
        "youtubeVideo",
        "errorMessage",
        "grounding",
        "isThought",
    }
)


def _collect_drive_docs(payload: object) -> list[JSONDocument | str]:
    return _collect_drive_docs_impl(payload)


def _attachment_from_doc(doc: JSONDocument | str, message_id: str | None) -> ParsedAttachment | None:
    return _attachment_from_doc_impl(doc, message_id)


def _gemini_content_block_payloads(message: GeminiMessage, text: str | None) -> list[JSONDocument]:
    content_block_payloads = [
        block_payload
        for content_block in message.extract_content_blocks()
        if (block_payload := _viewport_block_payload(content_block)) is not None
    ]
    if content_block_payloads:
        return content_block_payloads
    if not text:
        return []
    return [{"type": "thinking" if message.isThought else "text", "text": text}]


def _fallback_gemini_content_blocks(chunk_obj: JSONDocument, text: str | None) -> list[JSONDocument]:
    fallback_content_blocks: list[JSONDocument] = []
    if text:
        block_type = "thinking" if chunk_obj.get("isThought") else "text"
        fallback_content_blocks.append({"type": block_type, "text": text})

    exec_code = chunk_obj.get("executableCode")
    if isinstance(exec_code, dict) and exec_code:
        code = exec_code.get("code")
        if isinstance(code, str) and code:
            fallback_content_blocks.append({"type": "code", "text": code})

    exec_result = chunk_obj.get("codeExecutionResult")
    if isinstance(exec_result, dict) and exec_result:
        output = exec_result.get("output")
        outcome = exec_result.get("outcome")
        if isinstance(output, str) and output:
            fallback_content_blocks.append({"type": "tool_result", "text": output})
        elif isinstance(outcome, str) and outcome:
            fallback_content_blocks.append({"type": "tool_result", "text": f"[{outcome}]"})

    return fallback_content_blocks


def _append_attachment_blocks(
    content_blocks: list[JSONDocument], chunk_attachments: list[ParsedAttachment]
) -> list[JSONDocument]:
    # Attachment block payloads carry only JSON-valued fields (name/ids/mime);
    # cast bridges the DriveDocMetadata (dict[str, object]) alias to JSONDocument.
    return content_blocks + cast("list[JSONDocument]", _attachment_block_payloads(chunk_attachments))


def _string_field(payload: JSONDocument, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _non_negative_int_field(payload: JSONDocument, *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            return int(value) if value >= 0 else None
        if isinstance(value, str):
            try:
                parsed = int(float(value))
            except ValueError:
                continue
            return parsed if parsed >= 0 else None
    return None


def _usage_fields(chunk_obj: JSONDocument, *, role: Role) -> dict[str, int]:
    token_count = _non_negative_int_field(chunk_obj, "tokenCount", "token_count")
    if token_count is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if role is Role.USER:
        return {"input_tokens": token_count, "output_tokens": 0}
    return {"input_tokens": 0, "output_tokens": token_count}


def _branch_parent_provider_id(chunk_obj: JSONDocument) -> str | None:
    branch_parent = chunk_obj.get("branchParent")
    if isinstance(branch_parent, str) and branch_parent:
        return branch_parent
    branch_parent_obj = json_document(branch_parent)
    return _string_field(branch_parent_obj, "id", "promptId", "messageId")


def _branch_child_provider_id(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return _string_field(json_document(value), "id", "promptId", "messageId")


def _branch_child_parent_map(chunks: Sequence[object]) -> tuple[dict[str, str], frozenset[str]]:
    """Resolve unambiguous branch parents, plus the set of genuinely ambiguous child ids.

    A child declared under more than one parent's ``branchChildren`` has no
    real single-parent evidence -- it is excluded from the returned map (as
    before) AND returned in the ambiguous set so callers can keep it excluded
    from ``fill_linear_parent_chain``'s later gap-fill pass (bd polylogue-ksgg):
    that pass cannot distinguish "None because ambiguous" from "None because
    no branch data existed at all", so it would otherwise invent a parent for
    a case this function deliberately refused to resolve.
    """
    candidate_parents: dict[str, set[str]] = {}
    for chunk in chunks:
        chunk_obj = json_document(chunk)
        parent_id = _string_field(chunk_obj, "id")
        branch_children = chunk_obj.get("branchChildren")
        if parent_id is None or not isinstance(branch_children, list):
            continue
        for child in branch_children:
            child_id = _branch_child_provider_id(child)
            if child_id is not None:
                candidate_parents.setdefault(child_id, set()).add(parent_id)
    resolved = {child_id: next(iter(parents)) for child_id, parents in candidate_parents.items() if len(parents) == 1}
    ambiguous = frozenset(child_id for child_id, parents in candidate_parents.items() if len(parents) > 1)
    return resolved, ambiguous


def _instruction_text(payload: JSONDocument) -> str | None:
    instruction = payload.get("systemInstruction")
    if isinstance(instruction, str) and instruction:
        return instruction
    instruction_obj = json_document(instruction)
    direct = _string_field(instruction_obj, "text", "content")
    if direct is not None:
        return direct
    parts = instruction_obj.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts = [text for part in parts if (text := _string_field(json_document(part), "text", "content")) is not None]
    return "\n".join(text_parts) or None


def _model_config_event(
    run_settings: JSONDocument,
    *,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    if not run_settings:
        return None
    event_payload: dict[str, object] = {"runSettings": dict(run_settings)}
    model_name = _string_field(run_settings, "model", "modelName", "model_name")
    if model_name is not None:
        event_payload["model"] = model_name
    return ParsedSessionEvent(
        event_type="model_config",
        timestamp=timestamp,
        payload=event_payload,
    )


def _pending_drafts(pending_inputs: object) -> list[dict[str, object]]:
    """Extract non-blank ``chunkedPrompt.pendingInputs`` entries.

    AI Studio's Drive-synced JSON carries the operator's not-yet-submitted
    textbox content here -- draft prompts that never became a chunk and are
    otherwise unrecoverable once overwritten (polylogue-o4j2). Entries with
    blank/whitespace-only text are the overwhelmingly common case (the
    textbox was empty when synced) and carry no evidence, so they are
    skipped rather than kept as near-100%-empty noise.

    Deliberately returned as plain dicts for ``ParsedSession.pending_drafts``,
    NOT ``ParsedSessionEvent``s: a draft is mutable CURRENT state (the
    operator edits the same textbox in place, and the entry disappears
    entirely once submitted), not an append-only historical fact.
    ``session_events`` feeds ``session_revision_projection``'s
    message/attachment/event comparison axes (polylogue-aggz Invariant 1),
    which assume every axis only ever grows between two acquisitions of the
    same session; a mutable, disappearing item there reproduces the exact
    defect class polylogue-bu1i (acquisition state in identity) and
    polylogue-nuec (provider-remeasurement in identity) were fixed for --
    edits would compare as disjoint forks, and submission would shrink the
    event axis while the message axis grows, both misclassifying revision
    membership. ``pending_drafts`` stays outside every identity/hash
    computation in ``pipeline/ids.py`` (see ``sessions.pending_drafts_json``).
    """
    if not isinstance(pending_inputs, list):
        return []
    drafts: list[dict[str, object]] = []
    for entry in pending_inputs:
        entry_obj = json_document(entry)
        text = entry_obj.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        draft: dict[str, object] = {"text": text}
        role_val = _string_field(entry_obj, "role")
        if role_val is not None:
            draft["role"] = role_val
        token_count = _non_negative_int_field(entry_obj, "tokenCount", "token_count")
        if token_count is not None:
            draft["token_count"] = token_count
        drafts.append(draft)
    return drafts


def _delivery_status(chunk_obj: JSONDocument) -> str | None:
    if _string_field(chunk_obj, "errorMessage", "error_message") is not None:
        return "error"
    return _string_field(chunk_obj, "finishReason", "finish_reason")


def _gemini_usage_event(
    chunk_obj: JSONDocument,
    *,
    role: Role,
    message_id: str,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    token_count = _non_negative_int_field(chunk_obj, "tokenCount", "token_count")
    finish_reason = _string_field(chunk_obj, "finishReason", "finish_reason")
    if token_count is None and finish_reason is None:
        return None

    usage: dict[str, int] = {}
    if token_count is not None:
        if role is Role.USER:
            usage["input_tokens"] = token_count
        else:
            usage["output_tokens"] = token_count

    payload: dict[str, object] = {"type": "token_count"}
    if usage:
        payload["last_token_usage"] = usage
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    model_name = _string_field(chunk_obj, "model", "modelName", "model_name")
    if model_name is not None:
        payload["model"] = model_name
    return ParsedSessionEvent(
        event_type="token_count",
        timestamp=timestamp,
        source_message_provider_id=message_id,
        payload=payload,
    )


def parse_chunked_prompt(provider: Provider | str, payload: JSONDocument, fallback_id: str) -> ParsedSession:
    runtime_provider = Provider.from_string(provider)
    run_settings = json_document(payload.get("runSettings"))
    default_model_name = _string_field(run_settings, "model", "modelName", "model_name")
    prompt = json_document(payload.get("chunkedPrompt"))
    chunks: Sequence[object] = ()
    if prompt:
        prompt_chunks = prompt.get("chunks")
        chunks = prompt_chunks if isinstance(prompt_chunks, list) else []
    else:
        payload_chunks = payload.get("chunks")
        if isinstance(payload_chunks, list):
            chunks = payload_chunks

    # Fallback timestamp from session metadata
    create_time = payload.get("createTime")
    default_timestamp = str(create_time) if create_time else None

    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    attachments: list[ParsedAttachment] = []
    observed_timestamps: list[str | None] = []
    models_used: set[str] = set()
    if default_model_name is not None:
        models_used.add(default_model_name)
    if model_event := _model_config_event(run_settings, timestamp=default_timestamp):
        session_events.append(model_event)
    branch_child_parents, ambiguous_branch_child_ids = _branch_child_parent_map(chunks)
    message_position = 0
    for _idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            chunk_obj: JSONDocument = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            continue
        text = extract_text_from_chunk(chunk_obj)
        # Role is required - skip chunks without one
        role_val = chunk_obj.get("role") or chunk_obj.get("author")
        if not isinstance(role_val, str) or not role_val:
            continue
        role = Role.normalize(role_val)
        msg_id = str(chunk_obj.get("id") or "")
        message_timestamp = _chunk_timestamp(chunk_obj, default_timestamp)
        model_name = _string_field(chunk_obj, "model", "modelName", "model_name") or default_model_name
        if model_name is not None:
            models_used.add(model_name)
        usage_fields = _usage_fields(chunk_obj, role=role)
        if usage_event := _gemini_usage_event(
            chunk_obj,
            role=role,
            message_id=msg_id,
            timestamp=message_timestamp,
        ):
            session_events.append(usage_event)
        chunk_attachments = _collect_chunk_attachments(chunk_obj, msg_id)
        observed_timestamps.append(message_timestamp)
        used_typed_model = False

        # Try to parse via the rich GeminiMessage typed model for structured extraction.
        try:
            gemini_message = GeminiMessage.model_validate(chunk_obj)
            used_typed_model = True
            content_block_payloads = _gemini_content_block_payloads(gemini_message, text)
        except (ValidationError, Exception):
            content_block_payloads = _fallback_gemini_content_blocks(chunk_obj, text)

        if chunk_attachments and not used_typed_model:
            content_block_payloads = _append_attachment_blocks(content_block_payloads, chunk_attachments)

        if not text and not chunk_attachments and not content_block_payloads:
            continue

        session_events.extend(
            _session_events_from_meta_blocks(
                content_block_payloads,
                source_message_provider_id=msg_id,
                timestamp=message_timestamp,
            )
        )

        message_blocks = _parsed_blocks_from_meta(content_block_payloads)
        # A block-derived type (e.g. TOOL_RESULT for a codeExecutionResult
        # outcome) must be resolved BEFORE classify_material_origin runs --
        # otherwise the origin gets computed against an assumed plain
        # MESSAGE type and a genuine tool-result turn is misclassified as
        # ASSISTANT_AUTHORED instead of TOOL_RESULT (caught by
        # test_ai_studio_normalizes_identity_authorship_config_blocks_artifacts_usage_and_status).
        resolved_message_type = (
            classify_block_message_type(tuple(block.type for block in message_blocks)) or MessageType.MESSAGE
        )
        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=message_timestamp,
                blocks=message_blocks,
                message_type=resolved_message_type,
                position=message_position,
                variant_index=0,
                is_active_path=True,
                parent_message_provider_id=(_branch_parent_provider_id(chunk_obj) or branch_child_parents.get(msg_id)),
                input_tokens=usage_fields["input_tokens"],
                output_tokens=usage_fields["output_tokens"],
                model_name=model_name,
                duration_ms=_non_negative_int_field(chunk_obj, "durationMs", "duration_ms", "elapsed_ms"),
                delivery_status=_delivery_status(chunk_obj),
                end_turn=(
                    True
                    if _string_field(chunk_obj, "finishReason", "finish_reason", "errorMessage", "error_message")
                    is not None
                    else None
                ),
                # polylogue-gzgyl: AI-Studio/Drive has no agent/subagent
                # artifact ambiguity for a plain user turn -- positive-
                # evidence override for the shared classify_material_origin
                # no-fallthrough (#2502).
                material_origin=human_authored_override(
                    role,
                    resolved_message_type,
                    classify_material_origin(
                        role=role,
                        message_type=resolved_message_type,
                        text=text,
                        block_types=tuple(block.type for block in message_blocks),
                    ),
                ),
            )
        )
        message_position += 1
        attachments.extend(chunk_attachments)

    title_val = payload.get("title")
    title_source: TitleSource | None = TitleSource.ORIGIN
    if not title_val:
        title_val = payload.get("displayName")
        # polylogue-5dfu: None (not TitleSource.UNKNOWN) when neither field
        # carries a title -- NULL already means "no title evidence" here.
        title_source = TitleSource.ORIGIN if title_val else None
    title = str(title_val) if title_val else fallback_id
    create_time_str = (
        str(payload.get("createTime"))
        if payload.get("createTime")
        else _select_timestamp(observed_timestamps, latest=False)
    )
    update_time_str = (
        str(payload.get("updateTime"))
        if payload.get("updateTime")
        else _select_timestamp(observed_timestamps, latest=True)
    )
    pending_drafts = _pending_drafts(prompt.get("pendingInputs"))
    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]
    # bd polylogue-ksgg: real Gemini branch evidence (``_branch_parent_provider_id``
    # / ``branch_child_parents`` above) already sets ``parent_message_provider_id``
    # for messages that carry it; most AI Studio Drive sessions have none
    # (0% parented measured) because they're a plain linear chat with no
    # branch. Only fill the remaining gap -- chain to the previous message on
    # the active path -- without touching real branch evidence already set.
    # A genuinely ambiguous branch child (more than one declared parent, see
    # _branch_child_parent_map) must keep parent_message_provider_id=None --
    # fill_linear_parent_chain cannot tell that apart from "no branch data at
    # all" on its own, so those specific ids are excluded from the gap-fill
    # and restored to None afterward.
    # chunkedPrompt's array order is not guaranteed chronological (a Drive
    # payload can legitimately list chunks in a different order than they
    # occurred -- ai-studio-drive normalization laws require native facts,
    # including the reconstructed parent chain, to be order-independent).
    # fill_linear_parent_chain chains by LIST POSITION, so it must run
    # against a temporally-sorted view, not the raw chunk-input order.
    by_position = {id(message): position for position, message in enumerate(messages)}
    temporally_sorted = sorted(messages, key=lambda message: (message.timestamp or "", by_position[id(message)]))
    filled_sorted = fill_linear_parent_chain(temporally_sorted)
    parent_by_id = {message.provider_message_id: message.parent_message_provider_id for message in filled_sorted}
    messages = [
        message
        if message.provider_message_id in ambiguous_branch_child_ids
        else message.model_copy(update={"parent_message_provider_id": parent_by_id.get(message.provider_message_id)})
        for message in messages
    ]
    return ParsedSession(
        source_name=runtime_provider,
        provider_session_id=str(payload.get("id") or fallback_id),
        title=title,
        title_source=title_source,
        created_at=create_time_str,
        updated_at=update_time_str,
        messages=messages,
        session_events=session_events,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        instructions_text=_instruction_text(payload),
        models_used=sorted(models_used),
        # polylogue-2qx.4 / polylogue-cgfy: aistudio-drive's runSettings
        # (temperature/topP/topK/maxOutputTokens/thinkingLevel/
        # safetySettings/enable* flags) is genuinely per-session provider
        # config -- stored verbatim, not decomposed into columns, so it does
        # not couple the schema to one provider's knob set. Already read into
        # the ``model_config`` session_event above; this is the same value
        # landing on the session row itself.
        run_settings=dict(run_settings) if run_settings else None,
        # polylogue-o4j2: pendingInputs draft(s), kept off session_events on
        # purpose -- see _pending_drafts' docstring for why (mutable current
        # state must not enter session_revision_projection's comparison
        # axes).
        pending_drafts=pending_drafts,
    )


def looks_like_chunk(payload: object) -> bool:
    """Return whether a record has the minimum AI Studio chunk wire shape."""
    chunk = json_document(payload)
    role = chunk.get("role") or chunk.get("author")
    return isinstance(role, str) and bool(role.strip()) and any(key in chunk for key in _CHUNK_CONTENT_KEYS)


def _looks_like_chunks(value: object) -> bool:
    """Return whether ``value`` is a non-empty list of genuine chunks.

    An empty (or absent, or wrong-typed) ``chunks`` list is never sufficient
    evidence on its own (polylogue-mvcbi) -- both the named ``chunkedPrompt``
    envelope and the bare top-level ``chunks`` key require at least one
    chunk carrying role/content evidence.
    """
    if not isinstance(value, list) or not value:
        return False
    return all(looks_like_chunk(item) for item in value)


def has_chunk_container(payload: object) -> bool:
    """Return whether an explicitly selected payload carries a chunk list.

    This is intentionally more tolerant than :func:`looks_like`: explicit
    provider routes should still salvage valid chunks from partially malformed
    exports, while auto-detection must not claim generic ``chunks`` records.
    """
    record = json_document(payload)
    prompt = json_document(record.get("chunkedPrompt"))
    return isinstance(prompt.get("chunks"), list) or isinstance(record.get("chunks"), list)


def looks_like(payload: object) -> bool:
    """Return True if payload looks like a Drive / Gemini chunkedPrompt export.

    Called from ``dispatch._looks_like_gemini_mapping`` -- that is this
    detector's sole auto-detection call site (polylogue-zkmi); it is not
    dead code despite the differently-named wrapper.

    Tightened (polylogue-mvcbi, sibling fix to #3428/#3537): a bare
    ``chunkedPrompt`` envelope with a present-but-empty (or wrong-typed)
    ``chunks`` list used to be accepted on the strength of the key's mere
    presence (``allow_empty=True``) -- the same guess-instead-of-verify
    shape closed for Claude Code's bare ``type`` check and claude.ai's bare
    ``chat_messages`` check. This now requires at least one genuine chunk
    with role/content evidence (see ``looks_like_chunk``), matching the
    bare top-level ``chunks`` case below.
    """
    record = json_document(payload)
    prompt = json_document(record.get("chunkedPrompt"))
    if prompt and _looks_like_chunks(prompt.get("chunks")):
        return True
    # Older exports expose ``chunks`` at the document top level.
    return _looks_like_chunks(record.get("chunks"))
