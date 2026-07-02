from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.sources.providers.gemini import GeminiMessage

from .base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
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
    viewport_block_payload as _viewport_block_payload,
)

_logger = get_logger(__name__)


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
    message_position = 0
    for idx, chunk in enumerate(chunks, start=1):
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
        msg_id = str(chunk_obj.get("id") or f"chunk-{idx}")
        message_timestamp = _chunk_timestamp(chunk_obj, default_timestamp)
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

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=message_timestamp,
                blocks=_parsed_blocks_from_meta(content_block_payloads),
                position=message_position,
                variant_index=0,
                is_active_path=True,
                model_name=_string_field(chunk_obj, "model", "modelName", "model_name"),
                duration_ms=_non_negative_int_field(chunk_obj, "durationMs", "duration_ms", "elapsed_ms"),
            )
        )
        message_position += 1
        attachments.extend(chunk_attachments)

    title_val = payload.get("title")
    title_source: TitleSource = TitleSource.ORIGIN
    if not title_val:
        title_val = payload.get("displayName")
        title_source = TitleSource.ORIGIN if title_val else TitleSource.UNKNOWN
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
    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
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
    )


def looks_like(payload: object) -> bool:
    """Return True if payload looks like a Drive / Gemini chunkedPrompt export."""
    return "chunkedPrompt" in json_document(payload)
