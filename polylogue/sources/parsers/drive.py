from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from polylogue.archive.message.roles import Role
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.sources.providers.gemini import GeminiMessage
from polylogue.types import Provider

from .base import ParsedAttachment, ParsedConversation, ParsedMessage
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
    parsed_content_blocks_from_meta as _parsed_content_blocks_from_meta,
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


def _typed_gemini_meta(chunk_obj: JSONDocument, message: GeminiMessage, text: str | None) -> dict[str, object]:
    meta: dict[str, object] = {"raw": chunk_obj}

    if message.isThought:
        meta["isThought"] = True
    if message.tokenCount is not None:
        meta["tokenCount"] = message.tokenCount
    if message.finishReason:
        meta["finishReason"] = message.finishReason
    if message.thinkingBudget is not None:
        meta["thinkingBudget"] = message.thinkingBudget
    if message.safetyRatings:
        meta["safetyRatings"] = list(message.safetyRatings)
    if message.grounding:
        meta["grounding"] = (
            message.grounding.model_dump() if hasattr(message.grounding, "model_dump") else message.grounding
        )
    if message.branchParent:
        meta["branchParent"] = (
            message.branchParent.model_dump() if hasattr(message.branchParent, "model_dump") else message.branchParent
        )
    if message.branchChildren:
        meta["branchChildren"] = message.branchChildren
    if message.executableCode:
        meta["executableCode"] = message.executableCode
    if message.codeExecutionResult:
        meta["codeExecutionResult"] = message.codeExecutionResult
    if message.errorMessage:
        meta["errorMessage"] = message.errorMessage
    if message.isEdited:
        meta["isEdited"] = True

    meta["content_blocks"] = _gemini_content_block_payloads(message, text)

    reasoning_traces = message.extract_reasoning_traces()
    if reasoning_traces:
        meta["reasoning_traces"] = [
            {"text": trace.text, "token_count": trace.token_count, "provider": trace.provider}
            for trace in reasoning_traces
        ]
    return meta


def _fallback_gemini_meta(chunk_obj: JSONDocument, text: str | None) -> dict[str, object]:
    meta: dict[str, object] = {"raw": chunk_obj}
    if chunk_obj.get("isThought"):
        meta["isThought"] = True
    token_count = chunk_obj.get("tokenCount")
    if token_count:
        meta["tokenCount"] = token_count

    fallback_content_blocks: list[JSONDocument] = []
    if text:
        block_type = "thinking" if chunk_obj.get("isThought") else "text"
        fallback_content_blocks.append({"type": block_type, "text": text})

    exec_code = chunk_obj.get("executableCode")
    if isinstance(exec_code, dict) and exec_code:
        meta["executableCode"] = exec_code
        code = exec_code.get("code")
        if isinstance(code, str) and code:
            fallback_content_blocks.append({"type": "code", "text": code})

    exec_result = chunk_obj.get("codeExecutionResult")
    if isinstance(exec_result, dict) and exec_result:
        meta["codeExecutionResult"] = exec_result
        output = exec_result.get("output")
        outcome = exec_result.get("outcome")
        if isinstance(output, str) and output:
            fallback_content_blocks.append({"type": "tool_result", "text": output})
        elif isinstance(outcome, str) and outcome:
            fallback_content_blocks.append({"type": "tool_result", "text": f"[{outcome}]"})

    error_msg = chunk_obj.get("errorMessage")
    if isinstance(error_msg, str) and error_msg:
        meta["errorMessage"] = error_msg

    meta["content_blocks"] = fallback_content_blocks
    return meta


def _append_attachment_blocks(meta: dict[str, object], chunk_attachments: list[ParsedAttachment]) -> None:
    meta_blocks = meta.get("content_blocks")
    attachment_blocks = _attachment_block_payloads(chunk_attachments)
    if isinstance(meta_blocks, list):
        meta_blocks.extend(attachment_blocks)
        return
    meta["content_blocks"] = attachment_blocks


def parse_chunked_prompt(provider: Provider | str, payload: JSONDocument, fallback_id: str) -> ParsedConversation:
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

    # Fallback timestamp from conversation metadata
    create_time = payload.get("createTime")
    default_timestamp = str(create_time) if create_time else None

    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    observed_timestamps: list[str | None] = []
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
        chunk_attachments = _collect_chunk_attachments(chunk_obj, msg_id)
        observed_timestamps.append(message_timestamp)
        used_typed_model = False

        # Try to parse via the rich GeminiMessage typed model for structured extraction.
        try:
            gemini_message = GeminiMessage.model_validate(chunk_obj)
            used_typed_model = True
            meta = _typed_gemini_meta(chunk_obj, gemini_message, text)
        except (ValidationError, Exception):
            meta = _fallback_gemini_meta(chunk_obj, text)

        if chunk_attachments and not used_typed_model:
            _append_attachment_blocks(meta, chunk_attachments)

        if not text and not chunk_attachments and not meta.get("content_blocks"):
            continue

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=message_timestamp,
                content_blocks=_parsed_content_blocks_from_meta(meta.get("content_blocks")),
                provider_meta=meta,
            )
        )
        attachments.extend(chunk_attachments)

    title_val = payload.get("title")
    title_source = "imported:title"
    if not title_val:
        title_val = payload.get("displayName")
        title_source = "imported:displayName" if title_val else "fallback:id"
    title = str(title_val) if title_val else fallback_id
    provider_meta: dict[str, object] = {"title_source": title_source}
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
    return ParsedConversation(
        provider_name=runtime_provider,
        provider_conversation_id=str(payload.get("id") or fallback_id),
        title=title,
        created_at=create_time_str,
        updated_at=update_time_str,
        messages=messages,
        attachments=attachments,
        provider_meta=provider_meta,
    )


def looks_like(payload: object) -> bool:
    """Return True if payload looks like a Drive / Gemini chunkedPrompt export."""
    return "chunkedPrompt" in json_document(payload)
