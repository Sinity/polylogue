from __future__ import annotations

from pydantic import ValidationError

from polylogue.lib.roles import Role
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


def _collect_drive_docs(payload: object) -> list[dict[str, object] | str]:
    return _collect_drive_docs_impl(payload)


def _attachment_from_doc(doc: dict[str, object] | str, message_id: str | None) -> ParsedAttachment | None:
    return _attachment_from_doc_impl(doc, message_id)


def parse_chunked_prompt(provider: Provider | str, payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    runtime_provider = Provider.from_string(provider)
    prompt = payload.get("chunkedPrompt")
    chunks: list[str | dict[str, object]] = []
    if isinstance(prompt, dict):
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
            chunk_obj: dict[str, object] = {"text": chunk}
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

        # Try to parse via the rich GeminiMessage typed model for structured extraction
        meta: dict[str, object] = {"raw": chunk_obj}
        try:
            gem = GeminiMessage.model_validate(chunk_obj)
            used_typed_model = True
            # Extract rich metadata from the typed model
            if gem.isThought:
                meta["isThought"] = True
            if gem.tokenCount is not None:
                meta["tokenCount"] = gem.tokenCount
            if gem.finishReason:
                meta["finishReason"] = gem.finishReason
            if gem.thinkingBudget is not None:
                meta["thinkingBudget"] = gem.thinkingBudget
            if gem.safetyRatings:
                meta["safetyRatings"] = list(gem.safetyRatings)
            if gem.grounding:
                meta["grounding"] = (
                    gem.grounding.model_dump() if hasattr(gem.grounding, "model_dump") else gem.grounding
                )
            if gem.branchParent:
                meta["branchParent"] = (
                    gem.branchParent.model_dump() if hasattr(gem.branchParent, "model_dump") else gem.branchParent
                )
            if gem.branchChildren:
                meta["branchChildren"] = gem.branchChildren
            if gem.executableCode:
                meta["executableCode"] = gem.executableCode
            if gem.codeExecutionResult:
                meta["codeExecutionResult"] = gem.codeExecutionResult
            if gem.errorMessage:
                meta["errorMessage"] = gem.errorMessage
            if gem.isEdited:
                meta["isEdited"] = True

            # Extract structured content blocks via the typed model
            content_block_payloads = [
                block_payload
                for cb in gem.extract_content_blocks()
                if (block_payload := _viewport_block_payload(cb)) is not None
            ]
            if not content_block_payloads:
                # Fallback: basic block from text
                content_block_payloads = (
                    [{"type": "thinking" if gem.isThought else "text", "text": text}] if text else []
                )
            meta["content_blocks"] = content_block_payloads

            # Extract reasoning traces if present
            traces = gem.extract_reasoning_traces()
            if traces:
                meta["reasoning_traces"] = [
                    {"text": t.text, "token_count": t.token_count, "provider": t.provider} for t in traces
                ]
        except (ValidationError, Exception):
            # Fallback: basic extraction for non-conforming chunks
            if chunk_obj.get("isThought"):
                meta["isThought"] = True
            token_count = chunk_obj.get("tokenCount")
            if token_count:
                meta["tokenCount"] = token_count
            fallback_content_blocks: list[dict[str, object]] = []
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

        if chunk_attachments and not used_typed_model:
            meta_blocks = meta.get("content_blocks")
            attachment_blocks = _attachment_block_payloads(chunk_attachments)
            if isinstance(meta_blocks, list):
                meta_blocks.extend(attachment_blocks)
            else:
                meta["content_blocks"] = attachment_blocks

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

    title_val = payload.get("title") or payload.get("displayName")
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
    return ParsedConversation(
        provider_name=runtime_provider,
        provider_conversation_id=str(payload.get("id") or fallback_id),
        title=title,
        created_at=create_time_str,
        updated_at=update_time_str,
        messages=messages,
        attachments=attachments,
    )


def looks_like(payload: object) -> bool:
    """Return True if payload looks like a Drive / Gemini chunkedPrompt export."""
    return isinstance(payload, dict) and "chunkedPrompt" in payload
