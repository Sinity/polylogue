from __future__ import annotations

from pydantic import ValidationError

from polylogue.lib.log import get_logger
from polylogue.sources.providers.gemini import GeminiMessage

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, normalize_role

_logger = get_logger(__name__)


def extract_text_from_chunk(chunk: dict[str, object]) -> str | None:
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        return "\n".join(str(part) for part in parts if part)
    return None


def _collect_drive_docs(payload: object) -> list[dict[str, object] | str]:
    docs: list[dict[str, object] | str] = []
    if not isinstance(payload, dict):
        return docs
    for key in ("driveDocument", "driveDocuments", "drive_document"):
        value = payload.get(key)
        if isinstance(value, (dict, str)):
            docs.append(value)
        elif isinstance(value, list):
            docs.extend([item for item in value if isinstance(item, (dict, str))])
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        docs.extend(_collect_drive_docs(nested))
    return docs


def _attachment_from_doc(doc: dict[str, object] | str, message_id: str | None) -> ParsedAttachment | None:
    if isinstance(doc, str):
        doc_id: str = doc
        meta: dict[str, object] = {"id": doc_id}
        return ParsedAttachment(
            provider_attachment_id=doc_id,
            message_provider_id=message_id,
            name=None,
            mime_type=None,
            size_bytes=None,
            path=None,
            provider_meta=meta,
        )
    if not isinstance(doc, dict):
        return None
    doc_id_val = doc.get("id") or doc.get("fileId") or doc.get("driveId")
    if not isinstance(doc_id_val, str) or not doc_id_val:
        return None
    doc_id = doc_id_val
    size_raw = doc.get("sizeBytes") or doc.get("size")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    name_val = doc.get("name") or doc.get("title")
    mime_val = doc.get("mimeType") or doc.get("mime_type")
    return ParsedAttachment(
        provider_attachment_id=doc_id,
        message_provider_id=message_id,
        name=name_val if isinstance(name_val, str) else None,
        mime_type=mime_val if isinstance(mime_val, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_meta=doc,
    )


def parse_chunked_prompt(provider: str, payload: dict[str, object], fallback_id: str) -> ParsedConversation:
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
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            chunk_obj: dict[str, object] = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            continue
        text = extract_text_from_chunk(chunk_obj)
        if not text:
            continue
        # Role is required - skip chunks without one
        role_val = chunk_obj.get("role") or chunk_obj.get("author")
        if not isinstance(role_val, str) or not role_val:
            continue
        role = normalize_role(role_val)
        msg_id = str(chunk_obj.get("id") or f"chunk-{idx}")

        # Try to parse via the rich GeminiMessage typed model for structured extraction
        meta: dict[str, object] = {"raw": chunk_obj}
        try:
            gem = GeminiMessage.model_validate(chunk_obj)
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
            content_blocks = [
                {"type": cb.type.value if hasattr(cb.type, "value") else str(cb.type), "text": cb.text}
                for cb in gem.extract_content_blocks()
                if cb.text
            ]
            if not content_blocks:
                # Fallback: basic block from text
                content_blocks = [{"type": "thinking" if gem.isThought else "text", "text": text}]
            meta["content_blocks"] = content_blocks

            # Extract reasoning traces if present
            traces = gem.extract_reasoning_traces()
            if traces:
                meta["reasoning_traces"] = [
                    {"text": t.text, "token_count": t.token_count, "provider": t.provider}
                    for t in traces
                ]
        except (ValidationError, Exception):
            # Fallback: basic extraction for non-conforming chunks
            if chunk_obj.get("isThought"):
                meta["isThought"] = True
            token_count = chunk_obj.get("tokenCount")
            if token_count:
                meta["tokenCount"] = token_count
            block_type = "thinking" if chunk_obj.get("isThought") else "text"
            meta["content_blocks"] = [{"type": block_type, "text": text}]
            exec_code = chunk_obj.get("executableCode")
            if isinstance(exec_code, dict) and exec_code:
                meta["executableCode"] = exec_code
            exec_result = chunk_obj.get("codeExecutionResult")
            if isinstance(exec_result, dict) and exec_result:
                meta["codeExecutionResult"] = exec_result
            error_msg = chunk_obj.get("errorMessage")
            if isinstance(error_msg, str) and error_msg:
                meta["errorMessage"] = error_msg

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=default_timestamp,
                provider_meta=meta,
            )
        )
        for doc in _collect_drive_docs(chunk_obj):
            attachment = _attachment_from_doc(doc, msg_id)
            if attachment:
                attachments.append(attachment)

    title_val = payload.get("title") or payload.get("displayName")
    title = str(title_val) if title_val else fallback_id
    create_time_str = str(payload.get("createTime")) if payload.get("createTime") else None
    update_time_str = str(payload.get("updateTime")) if payload.get("updateTime") else None
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=str(payload.get("id") or fallback_id),
        title=title,
        created_at=create_time_str,
        updated_at=update_time_str,
        messages=messages,
        attachments=attachments,
    )
