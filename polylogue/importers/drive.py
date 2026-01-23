from __future__ import annotations

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, normalize_role


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
        role_val = chunk_obj.get("role") or chunk_obj.get("author")
        role = normalize_role(role_val if isinstance(role_val, str) else None)
        msg_id = str(chunk_obj.get("id") or f"chunk-{idx}")
        # Preserve useful metadata (isThought for Gemini thinking traces, tokenCount, etc.)
        meta: dict[str, object] = {"raw": chunk_obj}
        if chunk_obj.get("isThought"):
            meta["isThought"] = True
        token_count = chunk_obj.get("tokenCount")
        if token_count:
            meta["tokenCount"] = token_count

        # Extract structured content blocks for semantic detection
        content_blocks: list[dict[str, object]] = []
        if chunk_obj.get("isThought"):
            # Gemini thinking block
            content_blocks.append({
                "type": "thinking",
                "text": text,
            })
        else:
            # Regular text content
            content_blocks.append({
                "type": "text",
                "text": text,
            })
        if content_blocks:
            meta["content_blocks"] = content_blocks

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
