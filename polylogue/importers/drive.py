from __future__ import annotations

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, normalize_role


def extract_text_from_chunk(chunk: dict) -> str | None:
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        return "\n".join(str(part) for part in parts if part)
    return None


def _collect_drive_docs(payload: object) -> list[dict | str]:
    docs: list[dict | str] = []
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


def _attachment_from_doc(doc: dict | str, message_id: str | None) -> ParsedAttachment | None:
    if isinstance(doc, str):
        doc_id = doc
        meta = {"id": doc_id}
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
    doc_id = doc.get("id") or doc.get("fileId") or doc.get("driveId")
    if not isinstance(doc_id, str) or not doc_id:
        return None
    size_raw = doc.get("sizeBytes") or doc.get("size")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    return ParsedAttachment(
        provider_attachment_id=doc_id,
        message_provider_id=message_id,
        name=doc.get("name") or doc.get("title"),
        mime_type=doc.get("mimeType") or doc.get("mime_type"),
        size_bytes=size_bytes,
        path=None,
        provider_meta=doc,
    )


def parse_chunked_prompt(provider: str, payload: dict, fallback_id: str) -> ParsedConversation:
    prompt = payload.get("chunkedPrompt")
    chunks = []
    if isinstance(prompt, dict):
        chunks = prompt.get("chunks") or []
    elif isinstance(payload.get("chunks"), list):
        chunks = payload.get("chunks")

    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            chunk_obj = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            continue
        text = extract_text_from_chunk(chunk_obj)
        if not text:
            continue
        role = normalize_role(chunk_obj.get("role") or chunk_obj.get("author"))
        msg_id = str(chunk_obj.get("id") or f"chunk-{idx}")
        # Preserve useful metadata (isThought for Gemini thinking traces, tokenCount, etc.)
        meta = {"raw": chunk_obj}
        if chunk_obj.get("isThought"):
            meta["isThought"] = True
        if chunk_obj.get("tokenCount"):
            meta["tokenCount"] = chunk_obj.get("tokenCount")
        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=None,
                provider_meta=meta,
            )
        )
        for doc in _collect_drive_docs(chunk_obj):
            attachment = _attachment_from_doc(doc, msg_id)
            if attachment:
                attachments.append(attachment)

    title = payload.get("title") or payload.get("displayName") or fallback_id
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=str(payload.get("id") or fallback_id),
        title=str(title),
        created_at=str(payload.get("createTime")) if payload.get("createTime") else None,
        updated_at=str(payload.get("updateTime")) if payload.get("updateTime") else None,
        messages=messages,
        attachments=attachments,
    )
