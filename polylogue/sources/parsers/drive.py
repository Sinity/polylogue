from __future__ import annotations

import base64
import binascii

from pydantic import ValidationError

from polylogue.lib.hashing import hash_payload, hash_text_short
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.logging import get_logger
from polylogue.sources.providers.gemini import GeminiMessage
from polylogue.types import Provider

from .base import ParsedAttachment, ParsedContentBlock, ParsedConversation, ParsedMessage

_logger = get_logger(__name__)
_YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def extract_text_from_chunk(chunk: object) -> str | None:
    if not isinstance(chunk, dict):
        return None
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str) and part:
                texts.append(part)
            elif isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    texts.append(part_text)
        return "\n".join(texts) or None
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


def _chunk_timestamp(chunk: dict[str, object], default_timestamp: str | None) -> str | None:
    for key in ("createTime", "timestamp", "updateTime"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    return default_timestamp


def _select_timestamp(values: list[str | None], *, latest: bool) -> str | None:
    candidates: list[tuple[object, str]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        parsed = parse_timestamp(value)
        if parsed is None:
            continue
        seen.add(value)
        candidates.append((parsed, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1] if latest else candidates[0][1]


def _inline_file_size_bytes(data: object) -> int | None:
    if not isinstance(data, str) or not data:
        return None
    try:
        return len(base64.b64decode(data, validate=False))
    except (ValueError, binascii.Error):
        return len(data.encode("utf-8"))


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


def _attachment_from_inline_file(inline_file: object, message_id: str | None) -> ParsedAttachment | None:
    if not isinstance(inline_file, dict):
        return None
    mime_type = inline_file.get("mimeType")
    data = inline_file.get("data")
    attachment_key = data if isinstance(data, str) and data else hash_payload(inline_file)
    attachment_id = f"inline-file-{hash_text_short(str(attachment_key), length=24)}"
    size_bytes = _inline_file_size_bytes(data)
    provider_meta: dict[str, object] = {"attachment_kind": "inline_file"}
    if isinstance(mime_type, str) and mime_type:
        provider_meta["mimeType"] = mime_type
    if size_bytes is not None:
        provider_meta["sizeBytes"] = size_bytes
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        message_provider_id=message_id,
        name=None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_meta=provider_meta,
    )


def _attachment_from_youtube_video(video: object, message_id: str | None) -> ParsedAttachment | None:
    if not isinstance(video, dict):
        return None
    video_id = video.get("id")
    if isinstance(video_id, str) and video_id:
        attachment_id = f"youtube-video-{video_id}"
        url = _YOUTUBE_WATCH_URL.format(video_id=video_id)
    else:
        attachment_id = f"youtube-video-{hash_payload(video)}"
        url = None
    provider_meta: dict[str, object] = {"attachment_kind": "youtube_video", **video}
    if url is not None:
        provider_meta["url"] = url
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        message_provider_id=message_id,
        name=video_id if isinstance(video_id, str) else None,
        mime_type="video/youtube",
        size_bytes=None,
        path=None,
        provider_meta=provider_meta,
    )


def _collect_chunk_attachments(chunk: dict[str, object], message_id: str | None) -> list[ParsedAttachment]:
    attachments: list[ParsedAttachment] = []
    for doc in _collect_drive_docs(chunk):
        attachment = _attachment_from_doc(doc, message_id)
        if attachment is not None:
            attachments.append(attachment)
    inline_attachment = _attachment_from_inline_file(chunk.get("inlineFile"), message_id)
    if inline_attachment is not None:
        attachments.append(inline_attachment)
    youtube_attachment = _attachment_from_youtube_video(chunk.get("youtubeVideo"), message_id)
    if youtube_attachment is not None:
        attachments.append(youtube_attachment)
    return attachments


def _attachment_block_payloads(attachments: list[ParsedAttachment]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for attachment in attachments:
        metadata = dict(attachment.provider_meta or {})
        if attachment.name:
            metadata.setdefault("name", attachment.name)
        block: dict[str, object] = {
            "type": "document",
            "media_type": attachment.mime_type,
            "metadata": metadata,
        }
        if attachment.name:
            block["text"] = attachment.name
        blocks.append(block)
    return blocks


def _viewport_block_payload(block) -> dict[str, object] | None:
    raw_type = block.type.value if hasattr(block.type, "value") else str(block.type)
    block_type = {
        "file": "document",
        "audio": "document",
        "video": "document",
        "unknown": "text",
        "system": "text",
        "error": "text",
    }.get(raw_type, raw_type)
    if block_type not in {"text", "thinking", "tool_use", "tool_result", "image", "code", "document"}:
        return None
    payload: dict[str, object] = {"type": block_type}
    if block.text is not None:
        payload["text"] = block.text
    if block.language:
        payload["language"] = block.language
    if block.mime_type:
        payload["media_type"] = block.mime_type
    metadata: dict[str, object] = {}
    if isinstance(block.raw, dict) and block.raw:
        metadata.update(block.raw)
    if getattr(block, "url", None):
        metadata["url"] = block.url
    if metadata:
        payload["metadata"] = metadata
    return payload


def _parsed_content_blocks_from_meta(blocks: object) -> list[ParsedContentBlock]:
    if not isinstance(blocks, list):
        return []
    parsed: list[ParsedContentBlock] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type:
            continue
        metadata: dict[str, object] | None = (
            dict(block.get("metadata")) if isinstance(block.get("metadata"), dict) else None
        )
        language = block.get("language")
        if isinstance(language, str) and language:
            metadata = dict(metadata or {})
            metadata.setdefault("language", language)
        parsed.append(
            ParsedContentBlock(
                type=block_type,
                text=block.get("text") if isinstance(block.get("text"), str) else None,
                media_type=block.get("media_type") if isinstance(block.get("media_type"), str) else None,
                metadata=metadata,
            )
        )
    return parsed


def parse_chunked_prompt(provider: Provider | str, payload: dict[str, object], fallback_id: str) -> ParsedConversation:
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
            content_blocks = [
                block_payload
                for cb in gem.extract_content_blocks()
                if (block_payload := _viewport_block_payload(cb)) is not None
            ]
            if not content_blocks:
                # Fallback: basic block from text
                content_blocks = (
                    [{"type": "thinking" if gem.isThought else "text", "text": text}]
                    if text
                    else []
                )
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
            content_blocks: list[dict[str, object]] = []
            if text:
                block_type = "thinking" if chunk_obj.get("isThought") else "text"
                content_blocks.append({"type": block_type, "text": text})
            exec_code = chunk_obj.get("executableCode")
            if isinstance(exec_code, dict) and exec_code:
                meta["executableCode"] = exec_code
                code = exec_code.get("code")
                if isinstance(code, str) and code:
                    content_blocks.append({"type": "code", "text": code})
            exec_result = chunk_obj.get("codeExecutionResult")
            if isinstance(exec_result, dict) and exec_result:
                meta["codeExecutionResult"] = exec_result
                output = exec_result.get("output")
                outcome = exec_result.get("outcome")
                if isinstance(output, str) and output:
                    content_blocks.append({"type": "tool_result", "text": output})
                elif isinstance(outcome, str) and outcome:
                    content_blocks.append({"type": "tool_result", "text": f"[{outcome}]"})
            error_msg = chunk_obj.get("errorMessage")
            if isinstance(error_msg, str) and error_msg:
                meta["errorMessage"] = error_msg
            meta["content_blocks"] = content_blocks

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
        provider_name=provider,
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
