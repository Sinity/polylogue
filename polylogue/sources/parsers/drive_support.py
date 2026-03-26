"""Shared Gemini/Drive parser helpers."""

from __future__ import annotations

import base64
import binascii

from polylogue.lib.hashing import hash_payload, hash_text_short
from polylogue.lib.timestamps import parse_timestamp

from .base import ParsedAttachment, ParsedContentBlock

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


def chunk_timestamp(chunk: dict[str, object], default_timestamp: str | None) -> str | None:
    for key in ("createTime", "timestamp", "updateTime"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    return default_timestamp


def select_timestamp(values: list[str | None], *, latest: bool) -> str | None:
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
        provider_attachment_id=doc_id_val,
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


def collect_chunk_attachments(chunk: dict[str, object], message_id: str | None) -> list[ParsedAttachment]:
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


def attachment_block_payloads(attachments: list[ParsedAttachment]) -> list[dict[str, object]]:
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


def viewport_block_payload(block) -> dict[str, object] | None:
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


def parsed_content_blocks_from_meta(blocks: object) -> list[ParsedContentBlock]:
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


__all__ = [
    "attachment_block_payloads",
    "chunk_timestamp",
    "collect_chunk_attachments",
    "extract_text_from_chunk",
    "parsed_content_blocks_from_meta",
    "select_timestamp",
    "viewport_block_payload",
]
