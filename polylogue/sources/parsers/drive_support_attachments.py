"""Attachment helpers for Gemini/Drive parsing."""

from __future__ import annotations

import base64
import binascii
from typing import TypeAlias

from polylogue.core.hashing import hash_payload, hash_text_short
from polylogue.core.json import JSONDocument, JSONValue, is_json_document

from .base import ParsedAttachment

_YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


DrivePayload: TypeAlias = JSONDocument
DriveDocSource: TypeAlias = str | DrivePayload
DrivePayloadSequence: TypeAlias = list[DriveDocSource]
DriveDocMetadata: TypeAlias = dict[str, object]


def _as_drive_payload(payload: object) -> DrivePayload | None:
    return payload if is_json_document(payload) else None


def _as_document_metadata(document: JSONDocument) -> DriveDocMetadata:
    return dict(document)


def _collect_doc_fields(payload: DrivePayload) -> DrivePayloadSequence:
    docs: DrivePayloadSequence = []
    for key in ("driveDocument", "driveDocuments", "drive_document"):
        docs.extend(_docs_from_named_value(payload.get(key)))
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        docs.extend(collect_drive_docs(nested))
    return docs


def collect_drive_docs(payload: object) -> list[DriveDocSource]:
    payload_dict = _as_drive_payload(payload)
    if payload_dict is None:
        return []
    return _collect_doc_fields(payload_dict)


def _docs_from_named_value(value: JSONValue | None) -> list[DriveDocSource]:
    if isinstance(value, str):
        return [value]
    if is_json_document(value):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) or is_json_document(item)]
    return []


def inline_file_size_bytes(data: object) -> int | None:
    if not isinstance(data, str) or not data:
        return None
    try:
        return len(base64.b64decode(data, validate=False))
    except (ValueError, binascii.Error):
        return len(data.encode("utf-8"))


def _first_text(payload: DrivePayload, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def attachment_from_doc(doc: DriveDocSource, message_id: str | None) -> ParsedAttachment | None:
    if isinstance(doc, str):
        doc_id: str = doc
        meta: DrivePayload = {"id": doc_id}
        return ParsedAttachment(
            provider_attachment_id=doc_id,
            message_provider_id=message_id,
            name=None,
            mime_type=None,
            size_bytes=None,
            path=None,
            provider_meta=_as_document_metadata(meta),
        )
    if not is_json_document(doc):
        return None
    doc_id_val = _first_text(doc, "id", "fileId", "driveId")
    if not isinstance(doc_id_val, str) or not doc_id_val:
        return None
    size_bytes = _int_or_none(doc.get("sizeBytes") or doc.get("size"))
    name_val = _first_text(doc, "name", "title")
    mime_val = _first_text(doc, "mimeType", "mime_type")
    return ParsedAttachment(
        provider_attachment_id=doc_id_val,
        message_provider_id=message_id,
        name=name_val,
        mime_type=mime_val,
        size_bytes=size_bytes,
        path=None,
        provider_meta=_as_document_metadata(doc),
    )


def attachment_from_inline_file(
    inline_file: JSONValue | None,
    message_id: str | None,
) -> ParsedAttachment | None:
    inline_file = _as_drive_payload(inline_file)
    if inline_file is None:
        return None
    mime_type = _first_text(inline_file, "mimeType", "mime_type")
    data = inline_file.get("data")
    attachment_key = data if isinstance(data, str) and data else hash_payload(inline_file)
    attachment_id = f"inline-file-{hash_text_short(str(attachment_key), length=24)}"
    size_bytes = inline_file_size_bytes(data)
    provider_meta: DriveDocMetadata = {"attachment_kind": "inline_file"}
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


def attachment_from_youtube_video(
    video: JSONValue | None,
    message_id: str | None,
) -> ParsedAttachment | None:
    video_dict = _as_drive_payload(video)
    if video_dict is None:
        return None
    video_id = _first_text(video_dict, "id")
    if isinstance(video_id, str) and video_id:
        attachment_id = f"youtube-video-{video_id}"
        url = _YOUTUBE_WATCH_URL.format(video_id=video_id)
    else:
        attachment_id = f"youtube-video-{hash_payload(video_dict)}"
        url = None
    provider_meta: DriveDocMetadata = {"attachment_kind": "youtube_video"}
    provider_meta.update(_as_document_metadata(video_dict))
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


def collect_chunk_attachments(
    chunk: DrivePayload,
    message_id: str | None,
) -> list[ParsedAttachment]:
    attachments: list[ParsedAttachment] = []
    for doc in collect_drive_docs(chunk):
        attachment = attachment_from_doc(doc, message_id)
        if attachment is not None:
            attachments.append(attachment)
    inline_attachment = attachment_from_inline_file(chunk.get("inlineFile"), message_id)
    if inline_attachment is not None:
        attachments.append(inline_attachment)
    youtube_attachment = attachment_from_youtube_video(chunk.get("youtubeVideo"), message_id)
    if youtube_attachment is not None:
        attachments.append(youtube_attachment)
    return attachments


def attachment_block_payloads(attachments: list[ParsedAttachment]) -> list[DriveDocMetadata]:
    blocks: list[DriveDocMetadata] = []

    def _metadata_for_block(attachment: ParsedAttachment) -> DriveDocMetadata:
        metadata: DriveDocMetadata = dict(attachment.provider_meta or {})
        if attachment.name:
            metadata.setdefault("name", attachment.name)
        return metadata

    for attachment in attachments:
        metadata = _metadata_for_block(attachment)
        block: DriveDocMetadata = {
            "type": "document",
            "media_type": attachment.mime_type,
            "metadata": metadata,
        }
        if attachment.name:
            block["text"] = attachment.name
        blocks.append(block)
    return blocks


__all__ = [
    "attachment_block_payloads",
    "collect_chunk_attachments",
]
