"""Attachment helpers for Gemini/Drive parsing."""

from __future__ import annotations

import base64
import binascii
from typing import TypeAlias

from polylogue.core.hashing import hash_payload, hash_text_short
from polylogue.core.json import JSONDocument, JSONValue, is_json_document

from .base import ParsedAttachment

_YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

# Sidecar key injected by `polylogue.sources.drive.attachment_fetch` when a
# live DriveSourceAPI client fetched a Drive-hosted document/image/audio/video
# reference's bytes inside the acquisition iterator scope (polylogue-83u.2).
# Value is base64 of the fetched bytes; presence turns an otherwise-unfetched
# `upload_origin="drive"` attachment into one carrying real `inline_bytes`,
# reusing the same true-hash blob-publish path as paste/inline attachments.
DRIVE_LIVE_FETCH_DATA_KEY = "_polylogue_drive_live_bytes_b64"


DrivePayload: TypeAlias = JSONDocument
DriveDocSource: TypeAlias = str | DrivePayload
DrivePayloadSequence: TypeAlias = list[DriveDocSource]
DriveDocMetadata: TypeAlias = dict[str, object]

_DRIVE_MEDIA_FIELDS = (
    ("driveImage", "drive_image"),
    ("driveAudio", "drive_audio"),
    ("driveVideo", "drive_video"),
)

# Public field-name constants (no attachment_kind) for callers outside this
# module that only need to recognize the raw JSON shape — currently
# `polylogue.sources.drive.attachment_fetch`, which fetches live Drive bytes
# for these exact fields before the chunk parser below ever runs.
DRIVE_DOC_FIELD_NAMES: tuple[str, ...] = ("driveDocument", "driveDocuments", "drive_document")
DRIVE_MEDIA_FIELD_NAMES: tuple[str, ...] = tuple(name for name, _kind in _DRIVE_MEDIA_FIELDS)


def _as_drive_payload(payload: object) -> DrivePayload | None:
    return payload if is_json_document(payload) else None


def _collect_doc_fields(payload: DrivePayload) -> DrivePayloadSequence:
    docs: DrivePayloadSequence = []
    for key in DRIVE_DOC_FIELD_NAMES:
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
        return ParsedAttachment(
            provider_attachment_id=doc_id,
            message_provider_id=message_id,
            name=None,
            mime_type=None,
            size_bytes=None,
            path=None,
            provider_file_id=doc_id,
            upload_origin="drive",
        )
    if not is_json_document(doc):
        return None
    doc_id_val = _first_text(doc, "id", "fileId", "driveId")
    if not isinstance(doc_id_val, str) or not doc_id_val:
        return None
    size_bytes = _int_or_none(doc.get("sizeBytes") or doc.get("size"))
    name_val = _first_text(doc, "name", "title")
    mime_val = _first_text(doc, "mimeType", "mime_type")
    # #1252: promote drive native IDs into typed fields. `id`/`fileId` is the
    # Drive file identifier; `driveId` (when present) is the shared-drive
    # container.
    file_id_val = _first_text(doc, "fileId", "id")
    drive_id_val = _first_text(doc, "driveId")
    inline_bytes: bytes | None = None
    fetched_data = doc.get(DRIVE_LIVE_FETCH_DATA_KEY)
    if isinstance(fetched_data, str) and fetched_data:
        try:
            inline_bytes = base64.b64decode(fetched_data, validate=True)
        except (ValueError, binascii.Error):
            inline_bytes = None
    return ParsedAttachment(
        provider_attachment_id=doc_id_val,
        message_provider_id=message_id,
        name=name_val,
        mime_type=mime_val,
        size_bytes=size_bytes if size_bytes is not None else (len(inline_bytes) if inline_bytes is not None else None),
        path=None,
        provider_file_id=file_id_val,
        provider_drive_id=drive_id_val,
        upload_origin="drive",
        inline_bytes=inline_bytes,
    )


def attachment_from_inline_file(
    inline_file: JSONValue | None,
    message_id: str | None,
) -> ParsedAttachment | None:
    return _attachment_from_inline_data(
        inline_file,
        message_id,
        id_prefix="inline-file",
        attachment_kind="inline_file",
    )


def _attachment_from_inline_data(
    inline_file: JSONValue | None,
    message_id: str | None,
    *,
    id_prefix: str,
    attachment_kind: str,
) -> ParsedAttachment | None:
    inline_file = _as_drive_payload(inline_file)
    if inline_file is None:
        return None
    mime_type = _first_text(inline_file, "mimeType", "mime_type")
    data = inline_file.get("data")
    attachment_key = data if isinstance(data, str) and data else hash_payload(inline_file)
    attachment_id = f"{id_prefix}-{hash_text_short(str(attachment_key), length=24)}"
    size_bytes = inline_file_size_bytes(data)
    # Keep the decoded bytes so ingestion can store them in the blob store (#2468);
    # inline files are the one attachment kind whose bytes live in the source
    # export. A non-base64 string is not real file bytes — leave it unfetched.
    inline_bytes: bytes | None = None
    if isinstance(data, str) and data:
        try:
            inline_bytes = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            inline_bytes = None
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        message_provider_id=message_id,
        name=None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        attachment_kind=attachment_kind,
        upload_origin="paste",
        inline_bytes=inline_bytes,
    )


def attachment_from_file_data(
    file_data: JSONValue | None,
    message_id: str | None,
) -> ParsedAttachment | None:
    file_data_obj = _as_drive_payload(file_data)
    if file_data_obj is None:
        return None
    file_uri = _first_text(file_data_obj, "fileUri", "uri", "url")
    if file_uri is None and not file_data_obj:
        return None
    attachment_key = file_uri or hash_payload(file_data_obj)
    attachment_id = f"file-data-{hash_text_short(str(attachment_key), length=24)}"
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        message_provider_id=message_id,
        name=_first_text(file_data_obj, "displayName", "name"),
        mime_type=_first_text(file_data_obj, "mimeType", "mime_type"),
        size_bytes=_int_or_none(file_data_obj.get("sizeBytes") or file_data_obj.get("size")),
        path=None,
        attachment_kind="file_data",
        source_url=file_uri,
        upload_origin="url",
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
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        message_provider_id=message_id,
        name=video_id if isinstance(video_id, str) else None,
        mime_type="video/youtube",
        size_bytes=None,
        path=None,
        attachment_kind="youtube_video",
        source_url=url,
        upload_origin="url",
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
    for field_name, attachment_kind in _DRIVE_MEDIA_FIELDS:
        for doc in _docs_from_named_value(chunk.get(field_name)):
            attachment = attachment_from_doc(doc, message_id)
            if attachment is not None:
                attachments.append(attachment.model_copy(update={"attachment_kind": attachment_kind}))
    inline_attachment = attachment_from_inline_file(chunk.get("inlineFile"), message_id)
    if inline_attachment is not None:
        attachments.append(inline_attachment)
    inline_image = _attachment_from_inline_data(
        chunk.get("inlineImage"),
        message_id,
        id_prefix="inline-image",
        attachment_kind="inline_image",
    )
    if inline_image is not None:
        attachments.append(inline_image)
    parts = chunk.get("parts")
    if isinstance(parts, list):
        for part in parts:
            part_obj = _as_drive_payload(part)
            if part_obj is None:
                continue
            inline_part = _attachment_from_inline_data(
                part_obj.get("inlineData"),
                message_id,
                id_prefix="inline-data",
                attachment_kind="inline_data",
            )
            if inline_part is not None:
                attachments.append(inline_part)
            file_part = attachment_from_file_data(part_obj.get("fileData"), message_id)
            if file_part is not None:
                attachments.append(file_part)
    youtube_attachment = attachment_from_youtube_video(chunk.get("youtubeVideo"), message_id)
    if youtube_attachment is not None:
        attachments.append(youtube_attachment)
    return list({attachment.provider_attachment_id: attachment for attachment in attachments}.values())


def attachment_block_payloads(attachments: list[ParsedAttachment]) -> list[DriveDocMetadata]:
    blocks: list[DriveDocMetadata] = []

    def _metadata_for_block(attachment: ParsedAttachment) -> DriveDocMetadata:
        metadata: DriveDocMetadata = {}
        if attachment.name:
            metadata["name"] = attachment.name
        if attachment.provider_attachment_id:
            metadata["id"] = attachment.provider_attachment_id
        if attachment.provider_file_id:
            metadata["fileId"] = attachment.provider_file_id
        if attachment.provider_drive_id:
            metadata["driveId"] = attachment.provider_drive_id
        if attachment.mime_type:
            metadata["mimeType"] = attachment.mime_type
        if attachment.size_bytes is not None:
            metadata["sizeBytes"] = attachment.size_bytes
        if attachment.source_url:
            metadata["url"] = attachment.source_url
        return metadata

    for attachment in attachments:
        metadata = _metadata_for_block(attachment)
        block_type = (
            "image"
            if attachment.attachment_kind in {"drive_image", "inline_image"}
            or (attachment.mime_type or "").startswith("image/")
            else "document"
        )
        block: DriveDocMetadata = {
            "type": block_type,
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
