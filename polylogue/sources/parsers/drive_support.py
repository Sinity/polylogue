"""Shared Gemini/Drive parser helpers."""

from __future__ import annotations

from polylogue.sources.parsers.base import ParsedAttachment
from polylogue.sources.parsers.drive_support_attachments import (
    DriveDocSource,
    attachment_block_payloads,
    attachment_from_doc,
    collect_chunk_attachments,
    collect_drive_docs,
)
from polylogue.sources.parsers.drive_support_blocks import (
    parsed_content_blocks_from_meta,
    viewport_block_payload,
)
from polylogue.sources.parsers.drive_support_text import (
    chunk_timestamp,
    extract_text_from_chunk,
    select_timestamp,
)

__all__ = [
    "_attachment_from_doc",
    "_collect_drive_docs",
    "attachment_block_payloads",
    "chunk_timestamp",
    "collect_chunk_attachments",
    "collect_drive_docs",
    "extract_text_from_chunk",
    "parsed_content_blocks_from_meta",
    "select_timestamp",
    "viewport_block_payload",
]


def _collect_drive_docs(payload: object) -> list[DriveDocSource]:
    return collect_drive_docs(payload)


def _attachment_from_doc(doc: DriveDocSource, message_id: str | None) -> ParsedAttachment | None:
    return attachment_from_doc(doc, message_id)
