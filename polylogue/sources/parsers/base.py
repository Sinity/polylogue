"""Public parser contracts and shared extraction helpers."""

from __future__ import annotations

from polylogue.archive.message.roles import normalize_role

from .base_models import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedFileEdit,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
    ParsedSessionEvent,
    ParsedSessionRef,
    ParsedWebConstruct,
    RawSessionData,
)
from .base_support import (
    attachment_from_meta,
    content_blocks_from_segments,
    extract_messages_from_list,
    fill_linear_parent_chain,
    text_blocks_prose,
)

__all__ = [
    "ParsedContentBlock",
    "ParsedFileEdit",
    "ParsedMessage",
    "ParsedAttachment",
    "ParsedPasteEvidence",
    "ParsedWebConstruct",
    "ParsedSession",
    "ParsedSessionEvent",
    "ParsedSessionRef",
    "RawSessionData",
    "normalize_role",
    "content_blocks_from_segments",
    "extract_messages_from_list",
    "attachment_from_meta",
    "fill_linear_parent_chain",
    "text_blocks_prose",
]
