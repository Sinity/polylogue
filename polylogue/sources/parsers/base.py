"""Public parser contracts and shared extraction helpers."""

from __future__ import annotations

from polylogue.lib.roles import normalize_role

from .base_models import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
)
from .base_support import (
    attachment_from_meta,
    content_blocks_from_segments,
    extract_messages_from_list,
)

__all__ = [
    "ParsedContentBlock",
    "ParsedMessage",
    "ParsedAttachment",
    "ParsedConversation",
    "RawConversationData",
    "normalize_role",
    "content_blocks_from_segments",
    "extract_messages_from_list",
    "attachment_from_meta",
]
