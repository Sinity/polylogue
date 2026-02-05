"""Provider-specific parsers: JSON export → ParsedConversation."""

from .base import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
)

__all__ = [
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "RawConversationData",
]
