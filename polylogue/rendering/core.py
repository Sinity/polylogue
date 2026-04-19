"""Core rendering utilities shared across all renderers."""

from __future__ import annotations

from polylogue.rendering.block_models import RenderableBlock, coerce_renderable_blocks
from polylogue.rendering.core_formatter import (
    ConversationFormatter,
    FormattedConversation,
    FormattedConversationMetadata,
)
from polylogue.rendering.core_markdown import format_conversation_markdown
from polylogue.rendering.core_messages import (
    RenderedMessage,
    build_rendered_message,
    build_rendered_message_payload,
)

__all__ = [
    "ConversationFormatter",
    "FormattedConversation",
    "FormattedConversationMetadata",
    "RenderableBlock",
    "RenderedMessage",
    "build_rendered_message",
    "build_rendered_message_payload",
    "coerce_renderable_blocks",
    "format_conversation_markdown",
]
