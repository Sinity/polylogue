"""Conversation-oriented MCP payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from polylogue.lib.models import Conversation, ConversationSummary
from polylogue.mcp.payload_base import MCPPayload, MCPRootPayload, normalize_role


class MCPErrorPayload(MCPPayload):
    error: str
    tool: str | None = None
    conversation_id: str | None = None


class MCPMessagePayload(MCPPayload):
    id: str
    role: str
    text: str
    timestamp: datetime | None = None

    @classmethod
    def from_message(cls, message: Any) -> MCPMessagePayload:
        return cls(
            id=str(message.id),
            role=normalize_role(message.role),
            text=message.text or "",
            timestamp=message.timestamp,
        )


class MCPConversationSummaryPayload(MCPPayload):
    id: str
    provider: str
    title: str
    message_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationSummaryPayload:
        return cls(
            id=str(conversation.id),
            provider=conversation.provider,
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int | None = None,
    ) -> MCPConversationSummaryPayload:
        return cls(
            id=str(summary.id),
            provider=summary.provider,
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class MCPConversationDetailPayload(MCPConversationSummaryPayload):
    messages: list[MCPMessagePayload]

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> MCPConversationDetailPayload:
        summary = MCPConversationSummaryPayload.from_conversation(conversation)
        return cls(
            **summary.model_dump(),
            messages=[MCPMessagePayload.from_message(msg) for msg in conversation.messages],
        )


class MCPConversationSummaryListPayload(MCPRootPayload):
    root: list[MCPConversationSummaryPayload]


__all__ = [
    "MCPConversationDetailPayload",
    "MCPConversationSummaryListPayload",
    "MCPConversationSummaryPayload",
    "MCPErrorPayload",
    "MCPMessagePayload",
]
