"""Claude AI (web) provider-specific typed models.

These models match the Claude AI export format exactly.
Derived from schema: polylogue/schemas/providers/claude-ai.schema.json
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.roles import normalize_role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import ContentBlock, ContentType, MessageMeta, ReasoningTrace


class ClaudeAIChatMessage(BaseModel):
    """A single message in a Claude AI conversation."""

    model_config = ConfigDict(extra="allow")

    uuid: str
    """Message UUID."""

    text: str
    """Message text content."""

    sender: str
    """Sender: human or assistant."""

    created_at: str | None = None
    """Creation timestamp (ISO format)."""

    updated_at: str | None = None
    """Update timestamp (ISO format)."""

    attachments: list[dict[str, Any]] = Field(default_factory=list)
    """File attachments."""

    files: list[dict[str, Any]] = Field(default_factory=list)
    """Associated files."""

    @property
    def text_content(self) -> str:
        """Extract plain text content (viewport interface)."""""
        return self.text

    @property
    def role_normalized(self) -> str:
        """Normalize role to standard values."""
        try:
            return normalize_role(self.sender)
        except ValueError:
            return "unknown"

    @property
    def parsed_timestamp(self) -> datetime | None:
        """Parse timestamp to datetime."""
        return parse_timestamp(self.created_at)

    def to_meta(self) -> MessageMeta:
        """Convert to harmonized MessageMeta."""
        return MessageMeta(
            id=self.uuid,
            timestamp=self.parsed_timestamp,
            role=self.role_normalized,
            provider="claude-ai",
        )

    def to_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks."""
        return [ContentBlock(
            type=ContentType.TEXT,
            text=self.text,
            raw={"text": self.text, "sender": self.sender},
        )]

    def extract_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks (viewport interface alias for to_content_blocks)."""
        return self.to_content_blocks()

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        """Extract reasoning traces (Claude AI web does not expose reasoning; returns empty list)."""
        return []


class ClaudeAIConversation(BaseModel):
    """A complete Claude AI conversation export."""

    model_config = ConfigDict(extra="allow")

    uuid: str
    """Conversation UUID."""

    name: str
    """Conversation name/title."""

    created_at: str
    """Creation timestamp."""

    updated_at: str
    """Last update timestamp."""

    chat_messages: list[ClaudeAIChatMessage] = Field(default_factory=list)
    """Messages in the conversation."""

    account: dict[str, Any] | None = None
    """Account information."""

    summary: str | None = None
    """Conversation summary (if generated)."""

    @property
    def title(self) -> str:
        """Get conversation title."""
        return self.name

    @property
    def created_datetime(self) -> datetime | None:
        """Parse creation timestamp."""
        return parse_timestamp(self.created_at)

    @property
    def updated_datetime(self) -> datetime | None:
        """Parse update timestamp."""
        return parse_timestamp(self.updated_at)

    @property
    def messages(self) -> list[ClaudeAIChatMessage]:
        """Get messages (alias for chat_messages)."""
        return self.chat_messages
