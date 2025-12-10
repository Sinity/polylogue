"""Pydantic schemas for Claude.ai export format validation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClaudeAIAttachment(BaseModel):
    """Attachment in a Claude.ai message."""

    file_name: str
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    extracted_content: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ClaudeAIContent(BaseModel):
    """Content block in Claude.ai messages."""

    type: str
    text: Optional[str] = None
    # For tool use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # For tool results
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    is_error: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


class ClaudeAIMessage(BaseModel):
    """A single message in a Claude.ai conversation.

    This schema validates the expected structure from Claude.ai exports.
    """

    uuid: str
    text: str
    sender: str  # "human" or "assistant"
    index: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    edited_at: Optional[str] = None
    chat_feedback: Optional[Any] = None
    attachments: List[ClaudeAIAttachment] = Field(default_factory=list)
    # New format with content blocks
    content: Optional[List[ClaudeAIContent]] = None
    files: Optional[List[Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ClaudeAIConversation(BaseModel):
    """A complete Claude.ai conversation export.

    Schema version: 2024-01
    """

    uuid: str
    name: str
    created_at: str
    updated_at: str
    # Messages can be flat list or nested in chat_messages
    chat_messages: Optional[List[ClaudeAIMessage]] = Field(default_factory=list)
    messages: Optional[List[ClaudeAIMessage]] = Field(default_factory=list)
    # Project info
    project_uuid: Optional[str] = None
    model: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @property
    def all_messages(self) -> List[ClaudeAIMessage]:
        """Get all messages from either format."""
        if self.chat_messages:
            return self.chat_messages
        return self.messages or []
