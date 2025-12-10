"""Pydantic schemas for ChatGPT export format validation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatGPTAuthor(BaseModel):
    """Author information in ChatGPT messages."""

    role: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class ChatGPTContentPart(BaseModel):
    """A part of message content."""

    content_type: str
    parts: Optional[List[Any]] = None
    text: Optional[str] = None
    language: Optional[str] = None
    # For code execution results
    result: Optional[str] = None
    # For multimodal content
    asset_pointer: Optional[str] = None
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class ChatGPTContent(BaseModel):
    """Message content structure."""

    content_type: str
    parts: Optional[List[Any]] = None
    text: Optional[str] = None
    # Multimodal
    asset_pointer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class ChatGPTMessage(BaseModel):
    """A single message in a ChatGPT conversation.

    This schema validates the expected structure. If OpenAI changes their
    export format, Pydantic will raise a ValidationError with details about
    what field changed, making it easier to adapt the importer.
    """

    id: str
    author: ChatGPTAuthor
    content: ChatGPTContent
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    status: Optional[str] = None
    end_turn: Optional[bool] = None
    weight: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recipient: Optional[str] = None
    # Parent/child relationships for branching
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields to not break on minor additions
    )


class ChatGPTConversationMapping(BaseModel):
    """Mapping structure in ChatGPT exports."""

    id: str
    message: Optional[ChatGPTMessage] = None
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ChatGPTConversation(BaseModel):
    """A complete ChatGPT conversation export.

    Schema version: 2024-01 (update this when format changes are detected)
    """

    title: str
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    mapping: Dict[str, ChatGPTConversationMapping]
    moderation_results: List[Any] = Field(default_factory=list)
    current_node: Optional[str] = None
    plugin_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    conversation_template_id: Optional[str] = None
    id: Optional[str] = None
    # Additional metadata
    is_archived: bool = False
    safe_urls: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="allow",  # Don't break on new fields
    )

    @property
    def conversation_id_normalized(self) -> str:
        """Get the conversation ID, preferring 'id' over 'conversation_id'."""
        return self.id or self.conversation_id or "unknown"
