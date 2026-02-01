"""ChatGPT provider-specific typed models.

These models match the ChatGPT export format exactly.
Derived from schema: polylogue/schemas/providers/chatgpt.schema.json
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    ToolCall,
    ToolCategory,
    classify_tool,
)


class ChatGPTAuthor(BaseModel):
    """Author of a ChatGPT message."""

    model_config = ConfigDict(extra="allow")

    role: str
    """Role: user, assistant, system, tool"""

    name: str | None = None
    """Optional author name."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Author metadata."""


class ChatGPTContent(BaseModel):
    """Content of a ChatGPT message."""

    model_config = ConfigDict(extra="allow")

    content_type: str
    """Type: text, code, tether_browsing_display, etc."""

    parts: list[Any] | None = None
    """Content parts (usually strings, can be objects for multimodal)."""

    text: str | None = None
    """Direct text content (alternative to parts)."""

    language: str | None = None
    """Programming language (for code blocks)."""


class ChatGPTMessage(BaseModel):
    """A single ChatGPT message within a conversation node."""

    model_config = ConfigDict(extra="allow")

    id: str
    """Message ID."""

    author: ChatGPTAuthor
    """Message author."""

    create_time: float | None = None
    """Creation timestamp (Unix epoch)."""

    update_time: float | None = None
    """Update timestamp (Unix epoch)."""

    content: ChatGPTContent | None = None
    """Message content."""

    status: str | None = None
    """Message status (finished_successfully, etc.)."""

    end_turn: bool | None = None
    """Whether this message ends the turn."""

    weight: float = 0.0
    """Message weight."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Message metadata (attachments, citations, etc.)."""

    recipient: str | None = None
    """Message recipient."""

    # =========================================================================
    # Viewport extraction methods
    # =========================================================================

    @property
    def text_content(self) -> str:
        """Extract plain text content."""
        if not self.content:
            return ""

        if self.content.text:
            return self.content.text

        if self.content.parts:
            texts = []
            for part in self.content.parts:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
            return "\n".join(texts)

        return ""

    @property
    def timestamp(self) -> datetime | None:
        """Get message timestamp as datetime."""
        if self.create_time:
            try:
                return datetime.fromtimestamp(self.create_time)
            except (ValueError, OSError):
                pass
        return None

    @property
    def role_normalized(self) -> str:
        """Normalize role to standard values."""
        role = self.author.role.lower() if self.author.role else "unknown"
        mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
            "tool": "tool",
        }
        return mapping.get(role, "unknown")

    def to_meta(self) -> MessageMeta:
        """Convert to harmonized MessageMeta."""
        return MessageMeta(
            id=self.id,
            timestamp=self.timestamp,
            role=self.role_normalized,  # type: ignore
            model=self.metadata.get("model_slug"),
            provider="chatgpt",
        )

    def to_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks."""
        blocks = []

        if not self.content:
            return blocks

        content_type = self.content.content_type
        if content_type == "text":
            blocks.append(ContentBlock(
                type=ContentType.TEXT,
                text=self.text_content,
                raw=self.content.model_dump(),
            ))
        elif content_type == "code":
            blocks.append(ContentBlock(
                type=ContentType.CODE,
                text=self.text_content,
                language=self.content.language,
                raw=self.content.model_dump(),
            ))
        elif "tether" in content_type or "browse" in content_type:
            # Web browsing content
            blocks.append(ContentBlock(
                type=ContentType.TOOL_RESULT,
                text=self.text_content,
                raw=self.content.model_dump(),
            ))
        else:
            blocks.append(ContentBlock(
                type=ContentType.UNKNOWN,
                text=self.text_content,
                raw=self.content.model_dump(),
            ))

        return blocks


class ChatGPTNode(BaseModel):
    """A node in the ChatGPT conversation tree."""

    model_config = ConfigDict(extra="allow")

    id: str
    """Node ID."""

    message: ChatGPTMessage | None = None
    """Message at this node (None for root)."""

    parent: str | None = None
    """Parent node ID."""

    children: list[str] = Field(default_factory=list)
    """Child node IDs (for branching conversations)."""


class ChatGPTConversation(BaseModel):
    """A complete ChatGPT conversation export.

    This model matches the ChatGPT export format exactly.
    Use the viewport methods to extract harmonized data.
    """

    model_config = ConfigDict(extra="allow")

    # Core fields
    id: str
    """Conversation ID."""

    conversation_id: str
    """Alternate conversation ID."""

    title: str
    """Conversation title."""

    create_time: float
    """Creation timestamp (Unix epoch)."""

    update_time: float
    """Last update timestamp (Unix epoch)."""

    # Conversation structure
    mapping: dict[str, ChatGPTNode] = Field(default_factory=dict)
    """Node tree (keys are node IDs)."""

    current_node: str
    """Current/latest node ID."""

    # Settings and metadata
    default_model_slug: str | None = None
    """Model used (gpt-4, gpt-4o, etc.)."""

    is_archived: bool = False
    is_starred: bool | None = None
    is_read_only: bool | None = None

    gizmo_id: str | None = None
    """Custom GPT ID if applicable."""

    gizmo_type: str | None = None
    """Custom GPT type."""

    memory_scope: str = "global_enabled"
    """Memory scope setting."""

    safe_urls: list[str] = Field(default_factory=list)
    blocked_urls: list[str] = Field(default_factory=list)
    moderation_results: list[Any] = Field(default_factory=list)

    # =========================================================================
    # Viewport extraction methods
    # =========================================================================

    @property
    def messages(self) -> list[ChatGPTMessage]:
        """Extract messages in order (following the tree from root)."""
        messages = []

        # Find root node (no parent or parent is "client-created-root")
        root_id = None
        for node_id, node in self.mapping.items():
            if node.parent is None or node.parent == "client-created-root":
                root_id = node_id
                break

        if not root_id:
            return messages

        # Walk the tree (simple: follow first child path)
        current_id = root_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self.mapping.get(current_id)
            if not node:
                break

            if node.message:
                messages.append(node.message)

            # Follow first child (main conversation path)
            if node.children:
                current_id = node.children[0]
            else:
                break

        return messages

    @property
    def created_at(self) -> datetime | None:
        """Get creation timestamp as datetime."""
        try:
            return datetime.fromtimestamp(self.create_time)
        except (ValueError, OSError):
            return None

    @property
    def updated_at(self) -> datetime | None:
        """Get update timestamp as datetime."""
        try:
            return datetime.fromtimestamp(self.update_time)
        except (ValueError, OSError):
            return None

    def iter_user_assistant_pairs(self):
        """Iterate over (user_message, assistant_message) pairs."""
        messages = self.messages
        i = 0
        while i < len(messages) - 1:
            user_msg = messages[i]
            asst_msg = messages[i + 1]
            if (user_msg.role_normalized == "user" and
                asst_msg.role_normalized == "assistant"):
                yield (user_msg, asst_msg)
                i += 2
            else:
                i += 1
