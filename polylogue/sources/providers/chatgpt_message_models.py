"""ChatGPT message-level typed models and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.provider_semantics import extract_chatgpt_text
from polylogue.lib.roles import normalize_role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    ToolCall,
)


class ChatGPTAuthor(BaseModel):
    """Author of a ChatGPT message."""

    model_config = ConfigDict(extra="allow")

    role: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatGPTContent(BaseModel):
    """Content of a ChatGPT message."""

    model_config = ConfigDict(extra="allow")

    content_type: str
    parts: list[Any] | None = None
    text: str | None = None
    language: str | None = None


class ChatGPTMessage(BaseModel):
    """A single ChatGPT message within a conversation node."""

    model_config = ConfigDict(extra="allow")

    id: str
    author: ChatGPTAuthor
    create_time: float | None = None
    update_time: float | None = None
    content: ChatGPTContent | None = None
    status: str | None = None
    end_turn: bool | None = None
    weight: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    recipient: str | None = None

    @property
    def text_content(self) -> str:
        if not self.content:
            return ""
        return extract_chatgpt_text(self.content.model_dump(mode="python"))

    @property
    def timestamp(self) -> datetime | None:
        return parse_timestamp(self.create_time)

    @property
    def parsed_timestamp(self) -> datetime | None:
        return self.timestamp

    @property
    def role_normalized(self) -> str:
        role = self.author.role if self.author.role else "unknown"
        try:
            return normalize_role(role)
        except ValueError:
            return "unknown"

    def to_meta(self) -> MessageMeta:
        return MessageMeta(
            id=self.id,
            timestamp=self.timestamp,
            role=self.role_normalized,
            model=self.metadata.get("model_slug"),
            provider="chatgpt",
        )

    def to_content_blocks(self) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        if not self.content:
            return blocks

        content_type = self.content.content_type
        if content_type == "text":
            blocks.append(
                ContentBlock(
                    type=ContentType.TEXT,
                    text=self.text_content,
                    raw=self.content.model_dump(),
                )
            )
        elif content_type == "code":
            blocks.append(
                ContentBlock(
                    type=ContentType.CODE,
                    text=self.text_content,
                    language=self.content.language,
                    raw=self.content.model_dump(),
                )
            )
        elif "tether" in content_type or "browse" in content_type:
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_RESULT,
                    text=self.text_content,
                    raw=self.content.model_dump(),
                )
            )
        else:
            blocks.append(
                ContentBlock(
                    type=ContentType.UNKNOWN,
                    text=self.text_content,
                    raw=self.content.model_dump(),
                )
            )

        return blocks

    def extract_content_blocks(self) -> list[ContentBlock]:
        return self.to_content_blocks()

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        return []

    def extract_tool_calls(self) -> list[ToolCall]:
        return []
