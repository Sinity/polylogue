"""Codex (OpenAI CLI) provider-specific typed models.

These models match the Codex session JSONL format.
Derived from schema: polylogue/schemas/providers/codex.schema.json

Codex sessions can have multiple format generations:
- Envelope format: {"type": "...", "payload": {...}}
- Direct format: {"type": "message", "role": "...", ...}
- Legacy format: {"prompt": "...", "completion": "..."}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from polylogue.lib.roles import normalize_role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
)


class CodexContentBlock(BaseModel):
    """Content block within a Codex message."""

    model_config = ConfigDict(extra="allow")

    type: str
    """Block type: input_text, output_text, etc."""

    text: str | None = None
    """Text content."""


class CodexGitInfo(BaseModel):
    """Git context from Codex session."""

    model_config = ConfigDict(extra="allow")

    commit_hash: str | None = None
    branch: str | None = None
    repository_url: str | None = None


class CodexRecord(BaseModel):
    """A single record from a Codex JSONL session.

    Handles multiple format generations:
    - Envelope: {"type": "session_meta"|"response_item", "payload": {...}}
    - Direct: {"type": "message", "role": "...", "content": [...]}
    - State: {"record_type": "state"}
    """

    model_config = ConfigDict(extra="allow")

    # Envelope format fields
    type: str | None = None
    """Record type: session_meta, response_item, message, etc."""

    payload: dict[str, Any] | None = None
    """Payload for envelope format."""

    # Direct format fields
    record_type: str | None = None
    """Record type for state markers."""

    role: str | None = None
    """Role for direct message format."""

    content: list[CodexContentBlock | dict[str, Any]] | None = None
    """Content blocks for direct message format."""

    # Session metadata
    id: str | None = None
    """Session or message ID."""

    timestamp: str | None = None
    """ISO timestamp."""

    git: CodexGitInfo | None = None
    """Git context."""

    instructions: str | None = None
    """System instructions."""

    # =========================================================================
    # Format detection and normalization
    # =========================================================================

    @property
    def format_type(self) -> Literal["envelope", "direct", "state", "unknown"]:
        """Detect which format generation this record uses."""
        if self.payload is not None:
            return "envelope"
        if self.record_type == "state":
            return "state"
        if self.role is not None or self.type == "message":
            return "direct"
        return "unknown"

    @property
    def is_message(self) -> bool:
        """Check if this record contains an actual message."""
        if self.format_type == "envelope":
            return self.type == "response_item"
        if self.format_type == "direct":
            return self.type == "message" or self.role is not None
        return False

    @property
    def effective_role(self) -> str:
        """Get normalized role from any format."""
        if self.format_type == "envelope" and self.payload:
            return str(self.payload.get("role", "unknown"))
        if self.role:
            return self.role
        return "unknown"

    @property
    def role_normalized(self) -> str:
        """Normalize effective_role to standard viewport values."""
        try:
            return normalize_role(self.effective_role)
        except ValueError:
            return "unknown"

    @property
    def effective_content(self) -> list[dict[str, Any]]:
        """Get content blocks from any format."""
        if self.format_type == "envelope" and self.payload:
            content = self.payload.get("content", [])
            if isinstance(content, list):
                return content
            return []
        if self.content:
            return [
                c.model_dump() if isinstance(c, CodexContentBlock) else c
                for c in self.content
            ]
        return []

    @property
    def text_content(self) -> str:
        """Extract plain text from any format."""
        texts = []
        for block in self.effective_content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("input_text") or block.get("output_text")
                if text:
                    texts.append(text)
        return "\n".join(texts)

    @property
    def parsed_timestamp(self) -> datetime | None:
        """Parse timestamp to datetime."""
        return parse_timestamp(self.timestamp)

    # =========================================================================
    # Viewport extraction
    # =========================================================================

    def to_meta(self) -> MessageMeta:
        """Convert to harmonized MessageMeta."""
        role = self.effective_role
        try:
            role_normalized = normalize_role(role)
        except ValueError:
            role_normalized = "unknown"

        return MessageMeta(
            id=self.id,
            timestamp=self.parsed_timestamp,
            role=role_normalized,
            provider="codex",
        )

    def extract_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks."""
        blocks = []
        for raw in self.effective_content:
            if not isinstance(raw, dict):
                continue

            block_type = raw.get("type", "")

            if block_type in ("input_text", "output_text", "text"):
                blocks.append(ContentBlock(
                    type=ContentType.TEXT,
                    text=raw.get("text") or raw.get("input_text") or raw.get("output_text"),
                    raw=raw,
                ))
            elif "code" in block_type:
                blocks.append(ContentBlock(
                    type=ContentType.CODE,
                    text=raw.get("text") or raw.get("code"),
                    language=raw.get("language"),
                    raw=raw,
                ))
            else:
                blocks.append(ContentBlock(
                    type=ContentType.UNKNOWN,
                    text=raw.get("text"),
                    raw=raw,
                ))

        return blocks

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        """Extract reasoning traces (Codex does not expose reasoning; returns empty list)."""
        return []
