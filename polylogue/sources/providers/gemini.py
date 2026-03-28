"""Gemini AI Studio provider-specific typed models.

These models match the Gemini message format from polylogue imports.
Derived from schema: polylogue/schemas/providers/gemini.schema.json
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
)


class GeminiBranchParent(BaseModel):
    """Parent reference for branched conversations."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    index: int | None = None


class GeminiGrounding(BaseModel):
    """Grounding/search results from Gemini."""

    model_config = ConfigDict(extra="allow")

    # Structure varies - use extra="allow" to capture all


class GeminiThoughtSignature(BaseModel):
    """Signature for thinking blocks."""

    model_config = ConfigDict(extra="allow")

    # Structure varies


class GeminiPart(BaseModel):
    """Content part within a Gemini message."""

    model_config = ConfigDict(extra="allow")

    text: str | None = None
    # Can also contain inlineData, fileData, etc.


class GeminiMessage(BaseModel):
    """A single Gemini AI Studio message.

    This represents the raw message format as stored in polylogue's provider_meta.
    """

    model_config = ConfigDict(extra="allow")

    # Core fields
    text: str
    """Message text content."""

    role: str
    """Role: user or model."""

    # Token and completion info
    tokenCount: int | None = None
    """Token count for this message."""

    finishReason: str | None = None
    """Why generation stopped (STOP, MAX_TOKENS, etc.)."""

    # Thinking features
    isThought: bool = False
    """Whether this is a thinking/reasoning block."""

    thinkingBudget: int | None = None
    """Token budget allocated for thinking."""

    thoughtSignatures: list[GeminiThoughtSignature | dict[str, Any] | str] = Field(default_factory=list)
    """Signatures for thought verification (can be dicts, strings, or typed)."""

    # Content structure
    parts: list[GeminiPart | dict[str, Any]] = Field(default_factory=list)
    """Structured content parts."""

    # Grounding
    grounding: GeminiGrounding | dict[str, Any] | None = None
    """Web grounding/search results."""

    # Branching
    branchParent: GeminiBranchParent | dict[str, Any] | None = None
    """Parent for branched conversations."""

    branchChildren: list[dict[str, Any]] = Field(default_factory=list)
    """Child branches."""

    # Safety
    safetyRatings: list[dict[str, Any]] = Field(default_factory=list)
    """Safety rating results."""

    # =========================================================================
    # Viewport extraction methods
    # =========================================================================

    @property
    def role_normalized(self) -> str:
        """Normalize role to standard values."""
        role = self.role.lower() if self.role else "unknown"
        return {
            "user": "user",
            "model": "assistant",
            "assistant": "assistant",
            "system": "system",
        }.get(role, "unknown")

    @property
    def text_content(self) -> str:
        """Get full text content."""
        if self.text:
            return self.text

        # Fall back to parts
        texts = []
        for part in self.parts:
            if isinstance(part, GeminiPart) and part.text:
                texts.append(part.text)
            elif isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        return "\n".join(texts)

    def to_meta(self) -> MessageMeta:
        """Convert to harmonized MessageMeta."""
        tokens = None
        if self.tokenCount is not None:
            tokens = TokenUsage(output_tokens=self.tokenCount)

        return MessageMeta(
            role=self.role_normalized,  # type: ignore
            tokens=tokens,
            provider="gemini",
        )

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        """Extract thinking/reasoning traces."""
        traces = []
        if self.isThought:
            # Normalize signatures (can be strings, dicts, or models)
            sigs = []
            for s in self.thoughtSignatures:
                if isinstance(s, str):
                    sigs.append(s)
                elif isinstance(s, BaseModel):
                    sigs.append(s.model_dump())
                else:
                    sigs.append(s)

            traces.append(ReasoningTrace(
                text=self.text,
                token_count=self.thinkingBudget,
                provider="gemini",
                raw={
                    "isThought": True,
                    "thinkingBudget": self.thinkingBudget,
                    "thoughtSignatures": sigs,
                },
            ))
        return traces

    def extract_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks."""
        blocks = []

        if self.isThought:
            blocks.append(ContentBlock(
                type=ContentType.THINKING,
                text=self.text,
                raw={"isThought": True},
            ))
        elif self.text:
            blocks.append(ContentBlock(
                type=ContentType.TEXT,
                text=self.text,
                raw={"role": self.role},
            ))

        # Add parts if present
        for part in self.parts:
            if isinstance(part, GeminiPart):
                if part.text:
                    blocks.append(ContentBlock(
                        type=ContentType.TEXT,
                        text=part.text,
                        raw=part.model_dump(),
                    ))
            elif isinstance(part, dict):
                if "text" in part:
                    blocks.append(ContentBlock(
                        type=ContentType.TEXT,
                        text=part["text"],
                        raw=part,
                    ))
                elif "inlineData" in part or "fileData" in part:
                    blocks.append(ContentBlock(
                        type=ContentType.FILE,
                        raw=part,
                    ))

        return blocks
