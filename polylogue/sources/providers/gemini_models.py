"""Gemini support models shared by message extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GeminiBranchParent(BaseModel):
    """Parent reference for branched conversations."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    index: int | None = None


class GeminiGrounding(BaseModel):
    """Grounding/search results from Gemini."""

    model_config = ConfigDict(extra="allow")

    # Structure varies.


class GeminiThoughtSignature(BaseModel):
    """Signature for thinking blocks."""

    model_config = ConfigDict(extra="allow")

    # Structure varies.


class GeminiPart(BaseModel):
    """Content part within a Gemini message."""

    model_config = ConfigDict(extra="allow")

    text: str | None = None
    # Can also contain inlineData, fileData, etc.
