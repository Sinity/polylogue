from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    mimeType: Optional[str] = None
    items: Optional[List[Any]] = None
    rows: Optional[List[List[Any]]] = None

    class Config:
        extra = "allow"


class ChunkModel(BaseModel):
    role: str = Field(default="model")
    content: Optional[List[ContentPart]] = None
    text: Optional[str] = None
    token_count: Optional[int] = Field(default=None, alias="tokenCount")
    finish_reason: Optional[str] = Field(default=None, alias="finishReason")
    is_thought: Optional[bool] = Field(default=None, alias="isThought")

    class Config:
        extra = "allow"

    def to_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


def validate_chunks(raw_chunks: List[Any]) -> List[dict]:
    """Validate and normalise chunk payloads using Pydantic."""

    validated: List[dict] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        model = ChunkModel.model_validate(chunk)
        validated.append(model.to_dict())
    return validated
