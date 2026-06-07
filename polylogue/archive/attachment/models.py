"""Attachment domain models."""

from __future__ import annotations

from pydantic import BaseModel


class Attachment(BaseModel):
    id: str
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    source_url: str | None = None
    caption: str | None = None


__all__ = ["Attachment"]
