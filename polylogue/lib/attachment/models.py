"""Attachment domain models."""

from __future__ import annotations

from pydantic import BaseModel


class Attachment(BaseModel):
    id: str
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None


__all__ = ["Attachment"]
