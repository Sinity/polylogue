"""Attachment domain models."""

from __future__ import annotations

from pydantic import BaseModel

from polylogue.archive.attachment.availability import AttachmentAvailability


class Attachment(BaseModel):
    id: str
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    source_url: str | None = None
    caption: str | None = None
    upload_origin: str | None = None
    direction: str = "user_input"
    producer_ref: str | None = None
    availability: AttachmentAvailability | None = None


__all__ = ["Attachment"]
