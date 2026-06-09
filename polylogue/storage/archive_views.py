"""Archive-facing projections and lightweight identity views."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from polylogue.storage.runtime import AttachmentRecord, MessageRecord, SessionRecord


class ExistingSession(BaseModel):
    session_id: str
    content_hash: str


@dataclass(frozen=True)
class SessionRenderProjection:
    """Repository-owned render projection preserving raw attachment layout."""

    session: SessionRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


__all__ = ["SessionRenderProjection", "ExistingSession"]
