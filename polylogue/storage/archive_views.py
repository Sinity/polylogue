"""Archive-facing projections and lightweight identity views."""

from __future__ import annotations

from pydantic import BaseModel


class ExistingSession(BaseModel):
    session_id: str
    content_hash: str


__all__ = ["ExistingSession"]
