"""Provider-level event domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from polylogue.core.json import json_document
from polylogue.core.timestamps import parse_timestamp
from polylogue.types import MessageId, Provider, ProviderEventId, SessionId


class ProviderEvent(BaseModel):
    """Semantic provider artifact that is not itself a dialogue message."""

    id: ProviderEventId
    session_id: SessionId
    provider: Provider
    event_index: int
    event_type: str
    timestamp: datetime | None = None
    sort_key: float | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    source_message_id: MessageId | None = None
    raw_id: str | None = None
    materializer_version: int = 1

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> Provider:
        if isinstance(value, Provider):
            return value
        return Provider.from_string(str(value) if value is not None else None)

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_timestamp(cls, value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return parse_timestamp(str(value))

    @field_validator("payload", mode="before")
    @classmethod
    def coerce_payload(cls, value: object) -> dict[str, object]:
        if value is None:
            return {}
        return dict(json_document(value))


__all__ = ["ProviderEvent"]
