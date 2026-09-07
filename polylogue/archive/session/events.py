"""Session timeline event domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from polylogue.core.enums import Origin
from polylogue.core.json import json_document
from polylogue.core.sources import source_name_to_origin
from polylogue.core.timestamps import parse_timestamp
from polylogue.core.types import MessageId, SessionEventId, SessionId


class SessionEvent(BaseModel):
    """Semantic session artifact that is not itself a dialogue message."""

    id: SessionEventId
    session_id: SessionId
    origin: Origin
    event_index: int
    event_type: str
    timestamp: datetime | None = None
    sort_key: float | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    source_message_id: MessageId | None = None
    source_message_provider_id: str | None = None
    raw_id: str | None = None
    materializer_version: int = 1
    # Compaction boundary geometry (polylogue-4ts.5). For a ``compaction``
    # event, the inclusive message-position range the boundary replaced, in the
    # session's own position coordinate, plus the materialized summary message
    # that stands in for it. NULL on every event that replaces nothing, and on a
    # compaction whose parser could not resolve the range.
    boundary_start_position: int | None = None
    boundary_end_position: int | None = None
    boundary_message_id: MessageId | None = None

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, value: object) -> Origin:
        if isinstance(value, Origin):
            return value
        return Origin.from_string(source_name_to_origin(value))

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


__all__ = ["SessionEvent"]
