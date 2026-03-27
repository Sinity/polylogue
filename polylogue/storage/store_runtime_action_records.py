"""Action-event storage record models."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from polylogue.storage.store_constants import ACTION_EVENT_MATERIALIZER_VERSION
from polylogue.types import ConversationId, MessageId


class ActionEventRecord(BaseModel):
    event_id: str
    conversation_id: ConversationId
    message_id: MessageId
    materializer_version: int = ACTION_EVENT_MATERIALIZER_VERSION
    source_block_id: str | None = None
    timestamp: str | None = None
    sort_key: float | None = None
    sequence_index: int
    provider_name: str | None = None
    action_kind: str
    tool_name: str | None = None
    normalized_tool_name: str
    tool_id: str | None = None
    affected_paths: tuple[str, ...] = ()
    cwd_path: str | None = None
    branch_names: tuple[str, ...] = ()
    command: str | None = None
    query_text: str | None = None
    url: str | None = None
    output_text: str | None = None
    search_text: str

    @field_validator(
        "event_id",
        "conversation_id",
        "message_id",
        "action_kind",
        "normalized_tool_name",
        "search_text",
    )
    @classmethod
    def action_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
