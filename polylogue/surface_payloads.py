"""Shared surface payload models for CLI, MCP, and presentation adapters."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from polylogue.lib.json import JSONDocument, JSONValue, require_json_document

if TYPE_CHECKING:
    from collections.abc import Container

    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.lib.search_hits import ConversationSearchHit


def serialize_surface_payload(payload: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialize a surface payload model with stable JSON formatting."""
    return payload.model_dump_json(indent=2, exclude_none=exclude_none)


def model_json_document(payload: BaseModel, *, exclude_none: bool = False) -> JSONDocument:
    """Return a model dump constrained to the shared JSON document type."""
    return require_json_document(
        payload.model_dump(mode="json", exclude_none=exclude_none),
        context=f"{payload.__class__.__name__} JSON payload",
    )


class SurfacePayloadModel(BaseModel):
    """Shared base for immutable JSON payload models exposed by surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return serialize_surface_payload(self, exclude_none=exclude_none)


class MachineErrorEnvelope(TypedDict):
    """Serialized machine-error envelope with sparse optional keys."""

    status: Literal["error"]
    code: str
    message: str
    command: NotRequired[list[str]]
    details: NotRequired[JSONDocument]


class MachineSuccessEnvelope(TypedDict):
    """Serialized machine-success envelope."""

    status: Literal["ok"]
    result: JSONDocument


class MachineErrorPayload(SurfacePayloadModel):
    """Structured error payload for machine-readable CLI surfaces."""

    status: Literal["error"] = "error"
    code: str
    message: str
    command: Sequence[str] = ()
    details: Mapping[str, object] = Field(default_factory=dict)

    def to_dict(self) -> MachineErrorEnvelope:
        payload: MachineErrorEnvelope = {
            "status": self.status,
            "code": self.code,
            "message": self.message,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.details:
            payload["details"] = require_json_document(dict(self.details), context="machine error details")
        return payload

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.to_dict(), indent=2)

    def emit(self, *, exit_code: int = 1) -> None:
        """Write the payload to stdout and exit."""
        sys.stdout.write(self.to_json(exclude_none=True))
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise SystemExit(exit_code)


class MachineSuccessPayload(SurfacePayloadModel):
    """Structured success payload for machine-readable CLI surfaces."""

    status: Literal["ok"] = "ok"
    result: Mapping[str, object] = Field(default_factory=dict)

    def to_dict(self) -> MachineSuccessEnvelope:
        return {
            "status": self.status,
            "result": require_json_document(dict(self.result), context="machine success result"),
        }


def normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if isinstance(role, Enum):
        role = role.value
    return str(role)


class ConversationMessagePayload(SurfacePayloadModel):
    """Machine-readable message payload shared across CLI and MCP surfaces."""

    id: str
    role: str
    text: str
    timestamp: datetime | None = None

    @classmethod
    def from_message(cls, message: Message) -> ConversationMessagePayload:
        return cls(
            id=str(message.id),
            role=normalize_role(message.role),
            text=message.text or "",
            timestamp=message.timestamp,
        )


class ConversationSummaryPayload(SurfacePayloadModel):
    """Compact conversation summary payload used by MCP/search surfaces."""

    id: str
    provider: str
    title: str
    message_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationSummaryPayload:
        return cls(
            id=str(conversation.id),
            provider=str(conversation.provider),
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int | None = None,
    ) -> ConversationSummaryPayload:
        return cls(
            id=str(summary.id),
            provider=str(summary.provider),
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class ConversationDetailPayload(ConversationSummaryPayload):
    """Full conversation detail payload with serialized messages."""

    messages: tuple[ConversationMessagePayload, ...]

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationDetailPayload:
        summary = ConversationSummaryPayload.from_conversation(conversation)
        return cls(
            **summary.model_dump(),
            messages=tuple(ConversationMessagePayload.from_message(msg) for msg in conversation.messages),
        )


class ConversationListRowPayload(SurfacePayloadModel):
    """Conversation row payload used by CLI list, JSON, and YAML surfaces."""

    id: str
    provider: str
    title: str
    date: str | None = None
    messages: int
    tags: tuple[str, ...] = ()
    summary: str | None = None
    words: int | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationListRowPayload:
        return cls(
            id=str(conversation.id),
            provider=str(conversation.provider),
            title=conversation.display_title,
            date=conversation.display_date.isoformat() if conversation.display_date else None,
            messages=len(conversation.messages),
            tags=tuple(conversation.tags),
            summary=conversation.summary,
            words=sum(message.word_count for message in conversation.messages),
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int,
    ) -> ConversationListRowPayload:
        return cls(
            id=str(summary.id),
            provider=str(summary.provider),
            title=summary.display_title,
            date=summary.display_date.isoformat() if summary.display_date else None,
            messages=message_count,
            tags=tuple(summary.tags),
            summary=summary.summary,
        )

    def selected(self, fields: Container[str] | None = None) -> JSONDocument:
        data = model_json_document(self, exclude_none=True)
        if fields is None:
            return data
        return {key: value for key, value in data.items() if key in fields}


class ConversationSearchMatchPayload(SurfacePayloadModel):
    """Evidence explaining why a conversation appeared in search results."""

    rank: int
    retrieval_lane: str
    match_surface: str
    message_id: str | None = None
    snippet: str | None = None
    score: float | None = None


class ConversationSearchHitPayload(SurfacePayloadModel):
    """Search-hit payload with summary identity and match evidence."""

    conversation: ConversationSummaryPayload
    match: ConversationSearchMatchPayload

    @classmethod
    def from_search_hit(
        cls,
        hit: ConversationSearchHit,
        *,
        message_count: int | None = None,
    ) -> ConversationSearchHitPayload:
        return cls(
            conversation=ConversationSummaryPayload.from_summary(
                hit.summary,
                message_count=message_count if message_count is not None else hit.summary.message_count,
            ),
            match=ConversationSearchMatchPayload(
                rank=hit.rank,
                retrieval_lane=hit.retrieval_lane,
                match_surface=hit.match_surface,
                message_id=hit.message_id,
                snippet=hit.snippet,
                score=hit.score,
            ),
        )


__all__ = [
    "ConversationDetailPayload",
    "ConversationListRowPayload",
    "ConversationMessagePayload",
    "ConversationSearchHitPayload",
    "ConversationSearchMatchPayload",
    "ConversationSummaryPayload",
    "MachineErrorPayload",
    "MachineErrorEnvelope",
    "MachineSuccessEnvelope",
    "MachineSuccessPayload",
    "SurfacePayloadModel",
    "JSONDocument",
    "JSONValue",
    "model_json_document",
    "normalize_role",
    "serialize_surface_payload",
]
