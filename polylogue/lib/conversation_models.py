"""Conversation and summary domain models."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.lib.branch_type import BranchType
from polylogue.lib.message_models import DialoguePair, Message
from polylogue.lib.messages import MessageCollection
from polylogue.types import ConversationId, Provider

if TYPE_CHECKING:
    from polylogue.lib.projections import ConversationProjection


class ConversationSummary(BaseModel):
    """Lightweight conversation metadata without messages."""

    id: ConversationId
    provider: Provider
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    parent_id: ConversationId | None = None
    branch_type: BranchType | None = None
    message_count: int | None = None
    dialogue_count: int | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")

    @property
    def display_date(self) -> datetime | None:
        return self.updated_at or self.created_at

    @property
    def display_title(self) -> str:
        user_title = self.metadata.get("title")
        if user_title:
            return str(user_title)
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def tags(self) -> list[str]:
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        return []

    @property
    def summary(self) -> str | None:
        summary = self.metadata.get("summary")
        return str(summary) if summary is not None else None

    @property
    def is_continuation(self) -> bool:
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class Conversation(BaseModel):
    """Conversation with eagerly or lazily materialized message collection."""

    id: ConversationId
    provider: Provider
    title: str | None = None
    messages: MessageCollection
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    parent_id: ConversationId | None = None
    branch_type: BranchType | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def display_date(self) -> datetime | None:
        return self.updated_at or self.created_at

    @property
    def is_continuation(self) -> bool:
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def user_title(self) -> str | None:
        title = self.metadata.get("title")
        return str(title) if title is not None else None

    @property
    def display_title(self) -> str:
        if self.user_title:
            return self.user_title
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def summary(self) -> str | None:
        summary = self.metadata.get("summary")
        return str(summary) if summary is not None else None

    @property
    def tags(self) -> list[str]:
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        return []

    def filter(self, predicate: Callable[[Message], bool]) -> Conversation:
        filtered = [message for message in self.messages if predicate(message)]
        return self.model_copy(update={"messages": MessageCollection(messages=filtered)})

    def user_only(self) -> Conversation:
        return self.filter(lambda message: message.is_user)

    def assistant_only(self) -> Conversation:
        return self.filter(lambda message: message.is_assistant)

    def dialogue_only(self) -> Conversation:
        return self.filter(lambda message: message.is_dialogue)

    def without_noise(self) -> Conversation:
        return self.filter(lambda message: not message.is_noise)

    def substantive_only(self) -> Conversation:
        return self.filter(lambda message: message.is_substantive)

    def mainline_messages(self) -> list[Message]:
        return [message for message in self.messages if message.branch_index == 0]

    def iter_dialogue(self) -> Iterator[Message]:
        for message in self.messages:
            if message.is_dialogue:
                yield message

    def iter_substantive(self) -> Iterator[Message]:
        for message in self.messages:
            if message.is_substantive:
                yield message

    def iter_pairs(self) -> Iterator[DialoguePair]:
        substantive = [message for message in self.messages if message.is_substantive]
        index = 0
        while index < len(substantive) - 1:
            if substantive[index].is_user and substantive[index + 1].is_assistant:
                yield DialoguePair(user=substantive[index], assistant=substantive[index + 1])
                index += 2
            else:
                index += 1

    def iter_thinking(self) -> Iterator[str]:
        for message in self.messages:
            if message.is_thinking:
                thinking = message.extract_thinking()
                if thinking:
                    yield thinking

    def iter_branches(self) -> Iterator[tuple[str, list[Message]]]:
        from collections import defaultdict

        by_parent: dict[str, list[Message]] = defaultdict(list)
        for message in self.messages:
            if message.parent_id:
                by_parent[message.parent_id].append(message)

        for parent_id, children in by_parent.items():
            if len(children) > 1:
                yield parent_id, sorted(children, key=lambda message: message.branch_index)

    def to_text(self, include_role: bool = True) -> str:
        lines = []
        for message in self.messages:
            if not message.text:
                continue
            lines.append(f"{message.role}: {message.text}" if include_role else message.text)
        return "\n\n".join(lines)

    def to_clean_text(self) -> str:
        return self.substantive_only().to_text()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        return sum(1 for message in self.messages if message.is_user)

    @property
    def assistant_message_count(self) -> int:
        return sum(1 for message in self.messages if message.is_assistant)

    @property
    def word_count(self) -> int:
        return sum(message.word_count for message in self.messages)

    @property
    def total_cost_usd(self) -> float:
        from polylogue.lib.model_support import _coerce_optional_float

        message_total = sum(message.cost_usd or 0.0 for message in self.messages)
        if message_total > 0.0:
            return message_total
        if not self.provider_meta:
            return 0.0
        return _coerce_optional_float(self.provider_meta.get("total_cost_usd")) or 0.0

    @property
    def total_duration_ms(self) -> int:
        from polylogue.lib.model_support import _coerce_optional_int

        message_total = sum(message.duration_ms or 0 for message in self.messages)
        if message_total > 0:
            return message_total
        if not self.provider_meta:
            return 0
        return _coerce_optional_int(self.provider_meta.get("total_duration_ms")) or 0

    def project(self) -> ConversationProjection:
        from polylogue.lib.projections import ConversationProjection

        return ConversationProjection(self)


__all__ = ["Conversation", "ConversationSummary"]
