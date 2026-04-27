"""Runtime behavior helpers for ``Conversation`` models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Self, cast

from polylogue.lib.conversation.branch_type import BranchType
from polylogue.lib.message.messages import MessageCollection
from polylogue.lib.message.models import DialoguePair, Message
from polylogue.lib.message.roles import normalize_message_roles
from polylogue.lib.roles import Role
from polylogue.lib.tail_overlay import TailOverlayInfo, tail_overlay_from_provider_meta
from polylogue.types import ConversationId

if TYPE_CHECKING:
    from polylogue.lib.conversation.models import Conversation
    from polylogue.lib.projection.projections import ConversationProjection
    from polylogue.lib.semantic.content_projection import ContentProjectionSpec


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _provider_meta_string(provider_meta: dict[str, object] | None, key: str) -> str | None:
    if provider_meta is None:
        return None
    value = provider_meta.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_tags(metadata: dict[str, object]) -> list[str]:
    raw_tags = metadata.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags]


class ConversationRuntimeMixin:
    id: ConversationId
    title: str | None
    messages: MessageCollection
    created_at: datetime | None
    updated_at: datetime | None
    provider_meta: dict[str, object] | None
    metadata: dict[str, object]
    parent_id: ConversationId | None
    branch_type: BranchType | None

    if TYPE_CHECKING:

        def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self: ...

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
        return _metadata_string(self.metadata, "title")

    @property
    def display_title(self) -> str:
        user_title = self.user_title
        if user_title:
            return user_title
        provider_label = _provider_meta_string(self.provider_meta, "display_label")
        if provider_label:
            return provider_label
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def summary(self) -> str | None:
        return _metadata_string(self.metadata, "summary")

    @property
    def tags(self) -> list[str]:
        return _metadata_tags(self.metadata)

    @property
    def tail_overlay(self) -> TailOverlayInfo | None:
        return tail_overlay_from_provider_meta(self.provider_meta)

    def filter(self, predicate: Callable[[Message], bool]) -> Self:
        filtered_messages = [message for message in self.messages if predicate(message)]
        return self.model_copy(update={"messages": MessageCollection(messages=filtered_messages)})

    def with_roles(self, roles: object) -> Self:
        selected_roles = normalize_message_roles(roles)
        return self.filter(lambda message: message.role in selected_roles)

    def with_content_projection(
        self,
        projection: ContentProjectionSpec | Mapping[str, object] | None,
    ) -> Self:
        from polylogue.lib.semantic.content_projection import project_conversation_content

        return cast(Self, project_conversation_content(cast("Conversation", self), projection))

    def dialogue_only(self) -> Self:
        return self.with_roles((Role.USER, Role.ASSISTANT))

    def without_noise(self) -> Self:
        return self.filter(lambda message: not message.is_noise)

    def substantive_only(self) -> Self:
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
        substantive_messages = [message for message in self.messages if message.is_substantive]
        index = 0
        while index < len(substantive_messages) - 1:
            current = substantive_messages[index]
            next_message = substantive_messages[index + 1]
            if current.is_user and next_message.is_assistant:
                yield DialoguePair(user=current, assistant=next_message)
                index += 2
            else:
                index += 1

    def iter_thinking(self) -> Iterator[str]:
        for message in self.messages:
            if not message.is_thinking:
                continue
            thinking = message.extract_thinking()
            if thinking:
                yield thinking

    def iter_branches(self) -> Iterator[tuple[str, list[Message]]]:
        by_parent: dict[str, list[Message]] = defaultdict(list)
        for message in self.messages:
            if message.parent_id:
                by_parent[message.parent_id].append(message)

        for parent_id, children in by_parent.items():
            if len(children) > 1:
                yield parent_id, sorted(children, key=lambda message: message.branch_index)

    def to_text(self, include_role: bool = True) -> str:
        lines: list[str] = []
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
        from polylogue.lib.message.model_runtime import _coerce_optional_float

        message_total = sum((message.cost_usd or 0.0) for message in self.messages)
        if message_total > 0.0:
            return message_total
        if self.provider_meta is None:
            return 0.0
        return _coerce_optional_float(self.provider_meta.get("total_cost_usd")) or 0.0

    @property
    def total_duration_ms(self) -> int:
        from polylogue.lib.message.model_runtime import _coerce_optional_int

        message_total = sum((message.duration_ms or 0) for message in self.messages)
        if message_total > 0:
            return message_total
        if self.provider_meta is None:
            return 0
        return _coerce_optional_int(self.provider_meta.get("total_duration_ms")) or 0

    def project(self) -> ConversationProjection:
        from polylogue.lib.conversation.models import Conversation
        from polylogue.lib.projection.projections import ConversationProjection

        if not isinstance(self, Conversation):
            raise TypeError(f"projection requires Conversation, got {type(self).__name__}")
        return ConversationProjection(self)


__all__ = ["ConversationRuntimeMixin"]
