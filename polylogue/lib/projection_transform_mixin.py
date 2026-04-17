"""Transform/order mixin for conversation projections."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from polylogue.lib.message_models import Message

MessagePredicate = Callable[["Message"], bool]
MessageTransform = Callable[["Message"], "Message"]


class ProjectionTransformMixin:
    _transforms: list[MessageTransform]
    _limit: int | None
    _offset: int
    _reverse: bool

    def where(self, predicate: MessagePredicate) -> Self:
        raise NotImplementedError

    def transform(self, fn: MessageTransform) -> Self:
        self._transforms.append(fn)
        return self

    def strip_attachments(self) -> Self:
        return self.transform(lambda m: m.model_copy(update={"attachments": []}))

    def truncate_text(self, max_chars: int, suffix: str = "...") -> Self:
        def truncate(message: Message) -> Message:
            text = message.text
            if text and len(text) > max_chars:
                return message.model_copy(update={"text": text[:max_chars] + suffix})
            return message

        return self.transform(truncate)

    def strip_tools(self) -> Self:
        return self.where(lambda m: not m.is_tool_use)

    def strip_thinking(self) -> Self:
        return self.where(lambda m: not m.is_thinking)

    def strip_all(self) -> Self:
        return self.strip_tools().strip_thinking()

    def limit(self, n: int) -> Self:
        self._limit = n
        return self

    def offset(self, n: int) -> Self:
        self._offset = n
        return self

    def reverse(self) -> Self:
        self._reverse = not self._reverse
        return self

    def first_n(self, n: int) -> Self:
        return self.offset(0).limit(n)

    def last_n(self, n: int) -> Self:
        return self.reverse().limit(n)


__all__ = ["ProjectionTransformMixin"]
