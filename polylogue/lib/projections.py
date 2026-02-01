"""Fluent projection builder for conversation filtering and transformation.

Provides lazy-evaluated, composable projections over conversations.

Example usage:
    # Fluent composition
    result = conv.project()
        .substantive()
        .min_words(50)
        .since(datetime(2024, 1, 1))
        .limit(10)
        .execute()

    # Lazy iteration (memory efficient)
    for msg in conv.project().user_messages().contains("error").iter():
        print(msg.text)

    # Count without materialization
    error_count = conv.project().contains("error").count()
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, Message


class ConversationProjection:
    """Lazy-evaluated projection builder for conversations.

    Filters and transforms are accumulated and only applied when
    a terminal operation (execute, iter, count, etc.) is called.
    """

    def __init__(self, conversation: Conversation) -> None:
        self._conv = conversation
        self._filters: list[Callable[[Message], bool]] = []
        self._transforms: list[Callable[[Message], Message]] = []
        self._limit: int | None = None
        self._offset: int = 0
        self._reverse: bool = False

    # --- Filter methods (lazy, return self for chaining) ---

    def where(self, predicate: Callable[[Message], bool]) -> ConversationProjection:
        """Add a custom filter condition."""
        self._filters.append(predicate)
        return self

    def user_messages(self) -> ConversationProjection:
        """Filter to user messages only."""
        return self.where(lambda m: m.is_user)

    def assistant_messages(self) -> ConversationProjection:
        """Filter to assistant messages only."""
        return self.where(lambda m: m.is_assistant)

    def dialogue(self) -> ConversationProjection:
        """Filter to dialogue (user + assistant) messages."""
        return self.where(lambda m: m.is_dialogue)

    def substantive(self) -> ConversationProjection:
        """Filter to substantive messages (dialogue, not noise, meaningful length)."""
        return self.where(lambda m: m.is_substantive)

    def without_noise(self) -> ConversationProjection:
        """Exclude noise messages (tool use, context dumps, system)."""
        return self.where(lambda m: not m.is_noise)

    def with_attachments(self) -> ConversationProjection:
        """Filter to messages that have attachments."""
        return self.where(lambda m: len(m.attachments) > 0)

    def min_words(self, n: int) -> ConversationProjection:
        """Filter to messages with at least n words."""
        return self.where(lambda m: m.word_count >= n)

    def max_words(self, n: int) -> ConversationProjection:
        """Filter to messages with at most n words."""
        return self.where(lambda m: m.word_count <= n)

    def contains(self, text: str, case_sensitive: bool = False) -> ConversationProjection:
        """Filter to messages containing the given text."""
        if case_sensitive:
            return self.where(lambda m: m.text is not None and text in m.text)
        text_lower = text.lower()
        return self.where(lambda m: m.text is not None and text_lower in m.text.lower())

    def matches(self, pattern: str) -> ConversationProjection:
        """Filter to messages matching a regex pattern."""
        import re
        compiled = re.compile(pattern)
        return self.where(lambda m: m.text is not None and compiled.search(m.text) is not None)

    def since(self, timestamp: datetime) -> ConversationProjection:
        """Filter to messages after the given timestamp."""
        return self.where(lambda m: m.timestamp is not None and m.timestamp >= timestamp)

    def until(self, timestamp: datetime) -> ConversationProjection:
        """Filter to messages before the given timestamp."""
        return self.where(lambda m: m.timestamp is not None and m.timestamp <= timestamp)

    def between(self, start: datetime, end: datetime) -> ConversationProjection:
        """Filter to messages within a time range."""
        return self.since(start).until(end)

    def thinking_only(self) -> ConversationProjection:
        """Filter to messages that contain thinking/reasoning traces."""
        return self.where(lambda m: m.is_thinking)

    def tool_use_only(self) -> ConversationProjection:
        """Filter to messages that are tool use."""
        return self.where(lambda m: m.is_tool_use)

    # --- Transform methods (lazy, return self for chaining) ---

    def transform(self, fn: Callable[[Message], Message]) -> ConversationProjection:
        """Add a custom message transformation."""
        self._transforms.append(fn)
        return self

    def strip_attachments(self) -> ConversationProjection:
        """Transform: remove attachments from messages."""
        return self.transform(lambda m: m.model_copy(update={"attachments": []}))

    def truncate_text(self, max_chars: int, suffix: str = "...") -> ConversationProjection:
        """Transform: truncate message text to max_chars."""
        def truncate(m: Message) -> Message:
            if m.text and len(m.text) > max_chars:
                return m.model_copy(update={"text": m.text[:max_chars] + suffix})
            return m
        return self.transform(truncate)

    def strip_tools(self) -> ConversationProjection:
        """Filter: exclude tool use messages entirely."""
        return self.where(lambda m: not m.is_tool_use)

    def strip_thinking(self) -> ConversationProjection:
        """Filter: exclude thinking/reasoning trace messages entirely."""
        return self.where(lambda m: not m.is_thinking)

    def strip_all(self) -> ConversationProjection:
        """Filter: exclude both tool use and thinking messages."""
        return self.strip_tools().strip_thinking()

    # --- Ordering and pagination ---

    def limit(self, n: int) -> ConversationProjection:
        """Limit results to n messages."""
        self._limit = n
        return self

    def offset(self, n: int) -> ConversationProjection:
        """Skip first n messages."""
        self._offset = n
        return self

    def reverse(self) -> ConversationProjection:
        """Reverse message order."""
        self._reverse = not self._reverse
        return self

    def first_n(self, n: int) -> ConversationProjection:
        """Get first n messages (convenience for offset(0).limit(n))."""
        return self.offset(0).limit(n)

    def last_n(self, n: int) -> ConversationProjection:
        """Get last n messages."""
        return self.reverse().limit(n)

    # --- Terminal operations (eager, execute the projection) ---

    def execute(self) -> Conversation:
        """Execute projection and return a new Conversation with filtered/transformed messages."""
        messages = list(self.iter())
        return self._conv.model_copy(update={"messages": messages})

    def iter(self) -> Iterator[Message]:
        """Lazily iterate over filtered/transformed messages.

        Memory-efficient for large conversations as messages are
        processed one at a time without materializing the full list.
        """
        messages: Iterator[Message] = iter(self._conv.messages)

        # Apply reverse if needed
        if self._reverse:
            messages = iter(reversed(list(messages)))

        # Skip offset
        messages = itertools.islice(messages, self._offset, None)

        # Track limit
        remaining = self._limit

        # Early exit if limit is zero
        if remaining is not None and remaining <= 0:
            return

        for msg in messages:
            # Check all filters
            if not all(f(msg) for f in self._filters):
                continue

            # Apply transforms
            for transform_fn in self._transforms:
                msg = transform_fn(msg)

            yield msg

            # Respect limit
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    def count(self) -> int:
        """Count matching messages without fully materializing."""
        return sum(1 for _ in self.iter())

    def first(self) -> Message | None:
        """Get first matching message, or None if no matches."""
        for msg in self.iter():
            return msg
        return None

    def last(self) -> Message | None:
        """Get last matching message, or None if no matches."""
        result = None
        for msg in self.iter():
            result = msg
        return result

    def to_list(self) -> list[Message]:
        """Materialize projection as a list of messages."""
        return list(self.iter())

    def exists(self) -> bool:
        """Check if any messages match the projection."""
        return self.first() is not None

    def to_text(self, include_role: bool = True, separator: str = "\n\n") -> str:
        """Render matching messages as text."""
        parts = []
        for msg in self.iter():
            if include_role and msg.role:
                parts.append(f"{msg.role}: {msg.text or ''}")
            else:
                parts.append(msg.text or "")
        return separator.join(parts)
