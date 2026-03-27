"""Filter mixin for conversation projections."""

from __future__ import annotations

from datetime import datetime
import re


class ProjectionFilterMixin:
    def where(self, predicate):
        self._filters.append(predicate)
        return self

    def user_messages(self):
        return self.where(lambda m: m.is_user)

    def assistant_messages(self):
        return self.where(lambda m: m.is_assistant)

    def dialogue(self):
        return self.where(lambda m: m.is_dialogue)

    def substantive(self):
        return self.where(lambda m: m.is_substantive)

    def without_noise(self):
        return self.where(lambda m: not m.is_noise)

    def with_attachments(self):
        return self.where(lambda m: len(m.attachments) > 0)

    def min_words(self, n: int):
        return self.where(lambda m: m.word_count >= n)

    def max_words(self, n: int):
        return self.where(lambda m: m.word_count <= n)

    def contains(self, text: str, case_sensitive: bool = False):
        if case_sensitive:
            return self.where(lambda m: m.text is not None and text in m.text)
        text_lower = text.lower()
        return self.where(lambda m: m.text is not None and text_lower in m.text.lower())

    def matches(self, pattern: str):
        compiled = re.compile(pattern)
        return self.where(lambda m: m.text is not None and compiled.search(m.text) is not None)

    def since(self, timestamp: datetime):
        return self.where(lambda m: m.timestamp is not None and m.timestamp >= timestamp)

    def until(self, timestamp: datetime):
        return self.where(lambda m: m.timestamp is not None and m.timestamp <= timestamp)

    def between(self, start: datetime, end: datetime):
        return self.since(start).until(end)

    def thinking_only(self):
        return self.where(lambda m: m.is_thinking)

    def tool_use_only(self):
        return self.where(lambda m: m.is_tool_use)


__all__ = ["ProjectionFilterMixin"]
