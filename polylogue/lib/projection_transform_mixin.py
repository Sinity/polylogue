"""Transform/order mixin for conversation projections."""

from __future__ import annotations


class ProjectionTransformMixin:
    def transform(self, fn):
        self._transforms.append(fn)
        return self

    def strip_attachments(self):
        return self.transform(lambda m: m.model_copy(update={"attachments": []}))

    def truncate_text(self, max_chars: int, suffix: str = "..."):
        def truncate(m):
            if m.text and len(m.text) > max_chars:
                return m.model_copy(update={"text": m.text[:max_chars] + suffix})
            return m

        return self.transform(truncate)

    def strip_tools(self):
        return self.where(lambda m: not m.is_tool_use)

    def strip_thinking(self):
        return self.where(lambda m: not m.is_thinking)

    def strip_all(self):
        return self.strip_tools().strip_thinking()

    def limit(self, n: int):
        self._limit = n
        return self

    def offset(self, n: int):
        self._offset = n
        return self

    def reverse(self):
        self._reverse = not self._reverse
        return self

    def first_n(self, n: int):
        return self.offset(0).limit(n)

    def last_n(self, n: int):
        return self.reverse().limit(n)


__all__ = ["ProjectionTransformMixin"]
