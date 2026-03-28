"""Terminal operations for conversation projections."""

from __future__ import annotations

import itertools


class ProjectionTerminalMixin:
    def execute(self):
        messages = list(self.iter())
        return self._conv.model_copy(update={"messages": messages})

    def iter(self):
        messages = iter(self._conv.messages)
        if self._reverse:
            messages = iter(reversed(list(messages)))
        messages = itertools.islice(messages, self._offset, None)
        remaining = self._limit
        if remaining is not None and remaining <= 0:
            return
        for msg in messages:
            if not all(f(msg) for f in self._filters):
                continue
            for transform_fn in self._transforms:
                msg = transform_fn(msg)
            yield msg
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    def count(self) -> int:
        return sum(1 for _ in self.iter())

    def first(self):
        for msg in self.iter():
            return msg
        return None

    def last(self):
        result = None
        for msg in self.iter():
            result = msg
        return result

    def to_list(self):
        return list(self.iter())

    def exists(self) -> bool:
        return self.first() is not None

    def to_text(self, include_role: bool = True, separator: str = "\n\n") -> str:
        parts = []
        for msg in self.iter():
            if include_role and msg.role:
                parts.append(f"{msg.role}: {msg.text or ''}")
            else:
                parts.append(msg.text or "")
        return separator.join(parts)


__all__ = ["ProjectionTerminalMixin"]
