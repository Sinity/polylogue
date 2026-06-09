"""Shell-completion aggregates exposed through the archive facade.

Surface-side completion callbacks (CLI repo/cwd/tool name completions)
historically reached into ``session_profiles`` / ``actions`` via
raw SQL. These typed aggregates own the SQL inside the operation
boundary so the surface stays a leaf adapter (#860).
"""

from __future__ import annotations


class CompletionAggregate:
    """Single (value, count) pair for shell-completion aggregations."""

    __slots__ = ("value", "count")

    def __init__(self, value: str, count: int) -> None:
        self.value = value
        self.count = count

    def __repr__(self) -> str:
        return f"CompletionAggregate(value={self.value!r}, count={self.count})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionAggregate):
            return NotImplemented
        return self.value == other.value and self.count == other.count

    def __hash__(self) -> int:
        return hash((self.value, self.count))


__all__ = ["CompletionAggregate"]
