"""Typed query-spec construction/application errors."""

from __future__ import annotations


class QuerySpecError(ValueError):
    """Typed query-spec construction/application error."""

    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"invalid {field}: {value}")
        self.field = field
        self.value = value


__all__ = ["QuerySpecError"]
