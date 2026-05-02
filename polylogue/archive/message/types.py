"""Canonical message content-type labels."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias


class MessageType(str, Enum):
    """Normalized message type for filtering and read surfaces."""

    MESSAGE = "message"
    SUMMARY = "summary"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"

    @classmethod
    def normalize(cls, value: object) -> MessageType:
        """Coerce provider/parser message-type values to a canonical type.

        This is intentionally forgiving for provider data: unknown source
        values become normal messages instead of breaking ingestion.
        User/API query filters must use ``validate_filter_token``.
        """
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            return cls.MESSAGE
        for item in cls:
            if item.value == candidate:
                return item
        return cls.MESSAGE

    @classmethod
    def validate_filter_token(cls, value: object) -> MessageType:
        """Validate one user-supplied message-type filter token."""
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        for item in cls:
            if item.value == candidate:
                return item
        valid = ", ".join(item.value for item in cls)
        msg = f"Unknown message type {str(value)!r}. Valid message types: {valid}"
        raise ValueError(msg)


MessageTypeFilter: TypeAlias = MessageType | str | tuple[MessageType | str, ...] | list[MessageType | str]


def normalize_message_types(values: MessageTypeFilter | None) -> tuple[MessageType, ...]:
    """Coerce provider/parser message-type values to canonical types."""
    if values is None:
        return ()
    if isinstance(values, (MessageType, str)):
        raw_values: tuple[MessageType | str, ...] = (values,)
    else:
        raw_values = tuple(values)
    return tuple(MessageType.normalize(value) for value in raw_values)


def validate_message_type_filter(value: object) -> MessageType:
    """Validate one user/API/MCP-supplied message-type filter token."""
    return MessageType.validate_filter_token(value)


def message_type_sql_values(values: MessageTypeFilter | None) -> tuple[str, ...]:
    return tuple(message_type.value for message_type in normalize_message_types(values))


__all__ = [
    "MessageType",
    "MessageTypeFilter",
    "message_type_sql_values",
    "normalize_message_types",
    "validate_message_type_filter",
]
