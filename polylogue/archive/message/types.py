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
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            return cls.MESSAGE
        for item in cls:
            if item.value == candidate:
                return item
        return cls.MESSAGE


MessageTypeFilter: TypeAlias = MessageType | str | tuple[MessageType | str, ...] | list[MessageType | str]


def normalize_message_types(values: MessageTypeFilter | None) -> tuple[MessageType, ...]:
    if values is None:
        return ()
    if isinstance(values, (MessageType, str)):
        raw_values: tuple[MessageType | str, ...] = (values,)
    else:
        raw_values = tuple(values)
    return tuple(MessageType.normalize(value) for value in raw_values)


def message_type_sql_values(values: MessageTypeFilter | None) -> tuple[str, ...]:
    return tuple(message_type.value for message_type in normalize_message_types(values))


__all__ = ["MessageType", "MessageTypeFilter", "message_type_sql_values", "normalize_message_types"]
