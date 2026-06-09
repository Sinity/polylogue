"""Canonical message content-type labels."""

from __future__ import annotations

from typing import TypeAlias

from polylogue.core.enums import MessageType

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
