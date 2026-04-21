"""Shared message-role filtering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

from polylogue.lib.roles import Role

MessageRoleFilter: TypeAlias = tuple[Role, ...]

ROLE_SQL_VALUES: dict[Role, tuple[str, ...]] = {
    Role.USER: ("user", "human"),
    Role.ASSISTANT: ("assistant", "model", "ai"),
    Role.SYSTEM: ("system",),
    Role.TOOL: ("tool", "function", "tool_use", "tool_result", "progress", "result"),
    Role.UNKNOWN: ("unknown",),
}


def _split_role_tokens(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _iter_role_values(value: object) -> Iterable[object]:
    if value is None:
        return ()
    if isinstance(value, Role):
        return (value,)
    if isinstance(value, str):
        return _split_role_tokens(value)
    if isinstance(value, Iterable):
        return value
    return (value,)


def normalize_message_role_token(value: object) -> Role:
    """Normalize one user-supplied message role token."""
    if isinstance(value, Role):
        return value
    token = str(value).strip()
    role = Role.normalize(token)
    if role == Role.UNKNOWN and token.lower() != Role.UNKNOWN.value:
        valid = ", ".join(role.value for role in Role)
        msg = f"Unknown message role {token!r}. Valid roles: {valid}"
        raise ValueError(msg)
    return role


def normalize_message_roles(value: object) -> MessageRoleFilter:
    """Normalize repeated or comma-separated role values to canonical roles."""
    roles: list[Role] = []
    for raw in _iter_role_values(value):
        if isinstance(raw, str):
            candidates: Sequence[object] = _split_role_tokens(raw)
        else:
            candidates = (raw,)
        for candidate in candidates:
            role = normalize_message_role_token(candidate)
            if role not in roles:
                roles.append(role)
    return tuple(roles)


def message_role_labels(roles: Sequence[Role]) -> tuple[str, ...]:
    return tuple(role.value for role in roles)


def message_role_count_key(role: Role) -> str:
    return f"role_{role.value}_messages"


def message_role_sql_values(roles: Sequence[Role]) -> tuple[str, ...]:
    values: list[str] = []
    for role in roles:
        values.extend(ROLE_SQL_VALUES[role])
    return tuple(dict.fromkeys(values))


__all__ = [
    "MessageRoleFilter",
    "message_role_count_key",
    "message_role_labels",
    "message_role_sql_values",
    "normalize_message_role_token",
    "normalize_message_roles",
]
