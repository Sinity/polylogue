"""Deterministic archive identity helpers for the archive rewrite.

These helpers are the executable reference for #1743's generated-ID law.
The archive SQLite schema will compute these IDs with generated columns; Python
uses this module as the test oracle and for pre-DDL contract checks.
"""

from __future__ import annotations


def _required_origin(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("origin cannot be empty")
    if ":" in candidate:
        raise ValueError("origin cannot contain ':'")
    return candidate


def _required_text(field: str, value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{field} cannot be empty")
    return candidate


def _required_non_negative(field: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


def session_id(origin: str, native_id: str) -> str:
    """Return archive ``session_id``: ``origin:native_id``."""
    return f"{_required_origin(origin)}:{_required_text('native_id', native_id)}"


def message_local_id(
    native_id: str | None,
    *,
    position: int,
    variant_index: int = 0,
) -> str:
    """Return the message-local identity component.

    Provider-native message IDs win when present. When the provider omits a
    native ID, archive falls back to ``position.variant_index`` so sibling
    regeneration branches cannot collide.
    """
    if native_id is not None and native_id.strip():
        return _required_text("message native_id", native_id)
    return f"{_required_non_negative('position', position)}.{_required_non_negative('variant_index', variant_index)}"


def message_id(
    parent_session_id: str,
    native_id: str | None,
    *,
    position: int,
    variant_index: int = 0,
) -> str:
    """Return archive ``message_id`` under a session."""
    session = _required_text("session_id", parent_session_id)
    return f"{session}:{message_local_id(native_id, position=position, variant_index=variant_index)}"


def block_id(parent_message_id: str, *, position: int) -> str:
    """Return archive ``block_id`` under a message."""
    message = _required_text("message_id", parent_message_id)
    return f"{message}:{_required_non_negative('position', position)}"


__all__ = [
    "block_id",
    "message_id",
    "message_local_id",
    "session_id",
]
