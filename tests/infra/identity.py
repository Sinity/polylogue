"""Canonical archive identity constructors for hand-built test fixtures."""

from __future__ import annotations

from polylogue.core.identity_law import block_id as _block_id
from polylogue.core.identity_law import message_id as _message_id


def archive_message_id(
    session_id: str,
    native_id: str | None,
    *,
    position: int,
    variant_index: int = 0,
) -> str:
    """Construct the generated message id used by the archive schema."""
    return _message_id(session_id, native_id, position=position, variant_index=variant_index)


def archive_block_id(message_id: str, *, position: int) -> str:
    """Construct the generated block id used by the archive schema."""
    return _block_id(message_id, position=position)


__all__ = ["archive_block_id", "archive_message_id"]
