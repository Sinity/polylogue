"""Serialization helpers for storage query modules."""

from __future__ import annotations

import hashlib

from polylogue.lib.json import dumps as json_dumps
from polylogue.types import AttachmentId, ConversationId, MessageId


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _json_array_or_none(value: tuple[str, ...] | list[str] | None) -> str | None:
    if not value:
        return None
    return json_dumps(list(value))


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


__all__ = ["_json_array_or_none", "_json_or_none", "_make_ref_id"]
