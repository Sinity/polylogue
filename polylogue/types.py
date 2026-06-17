"""Semantic ID types for polylogue."""

from __future__ import annotations

from typing import NewType

SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)
SessionEventId = NewType("SessionEventId", str)


__all__ = [
    "AttachmentId",
    "ContentHash",
    "MessageId",
    "SessionId",
    "SessionEventId",
]
