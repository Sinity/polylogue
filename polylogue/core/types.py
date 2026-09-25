"""Semantic ID types for polylogue."""

from __future__ import annotations

from typing import Literal, NewType

SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)
SessionEventId = NewType("SessionEventId", str)

# ``messages.identity_source`` records which durable message-ID path fired.
# Keep the closed domain vocabulary beside the semantic identity types so
# index DDL and identity-producing code can share one owner.
MessageIdentitySource = Literal["native", "content"]


__all__ = [
    "AttachmentId",
    "ContentHash",
    "MessageId",
    "MessageIdentitySource",
    "SessionId",
    "SessionEventId",
]
