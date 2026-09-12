"""Typed outcomes for attachment ownership evidence."""

from __future__ import annotations

from enum import StrEnum


class AttachmentOwnerResolutionReason(StrEnum):
    """Why a parsed attachment could not receive an owner reference."""

    OWNER_AMBIGUOUS = "owner_ambiguous"
    PROVIDER_NEVER_LINKED = "provider_never_linked"
    SOURCE_OMITTED = "source_omitted"
    MESSAGE_MISSING = "message_missing"


__all__ = ["AttachmentOwnerResolutionReason"]
