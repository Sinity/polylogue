"""Canonical archive identity constructors for hand-built test fixtures."""

from __future__ import annotations

from hashlib import sha256

from polylogue.core.identity_law import block_id as _block_id
from polylogue.core.identity_law import message_id as _message_id
from polylogue.core.text_identity import nfc
from polylogue.pipeline.ids import MESSAGE_CONTENT_IDENTITY_HEX_CHARS


def archive_message_id(
    session_id: str,
    native_id: str | None,
    *,
    content_identity: str | None = None,
    content_occurrence: int = 0,
) -> str:
    """Construct the generated message id used by the archive schema."""
    return _message_id(
        session_id,
        native_id,
        content_identity=content_identity,
        content_occurrence=content_occurrence,
    )


def fixture_content_identity(*parts: str) -> str:
    """Return a content-derived identity digest for a hand-built message row.

    Production derives ``messages.content_identity`` from a ``ParsedMessage``'s
    declared semantic fields (``pipeline.ids.message_content_identity``). A
    fixture that writes SQL directly has no ``ParsedMessage``, so it states the
    same thing in the only form available to it: a digest over the semantic
    text it is modelling, truncated to the same width the production digest
    uses. Two fixture rows with the same declared content therefore share a
    digest -- and must separate themselves with ``content_occurrence``, exactly
    as production does.
    """
    payload = "\u0000".join(nfc(part) for part in parts).encode("utf-8")
    return sha256(payload).hexdigest()[:MESSAGE_CONTENT_IDENTITY_HEX_CHARS]


def archive_block_id(message_id: str, *, position: int) -> str:
    """Construct the generated block id used by the archive schema."""
    return _block_id(message_id, position=position)


__all__ = ["archive_block_id", "archive_message_id", "fixture_content_identity"]
