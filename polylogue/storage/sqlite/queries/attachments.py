"""Attachment and block queries."""

from __future__ import annotations

from polylogue.storage.sqlite.queries.attachment_blocks import (
    get_blocks,
)
from polylogue.storage.sqlite.queries.attachment_mutations import (
    prune_attachments,
)
from polylogue.storage.sqlite.queries.attachment_records import (
    get_attachment_library_page,
    get_attachments,
    get_attachments_batch,
    get_message_attachments,
    search_attachment_identity_evidence_hits,
)

__all__ = [
    "get_blocks",
    "get_attachments",
    "get_attachments_batch",
    "get_message_attachments",
    "get_attachment_library_page",
    "search_attachment_identity_evidence_hits",
    "prune_attachments",
]
