"""Attachment and content block queries."""

from __future__ import annotations

from polylogue.storage.backends.queries.attachment_content_blocks import (
    get_content_blocks,
    save_content_blocks,
)
from polylogue.storage.backends.queries.attachment_mutations import (
    prune_attachments,
    save_attachments,
)
from polylogue.storage.backends.queries.attachment_records import (
    get_attachments,
    get_attachments_batch,
)

__all__ = [
    "get_content_blocks",
    "save_content_blocks",
    "get_attachments",
    "get_attachments_batch",
    "save_attachments",
    "prune_attachments",
]
