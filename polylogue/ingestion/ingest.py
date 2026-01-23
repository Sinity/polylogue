from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..storage.store import AttachmentRecord, ConversationRecord, MessageRecord, store_records

if TYPE_CHECKING:
    from ..storage.repository import StorageRepository


class IngestBundle(BaseModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


class IngestResult(BaseModel):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


def ingest_bundle(
    bundle: IngestBundle,
    *,
    conn: sqlite3.Connection | None = None,
    repository: StorageRepository | None = None,
) -> IngestResult:
    """Ingest a conversation bundle into storage.

    Args:
        bundle: The conversation bundle to ingest
        conn: Optional database connection
        repository: Optional storage repository (recommended for thread-safe operations)

    Returns:
        IngestResult with counts of inserted/skipped records

    Note:
        If repository is provided, it will be used for storage operations (recommended).
        Otherwise, falls back to legacy store_records() function.
    """
    if repository:
        counts = repository.save_conversation(
            conversation=bundle.conversation,
            messages=bundle.messages,
            attachments=bundle.attachments,
            conn=conn,
        )
    else:
        counts = store_records(
            conversation=bundle.conversation,
            messages=bundle.messages,
            attachments=bundle.attachments,
            conn=conn,
        )

    return IngestResult(
        conversations=counts["conversations"],
        messages=counts["messages"],
        attachments=counts["attachments"],
        skipped_conversations=counts["skipped_conversations"],
        skipped_messages=counts["skipped_messages"],
        skipped_attachments=counts["skipped_attachments"],
    )


__all__ = ["IngestBundle", "IngestResult", "ingest_bundle"]
