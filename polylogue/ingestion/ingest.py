from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..storage.store import AttachmentRecord, ConversationRecord, MessageRecord

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
    repository: StorageRepository | None = None,
    conn: sqlite3.Connection | None = None,
) -> IngestResult:
    """Ingest a conversation bundle into storage.

    Args:
        bundle: The conversation bundle to ingest
        repository: Storage repository for thread-safe operations (creates default if None)
        conn: Optional database connection (unused - kept for backwards compatibility)

    Returns:
        IngestResult with counts of inserted/skipped records
    """
    # Create default repository if none provided
    if repository is None:
        from ..storage.backends.sqlite import create_default_backend
        from ..storage.repository import StorageRepository
        backend = create_default_backend()
        repository = StorageRepository(backend=backend)

    counts = repository.save_conversation(
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
