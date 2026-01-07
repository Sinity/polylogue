from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from .store import AttachmentRecord, ConversationRecord, MessageRecord, store_records


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


def ingest_bundle(bundle: IngestBundle, *, conn: sqlite3.Connection | None = None) -> IngestResult:
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
