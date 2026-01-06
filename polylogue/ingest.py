from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import List, Optional

from .store import AttachmentRecord, ConversationRecord, MessageRecord, store_records


@dataclass
class IngestBundle:
    conversation: ConversationRecord
    messages: List[MessageRecord]
    attachments: List[AttachmentRecord]


@dataclass
class IngestResult:
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


def ingest_bundle(bundle: IngestBundle, *, conn: Optional[sqlite3.Connection] = None) -> IngestResult:
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
