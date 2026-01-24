"""Ingest preparation logic."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.content_enrichment import enrich_message_metadata
from polylogue.ingestion import IngestBundle, ParsedConversation, ingest_bundle
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    message_content_hash,
)
from polylogue.pipeline.ids import (
    conversation_id as make_conversation_id,
)
from polylogue.pipeline.ids import (
    message_id as make_message_id,
)
from polylogue.pipeline.models import ExistingConversation
from polylogue.storage.db import connection_context
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import AttachmentId, ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.storage.repository import StorageRepository


def _existing_message_map(conversation_id: str) -> dict[str, str]:
    with connection_context(None) as conn:
        rows = conn.execute(
            """
            SELECT provider_message_id, message_id
            FROM messages
            WHERE conversation_id = ? AND provider_message_id IS NOT NULL
            """,
            (conversation_id,),
        ).fetchall()
    return {str(row["provider_message_id"]): row["message_id"] for row in rows if row["provider_message_id"]}


def prepare_ingest(
    convo: ParsedConversation,
    source_name: str,
    *,
    archive_root: Path,
    conn: sqlite3.Connection | None = None,
    repository: StorageRepository | None = None,
) -> tuple[str, dict[str, int], bool]:
    # Create default repository if none provided
    if repository is None:
        from polylogue.storage.backends.sqlite import create_default_backend
        from polylogue.storage.repository import StorageRepository
        backend = create_default_backend()
        repository = StorageRepository(backend=backend)

    content_hash = conversation_content_hash(convo)

    # Use the passed connection for lookups
    existing = None
    if conn:
        row = conn.execute(
            """
            SELECT conversation_id, content_hash
            FROM conversations
            WHERE provider_name = ? AND provider_conversation_id = ?
            ORDER BY updated_at DESC, rowid DESC
            LIMIT 1
            """,
            (convo.provider_name, convo.provider_conversation_id),
        ).fetchone()
        if row:
            existing = ExistingConversation(conversation_id=row["conversation_id"], content_hash=row["content_hash"])

    if existing:
        cid: ConversationId = ConversationId(existing.conversation_id)
        changed = existing.content_hash != content_hash
    else:
        cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
        changed = False

    conversation_record = ConversationRecord(
        conversation_id=cid,
        provider_name=convo.provider_name,
        provider_conversation_id=convo.provider_conversation_id,
        title=convo.title,
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        content_hash=content_hash,
        provider_meta={"source": source_name},
    )

    messages: list[MessageRecord] = []
    message_ids: dict[str, MessageId] = {}

    # Retrieve existing message ID mapping using the same connection
    existing_message_ids: dict[str, MessageId] = {}
    if conn:
        # Optimization: Reuse conn if available
        # But _existing_message_map uses its own connection context.
        # Use inline query here to reuse `conn`?
        rows = conn.execute(
            """
            SELECT provider_message_id, message_id
            FROM messages
            WHERE conversation_id = ? AND provider_message_id IS NOT NULL
            """,
            (cid,),
        ).fetchall()
        existing_message_ids = {
            str(row["provider_message_id"]): MessageId(row["message_id"]) for row in rows if row["provider_message_id"]
        }

    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid: MessageId = existing_message_ids.get(provider_message_id) or make_message_id(cid, provider_message_id)
        message_hash = message_content_hash(msg, provider_message_id)
        message_ids[str(provider_message_id)] = mid

        # Enrich provider_meta with code language detection
        enriched_meta = enrich_message_metadata(msg.provider_meta)

        messages.append(
            MessageRecord(
                message_id=mid,
                conversation_id=cid,
                provider_message_id=provider_message_id,
                role=msg.role,
                text=msg.text,
                timestamp=msg.timestamp,
                content_hash=message_hash,
                provider_meta=enriched_meta,
            )
        )

    attachments: list[AttachmentRecord] = []
    for att in convo.attachments:
        aid, updated_meta, updated_path = attachment_content_id(
            convo.provider_name, att, archive_root=archive_root
        )
        # Merge updated metadata with provider_id if present
        meta: dict[str, object] = dict(updated_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        message_id_val: MessageId | None = message_ids.get(att.message_provider_id or "") if att.message_provider_id else None
        attachments.append(
            AttachmentRecord(
                attachment_id=AttachmentId(aid),
                conversation_id=cid,
                message_id=message_id_val,
                mime_type=att.mime_type,
                size_bytes=att.size_bytes,
                path=updated_path,
                provider_meta=meta,
            )
        )

    result = ingest_bundle(
        IngestBundle(
            conversation=conversation_record,
            messages=messages,
            attachments=attachments,
        ),
        repository=repository,
    )
    return (
        cid,
        {
            "conversations": result.conversations,
            "messages": result.messages,
            "attachments": result.attachments,
            "skipped_conversations": result.skipped_conversations,
            "skipped_messages": result.skipped_messages,
            "skipped_attachments": result.skipped_attachments,
        },
        changed,
    )
