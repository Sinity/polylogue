"""Async parse preparation logic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.pipeline.enrichment import enrich_message_metadata
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
from polylogue.sources.source import RecordBundle, SaveResult, save_bundle
from polylogue.storage.store import AttachmentRecord, ConversationRecord, ExistingConversation, MessageRecord
from polylogue.types import AttachmentId, ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


async def prepare_records(
    convo,
    source_name: str,
    *,
    archive_root: Path,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
    raw_id: str | None = None,
) -> tuple[str, dict[str, int], bool]:
    """Async version of prepare_records for converting ParsedConversation to records.

    Args:
        convo: ParsedConversation to prepare
        source_name: Name of the source
        archive_root: Root directory for archived conversations
        backend: SQLiteBackend for database lookups
        repository: ConversationRepository for saving
        raw_id: Optional raw conversation ID

    Returns:
        Tuple of (conversation_id, result_counts, content_changed)
    """
    # Create default repository if none provided
    if repository is None:
        from polylogue.storage.repository import ConversationRepository
        from polylogue.storage.backends.async_sqlite import SQLiteBackend

        backend = SQLiteBackend()
        repository = ConversationRepository(backend=backend)

    content_hash = conversation_content_hash(convo)

    # Use the passed backend for lookups
    existing = None
    if backend:
        async with backend._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT conversation_id, content_hash
                FROM conversations
                WHERE provider_name = ? AND provider_conversation_id = ?
                ORDER BY updated_at DESC, rowid DESC
                LIMIT 1
                """,
                (convo.provider_name, convo.provider_conversation_id),
            )
            row = await cursor.fetchone()
        if row:
            existing = ExistingConversation(conversation_id=row["conversation_id"], content_hash=row["content_hash"])

    if existing:
        cid: ConversationId = ConversationId(existing.conversation_id)
        changed = existing.content_hash != content_hash
    else:
        cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
        changed = False

    # Resolve parent conversation ID if present (provider ID → internal polylogue ID)
    parent_conversation_id = None
    if convo.parent_conversation_provider_id:
        parent_conversation_id = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)

    # Merge source into provider_meta rather than overwriting
    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

    conversation_record = ConversationRecord(
        conversation_id=cid,
        provider_name=convo.provider_name,
        provider_conversation_id=convo.provider_conversation_id,
        title=convo.title,
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        content_hash=content_hash,
        provider_meta=merged_provider_meta,
        parent_conversation_id=parent_conversation_id,
        branch_type=convo.branch_type,
        raw_id=raw_id,
    )

    messages: list[MessageRecord] = []
    message_ids: dict[str, MessageId] = {}

    # Retrieve existing message ID mapping using the same backend
    existing_message_ids: dict[str, MessageId] = {}
    if backend:
        async with backend._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT provider_message_id, message_id
                FROM messages
                WHERE conversation_id = ? AND provider_message_id IS NOT NULL
                """,
                (cid,),
            )
            rows = await cursor.fetchall()
        existing_message_ids = {
            str(row["provider_message_id"]): MessageId(row["message_id"]) for row in rows if row["provider_message_id"]
        }

    # First pass: build complete message_ids mapping (needed for parent resolution)
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid: MessageId = existing_message_ids.get(provider_message_id) or make_message_id(cid, provider_message_id)
        message_ids[str(provider_message_id)] = mid

    # Second pass: create MessageRecords with resolved parent IDs
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_ids[str(provider_message_id)]
        message_hash = message_content_hash(msg, provider_message_id)

        # Resolve parent message ID if present
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = message_ids.get(str(msg.parent_message_provider_id))

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
                parent_message_id=parent_message_id,
                branch_index=msg.branch_index,
            )
        )

    attachments: list[AttachmentRecord] = []
    for att in convo.attachments:
        aid, updated_meta, updated_path = attachment_content_id(convo.provider_name, att, archive_root=archive_root)
        # Merge updated metadata with provider_id if present
        meta: dict[str, object] = dict(updated_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        message_id_val: MessageId | None = (
            message_ids.get(att.message_provider_id or "") if att.message_provider_id else None
        )
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

    result = await save_bundle(
        RecordBundle(
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


__all__ = [
    "prepare_records",
]
