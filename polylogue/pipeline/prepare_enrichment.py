"""DB-aware enrichment for transformed record bundles."""

from __future__ import annotations

from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.prepare_models import (
    EnrichedBundle,
    PrepareCache,
    RecordBundle,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    ExistingConversation,
    MessageRecord,
)
from polylogue.types import MessageId


def enrich_bundle_from_db(convo, source_name: str, transform, cache: PrepareCache, *, raw_id: str | None = None) -> EnrichedBundle:
    candidate_cid = transform.candidate_cid
    content_hash = transform.content_hash

    existing = cache.existing.get(candidate_cid)
    if existing:
        cid = existing.conversation_id
        changed = existing.content_hash != content_hash
    else:
        cid = candidate_cid
        changed = False

    parent_conversation_id = None
    if convo.parent_conversation_provider_id:
        candidate_parent = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)
        if candidate_parent in cache.known_ids:
            parent_conversation_id = candidate_parent

    existing_message_ids = cache.message_ids.get(cid, {})
    stable_message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        key = str(provider_message_id)
        stable_message_id_map[key] = existing_message_ids.get(key) or transform.message_id_map[key]

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
        sort_key=transform.bundle.conversation.sort_key,
        content_hash=content_hash,
        provider_meta=merged_provider_meta,
        parent_conversation_id=parent_conversation_id,
        branch_type=convo.branch_type,
        raw_id=raw_id,
    )

    patched_messages: list[MessageRecord] = []
    for idx, (msg_rec, msg) in enumerate(zip(transform.bundle.messages, convo.messages, strict=True), start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = stable_message_id_map[str(provider_message_id)]
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = stable_message_id_map.get(str(msg.parent_message_provider_id))
        patched_messages.append(
            MessageRecord(
                message_id=mid,
                conversation_id=cid,
                provider_message_id=msg_rec.provider_message_id,
                role=msg_rec.role,
                text=msg_rec.text,
                sort_key=msg_rec.sort_key,
                content_hash=msg_rec.content_hash,
                parent_message_id=parent_message_id,
                branch_index=msg_rec.branch_index,
                provider_name=msg_rec.provider_name,
                word_count=msg_rec.word_count,
                has_tool_use=msg_rec.has_tool_use,
                has_thinking=msg_rec.has_thinking,
            )
        )

    reverse_mid: dict[MessageId, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        old = transform.message_id_map[str(provider_message_id)]
        reverse_mid[old] = stable_message_id_map[str(provider_message_id)]

    patched_blocks = [
        ContentBlockRecord(
            block_id=ContentBlockRecord.make_id(reverse_mid.get(block.message_id, block.message_id), block.block_index),
            message_id=reverse_mid.get(block.message_id, block.message_id),
            conversation_id=cid,
            block_index=block.block_index,
            type=block.type,
            text=block.text,
            tool_name=block.tool_name,
            tool_id=block.tool_id,
            tool_input=block.tool_input,
            media_type=block.media_type,
            metadata=block.metadata,
            semantic_type=block.semantic_type,
        )
        for block in transform.bundle.content_blocks
    ]

    patched_attachments: list[AttachmentRecord] = []
    for attachment in transform.bundle.attachments:
        att_message_id: MessageId | None = None
        if attachment.message_id is not None:
            att_message_id = reverse_mid.get(attachment.message_id, attachment.message_id)
        patched_attachments.append(
            AttachmentRecord(
                attachment_id=attachment.attachment_id,
                conversation_id=cid,
                message_id=att_message_id,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                path=attachment.path,
                provider_meta=attachment.provider_meta,
            )
        )

    return EnrichedBundle(
        bundle=RecordBundle(
            conversation=conversation_record,
            messages=patched_messages,
            attachments=patched_attachments,
            content_blocks=patched_blocks,
        ),
        materialization_plan=transform.materialization_plan,
        cid=cid,
        changed=changed,
    )


async def _build_single_cache(backend, convo, candidate_cid, _unused) -> PrepareCache:
    cache = PrepareCache()

    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT conversation_id, content_hash FROM conversations WHERE conversation_id = ? LIMIT 1",
            (candidate_cid,),
        )
        row = await cursor.fetchone()
    if row:
        cid = row["conversation_id"]
        cache.existing[cid] = ExistingConversation(conversation_id=cid, content_hash=row["content_hash"])
        cache.known_ids.add(cid)

    if convo.parent_conversation_provider_id:
        candidate_parent = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (candidate_parent,),
            )
            if await cursor.fetchone():
                cache.known_ids.add(candidate_parent)

    existing_cid = candidate_cid if candidate_cid in cache.known_ids else None
    if existing_cid:
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT provider_message_id, message_id FROM messages "
                "WHERE conversation_id = ? AND provider_message_id IS NOT NULL",
                (existing_cid,),
            )
            rows = await cursor.fetchall()
        cache.message_ids[existing_cid] = {
            str(row["provider_message_id"]): MessageId(row["message_id"])
            for row in rows
            if row["provider_message_id"]
        }

    return cache


__all__ = [
    "_build_single_cache",
    "enrich_bundle_from_db",
]
