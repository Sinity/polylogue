"""Conversation write helpers for the repository mixin."""

from __future__ import annotations

import builtins

from polylogue.lib.conversation_models import Conversation
from polylogue.storage.action_event_rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.session_product_refresh_deletes import (
    delete_session_products_for_conversation_async,
    refresh_thread_after_conversation_delete_async,
)
from polylogue.storage.session_product_refresh_updates import (
    refresh_session_products_for_conversation_async,
)
from polylogue.storage.session_product_threads import thread_root_id_async
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)


def provider_conversation_id(conversation_id: str, provider: str | None) -> str:
    """Strip only the canonical provider prefix from conversation IDs."""
    if not provider:
        return conversation_id
    prefix = f"{provider}:"
    return conversation_id[len(prefix) :] if conversation_id.startswith(prefix) else conversation_id


def conversation_to_record(conversation: Conversation) -> ConversationRecord:
    from typing import cast

    from polylogue.types import ContentHash, ConversationId

    created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
    updated_at_str = conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)

    return ConversationRecord(
        conversation_id=cast(ConversationId, str(conversation.id)),
        provider_name=conversation.provider,
        provider_conversation_id=provider_conversation_id(
            conversation_id=str(conversation.id),
            provider=conversation.provider,
        ),
        title=conversation.title or "",
        created_at=created_at_str,
        updated_at=updated_at_str,
        content_hash=cast(ContentHash, conversation.metadata.get("content_hash", "")),
        provider_meta=cast(dict[str, object], conversation.metadata.get("provider_meta", {})),
        metadata=conversation.metadata,
    )


async def save_via_backend(
    backend,
    conversation: ConversationRecord,
    messages: builtins.list[MessageRecord],
    attachments: builtins.list[AttachmentRecord],
    content_blocks: builtins.list[ContentBlockRecord] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    existing_hash: str | None = None
    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = ?",
            (conversation.conversation_id,),
        )
        row = await cursor.fetchone()
        if row:
            existing_hash = row["content_hash"]

    content_unchanged = existing_hash is not None and existing_hash == conversation.content_hash

    async with backend.transaction():
        await backend.save_conversation_record(conversation)

        if content_unchanged:
            counts["skipped_conversations"] = 1
            counts["skipped_messages"] = len(messages)
            counts["skipped_attachments"] = len(attachments)
        else:
            counts["conversations"] = 1

            if messages:
                pname = conversation.provider_name
                if pname:
                    messages = [
                        message.model_copy(update={"provider_name": pname})
                        if not message.provider_name
                        else message
                        for message in messages
                    ]
                await backend.save_messages(messages)
                counts["messages"] = len(messages)
                await backend.upsert_conversation_stats(
                    conversation.conversation_id,
                    pname,
                    messages,
                )

            all_blocks: builtins.list[ContentBlockRecord] = list(content_blocks or [])
            for message in messages:
                all_blocks.extend(message.content_blocks)
            if all_blocks:
                await backend.save_content_blocks(all_blocks)
            action_messages = attach_blocks_to_messages(messages, all_blocks)
            action_records = build_action_event_records(conversation, action_messages)
            await backend.replace_action_events(conversation.conversation_id, action_records)

            new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
            await backend.prune_attachments(conversation.conversation_id, new_attachment_ids)

            if attachments:
                await backend.save_attachments(attachments)
                counts["attachments"] = len(attachments)

            async with backend.connection() as conn:
                await refresh_session_products_for_conversation_async(
                    conn,
                    str(conversation.conversation_id),
                    transaction_depth=backend.transaction_depth,
                )

    invalidate_search_cache()
    return counts


async def delete_conversation_via_backend(backend, conversation_id: str) -> bool:
    from polylogue.storage.backends.queries import conversations as conversations_q

    async with backend.transaction(), backend.connection() as conn:
        parent_row = await (
            await conn.execute(
                "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        ).fetchone()
        child_rows = await (
            await conn.execute(
                "SELECT conversation_id FROM conversations WHERE parent_conversation_id = ?",
                (conversation_id,),
            )
        ).fetchall()
        existing_row = await (
            await conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        ).fetchone()
        deleted = False
        if existing_row is not None:
            await delete_session_products_for_conversation_async(
                conn,
                conversation_id,
                transaction_depth=backend.transaction_depth,
            )
            deleted = await conversations_q.delete_conversation_sql(
                conn,
                conversation_id,
                backend.transaction_depth,
            )
        if deleted:
            affected_seeds = {str(row["conversation_id"]) for row in child_rows}
            if parent_row is not None and parent_row["parent_conversation_id"] is not None:
                affected_seeds.add(str(parent_row["parent_conversation_id"]))
            affected_roots: set[str] = set()
            for seed in affected_seeds:
                root_id = await thread_root_id_async(conn, seed)
                if root_id is not None:
                    affected_roots.add(root_id)
            for root_id in affected_roots:
                await refresh_thread_after_conversation_delete_async(
                    conn,
                    root_id,
                    transaction_depth=backend.transaction_depth,
                )
    if deleted:
        invalidate_search_cache()
    return deleted
