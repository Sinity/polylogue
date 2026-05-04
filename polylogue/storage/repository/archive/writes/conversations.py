"""Conversation write helpers for the repository mixin."""

from __future__ import annotations

import builtins

from polylogue.archive.conversation.models import Conversation
from polylogue.core.hashing import hash_payload
from polylogue.core.json import json_document
from polylogue.pipeline.ids import _conversation_hash_payload, _normalize_for_hash
from polylogue.storage.action_events.rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.conversation_replacement import (
    recount_and_prune_attachments_async,
    replace_conversation_runtime_state_async,
)
from polylogue.storage.insights.session.refresh import (
    delete_session_insights_for_conversation_async,
    refresh_thread_after_conversation_delete_async,
)
from polylogue.storage.insights.session.threads import thread_root_id_async
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.sqlite.queries import action_events as action_events_q
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import conversations as conversations_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import stats as stats_q


def provider_conversation_id(conversation_id: str, provider: str | None) -> str:
    """Strip only the canonical provider prefix from conversation IDs."""
    if not provider:
        return conversation_id
    prefix = f"{provider}:"
    return conversation_id[len(prefix) :] if conversation_id.startswith(prefix) else conversation_id


def _normalize_hash_value(value: object) -> JSONValue:
    if value is None:
        return "__POLYLOGUE_NULL__"
    if value == "":
        return "__POLYLOGUE_EMPTY__"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _content_hash_from_metadata_or_domain(conversation: Conversation, metadata: dict[str, object]) -> str:
    existing = metadata.get("content_hash")
    if isinstance(existing, str) and existing.strip():
        return existing

    messages_payload: list[dict[str, JSONValue]] = []
    for message in conversation.messages:
        messages_payload.append(
            {
                "id": str(message.id),
                "role": str(message.role),
                "text": _normalize_hash_value(message.text),
                "timestamp": _normalize_hash_value(message.timestamp.isoformat() if message.timestamp else None),
            }
        )
    attachments_payload = sorted(
        [
            {
                "id": _normalize_hash_value(attachment.id),
                "message_id": _normalize_hash_value(message.id),
                "name": _normalize_hash_value(attachment.name),
                "mime_type": _normalize_hash_value(attachment.mime_type),
                "size_bytes": _normalize_hash_value(attachment.size_bytes),
            }
            for message in conversation.messages
            for attachment in message.attachments
        ],
        key=lambda item: (
            str(item.get("message_id") or ""),
            str(item.get("id") or ""),
            str(item.get("name") or ""),
        ),
    )
    return hash_payload(
        {
            "title": _normalize_hash_value(conversation.title),
            "created_at": _normalize_hash_value(
                conversation.created_at.isoformat() if conversation.created_at else None
            ),
            "updated_at": _normalize_hash_value(
                conversation.updated_at.isoformat() if conversation.updated_at else None
            ),
            "messages": messages_payload,
            "attachments": attachments_payload,
        }
    )


def conversation_to_record(conversation: Conversation) -> ConversationRecord:
    from polylogue.types import ContentHash, ConversationId

    created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
    updated_at_str = conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)
    metadata = json_document(conversation.metadata)
    metadata_record: dict[str, object] = {}
    metadata_record.update(metadata)
    provider_meta: dict[str, object] = {}
    provider_meta.update(json_document(metadata.get("provider_meta")))

    return ConversationRecord(
        conversation_id=ConversationId(str(conversation.id)),
        provider_name=conversation.provider,
        provider_conversation_id=provider_conversation_id(
            conversation_id=str(conversation.id),
            provider=conversation.provider,
        ),
        title=conversation.title or "",
        created_at=created_at_str,
        updated_at=updated_at_str,
        content_hash=ContentHash(_content_hash_from_metadata_or_domain(conversation, metadata_record)),
        provider_meta=provider_meta,
        metadata=metadata_record,
    )


async def save_via_backend(
    backend: RepositoryBackendProtocol,
    conversation: ConversationRecord,
    messages: builtins.list[MessageRecord],
    attachments: builtins.list[AttachmentRecord],
    content_blocks: builtins.list[ContentBlockRecord] | None = None,
) -> dict[str, int]:
    import time as _time

    t_start = _time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    async with backend.transaction(), backend.connection() as conn:
        t0 = _time.perf_counter()
        existing_hash: str | None = None
        cursor = await conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = ?",
            (conversation.conversation_id,),
        )
        row = await cursor.fetchone()
        if row:
            existing_hash = row["content_hash"]
        timings["hash_check"] = _time.perf_counter() - t0

        content_unchanged = existing_hash is not None and existing_hash == conversation.content_hash

        t0 = _time.perf_counter()
        await conversations_q.save_conversation_record(conn, conversation, backend.transaction_depth)
        timings["save_conv"] = _time.perf_counter() - t0

        if content_unchanged:
            counts["skipped_conversations"] = 1
            counts["skipped_messages"] = len(messages)
            counts["skipped_attachments"] = len(attachments)
        else:
            counts["conversations"] = 1
            t0 = _time.perf_counter()
            affected_attachment_ids = await replace_conversation_runtime_state_async(conn, conversation.conversation_id)
            timings["replace_runtime"] = _time.perf_counter() - t0

            if messages:
                t0 = _time.perf_counter()
                await messages_q.save_messages(conn, messages, backend.transaction_depth)
                timings["save_msgs"] = _time.perf_counter() - t0
                counts["messages"] = len(messages)

                t0 = _time.perf_counter()
                await stats_q.upsert_conversation_stats(
                    conn,
                    conversation.conversation_id,
                    conversation.provider_name,
                    messages,
                    backend.transaction_depth,
                )
                timings["upsert_stats"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            all_blocks: builtins.list[ContentBlockRecord] = list(content_blocks or [])
            for message in messages:
                all_blocks.extend(message.content_blocks)
            if all_blocks:
                await attachments_q.save_content_blocks(conn, all_blocks, backend.transaction_depth)
            timings["save_blocks"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            action_messages = attach_blocks_to_messages(messages, all_blocks)
            action_records = build_action_event_records(conversation, action_messages)
            await action_events_q.replace_action_events(
                conn,
                conversation.conversation_id,
                action_records,
                backend.transaction_depth,
            )
            timings["action_events"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
            affected_attachment_ids |= new_attachment_ids
            if attachments:
                await attachments_q.save_attachments(conn, attachments, backend.transaction_depth)
                counts["attachments"] = len(attachments)
            await recount_and_prune_attachments_async(conn, affected_attachment_ids)
            timings["attachments"] = _time.perf_counter() - t0

    total = _time.perf_counter() - t_start
    if total > 2.0:
        from polylogue.logging import get_logger

        get_logger(__name__).info(
            "slow_save",
            total_s=round(total, 2),
            msgs=len(messages),
            blocks=counts.get("messages", 0),
            cid=str(conversation.conversation_id)[:20],
            **{k: round(v, 3) for k, v in timings.items()},
        )

    invalidate_search_cache()
    return counts


async def delete_conversation_via_backend(
    backend: RepositoryBackendProtocol,
    conversation_id: str,
) -> bool:
    from polylogue.storage.sqlite.queries import conversations as conversations_q

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
            await delete_session_insights_for_conversation_async(
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
