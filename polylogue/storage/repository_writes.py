"""Write/admin method mixin for the conversation repository."""

from __future__ import annotations

import builtins
from collections.abc import Callable

from polylogue.lib.models import Conversation
from polylogue.storage.action_event_rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.backends.queries import conversations as conversations_q
from polylogue.storage.backends.queries import publications as publications_q
from polylogue.storage.backends.queries import runs as runs_q
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
)

from .repository_support import provider_conversation_id


class RepositoryWriteMixin:
    async def save_conversation(
        self,
        conversation: Conversation | ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
        content_blocks: builtins.list[ContentBlockRecord] | None = None,
    ) -> dict[str, int]:
        if isinstance(conversation, Conversation):
            conv_record = self._conversation_to_record(conversation)
        else:
            conv_record = conversation

        return await self._save_via_backend(
            conv_record,
            messages,
            attachments,
            content_blocks or [],
        )

    def _conversation_to_record(self, conversation: Conversation) -> ConversationRecord:
        from typing import cast

        from polylogue.types import ContentHash, ConversationId

        created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
        updated_at_str = (
            conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)
        )

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

    async def _save_via_backend(
        self,
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

        backend = self._backend
        if backend is None:
            raise RuntimeError("Backend is not initialized")

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

        invalidate_search_cache()
        return counts

    async def record_run(self, record: RunRecord) -> None:
        async with self._backend.transaction(), self._backend.connection() as conn:
            await runs_q.record_run(conn, record, self._backend.transaction_depth)

    async def record_publication(self, record: PublicationRecord) -> None:
        async with self._backend.transaction(), self._backend.connection() as conn:
            await publications_q.record_publication(conn, record, self._backend.transaction_depth)

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        async with self._backend.connection() as conn:
            return await publications_q.get_latest_publication(conn, publication_kind)

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        async with self._backend.connection() as conn:
            return await conversations_q.get_metadata(conn, conversation_id)

    async def _metadata_read_modify_write(
        self,
        conversation_id: str,
        mutator: Callable[[dict[str, object]], bool],
    ) -> None:
        async with self._backend.transaction(), self._backend.connection() as conn:
            current = await conversations_q.get_metadata(conn, conversation_id)
            if mutator(current):
                await conversations_q.update_metadata_raw(
                    conn,
                    conversation_id,
                    current,
                )

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        def _set(meta: dict[str, object]) -> bool:
            meta[key] = value
            return True

        await self._metadata_read_modify_write(conversation_id, _set)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        def _delete(meta: dict[str, object]) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _delete)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        def _add(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            if tag not in tags:
                tags.append(tag)
                meta["tags"] = tags
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _add)

    async def remove_tag(self, conversation_id: str, tag: str) -> None:
        def _remove(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if isinstance(tags, list) and tag in tags:
                tags.remove(tag)
                meta["tags"] = tags
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _remove)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        async with self._backend.connection() as conn:
            return await conversations_q.list_tags(conn, provider=provider)

    async def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        async with self._backend.connection() as conn:
            await conversations_q.set_metadata(
                conn,
                conversation_id,
                metadata,
                self._backend.transaction_depth,
            )

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self._backend.transaction(), self._backend.connection() as conn:
            deleted = await conversations_q.delete_conversation_sql(
                conn,
                conversation_id,
                self._backend.transaction_depth,
            )
        if deleted:
            invalidate_search_cache()
        return deleted
