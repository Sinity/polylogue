"""Write/admin method mixin for the conversation repository."""

from __future__ import annotations

import builtins

from polylogue.lib.models import Conversation
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

                new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
                await backend.prune_attachments(conversation.conversation_id, new_attachment_ids)

                if attachments:
                    await backend.save_attachments(attachments)
                    counts["attachments"] = len(attachments)

        invalidate_search_cache()
        return counts

    async def record_run(self, record: RunRecord) -> None:
        await self._backend.record_run(record)

    async def record_publication(self, record: PublicationRecord) -> None:
        await self._backend.record_publication(record)

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        return await self._backend.get_latest_publication(publication_kind)

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        return await self._backend.get_metadata(conversation_id)

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        await self._backend.update_metadata(conversation_id, key, value)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        await self._backend.delete_metadata(conversation_id, key)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        await self._backend.add_tag(conversation_id, tag)

    async def remove_tag(self, conversation_id: str, tag: str) -> None:
        await self._backend.remove_tag(conversation_id, tag)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self._backend.list_tags(provider=provider)

    async def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        await self._backend.set_metadata(conversation_id, metadata)

    async def delete_conversation(self, conversation_id: str) -> bool:
        deleted = await self._backend.delete_conversation(conversation_id)
        if deleted:
            invalidate_search_cache()
        return deleted
