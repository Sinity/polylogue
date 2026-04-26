"""Write/admin method mixin for the conversation repository."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import TYPE_CHECKING

from polylogue.lib.conversation_models import Conversation
from polylogue.lib.json import JSONDocument, JSONValue
from polylogue.lib.payload_coercion import string_sequence
from polylogue.storage.backends.queries import conversations as conversations_q
from polylogue.storage.backends.queries import publications as publications_q
from polylogue.storage.backends.queries import runs as runs_q
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
)

from .repository_contracts import RepositoryBackendProtocol
from .repository_write_conversations import (
    conversation_to_record,
    delete_conversation_via_backend,
    save_via_backend,
)
from .repository_write_metadata import metadata_read_modify_write


class RepositoryWriteMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def save_conversation(
        self,
        conversation: Conversation | ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
        content_blocks: builtins.list[ContentBlockRecord] | None = None,
    ) -> dict[str, int]:
        if isinstance(conversation, Conversation):
            if conversation.messages and not messages:
                raise ValueError(
                    "save_conversation() received a domain Conversation with messages but no "
                    "MessageRecord rows; pass transformed records instead of risking runtime-state loss"
                )
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
        return conversation_to_record(conversation)

    async def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
        content_blocks: builtins.list[ContentBlockRecord] | None = None,
    ) -> dict[str, int]:
        backend = self._backend
        if backend is None:
            raise RuntimeError("Backend is not initialized")
        return await save_via_backend(
            backend,
            conversation,
            messages,
            attachments,
            content_blocks,
        )

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

    async def get_metadata(self, conversation_id: str) -> JSONDocument:
        async with self._backend.connection() as conn:
            return await conversations_q.get_metadata(conn, conversation_id)

    async def _metadata_read_modify_write(
        self,
        conversation_id: str,
        mutator: Callable[[JSONDocument], bool],
    ) -> None:
        await metadata_read_modify_write(self._backend, conversation_id, mutator)

    async def update_metadata(self, conversation_id: str, key: str, value: JSONValue) -> None:
        def _set(meta: JSONDocument) -> bool:
            meta[key] = value
            return True

        await self._metadata_read_modify_write(conversation_id, _set)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        def _delete(meta: JSONDocument) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _delete)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        if not tag or not tag.strip():
            raise ValueError("tag must be a non-empty string")
        if len(tag) > 200:
            raise ValueError("tag must be at most 200 characters")

        def _add(meta: JSONDocument) -> bool:
            tags = list(string_sequence(meta.get("tags")))
            if tag not in tags:
                tags.append(tag)
                tag_payload: list[JSONValue] = list(tags)
                meta["tags"] = tag_payload
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _add)

    async def bulk_add_tags(self, conversation_ids: list[str], tags: list[str]) -> int:
        """Add tags to multiple conversations within a single transaction.

        Args:
            conversation_ids: List of conversation IDs to tag.
            tags: List of tag strings to apply to each conversation.

        Returns:
            Number of conversations whose tags were actually changed.
        """
        backend = self._backend
        applied_count = 0
        async with backend.transaction(), backend.connection() as conn:
            for conversation_id in conversation_ids:
                exists = await conn.execute(
                    "SELECT 1 FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )
                if not await exists.fetchone():
                    continue
                current = await conversations_q.get_metadata(conn, conversation_id)
                meta: JSONDocument = dict(current) if current else {}
                existing_tags = list(string_sequence(meta.get("tags")))
                changed = False
                for tag in tags:
                    if tag not in existing_tags:
                        existing_tags.append(tag)
                        changed = True
                if changed:
                    tag_payload: list[JSONValue] = list(existing_tags)
                    meta["tags"] = tag_payload
                    await conversations_q.update_metadata_raw(
                        conn,
                        conversation_id,
                        meta,
                    )
                    applied_count += 1
        return applied_count

    async def remove_tag(self, conversation_id: str, tag: str) -> None:
        def _remove(meta: JSONDocument) -> bool:
            tags = list(string_sequence(meta.get("tags")))
            if tag in tags:
                tags.remove(tag)
                tag_payload: list[JSONValue] = list(tags)
                meta["tags"] = tag_payload
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _remove)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        async with self._backend.connection() as conn:
            return await conversations_q.list_tags(conn, provider=provider)

    async def set_metadata(self, conversation_id: str, metadata: JSONDocument) -> None:
        async with self._backend.connection() as conn:
            await conversations_q.set_metadata(
                conn,
                conversation_id,
                metadata,
                self._backend.transaction_depth,
            )

    async def delete_conversation(self, conversation_id: str) -> bool:
        return await delete_conversation_via_backend(self._backend, conversation_id)
