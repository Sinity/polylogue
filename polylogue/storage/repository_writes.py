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
        def _add(meta: JSONDocument) -> bool:
            tags = list(string_sequence(meta.get("tags")))
            if tag not in tags:
                tags.append(tag)
                tag_payload: list[JSONValue] = list(tags)
                meta["tags"] = tag_payload
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _add)

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
