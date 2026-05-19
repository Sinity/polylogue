"""Write/admin method mixin for the conversation repository."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import TYPE_CHECKING

from polylogue.archive.conversation.models import Conversation
from polylogue.archive.topology.edge import TopologyEdgeRecord
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.insights.feedback import LearningCorrection
from polylogue.storage.repository.archive.writes.conversations import (
    conversation_to_record,
    delete_conversation_via_backend,
    provider_event_to_record,
    save_via_backend,
)
from polylogue.storage.repository.archive.writes.metadata import metadata_read_modify_write
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    ProviderEventRecord,
)
from polylogue.storage.sqlite.queries import conversations as conversations_q


class RepositoryWriteMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def save_conversation(
        self,
        conversation: Conversation | ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
        content_blocks: builtins.list[ContentBlockRecord] | None = None,
        provider_events: builtins.list[ProviderEventRecord] | None = None,
        topology_edges: builtins.list[TopologyEdgeRecord] | None = None,
    ) -> dict[str, int]:
        if isinstance(conversation, Conversation):
            if conversation.messages and not messages:
                raise ValueError(
                    "save_conversation() received a domain Conversation with messages but no "
                    "MessageRecord rows; pass transformed records instead of risking runtime-state loss"
                )
            if provider_events is None:
                provider_events = [provider_event_to_record(event) for event in conversation.provider_events]
            conv_record = self._conversation_to_record(conversation)
        else:
            conv_record = conversation

        return await self._save_via_backend(
            conv_record,
            messages,
            attachments,
            content_blocks or [],
            provider_events,
            topology_edges,
        )

    def _conversation_to_record(self, conversation: Conversation) -> ConversationRecord:
        return conversation_to_record(conversation)

    async def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
        content_blocks: builtins.list[ContentBlockRecord] | None = None,
        provider_events: builtins.list[ProviderEventRecord] | None = None,
        topology_edges: builtins.list[TopologyEdgeRecord] | None = None,
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
            provider_events,
            topology_edges,
        )

    async def get_metadata(self, conversation_id: str) -> JSONDocument:
        async with self._backend.connection() as conn:
            return await conversations_q.get_metadata(conn, conversation_id)

    async def _upsert_normalized_tag(self, conversation_id: str, tag_name: str) -> None:
        """Upsert a tag into the normalized tags table and link to conversation."""
        async with self._backend.transaction(), self._backend.connection() as conn:
            await conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            if row is not None:
                tag_id = row["id"]
                await conn.execute(
                    "INSERT OR IGNORE INTO conversation_tags (conversation_id, tag_id) VALUES (?, ?)",
                    (conversation_id, tag_id),
                )

    async def _delete_normalized_tag(self, conversation_id: str, tag_name: str) -> None:
        """Remove a tag link from the normalized tables for a conversation."""
        async with self._backend.transaction(), self._backend.connection() as conn:
            cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            if row is not None:
                await conn.execute(
                    "DELETE FROM conversation_tags WHERE conversation_id = ? AND tag_id = ?",
                    (conversation_id, row["id"]),
                )

    async def _metadata_read_modify_write(
        self,
        conversation_id: str,
        mutator: Callable[[JSONDocument], bool],
    ) -> bool:
        return await metadata_read_modify_write(self._backend, conversation_id, mutator)

    async def update_metadata(self, conversation_id: str, key: str, value: JSONValue) -> bool:
        def _set(meta: JSONDocument) -> bool:
            if key in meta and meta[key] == value:
                return False
            meta[key] = value
            return True

        return await self._metadata_read_modify_write(conversation_id, _set)

    async def delete_metadata(self, conversation_id: str, key: str) -> bool:
        def _delete(meta: JSONDocument) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        return await self._metadata_read_modify_write(conversation_id, _delete)

    async def add_tag(self, conversation_id: str, tag: str) -> bool:
        # #1240: tags are stored only in the M2M tables (tags + conversation_tags).
        # The previous dual-write into ``conversations.metadata['tags']`` was
        # legacy bookkeeping for the JSON read-fallback that has been removed.
        if not tag or not tag.strip():
            raise ValueError("tag must be a non-empty string")
        if len(tag) > 200:
            raise ValueError("tag must be at most 200 characters")

        async with self._backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM conversation_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.conversation_id = ? AND t.name = ?
                """,
                (conversation_id, tag),
            )
            already_present = await cursor.fetchone() is not None

        await self._upsert_normalized_tag(conversation_id, tag)
        return not already_present

    async def bulk_add_tags(self, conversation_ids: list[str], tags: list[str]) -> int:
        """Add tags to multiple conversations within a single transaction.

        Args:
            conversation_ids: List of conversation IDs to tag.
            tags: List of tag strings to apply to each conversation.

        Returns:
            Number of conversations whose tag set was actually changed.
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
                changed = False
                for tag in tags:
                    await conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                    cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    row = await cursor.fetchone()
                    if row is None:
                        continue
                    insert_result = await conn.execute(
                        "INSERT OR IGNORE INTO conversation_tags (conversation_id, tag_id) VALUES (?, ?)",
                        (conversation_id, row["id"]),
                    )
                    if insert_result.rowcount > 0:
                        changed = True
                if changed:
                    applied_count += 1
        return applied_count

    async def remove_tag(self, conversation_id: str, tag: str) -> bool:
        async with self._backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM conversation_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.conversation_id = ? AND t.name = ?
                """,
                (conversation_id, tag),
            )
            had_link = await cursor.fetchone() is not None

        await self._delete_normalized_tag(conversation_id, tag)
        return had_link

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

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    async def add_mark(
        self,
        conversation_id: str,
        mark_type: str,
        *,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Add a mark to any supported user-state target. Returns True if newly inserted.

        Supported kinds: see ``polylogue.core.user_state_targets`` (#1113). The
        caller is expected to have resolved/validated ``target_id`` via the
        archive facade resolver; this method enforces the substrate-level
        invariants only.
        """
        import datetime as _dt

        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        if mark_type not in ("star", "pin", "archive"):
            raise ValueError(f"invalid mark_type: {mark_type!r}")
        if target_type not in TARGET_KIND_NAMES:
            raise ValueError(f"invalid target_type: {target_type!r}. Supported: {', '.join(TARGET_KIND_NAMES)}")
        resolved_target_id: str | None
        if target_type == "conversation":
            resolved_target_id = target_id or conversation_id
        elif target_type == "message":
            resolved_target_id = target_id or message_id
        else:
            resolved_target_id = target_id
        if resolved_target_id is None:
            raise ValueError(f"target_id is required for {target_type!r} marks")
        async with self._backend.connection() as conn:
            return await conversations_q.add_mark(
                conn,
                target_type=target_type,
                target_id=resolved_target_id,
                conversation_id=conversation_id,
                message_id=message_id,
                mark_type=mark_type,
                created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )

    async def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove a mark from a conversation or message target. Returns True if deleted."""
        async with self._backend.connection() as conn:
            return await conversations_q.remove_mark(conn, target_type, target_id, mark_type)

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks, optionally filtered by type, target, conversation, or message."""
        async with self._backend.connection() as conn:
            return await conversations_q.list_marks(
                conn,
                mark_type=mark_type,
                conversation_id=conversation_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )

    async def save_annotation(
        self,
        *,
        annotation_id: str,
        target_type: str,
        target_id: str,
        conversation_id: str,
        note_text: str,
        message_id: str | None = None,
    ) -> bool:
        """Insert or update an annotation. Returns True if newly inserted."""
        import datetime as _dt

        now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        async with self._backend.connection() as conn:
            return await conversations_q.save_annotation(
                conn,
                annotation_id=annotation_id,
                target_type=target_type,
                target_id=target_id,
                conversation_id=conversation_id,
                message_id=message_id,
                note_text=note_text,
                now=now,
            )

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Get an annotation by ID."""
        async with self._backend.connection() as conn:
            return await conversations_q.get_annotation(conn, annotation_id)

    async def list_annotations(
        self,
        *,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations, optionally filtered by target, conversation, or message."""
        async with self._backend.connection() as conn:
            return await conversations_q.list_annotations(
                conn,
                conversation_id=conversation_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )

    async def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns True if deleted."""
        async with self._backend.connection() as conn:
            return await conversations_q.delete_annotation(conn, annotation_id)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Save a named query view. Returns True if newly created."""
        import datetime as _dt

        async with self._backend.connection() as conn:
            return await conversations_q.save_view(
                conn, view_id, name, query_json, _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            )

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        async with self._backend.connection() as conn:
            return await conversations_q.get_view(conn, view_id)

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get a saved view by name."""
        async with self._backend.connection() as conn:
            return await conversations_q.get_view_by_name(conn, name)

    async def list_views(self) -> list[dict[str, str]]:
        """List all saved views."""
        async with self._backend.connection() as conn:
            return await conversations_q.list_views(conn)

    async def delete_view(self, view_id: str) -> bool:
        """Delete a saved view. Returns True if deleted."""
        async with self._backend.connection() as conn:
            return await conversations_q.delete_view(conn, view_id)

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def save_recall_pack(self, pack_id: str, label: str, conversation_ids_json: str, payload_json: str) -> bool:
        """Save a recall pack. Returns True if newly created."""
        import datetime as _dt

        async with self._backend.connection() as conn:
            return await conversations_q.save_recall_pack(
                conn,
                pack_id,
                label,
                conversation_ids_json,
                payload_json,
                _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get a recall pack by ID."""
        async with self._backend.connection() as conn:
            return await conversations_q.get_recall_pack(conn, pack_id)

    async def list_recall_packs(self) -> list[dict[str, str]]:
        """List all recall packs."""
        async with self._backend.connection() as conn:
            return await conversations_q.list_recall_packs(conn)

    async def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete a recall pack. Returns True if deleted."""
        async with self._backend.connection() as conn:
            return await conversations_q.delete_recall_pack(conn, pack_id)

    # ------------------------------------------------------------------
    # Reader workspaces
    # ------------------------------------------------------------------

    async def save_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        """Save a durable reader workspace. Returns True if newly created."""
        import datetime as _dt

        async with self._backend.connection() as conn:
            return await conversations_q.save_workspace(
                conn,
                workspace_id=workspace_id,
                name=name,
                mode=mode,
                open_targets_json=open_targets_json,
                layout_json=layout_json,
                active_target_json=active_target_json,
                now=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get a reader workspace by ID."""
        async with self._backend.connection() as conn:
            return await conversations_q.get_workspace(conn, workspace_id)

    async def list_workspaces(self) -> list[dict[str, str]]:
        """List durable reader workspaces."""
        async with self._backend.connection() as conn:
            return await conversations_q.list_workspaces(conn)

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a reader workspace. Returns True if deleted."""
        async with self._backend.connection() as conn:
            return await conversations_q.delete_workspace(conn, workspace_id)

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # Persisted in ``user_corrections``. Lives outside the content-hash
    # boundary: applying or removing a correction never touches the
    # ``conversations.content_hash`` column. See
    # :mod:`polylogue.storage.insights.feedback` for the SQL surface and
    # :mod:`polylogue.insights.feedback` for the merge semantics.
    # ------------------------------------------------------------------

    async def record_correction(
        self,
        conversation_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        from polylogue.insights.feedback import parse_correction_kind
        from polylogue.storage.insights.feedback import upsert_correction

        typed_kind = parse_correction_kind(kind)
        async with self._backend.connection() as conn:
            return await upsert_correction(
                conn,
                conversation_id=conversation_id,
                kind=typed_kind,
                payload=payload,
                note=note,
            )

    async def list_corrections(
        self,
        *,
        conversation_id: str | None = None,
        kind: str | None = None,
    ) -> builtins.list[LearningCorrection]:
        from polylogue.insights.feedback import parse_correction_kind
        from polylogue.storage.insights.feedback import list_corrections

        typed_kind = parse_correction_kind(kind) if kind is not None else None
        async with self._backend.connection() as conn:
            return await list_corrections(
                conn,
                conversation_id=conversation_id,
                kind=typed_kind,
            )

    async def delete_correction(self, conversation_id: str, kind: str) -> bool:
        from polylogue.insights.feedback import parse_correction_kind
        from polylogue.storage.insights.feedback import delete_correction

        typed_kind = parse_correction_kind(kind)
        async with self._backend.connection() as conn:
            return await delete_correction(
                conn,
                conversation_id=conversation_id,
                kind=typed_kind,
            )

    async def clear_corrections(self, conversation_id: str) -> int:
        from polylogue.storage.insights.feedback import clear_corrections

        async with self._backend.connection() as conn:
            return await clear_corrections(conn, conversation_id=conversation_id)
