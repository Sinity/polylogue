"""Mutation operations: tags, metadata, marks, annotations, views, recall packs, workspaces.

This mixin is consumed by ``ArchiveOperations``. It exists in its own
module to keep ``operations/archive.py`` below the documented file-size
budget while still routing every surface-side mutation through the
shared operation contract (#860). Each method delegates to the
``ConversationRepository`` write mixin via a localized import so the
surface layer never has to reach into ``polylogue.storage.repository``
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from polylogue.surfaces.payloads import (
    BulkTagMutationResult,
    DeleteConversationResult,
    MetadataMutationResult,
    validate_metadata_key,
)

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


class MetadataKeyValidationError(ValueError):
    """Raised when a user metadata key fails the centralized validation rules."""


class ArchiveMutationsMixin:
    """Tag, metadata, mark, annotation, view, recall-pack, and workspace writes."""

    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

    # ------------------------------------------------------------------
    # Validation helpers shared across surfaces
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_metadata_key(key: str) -> None:
        """Raise :class:`MetadataKeyValidationError` for invalid metadata keys."""
        error = validate_metadata_key(key)
        if error is not None:
            raise MetadataKeyValidationError(error)

    async def _resolve_or_none(self, conversation_id: str) -> str | None:
        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        return str(resolved) if resolved else None

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    async def add_tag(self, conversation_id: str, tag: str) -> bool:
        """Add a tag to a conversation. Returns ``True`` if newly added."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.add_tag(conversation_id, tag)

    async def remove_tag(self, conversation_id: str, tag: str) -> bool:
        """Remove a tag. Returns ``True`` if it was present."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.remove_tag(conversation_id, tag)

    async def bulk_add_tags(self, conversation_ids: list[str], tags: list[str]) -> int:
        """Add tags to many conversations in one transaction."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.bulk_add_tags(conversation_ids, tags)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Return raw metadata document for a conversation."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        doc = await store.get_metadata(conversation_id)
        return dict(doc)

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> bool:
        """Set ``key=value`` on a conversation's metadata document."""
        from polylogue.core.json import JSONValue
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.update_metadata(conversation_id, key, cast(JSONValue, value))

    async def delete_metadata(self, conversation_id: str, key: str) -> bool:
        """Delete ``key`` from a conversation's metadata document."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_metadata(conversation_id, key)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Permanently delete a conversation. Returns ``True`` if it existed."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_conversation(conversation_id)

    # ------------------------------------------------------------------
    # Validated, resolve-aware entrypoints used by all surfaces (#862)
    # ------------------------------------------------------------------

    async def set_metadata_validated(self, conversation_id: str, key: str, value: object) -> MetadataMutationResult:
        """Validate, resolve, and set a metadata key.

        Raises :class:`MetadataKeyValidationError` on bad keys and
        :class:`~polylogue.api.archive.ConversationNotFoundError` when the
        conversation is missing. Always returns a typed
        :class:`MetadataMutationResult`.
        """
        from polylogue.api.archive import ConversationNotFoundError

        self._validate_metadata_key(key)
        resolved = await self._resolve_or_none(conversation_id)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        changed = await self.update_metadata(resolved, key, value)
        return MetadataMutationResult(
            outcome="set" if changed else "unchanged",
            conversation_id=resolved,
            key=key,
            detail=None if changed else "value_unchanged",
        )

    async def delete_metadata_validated(self, conversation_id: str, key: str) -> MetadataMutationResult:
        """Validate, resolve, and delete a metadata key.

        Returns a typed result with ``outcome="deleted"`` if the key existed
        and ``outcome="not_found"`` otherwise. Raises
        :class:`MetadataKeyValidationError` on bad keys and
        :class:`~polylogue.api.archive.ConversationNotFoundError` when the
        conversation is missing.
        """
        from polylogue.api.archive import ConversationNotFoundError

        self._validate_metadata_key(key)
        resolved = await self._resolve_or_none(conversation_id)
        if resolved is None:
            raise ConversationNotFoundError(conversation_id)
        deleted = await self.delete_metadata(resolved, key)
        return MetadataMutationResult(
            outcome="deleted" if deleted else "not_found",
            conversation_id=resolved,
            key=key,
            detail=None if deleted else "key_not_found",
        )

    async def delete_conversation_safe(self, conversation_id: str) -> DeleteConversationResult:
        """Idempotent typed delete that never raises on missing conversations.

        Returns a :class:`DeleteConversationResult` with ``outcome="deleted"``
        or ``outcome="not_found"``. Surfaces can rely on this contract instead
        of mapping booleans into status strings themselves.
        """
        resolved = await self._resolve_or_none(conversation_id)
        if resolved is None:
            return DeleteConversationResult(
                outcome="not_found",
                conversation_id=conversation_id,
                detail="conversation_not_found",
            )
        deleted = await self.delete_conversation(resolved)
        if not deleted:
            return DeleteConversationResult(
                outcome="not_found",
                conversation_id=resolved,
                detail="conversation_not_found",
            )
        return DeleteConversationResult(outcome="deleted", conversation_id=resolved)

    async def bulk_tag_conversations(
        self,
        conversation_ids: list[str],
        tags: list[str],
        *,
        max_conversations: int = 100,
        max_tags: int = 20,
    ) -> BulkTagMutationResult:
        """Validate and apply a bulk-tag operation.

        Raises :class:`ValueError` for empty or oversize inputs. Returns a
        :class:`BulkTagMutationResult` with the affected and skipped counts.
        """
        if not conversation_ids:
            raise ValueError("bulk_tag_conversations requires at least one conversation_id")
        if not tags:
            raise ValueError("bulk_tag_conversations requires at least one tag")
        if len(conversation_ids) > max_conversations:
            raise ValueError(f"bulk_tag_conversations supports at most {max_conversations} conversation_ids")
        if len(tags) > max_tags:
            raise ValueError(f"bulk_tag_conversations supports at most {max_tags} tags")
        applied = await self.bulk_add_tags(conversation_ids, tags)
        skipped = len(conversation_ids) - applied
        return BulkTagMutationResult(
            conversation_count=len(conversation_ids),
            tag_count=len(tags),
            affected_count=applied,
            skipped_count=skipped,
        )

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
        """Add a mark (star/pin/archive) to a conversation or message target."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.add_mark(
            conversation_id,
            mark_type,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    async def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove a mark."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.remove_mark(target_type, target_id, mark_type)

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks with optional filters."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.list_marks(
            mark_type=mark_type,
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

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
        """Insert or update an annotation. Returns ``True`` if newly inserted."""
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.save_annotation(
            annotation_id=annotation_id,
            target_type=target_type,
            target_id=target_id,
            conversation_id=conversation_id,
            message_id=message_id,
            note_text=note_text,
        )

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.get_annotation(annotation_id)

    async def list_annotations(
        self,
        *,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.list_annotations(
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    async def delete_annotation(self, annotation_id: str) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_annotation(annotation_id)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.save_view(view_id, name, query_json)

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.get_view(view_id)

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.get_view_by_name(name)

    async def list_views(self) -> list[dict[str, str]]:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.list_views()

    async def delete_view(self, view_id: str) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_view(view_id)

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def save_recall_pack(
        self,
        pack_id: str,
        label: str,
        conversation_ids_json: str,
        payload_json: str,
    ) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.save_recall_pack(pack_id, label, conversation_ids_json, payload_json)

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.get_recall_pack(pack_id)

    async def list_recall_packs(self) -> list[dict[str, str]]:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.list_recall_packs()

    async def delete_recall_pack(self, pack_id: str) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_recall_pack(pack_id)

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
        active_target_json: str = "{}",
    ) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.save_workspace(
            workspace_id=workspace_id,
            name=name,
            mode=mode,
            open_targets_json=open_targets_json,
            layout_json=layout_json,
            active_target_json=active_target_json,
        )

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.get_workspace(workspace_id)

    async def list_workspaces(self) -> list[dict[str, str]]:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.list_workspaces()

    async def delete_workspace(self, workspace_id: str) -> bool:
        from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

        store: RepositoryWriteMixin = cast("RepositoryWriteMixin", self.repository)
        return await store.delete_workspace(workspace_id)


__all__ = ["ArchiveMutationsMixin", "MetadataKeyValidationError"]
