"""Centralized archive mutation operations.

This module owns the *semantics* of every user-driven archive mutation —
tag add/remove (single and bulk), metadata set/delete, and conversation
delete. Every surface (Python API, CLI, MCP, daemon HTTP) routes through
the same :class:`ArchiveMutations` boundary so validation, idempotency,
and not-found behavior are defined exactly once.

The shared result envelopes live in :mod:`polylogue.surfaces.payloads`
(``TagMutationResult``, ``MetadataMutationResult``,
``DeleteConversationResult``, ``BulkTagMutationResult``). Surfaces adapt
those typed results into their own transport (JSON, click output,
HTTP envelope) without re-implementing the bool→outcome mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.errors import PolylogueError
from polylogue.surfaces.payloads import (
    BulkTagMutationResult,
    DeleteConversationResult,
    MetadataMutationResult,
    TagMutationResult,
)


def _import_conversation_not_found_error() -> type[PolylogueError]:
    """Lazy import to avoid a circular dependency with ``polylogue.api``.

    The canonical ``ConversationNotFoundError`` lives on the API surface
    because surfaces depend on it; this module is imported by the API
    module to install mutation behavior, so we can't import the
    error at module load time without creating a cycle.
    """
    from polylogue.api.archive import ConversationNotFoundError

    return ConversationNotFoundError


if TYPE_CHECKING:
    from polylogue.core.json import JSONDocument
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin


_METADATA_KEY_MAX_LENGTH = 200


class InvalidMutationInputError(PolylogueError):
    """Raised when mutation input fails shared validation."""

    http_status_code = 400


class ArchiveMutations:
    """User-mutation operations bound to a :class:`ConversationRepository`.

    All methods take already-trusted conversation IDs (resolved via the
    underlying repository) and return typed result payloads. Validation
    that is the same regardless of surface — empty/oversized metadata
    keys, bulk-size caps, conversation existence — lives here so each
    surface adapter only handles transport-layer concerns.
    """

    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> ConversationRepository:
        return self._repository

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_or_raise(self, conversation_id: str) -> str:
        resolved = await self._repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            raise _import_conversation_not_found_error()(conversation_id)
        return str(resolved)

    def _write_store(self) -> RepositoryWriteMixin:
        store: RepositoryWriteMixin = self._repository
        return store

    @staticmethod
    def _validate_metadata_key(key: str) -> None:
        if not key or not key.strip():
            raise InvalidMutationInputError("metadata key must not be empty")
        if len(key) > _METADATA_KEY_MAX_LENGTH:
            raise InvalidMutationInputError(f"metadata key exceeds {_METADATA_KEY_MAX_LENGTH} characters")

    # ------------------------------------------------------------------
    # Tag mutations
    # ------------------------------------------------------------------

    async def add_tag(self, conversation_id: str, tag: str) -> TagMutationResult:
        """Add ``tag`` to ``conversation_id``.

        Raises :class:`ConversationNotFoundError` if the conversation
        does not exist. Returns ``outcome="added"`` if the tag was newly
        attached, ``outcome="no_op"`` (detail ``already_present``) if
        the tag was already on the conversation.
        """
        resolved = await self._resolve_or_raise(conversation_id)
        added = await self._write_store().add_tag(resolved, tag)
        return TagMutationResult(
            outcome="added" if added else "no_op",
            detail=None if added else "already_present",
        )

    async def remove_tag(self, conversation_id: str, tag: str) -> TagMutationResult:
        """Remove ``tag`` from ``conversation_id``.

        Raises :class:`ConversationNotFoundError` if the conversation
        does not exist. Returns ``outcome="removed"`` if the tag was
        deleted, ``outcome="not_present"`` (detail ``tag_not_present``)
        if the conversation did not carry the tag.
        """
        resolved = await self._resolve_or_raise(conversation_id)
        removed = await self._write_store().remove_tag(resolved, tag)
        return TagMutationResult(
            outcome="removed" if removed else "not_present",
            detail=None if removed else "tag_not_present",
        )

    async def bulk_add_tags(
        self,
        conversation_ids: list[str],
        tags: list[str],
        *,
        max_conversations: int = 100,
        max_tags: int = 20,
    ) -> BulkTagMutationResult:
        """Apply each tag in ``tags`` to each id in ``conversation_ids``.

        ``applied_count`` is the number of (conversation, tag) pairs the
        underlying tag store reports as freshly attached. ``skipped_count``
        is ``len(conversation_ids) - applied_count`` — kept as a simple
        difference for surface backward compatibility; do not interpret
        it as "skipped pairs".
        """
        if not conversation_ids:
            raise InvalidMutationInputError("bulk_add_tags requires at least one conversation_id")
        if not tags:
            raise InvalidMutationInputError("bulk_add_tags requires at least one tag")
        if len(conversation_ids) > max_conversations:
            raise InvalidMutationInputError(f"bulk_add_tags supports at most {max_conversations} conversation_ids")
        if len(tags) > max_tags:
            raise InvalidMutationInputError(f"bulk_add_tags supports at most {max_tags} tags")

        applied = await self._write_store().bulk_add_tags(conversation_ids, tags)
        return BulkTagMutationResult(
            conversation_count=len(conversation_ids),
            tag_count=len(tags),
            applied_count=applied,
            skipped_count=len(conversation_ids) - applied,
        )

    # ------------------------------------------------------------------
    # Metadata mutations
    # ------------------------------------------------------------------

    async def get_metadata(self, conversation_id: str) -> dict[str, str]:
        """Return all metadata key/value pairs on ``conversation_id``.

        Does not raise on missing conversation; returns an empty mapping
        so callers can distinguish "no metadata" from "no conversation"
        with a separate existence check when needed.
        """
        store = self._write_store()
        doc: JSONDocument = await store.get_metadata(conversation_id)
        return {str(k): (v if isinstance(v, str) else str(v)) for k, v in doc.items()}

    async def set_metadata(
        self,
        conversation_id: str,
        key: str,
        value: str,
    ) -> MetadataMutationResult:
        """Set ``key`` on ``conversation_id`` to ``value``.

        Validates the key (non-empty, length cap) and resolves the
        conversation. Returns ``outcome="set"`` when the value changed,
        ``outcome="unchanged"`` (detail ``value_unchanged``) when the
        stored value already matched.
        """
        self._validate_metadata_key(key)
        resolved = await self._resolve_or_raise(conversation_id)
        changed = await self._write_store().update_metadata(resolved, key, value)
        return MetadataMutationResult(
            outcome="set" if changed else "unchanged",
            key=key,
            detail=None if changed else "value_unchanged",
        )

    async def delete_metadata(
        self,
        conversation_id: str,
        key: str,
    ) -> MetadataMutationResult:
        """Delete metadata ``key`` from ``conversation_id``.

        Returns ``outcome="deleted"`` when the key was removed,
        ``outcome="not_found"`` (detail ``key_not_found``) when the
        conversation did not carry that key.
        """
        self._validate_metadata_key(key)
        resolved = await self._resolve_or_raise(conversation_id)
        deleted = await self._write_store().delete_metadata(resolved, key)
        return MetadataMutationResult(
            outcome="deleted" if deleted else "not_found",
            key=key,
            detail=None if deleted else "key_not_found",
        )

    # ------------------------------------------------------------------
    # Conversation deletion
    # ------------------------------------------------------------------

    async def delete_conversation(self, conversation_id: str) -> DeleteConversationResult:
        """Delete ``conversation_id`` and all derived rows.

        Returns ``outcome="deleted"`` when at least one conversation row
        was removed. Returns ``outcome="not_found"`` (detail
        ``conversation_not_found``) when no row matched — this is the
        idempotent semantics: deleting an already-missing conversation
        is not an error. The conversation id is resolved before the
        delete, but the resolve is non-strict so legitimate
        "already deleted" requests still produce ``not_found`` rather
        than raising.
        """
        resolved = await self._repository.resolve_id(conversation_id, strict=False)
        target_id = str(resolved) if resolved is not None else conversation_id
        deleted_count = await self._repository.delete_conversation(target_id)
        if isinstance(deleted_count, bool):
            deleted_any = deleted_count
            removed_rows: int | None = None
        else:
            deleted_any = deleted_count > 0
            removed_rows = int(deleted_count)
        return DeleteConversationResult(
            conversation_id=target_id,
            outcome="deleted" if deleted_any else "not_found",
            detail=None if deleted_any else "conversation_not_found",
            removed_count=removed_rows,
        )


__all__ = [
    "ArchiveMutations",
    "InvalidMutationInputError",
]
