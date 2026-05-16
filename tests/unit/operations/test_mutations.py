"""Unit tests for the centralized mutation operations boundary (#862).

The :class:`ArchiveMutations` class owns the bool→outcome mapping, the
metadata-key validation, and the conversation-not-found semantics that
were previously duplicated across CLI, MCP, API, and daemon surfaces.
These tests pin the contract so a regression on the operation boundary
is caught before it leaks into the adapters.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.api.archive import ConversationNotFoundError
from polylogue.operations.mutations import (
    ArchiveMutations,
    InvalidMutationInputError,
)
from polylogue.surfaces.payloads import (
    BulkTagMutationResult,
    DeleteConversationResult,
    MetadataMutationResult,
    TagMutationResult,
)


def _repo(resolved: str | None = "conv-1") -> MagicMock:
    repo: MagicMock = MagicMock()
    repo.resolve_id = AsyncMock(return_value=resolved)
    repo.add_tag = AsyncMock(return_value=True)
    repo.remove_tag = AsyncMock(return_value=True)
    repo.bulk_add_tags = AsyncMock(return_value=0)
    repo.get_metadata = AsyncMock(return_value={})
    repo.update_metadata = AsyncMock(return_value=True)
    repo.delete_metadata = AsyncMock(return_value=True)
    repo.delete_conversation = AsyncMock(return_value=1)
    return repo


def _mut(repo: Any) -> ArchiveMutations:
    return ArchiveMutations(repo)


# ---------------------------------------------------------------------------
# Tag mutations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_tag_newly_attached_returns_added() -> None:
    repo = _repo()
    repo.add_tag.return_value = True
    result = await _mut(repo).add_tag("conv-1", "review")
    assert result == TagMutationResult(outcome="added")
    assert bool(result) is True
    repo.add_tag.assert_awaited_once_with("conv-1", "review")


@pytest.mark.asyncio
async def test_add_tag_already_present_returns_no_op_with_detail() -> None:
    repo = _repo()
    repo.add_tag.return_value = False
    result = await _mut(repo).add_tag("conv-1", "review")
    assert result == TagMutationResult(outcome="no_op", detail="already_present")
    assert bool(result) is False


@pytest.mark.asyncio
async def test_add_tag_raises_when_conversation_missing() -> None:
    repo = _repo(resolved=None)
    with pytest.raises(ConversationNotFoundError):
        await _mut(repo).add_tag("conv-missing", "review")
    repo.add_tag.assert_not_called()


@pytest.mark.asyncio
async def test_remove_tag_present_returns_removed() -> None:
    repo = _repo()
    repo.remove_tag.return_value = True
    result = await _mut(repo).remove_tag("conv-1", "review")
    assert result == TagMutationResult(outcome="removed")


@pytest.mark.asyncio
async def test_remove_tag_absent_returns_not_present_with_detail() -> None:
    repo = _repo()
    repo.remove_tag.return_value = False
    result = await _mut(repo).remove_tag("conv-1", "review")
    assert result == TagMutationResult(outcome="not_present", detail="tag_not_present")
    assert bool(result) is False


@pytest.mark.asyncio
async def test_remove_tag_raises_when_conversation_missing() -> None:
    repo = _repo(resolved=None)
    with pytest.raises(ConversationNotFoundError):
        await _mut(repo).remove_tag("conv-missing", "review")


# ---------------------------------------------------------------------------
# Bulk tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_add_tags_returns_counts() -> None:
    repo = _repo()
    repo.bulk_add_tags.return_value = 5
    result = await _mut(repo).bulk_add_tags(["a", "b", "c"], ["x", "y"])
    assert result == BulkTagMutationResult(conversation_count=3, tag_count=2, applied_count=5, skipped_count=-2)
    repo.bulk_add_tags.assert_awaited_once_with(["a", "b", "c"], ["x", "y"])


@pytest.mark.asyncio
async def test_bulk_add_tags_rejects_empty_conversation_ids() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError, match="at least one conversation_id"):
        await _mut(repo).bulk_add_tags([], ["x"])
    repo.bulk_add_tags.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_add_tags_rejects_empty_tags() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError, match="at least one tag"):
        await _mut(repo).bulk_add_tags(["a"], [])


@pytest.mark.asyncio
async def test_bulk_add_tags_enforces_caps() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError, match="at most 100"):
        await _mut(repo).bulk_add_tags(["a"] * 101, ["x"])
    with pytest.raises(InvalidMutationInputError, match="at most 20"):
        await _mut(repo).bulk_add_tags(["a"], ["x"] * 21)


# ---------------------------------------------------------------------------
# Metadata mutations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_metadata_value_changed_returns_set() -> None:
    repo = _repo()
    repo.update_metadata.return_value = True
    result = await _mut(repo).set_metadata("conv-1", "author", "alice")
    assert result == MetadataMutationResult(outcome="set", key="author")
    assert bool(result) is True


@pytest.mark.asyncio
async def test_set_metadata_value_unchanged_returns_unchanged() -> None:
    repo = _repo()
    repo.update_metadata.return_value = False
    result = await _mut(repo).set_metadata("conv-1", "author", "alice")
    assert result == MetadataMutationResult(outcome="unchanged", key="author", detail="value_unchanged")
    assert bool(result) is False


@pytest.mark.asyncio
async def test_set_metadata_rejects_empty_key() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError, match="must not be empty"):
        await _mut(repo).set_metadata("conv-1", "", "v")
    with pytest.raises(InvalidMutationInputError, match="must not be empty"):
        await _mut(repo).set_metadata("conv-1", "   ", "v")
    repo.update_metadata.assert_not_called()


@pytest.mark.asyncio
async def test_set_metadata_rejects_overlong_key() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError, match="200 characters"):
        await _mut(repo).set_metadata("conv-1", "k" * 201, "v")


@pytest.mark.asyncio
async def test_set_metadata_raises_when_conversation_missing() -> None:
    repo = _repo(resolved=None)
    with pytest.raises(ConversationNotFoundError):
        await _mut(repo).set_metadata("conv-missing", "author", "alice")


@pytest.mark.asyncio
async def test_delete_metadata_present_returns_deleted() -> None:
    repo = _repo()
    repo.delete_metadata.return_value = True
    result = await _mut(repo).delete_metadata("conv-1", "author")
    assert result == MetadataMutationResult(outcome="deleted", key="author")
    assert bool(result) is True


@pytest.mark.asyncio
async def test_delete_metadata_absent_returns_not_found_with_detail() -> None:
    repo = _repo()
    repo.delete_metadata.return_value = False
    result = await _mut(repo).delete_metadata("conv-1", "author")
    assert result == MetadataMutationResult(outcome="not_found", key="author", detail="key_not_found")
    assert bool(result) is False


@pytest.mark.asyncio
async def test_delete_metadata_validates_key() -> None:
    repo = _repo()
    with pytest.raises(InvalidMutationInputError):
        await _mut(repo).delete_metadata("conv-1", "")


@pytest.mark.asyncio
async def test_delete_metadata_raises_when_conversation_missing() -> None:
    repo = _repo(resolved=None)
    with pytest.raises(ConversationNotFoundError):
        await _mut(repo).delete_metadata("conv-missing", "author")


# ---------------------------------------------------------------------------
# Conversation delete (idempotent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_conversation_existing_returns_deleted() -> None:
    repo = _repo()
    repo.delete_conversation.return_value = 1
    result = await _mut(repo).delete_conversation("conv-1")
    assert result.outcome == "deleted"
    assert result.conversation_id == "conv-1"
    assert result.detail is None
    assert result.removed_count == 1
    assert bool(result) is True


@pytest.mark.asyncio
async def test_delete_conversation_missing_returns_not_found_idempotently() -> None:
    """Deleting an already-missing conversation must not raise; it is a
    successful idempotent no-op with detail ``conversation_not_found``."""
    repo = _repo(resolved=None)
    repo.delete_conversation.return_value = 0
    result = await _mut(repo).delete_conversation("conv-missing")
    assert result.outcome == "not_found"
    assert result.detail == "conversation_not_found"
    assert result.conversation_id == "conv-missing"
    assert bool(result) is False


@pytest.mark.asyncio
async def test_delete_conversation_accepts_bool_backend_return() -> None:
    """Backends that report a bool instead of a row count are normalized."""
    repo = _repo()
    repo.delete_conversation.return_value = True
    result = await _mut(repo).delete_conversation("conv-1")
    assert result.outcome == "deleted"
    assert result.removed_count is None


# ---------------------------------------------------------------------------
# Cross-surface parity — every surface should produce the same envelope
# shape for the same operation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_routes_through_mutations_boundary() -> None:
    """The ``Polylogue`` facade methods must be thin delegates.

    A regression that bypasses the centralized mutation boundary
    (re-implementing existence checks in the facade, returning raw
    bools, etc.) shows up here as a type mismatch.
    """
    from polylogue.api import Polylogue

    poly = Polylogue.__new__(Polylogue)
    # Inject a stub repository so ``poly.mutations`` is exercisable.
    repo = _repo()
    repo.add_tag.return_value = True
    poly._services = MagicMock()
    poly._services.get_repository = MagicMock(return_value=repo)

    result = await poly.add_tag("conv-1", "review")
    assert isinstance(result, TagMutationResult)
    assert result.outcome == "added"

    repo.delete_conversation.return_value = 1
    delete_result = await poly.delete_conversation("conv-1")
    assert isinstance(delete_result, DeleteConversationResult)
    assert delete_result.outcome == "deleted"

    repo.update_metadata.return_value = True
    meta_result = await poly.update_metadata("conv-1", "author", "alice")
    assert isinstance(meta_result, MetadataMutationResult)
    assert meta_result.outcome == "set"

    repo.delete_metadata.return_value = True
    delete_meta = await poly.delete_metadata("conv-1", "author")
    assert isinstance(delete_meta, MetadataMutationResult)
    assert delete_meta.outcome == "deleted"

    repo.bulk_add_tags.return_value = 4
    bulk_result = await poly.bulk_add_tags(["a", "b"], ["x", "y"])
    assert isinstance(bulk_result, BulkTagMutationResult)
    assert bulk_result.applied_count == 4
