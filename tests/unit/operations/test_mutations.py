"""Centralized mutation-contract tests for ``ArchiveMutationsMixin`` (#862).

These tests pin the typed-result entrypoints used by every surface (CLI,
MCP, daemon, API) so the validation, resolution, and idempotency behavior
stays in one place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.api.archive import SessionNotFoundError
from polylogue.surfaces.payloads import (
    BulkTagMutationResult,
    DeleteSessionResult,
    MetadataKeyValidationError,
    MetadataMutationResult,
    validate_metadata_key,
)
from tests.infra.storage_records import SessionBuilder, db_setup


# Archive session ids derive from the builder's provider_session_id
# (``ext-<conv_id>``) and the claude-code origin. Mutation entrypoints
# resolve and echo the archive session id, so tests address the store by it.
def _native(token: str) -> str:
    return f"claude-code-session:ext-{token}"


def _seed(workspace_env: dict[str, Path], session_id: str = "conv-mut") -> Path:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, session_id).provider("claude-code").add_message(
        message_id=f"{session_id}-msg",
        text="hello world",
    ).save()
    return db_path


# ---------------------------------------------------------------------------
# Pure validator
# ---------------------------------------------------------------------------


class TestValidateMetadataKey:
    def test_accepts_well_formed_key(self) -> None:
        assert validate_metadata_key("status") is None
        assert validate_metadata_key("a" * 200) is None

    def test_rejects_empty_or_whitespace(self) -> None:
        assert validate_metadata_key("") is not None
        assert validate_metadata_key("   ") is not None

    def test_rejects_oversize(self) -> None:
        message = validate_metadata_key("k" * 201)
        assert message is not None
        assert "200" in message

    def test_rejects_non_string(self) -> None:
        assert validate_metadata_key(None) is not None
        assert validate_metadata_key(42) is not None


# ---------------------------------------------------------------------------
# Metadata: set + delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSetMetadataValidated:
    async def test_set_then_unchanged_then_overwrite(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            first = await poly.set_metadata(_native("conv-mut"), "status", "ready")
            second = await poly.set_metadata(_native("conv-mut"), "status", "ready")
            third = await poly.set_metadata(_native("conv-mut"), "status", "shipped")

        assert isinstance(first, MetadataMutationResult)
        assert first.outcome == "set"
        assert first.session_id == _native("conv-mut")
        assert first.key == "status"
        assert second.outcome == "unchanged"
        assert second.detail == "value_unchanged"
        assert third.outcome == "set"

    async def test_invalid_key_raises_validation_error(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(MetadataKeyValidationError):
                await poly.set_metadata(_native("conv-mut"), "", "value")
            with pytest.raises(MetadataKeyValidationError):
                await poly.set_metadata(_native("conv-mut"), "k" * 201, "value")

    async def test_missing_session_raises(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(SessionNotFoundError):
                await poly.set_metadata("missing-id", "status", "ready")


@pytest.mark.asyncio
class TestDeleteMetadataValidated:
    async def test_deleted_then_not_found(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            await poly.set_metadata(_native("conv-mut"), "status", "ready")
            deleted = await poly.delete_metadata(_native("conv-mut"), "status")
            second = await poly.delete_metadata(_native("conv-mut"), "status")

        assert deleted.outcome == "deleted"
        assert deleted.session_id == _native("conv-mut")
        assert deleted.key == "status"
        assert second.outcome == "not_found"
        assert second.detail == "key_not_found"

    async def test_invalid_key_raises_validation_error(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(MetadataKeyValidationError):
                await poly.delete_metadata(_native("conv-mut"), "")

    async def test_missing_session_raises(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(SessionNotFoundError):
                await poly.delete_metadata("missing-id", "status")


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeleteSessionSafe:
    async def test_delete_then_not_found(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env, session_id="conv-del")
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            first = await poly.delete_session_safe(_native("conv-del"))
            second = await poly.delete_session_safe(_native("conv-del"))

        assert isinstance(first, DeleteSessionResult)
        assert first.outcome == "deleted"
        assert first.session_id == _native("conv-del")
        assert second.outcome == "not_found"
        assert second.detail == "session_not_found"

    async def test_missing_session_returns_not_found(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            result = await poly.delete_session_safe("never-existed")
        assert result.outcome == "not_found"
        assert result.session_id == "never-existed"

    async def test_bool_wrapper_still_returns_bool(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env, session_id="conv-bool")
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.delete_session(_native("conv-bool")) is True
            assert await poly.delete_session(_native("conv-bool")) is False


# ---------------------------------------------------------------------------
# Bulk tag sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBulkTagSessions:
    async def test_applies_and_counts_affected(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        for cid in ("conv-a", "conv-b"):
            SessionBuilder(db_path, cid).provider("claude-code").add_message(
                message_id=f"{cid}-msg",
                text="x",
            ).save()
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            first = await poly.bulk_tag_sessions([_native("conv-a"), _native("conv-b")], ["important"])
            # Re-applying the same tag should report zero affected.
            second = await poly.bulk_tag_sessions([_native("conv-a"), _native("conv-b")], ["important"])

        assert isinstance(first, BulkTagMutationResult)
        assert first.session_count == 2
        assert first.tag_count == 1
        assert first.affected_count == 2
        assert first.skipped_count == 0
        assert second.affected_count == 0
        assert second.skipped_count == 2

    async def test_empty_session_ids_raises(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(ValueError, match="session_id"):
                await poly.bulk_tag_sessions([], ["t"])

    async def test_empty_tags_raises(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            with pytest.raises(ValueError, match="tag"):
                await poly.bulk_tag_sessions([_native("conv-mut")], [])

    async def test_oversize_inputs_raise(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed(workspace_env)
        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            many_ids = [f"id-{i}" for i in range(101)]
            with pytest.raises(ValueError, match="at most 100"):
                await poly.bulk_tag_sessions(many_ids, ["t"])
            many_tags = [f"t{i}" for i in range(21)]
            with pytest.raises(ValueError, match="at most 20"):
                await poly.bulk_tag_sessions([_native("conv-mut")], many_tags)
