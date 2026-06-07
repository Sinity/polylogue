"""Archive lifecycle laws (archive).

The legacy ``SessionRepository`` write surface (``save_session`` with
record/domain overloads, session-event re-save, hash-skip counts) is gone. The
archive write path is ``ArchiveStore.write_parsed`` (driven here through
``SessionBuilder``); the user-facing mutation surface is the async
``Polylogue`` facade (``add_tag`` / ``remove_tag`` / ``set_metadata`` /
``update_metadata`` / ``delete_metadata`` / ``delete_session``).

These tests pin the lifecycle invariants of that archive surface:
  * tag add/remove is idempotent and reflected in ``list_tags``
  * metadata set/update/get/delete behave as a key/value store
  * tags are M2M-only and do NOT leak into the metadata JSON (#1240)
  * mutating one session leaves its neighbors untouched
  * deleting a session removes it and its tags; re-ingest is idempotent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from tests.infra.storage_records import SessionBuilder, db_setup


def _seed_one(workspace_env: dict[str, Path], conv_id: str, provider: str = "claude-code") -> str:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, conv_id)
        .provider(provider)
        .title(f"Lifecycle {conv_id}")
        .add_message(role="user", text="Inspect lifecycle")
        .add_message(role="assistant", text="Acknowledged")
    )
    builder.save()
    return builder.native_session_id()


# ---------------------------------------------------------------------------
# Tag lifecycle
# ---------------------------------------------------------------------------


class TestTagLifecycle:
    @pytest.mark.asyncio
    async def test_add_tag_is_visible_and_idempotent(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "tag-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            first = await poly.add_tag(session_id, "review")
            assert first.outcome == "added"
            second = await poly.add_tag(session_id, "review")
            assert second.outcome == "no_op"
            assert await poly.list_tags() == {"review": 1}

    @pytest.mark.asyncio
    async def test_remove_tag_is_visible_and_idempotent(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "tag-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            await poly.add_tag(session_id, "review")
            removed = await poly.remove_tag(session_id, "review")
            assert removed.outcome == "removed"
            again = await poly.remove_tag(session_id, "review")
            assert again.outcome != "removed"
            assert await poly.list_tags() == {}

    @pytest.mark.asyncio
    async def test_tags_do_not_leak_into_metadata(self, workspace_env: dict[str, Path]) -> None:
        """#1240: tags are M2M-only; add_tag must not write into the metadata JSON."""
        session_id = _seed_one(workspace_env, "tag-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            await poly.update_metadata(session_id, "summary", "after")
            await poly.add_tag(session_id, "review")
            metadata = await poly.get_metadata(session_id)
            assert "tags" not in metadata
            assert metadata == {"summary": "after"}
            assert await poly.list_tags() == {"review": 1}


# ---------------------------------------------------------------------------
# Metadata lifecycle
# ---------------------------------------------------------------------------


class TestMetadataLifecycle:
    @pytest.mark.asyncio
    async def test_set_update_get_metadata(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "meta-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.update_metadata(session_id, "summary", "after") is True
            set_result = await poly.set_metadata(session_id, "audit", "checked")
            assert set_result.outcome == "set"
            metadata = await poly.get_metadata(session_id)
            assert metadata["summary"] == "after"
            assert metadata["audit"] == "checked"

    @pytest.mark.asyncio
    async def test_delete_metadata_missing_key_is_explicit(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "meta-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            result = await poly.delete_metadata(session_id, "missing")
            assert result.outcome == "not_found"

    @pytest.mark.asyncio
    async def test_delete_metadata_removes_present_key(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "meta-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            await poly.update_metadata(session_id, "summary", "after")
            result = await poly.delete_metadata(session_id, "summary")
            assert result.outcome != "not_found"
            assert await poly.get_metadata(session_id) == {}


# ---------------------------------------------------------------------------
# Neighbor isolation
# ---------------------------------------------------------------------------


class TestNeighborIsolation:
    @pytest.mark.asyncio
    async def test_state_changes_do_not_disturb_neighbors(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        target = (
            SessionBuilder(db_path, "target")
            .provider("claude-code")
            .title("Target")
            .add_message(role="user", text="Mutate me")
        )
        target.save()
        neighbor = (
            SessionBuilder(db_path, "neighbor")
            .provider("chatgpt")
            .title("Neighbor")
            .add_message(role="assistant", text="Keep me")
        )
        neighbor.save()
        target_id = target.native_session_id()
        neighbor_id = neighbor.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            await poly.add_tag(target_id, "review")
            await poly.update_metadata(target_id, "summary", "reviewed")

            assert await poly.get_metadata(target_id) == {"summary": "reviewed"}
            assert await poly.get_metadata(neighbor_id) == {}
            assert await poly.list_tags() == {"review": 1}

            # Deleting the target leaves the neighbor and drops the target's tag.
            assert await poly.delete_session(target_id) is True
            remaining = {str(c.id) for c in await poly.list_sessions(limit=100)}
            assert remaining == {neighbor_id}
            assert await poly.list_tags() == {}


# ---------------------------------------------------------------------------
# Delete + re-ingest
# ---------------------------------------------------------------------------


class TestDeleteAndReingest:
    @pytest.mark.asyncio
    async def test_delete_removes_session_and_count_goes_to_zero(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_one(workspace_env, "del-conv")
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.delete_session(session_id) is True
            assert await poly.list_sessions(limit=100) == []
            assert await poly.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_reingest_is_idempotent(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = (
            SessionBuilder(db_path, "resave-conv")
            .provider("chatgpt")
            .title("Resave")
            .add_message(role="user", text="Question")
            .add_message(role="assistant", text="Answer")
        )
        builder.save()
        builder.save()  # identical re-ingest
        session_id = builder.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            convos = await poly.list_sessions(limit=100)
            assert [str(c.id) for c in convos] == [session_id]
            session = await poly.get_session(session_id)
            assert session is not None
            assert len(session.messages) == 2


# ---------------------------------------------------------------------------
# Content-changing re-ingest
# ---------------------------------------------------------------------------


class TestContentChangeReingest:
    @pytest.mark.asyncio
    async def test_changed_content_updates_in_place(self, workspace_env: dict[str, Path]) -> None:
        """Re-ingesting the same logical session with changed content updates it."""
        db_path = db_setup(workspace_env)
        first = (
            SessionBuilder(db_path, "evolve")
            .provider("claude-code")
            .add_message(role="user", text="Run a command")
            .add_message(role="assistant", text="I will run it")
        )
        first.save()
        session_id = first.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            before = await poly.get_session(session_id)
            assert before is not None
            assert before.messages.to_list()[1].text == "I will run it"

        # Same logical id (deterministic provider_session_id), changed text.
        SessionBuilder(db_path, "evolve").provider("claude-code").add_message(
            role="user", text="Run a command"
        ).add_message(role="assistant", text="Running it differently").save()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            convos = await poly.list_sessions(limit=100)
            assert [str(c.id) for c in convos] == [session_id]
            after = await poly.get_session(session_id)
            assert after is not None
            assert after.messages.to_list()[1].text == "Running it differently"
