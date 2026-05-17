"""Identity-preserving marks and annotations across reimport (#1114).

These tests pin the surface contract: user marks and annotations live outside
the conversation row's lifecycle, keyed by the stable identity key
(``conversation:{cid}`` and ``message:{cid}:{mid}``), and are rebound to the
re-imported conversation/message when ingest puts the rows back.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.api import Polylogue
from polylogue.storage.sqlite.queries.conversations_identity import (
    repoint_user_state_by_identity,
    user_state_identity_key,
)
from tests.infra.storage_records import ConversationBuilder, db_setup


def _mark_row(db_path: Path) -> tuple[str, str | None, str | None]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT identity_key, conversation_id, message_id FROM user_marks WHERE mark_type = 'star'"
        ).fetchone()
    assert row is not None
    return row[0], row[1], row[2]


def _annotation_row(db_path: Path, annotation_id: str) -> tuple[str, str | None, str | None]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT identity_key, conversation_id, message_id FROM user_annotations WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()
    assert row is not None
    return row[0], row[1], row[2]


def test_user_state_identity_key_matches_payload_convention() -> None:
    """``identity_key`` storage matches the ``TargetRefPayload`` surface token."""
    assert (
        user_state_identity_key(
            target_type="conversation",
            conversation_id="chatgpt:1",
            message_id=None,
        )
        == "conversation:chatgpt:1"
    )
    assert (
        user_state_identity_key(
            target_type="message",
            conversation_id="chatgpt:1",
            message_id="chatgpt:1:m1",
        )
        == "message:chatgpt:1:chatgpt:1:m1"
    )


def test_user_state_identity_key_rejects_message_without_message_id() -> None:
    with pytest.raises(ValueError, match="message_id is required"):
        user_state_identity_key(target_type="message", conversation_id="c", message_id=None)


@pytest.mark.asyncio
async def test_add_mark_records_identity_key(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark("conv-id", "star") is True

    identity_key, conv_id, msg_id = _mark_row(db_path)
    assert identity_key == "conversation:conv-id"
    assert conv_id == "conv-id"
    assert msg_id is None


@pytest.mark.asyncio
async def test_marks_survive_conversation_delete_and_repoint_on_reimport(
    workspace_env: dict[str, Path],
) -> None:
    """The core #1114 acceptance: mark survives hard delete and rebinds on reimport."""
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark("conv-id", "star") is True
        assert await poly.save_annotation("ann-1", "conv-id", "important") is True

        # Hard-delete the conversation (FK SET NULL must keep marks alive).
        assert await poly.delete_conversation("conv-id") is True

        marks_after_delete = await poly.list_marks(conversation_id=None, mark_type="star")
        annotations_after_delete = await poly.list_annotations()
        assert {row["target_id"] for row in marks_after_delete} == {"conv-id"}
        assert {row["annotation_id"] for row in annotations_after_delete} == {"ann-1"}

    identity_key, conv_id, _ = _mark_row(db_path)
    assert identity_key == "conversation:conv-id"
    assert conv_id is None, "Hard delete should null the resolved pointer until reimport"

    # Reimport the same logical conversation (deterministic id from provider).
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    identity_key, conv_id, _ = _mark_row(db_path)
    assert identity_key == "conversation:conv-id"
    assert conv_id == "conv-id", "Reimport must repoint the resolved conversation_id"

    annotation_id_key, annotation_conv_id, _ = _annotation_row(db_path, "ann-1")
    assert annotation_id_key == "conversation:conv-id"
    assert annotation_conv_id == "conv-id"


@pytest.mark.asyncio
async def test_message_target_marks_repoint_on_reimport(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert (
            await poly.add_mark(
                "conv-id",
                "pin",
                target_type="message",
                message_id="msg-id",
            )
            is True
        )
        assert await poly.delete_conversation("conv-id") is True

    # Reimport the same conversation with the same message.
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT identity_key, conversation_id, message_id FROM user_marks WHERE mark_type='pin'"
        ).fetchone()
    assert row == ("message:conv-id:msg-id", "conv-id", "msg-id")


@pytest.mark.asyncio
async def test_repoint_leaves_orphan_when_message_disappears_on_reimport(
    workspace_env: dict[str, Path],
) -> None:
    """Message-target marks whose target disappears stay orphaned (NULL pointers)."""
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert (
            await poly.add_mark(
                "conv-id",
                "pin",
                target_type="message",
                message_id="msg-id",
            )
            is True
        )
        assert await poly.delete_conversation("conv-id") is True

    # Reimport WITHOUT the original message — the message-level mark cannot be
    # repointed and should remain orphaned (NULL conversation_id/message_id).
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="other-msg",
        text="different",
    ).save()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT identity_key, conversation_id, message_id FROM user_marks WHERE mark_type='pin'"
        ).fetchone()
    assert row == ("message:conv-id:msg-id", None, None)


@pytest.mark.asyncio
async def test_repoint_is_idempotent(workspace_env: dict[str, Path]) -> None:
    """Repeated repoint calls do not duplicate or re-touch already-bound rows."""
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark("conv-id", "star") is True

    # Direct repoint call without a delete cycle — nothing should change.
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        first = await repoint_user_state_by_identity(conn, "conv-id")
        await conn.commit()
        second = await repoint_user_state_by_identity(conn, "conv-id")
        await conn.commit()
    assert first == (0, 0)
    assert second == (0, 0)


@pytest.mark.asyncio
async def test_user_state_does_not_affect_conversation_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    """User state remains outside the content-hash boundary post-#1114."""
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    with sqlite3.connect(db_path) as conn:
        before_hash = conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = 'conv-id'"
        ).fetchone()[0]

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark("conv-id", "star") is True
        assert await poly.save_annotation("ann-x", "conv-id", "note") is True

    with sqlite3.connect(db_path) as conn:
        after_hash = conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = 'conv-id'"
        ).fetchone()[0]

    assert before_hash == after_hash
