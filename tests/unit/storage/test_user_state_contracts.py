"""Durable user-state storage contracts (#867)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from tests.infra.storage_records import ConversationBuilder, db_setup


def _conversation_content_hash(db_path: Path, conversation_id: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.asyncio
async def test_target_aware_marks_and_annotations_do_not_change_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-user-state").provider("claude-code").add_message(
        message_id="msg-user-state",
        text="Important message",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        before = await poly.get_conversation("conv-user-state")
        assert before is not None
        before_hash = _conversation_content_hash(db_path, "conv-user-state")

        assert await poly.add_mark("conv-user-state", "star") is True
        assert await poly.add_mark("conv-user-state", "star") is False
        assert (
            await poly.add_mark(
                "conv-user-state",
                "pin",
                target_type="message",
                message_id="msg-user-state",
            )
            is True
        )
        assert await poly.save_annotation("ann-conv", "conv-user-state", "Conversation note") is True
        assert (
            await poly.save_annotation(
                "ann-msg",
                "conv-user-state",
                "Message note",
                target_type="message",
                message_id="msg-user-state",
            )
            is True
        )
        assert (
            await poly.save_annotation(
                "ann-msg",
                "conv-user-state",
                "Updated message note",
                target_type="message",
                message_id="msg-user-state",
            )
            is False
        )

        marks = await poly.list_marks(conversation_id="conv-user-state")
        annotations = await poly.list_annotations(conversation_id="conv-user-state")
        after = await poly.get_conversation("conv-user-state")
        after_hash = _conversation_content_hash(db_path, "conv-user-state")

    assert after is not None
    assert after_hash == before_hash
    assert {(row["target_type"], row["target_id"], row["mark_type"]) for row in marks} == {
        ("conversation", "conv-user-state", "star"),
        ("message", "msg-user-state", "pin"),
    }
    assert {(row["target_type"], row["target_id"], row["note_text"]) for row in annotations} == {
        ("conversation", "conv-user-state", "Conversation note"),
        ("message", "msg-user-state", "Updated message note"),
    }


@pytest.mark.asyncio
async def test_message_target_user_state_rejects_unknown_messages(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-user-state").provider("claude-code").add_message(
        message_id="msg-user-state",
        text="Important message",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        with pytest.raises(ValueError, match="not in conversation"):
            await poly.add_mark(
                "conv-user-state",
                "pin",
                target_type="message",
                message_id="missing-message",
            )
        with pytest.raises(ValueError, match="not in conversation"):
            await poly.save_annotation(
                "ann-missing",
                "conv-user-state",
                "Missing",
                target_type="message",
                message_id="missing-message",
            )


@pytest.mark.asyncio
async def test_recall_pack_items_resolve_and_degrade_explicitly(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-user-state").provider("claude-code").add_message(
        message_id="msg-user-state",
        text="Important message",
    ).save()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark("conv-user-state", "pin", target_type="message", message_id="msg-user-state")
        assert await poly.save_annotation(
            "ann-msg", "conv-user-state", "Message note", target_type="message", message_id="msg-user-state"
        )
        created = await poly.create_recall_pack(
            "pack-user-state",
            "User state pack",
            '["conv-user-state","missing-conv"]',
            (
                '{"items":['
                '{"target_type":"message","conversation_id":"conv-user-state","message_id":"msg-user-state"},'
                '{"target_type":"message","conversation_id":"conv-user-state","message_id":"missing-msg"},'
                '{"target_type":"mark","mark_target_type":"message","mark_target_id":"msg-user-state","mark_type":"pin","conversation_id":"conv-user-state"},'
                '{"target_type":"annotation","annotation_id":"ann-msg"},'
                '{"target_type":"annotation","annotation_id":"missing-ann"},'
                '{"target_type":"topology_edge","target_id":"edge-1"}'
                '],"summary":"handoff"}'
            ),
        )
        saved = await poly.get_recall_pack("pack-user-state")

    assert created is True
    assert saved is not None
    import json

    conversation_ids = json.loads(saved["conversation_ids_json"])
    payload = json.loads(saved["payload_json"])
    assert conversation_ids == ["conv-user-state"]
    assert payload["schema_version"] == 1
    assert payload["summary"] == "handoff"
    assert payload["resolved_count"] == 4
    assert payload["degraded_count"] == 4
    item_statuses = {(item["target_type"], item["target_id"]): item["status"] for item in payload["items"]}
    assert item_statuses[("conversation", "conv-user-state")] == "resolved"
    assert item_statuses[("conversation", "missing-conv")] == "missing"
    assert item_statuses[("message", "msg-user-state")] == "resolved"
    assert item_statuses[("message", "missing-msg")] == "missing"
    assert item_statuses[("annotation", "ann-msg")] == "resolved"
    assert item_statuses[("annotation", "missing-ann")] == "missing"
    assert item_statuses[("topology_edge", "edge-1")] == "unsupported"
