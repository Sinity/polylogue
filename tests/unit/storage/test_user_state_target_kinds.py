"""Storage contracts for the user-state target kinds added in #1113."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.user_state_targets import (
    TARGET_KIND_NAMES,
    identity_key,
    is_supported,
)
from tests.infra.storage_records import ConversationBuilder, db_setup


def _conversation_content_hash(db_path: Path, conversation_id: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def _seed_content_block(db_path: Path, conversation_id: str, message_id: str) -> None:
    """Insert one ``text`` content block at block_index=0 for the message."""

    with sqlite3.connect(db_path) as conn:
        notnull_cols = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(content_blocks)").fetchall()
            if row[3] == 1 and row[4] is None
        }
        values: dict[str, object] = {
            "block_id": f"{message_id}:0",
            "message_id": message_id,
            "conversation_id": conversation_id,
            "block_index": 0,
            "type": "text",
            "text": "Block payload",
        }
        for col, col_type in notnull_cols.items():
            if col in values:
                continue
            values[col] = 0 if "INT" in col_type.upper() or "REAL" in col_type.upper() else ""
        placeholders = ", ".join("?" * len(values))
        conn.execute(
            f"INSERT INTO content_blocks ({', '.join(values)}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        conn.commit()


def _seed_insight_rows(db_path: Path, conversation_id: str) -> tuple[str, str]:
    """Seed minimal session_profiles / session_work_events / work_threads rows."""

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_profiles (
                conversation_id, logical_conversation_id, materializer_version, materialized_at,
                provider_name, search_text
            )
            VALUES (?, ?, 5, '2026-05-17T00:00:00Z', 'claude-code', '')
            """,
            (conversation_id, conversation_id),
        )
        # session_work_events columns vary by version; insert only the
        # PK/FK/required event_id pair to keep this fixture decoupled
        # from the wider work-event schema.
        notnull_cols = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(session_work_events)").fetchall()
            if row[3] == 1 and row[4] is None  # notnull, no default
        }
        values: dict[str, object] = {
            "event_id": "evt-1",
            "conversation_id": conversation_id,
        }
        for col, col_type in notnull_cols.items():
            if col in values:
                continue
            values[col] = 0 if "INT" in col_type.upper() or "REAL" in col_type.upper() else ""
        placeholders = ", ".join("?" * len(values))
        conn.execute(
            f"INSERT INTO session_work_events ({', '.join(values)}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        # Same defensive insert for work_threads.
        notnull_cols = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(work_threads)").fetchall()
            if row[3] == 1 and row[4] is None
        }
        values = {"thread_id": "thr-1", "root_id": conversation_id}
        for col, col_type in notnull_cols.items():
            if col in values:
                continue
            values[col] = 0 if "INT" in col_type.upper() or "REAL" in col_type.upper() else ""
        placeholders = ", ".join("?" * len(values))
        conn.execute(
            f"INSERT INTO work_threads ({', '.join(values)}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        conn.commit()
    return ("evt-1", "thr-1")


def test_target_kinds_registry_admits_documented_kinds() -> None:
    expected = {
        "conversation",
        "message",
        "session",
        "work_event",
        "thread",
        "content_block",
        "attachment",
        "paste_span",
    }
    assert set(TARGET_KIND_NAMES) == expected
    for name in expected:
        assert is_supported(name)
    assert not is_supported("topology_edge")


def test_identity_key_renders_distinct_keys_per_kind() -> None:
    assert identity_key("session", conversation_id="conv", target_id="conv") == "session:conv"
    assert identity_key("work_event", conversation_id="conv", target_id="evt-1") == "work_event:conv:evt-1"
    assert identity_key("thread", conversation_id="conv", target_id="thr-1") == "thread:thr-1"
    assert identity_key("content_block", conversation_id="conv", target_id="msg-1:0") == "content_block:conv:msg-1:0"


@pytest.mark.asyncio
async def test_marks_admit_insight_target_kinds_without_changing_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-target-kinds").provider("claude-code").add_message(
        message_id="msg-1",
        text="Block payload",
    ).save()
    _seed_content_block(db_path, "conv-target-kinds", "msg-1")
    event_id, thread_id = _seed_insight_rows(db_path, "conv-target-kinds")

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        before_hash = _conversation_content_hash(db_path, "conv-target-kinds")

        assert await poly.add_mark("conv-target-kinds", "star", target_type="session")
        assert await poly.add_mark(
            "conv-target-kinds",
            "pin",
            target_type="work_event",
            target_id=event_id,
        )
        assert await poly.add_mark(
            "conv-target-kinds",
            "archive",
            target_type="thread",
            target_id=thread_id,
        )
        assert await poly.add_mark(
            "conv-target-kinds",
            "star",
            target_type="content_block",
            target_id="msg-1:0",
        )
        assert await poly.add_mark(
            "conv-target-kinds",
            "pin",
            target_type="attachment",
            target_id="att-1",
        )
        assert await poly.add_mark(
            "conv-target-kinds",
            "archive",
            target_type="paste_span",
            target_id="paste-1",
        )

        # Idempotency on the new kinds.
        assert not await poly.add_mark(
            "conv-target-kinds",
            "pin",
            target_type="work_event",
            target_id=event_id,
        )

        marks = await poly.list_marks(conversation_id="conv-target-kinds")
        after_hash = _conversation_content_hash(db_path, "conv-target-kinds")

    target_types = {(row["target_type"], row["target_id"]) for row in marks}
    assert target_types == {
        ("session", "conv-target-kinds"),
        ("work_event", event_id),
        ("thread", thread_id),
        ("content_block", "msg-1:0"),
        ("attachment", "att-1"),
        ("paste_span", "paste-1"),
    }
    assert after_hash == before_hash


@pytest.mark.asyncio
async def test_annotations_admit_insight_target_kinds_with_crud(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-ann").provider("claude-code").add_message(
        message_id="msg-1",
        text="Block payload",
    ).save()
    _seed_content_block(db_path, "conv-ann", "msg-1")
    event_id, thread_id = _seed_insight_rows(db_path, "conv-ann")

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.save_annotation("ann-session", "conv-ann", "Session note", target_type="session")
        assert await poly.save_annotation(
            "ann-event",
            "conv-ann",
            "Event note",
            target_type="work_event",
            target_id=event_id,
        )
        assert await poly.save_annotation(
            "ann-thread",
            "conv-ann",
            "Thread note",
            target_type="thread",
            target_id=thread_id,
        )
        assert await poly.save_annotation(
            "ann-block",
            "conv-ann",
            "Block note",
            target_type="content_block",
            target_id="msg-1:0",
        )

        # Updating an existing annotation returns False (not newly created).
        assert (
            await poly.save_annotation(
                "ann-session",
                "conv-ann",
                "Session note (updated)",
                target_type="session",
            )
            is False
        )

        rows = await poly.list_annotations(conversation_id="conv-ann")
        assert {row["target_type"] for row in rows} == {
            "session",
            "work_event",
            "thread",
            "content_block",
        }
        deleted = await poly.delete_annotation("ann-event")
        assert deleted is True


@pytest.mark.asyncio
async def test_insight_target_kinds_reject_missing_units(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-reject").provider("claude-code").add_message(
        message_id="msg-1", text="Hi"
    ).save()
    # Note: deliberately NOT seeding insight rows here so the resolver must
    # refuse to write marks/annotations that point at non-existent insight units.

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        with pytest.raises(ValueError, match="session profile"):
            await poly.add_mark("conv-reject", "star", target_type="session")
        with pytest.raises(ValueError, match="work_event"):
            await poly.add_mark("conv-reject", "pin", target_type="work_event", target_id="bogus")
        with pytest.raises(ValueError, match="thread"):
            await poly.add_mark("conv-reject", "archive", target_type="thread", target_id="bogus")
        with pytest.raises(ValueError, match="content_block"):
            await poly.add_mark(
                "conv-reject",
                "star",
                target_type="content_block",
                target_id="msg-1:99",
            )
        with pytest.raises(ValueError, match="content_block target_id"):
            await poly.add_mark(
                "conv-reject",
                "star",
                target_type="content_block",
                target_id="malformed-no-colon",
            )
        with pytest.raises(ValueError, match="target_type must be one of"):
            await poly.add_mark("conv-reject", "star", target_type="topology_edge")


@pytest.mark.asyncio
async def test_recall_pack_resolves_insight_targets_and_degrades_explicitly(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-pack").provider("claude-code").add_message(message_id="msg-1", text="Hi").save()
    _seed_content_block(db_path, "conv-pack", "msg-1")
    event_id, thread_id = _seed_insight_rows(db_path, "conv-pack")

    items_json = json.dumps(
        {
            "items": [
                {"target_type": "session", "conversation_id": "conv-pack"},
                {
                    "target_type": "work_event",
                    "conversation_id": "conv-pack",
                    "target_id": event_id,
                },
                {
                    "target_type": "thread",
                    "conversation_id": "conv-pack",
                    "target_id": thread_id,
                },
                {
                    "target_type": "content_block",
                    "conversation_id": "conv-pack",
                    "target_id": "msg-1:0",
                },
                {
                    "target_type": "work_event",
                    "conversation_id": "conv-pack",
                    "target_id": "missing-event",
                },
                {
                    "target_type": "session",
                    "conversation_id": "missing-conv",
                },
                {"target_type": "topology_edge", "target_id": "edge-1"},
            ],
            "summary": "kinds",
        }
    )

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.create_recall_pack("pack-kinds", "Kinds pack", items_json)
        saved = await poly.get_recall_pack("pack-kinds")

    assert saved is not None
    payload = json.loads(saved["payload_json"])
    statuses = {(item["target_type"], item["target_id"]): item["status"] for item in payload["items"]}
    assert statuses[("session", "conv-pack")] == "resolved"
    assert statuses[("work_event", event_id)] == "resolved"
    assert statuses[("thread", thread_id)] == "resolved"
    assert statuses[("content_block", "msg-1:0")] == "resolved"
    assert statuses[("work_event", "missing-event")] == "missing"
    assert statuses[("session", "missing-conv")] == "missing"
    assert statuses[("topology_edge", "edge-1")] == "unsupported"

    identity_keys = {
        (item["target_type"], item["target_id"]): item.get("identity_key")
        for item in payload["items"]
        if item["status"] == "resolved"
    }
    assert identity_keys[("session", "conv-pack")] == "session:conv-pack"
    assert identity_keys[("work_event", event_id)] == f"work_event:conv-pack:{event_id}"
    assert identity_keys[("thread", thread_id)] == f"thread:{thread_id}"
    assert identity_keys[("content_block", "msg-1:0")] == "content_block:conv-pack:msg-1:0"


@pytest.mark.asyncio
async def test_workspace_open_targets_round_trip_insight_kinds(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "conv-ws").provider("claude-code").add_message(message_id="msg-1", text="Hi").save()
    event_id, _ = _seed_insight_rows(db_path, "conv-ws")

    open_targets_json = json.dumps(
        [
            {"target_type": "session", "conversation_id": "conv-ws"},
            {
                "target_type": "work_event",
                "conversation_id": "conv-ws",
                "target_id": event_id,
            },
            {
                "target_type": "work_event",
                "conversation_id": "conv-ws",
                "target_id": "missing-event",
            },
        ]
    )

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.save_workspace(
            "workspace-kinds",
            "Kinds workspace",
            "tabs",
            open_targets_json,
            "{}",
            json.dumps({"target_type": "session", "conversation_id": "conv-ws"}),
        )
        saved = await poly.get_workspace("workspace-kinds")

    assert saved is not None
    targets = json.loads(saved["open_targets_json"])
    statuses = {(item["target_type"], item["target_id"]): item["status"] for item in targets}
    assert statuses[("session", "conv-ws")] == "resolved"
    assert statuses[("work_event", event_id)] == "resolved"
    assert statuses[("work_event", "missing-event")] == "missing"
    active = json.loads(saved["active_target_json"])
    assert active["identity_key"] == "session:conv-ws"
