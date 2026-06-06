"""Durable user-state storage contracts (#867)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.types import ContentBlockType, Provider

USER_STATE_SESSION_ID = "claude-code-session:conv-user-state"
ARCHIVE_USER_STATE_SESSION_ID = "claude-code-session:conv-v1-user-state"


def _session_content_hash(index_db: Path, session_id: str) -> str:
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT hex(content_hash) FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def _seed_user_state_session(
    archive_root: Path,
    *,
    native_id: str = "conv-user-state",
    message_native_id: str = "msg-user-state",
    text: str = "Important message",
) -> tuple[str, str]:
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id=message_native_id,
                        role=Role.USER,
                        text=text,
                        content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
                    )
                ],
            )
        )
        envelope = archive.read_session(session_id)
    return session_id, envelope.messages[0].message_id


@pytest.mark.asyncio
async def test_target_aware_marks_and_annotations_do_not_change_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    _session_id, message_id = _seed_user_state_session(archive_root)

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        before = await poly.get_session(USER_STATE_SESSION_ID)
        assert before is not None
        before_hash = _session_content_hash(archive_root / "index.db", USER_STATE_SESSION_ID)

        assert await poly.add_mark(USER_STATE_SESSION_ID, "star") is True
        assert await poly.add_mark(USER_STATE_SESSION_ID, "star") is False
        assert (
            await poly.add_mark(
                USER_STATE_SESSION_ID,
                "pin",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        assert await poly.save_annotation("ann-conv", USER_STATE_SESSION_ID, "Session note") is True
        assert (
            await poly.save_annotation(
                "ann-msg",
                USER_STATE_SESSION_ID,
                "Message note",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        assert (
            await poly.save_annotation(
                "ann-msg",
                USER_STATE_SESSION_ID,
                "Updated message note",
                target_type="message",
                message_id=message_id,
            )
            is False
        )

        marks = await poly.list_marks()
        annotations = await poly.list_annotations()
        after = await poly.get_session(USER_STATE_SESSION_ID)
        after_hash = _session_content_hash(archive_root / "index.db", USER_STATE_SESSION_ID)

    assert after is not None
    assert after_hash == before_hash
    assert {(row["target_type"], row["target_id"], row["mark_type"]) for row in marks} == {
        ("session", USER_STATE_SESSION_ID, "star"),
        ("message", message_id, "pin"),
    }
    assert {(row["target_type"], row["target_id"], row["note_text"]) for row in annotations} == {
        ("session", USER_STATE_SESSION_ID, "Session note"),
        ("message", message_id, "Updated message note"),
    }


@pytest.mark.asyncio
async def test_user_state_mutations_write_archive_user_tier(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    _seed_user_state_session(archive_root, native_id="conv-v1-user-state", message_native_id="msg-v1")

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        assert await poly.add_mark(ARCHIVE_USER_STATE_SESSION_ID, "star") is True
        assert await poly.save_annotation("ann-v1", ARCHIVE_USER_STATE_SESSION_ID, "Stored in user.db") is True
        assert await poly.save_view("view-v1", "Archive view", '{"query":"storage","limit":5}') is True
        assert await poly.create_recall_pack(
            "pack-v1",
            "Archive pack",
            f'{{"items":[{{"target_type":"session","session_id":"{ARCHIVE_USER_STATE_SESSION_ID}"}}]}}',
        )
        assert await poly.save_workspace(
            "workspace-v1",
            "Archive workspace",
            "tabs",
            f'[{{"target_type":"session","session_id":"{ARCHIVE_USER_STATE_SESSION_ID}"}}]',
            '{"density":"compact"}',
        )
        correction = await poly.record_correction(ARCHIVE_USER_STATE_SESSION_ID, "tag_accept", {"tag": "archive"})
        assert correction.session_id == ARCHIVE_USER_STATE_SESSION_ID

    user_db = archive_root / "user.db"
    assert user_db.exists()
    with sqlite3.connect(user_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == archive_tier_spec(ArchiveTier.USER).version
        mark = conn.execute(
            "SELECT target_type, target_id, mark_type, json_extract(metadata_json, '$.public_target_type') FROM marks"
        ).fetchone()
        annotation = conn.execute(
            "SELECT target_type, target_id, body FROM annotations WHERE annotation_id = 'ann-v1'"
        ).fetchone()
        saved_view = conn.execute(
            "SELECT view_id, name, query_json FROM saved_views WHERE view_id = 'view-v1'"
        ).fetchone()
        recall_pack = conn.execute(
            "SELECT recall_pack_id, name, payload_json FROM recall_packs WHERE recall_pack_id = 'pack-v1'"
        ).fetchone()
        workspace = conn.execute(
            "SELECT workspace_id, name, settings_json FROM workspaces WHERE workspace_id = 'workspace-v1'"
        ).fetchone()
        correction_row = conn.execute(
            "SELECT target_type, target_id, correction_type, payload_json FROM corrections"
        ).fetchone()

    assert mark == ("session", ARCHIVE_USER_STATE_SESSION_ID, "star", None)
    assert annotation is not None
    assert annotation[0:2] == ("session", ARCHIVE_USER_STATE_SESSION_ID)
    assert annotation[2] == "Stored in user.db"
    assert saved_view is not None
    assert saved_view[0:2] == ("view-v1", "Archive view")
    assert json.loads(saved_view[2]) == {"query": "storage", "limit": 5}
    assert recall_pack is not None
    assert recall_pack[0:2] == ("pack-v1", "Archive pack")
    assert json.loads(json.loads(recall_pack[2])["session_ids_json"]) == [ARCHIVE_USER_STATE_SESSION_ID]
    assert workspace is not None
    assert workspace[0:2] == ("workspace-v1", "Archive workspace")
    assert json.loads(workspace[2])["mode"] == "tabs"
    assert correction_row is not None
    assert correction_row[0:3] == ("session", ARCHIVE_USER_STATE_SESSION_ID, "tag_accept")
    assert json.loads(correction_row[3])["payload"] == {"tag": "archive"}

    db_anchor = archive_root / "polylogue.db"
    assert not db_anchor.exists()


@pytest.mark.asyncio
async def test_user_state_target_resolution_reads_archive_file_set_without_polylogue_db(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="user-state-v1-only",
                title="User state archive only",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="mark me",
                        content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="mark me")],
                    )
                ],
            )
        )
        envelope = archive.read_session(session_id)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute(
            "INSERT INTO session_profiles (session_id, search_text) VALUES (?, '')",
            (session_id,),
        )
        conn.commit()
    message_id = envelope.messages[0].message_id

    async with Polylogue(db_path=archive_root / "polylogue.db", archive_root=archive_root) as poly:
        assert await poly.add_mark(session_id, "star", target_type="session")
        assert await poly.add_mark(session_id, "pin", target_type="message", message_id=message_id)

    assert not (archive_root / "polylogue.db").exists()
    with sqlite3.connect(archive_root / "user.db") as conn:
        rows = conn.execute(
            """
            SELECT target_type, target_id, mark_type
            FROM marks
            ORDER BY target_type, mark_type
            """
        ).fetchall()
    assert rows == [
        ("message", message_id, "pin"),
        ("session", session_id, "star"),
    ]


@pytest.mark.asyncio
async def test_message_target_user_state_rejects_unknown_messages(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    _seed_user_state_session(archive_root)

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        with pytest.raises(ValueError, match="not in session"):
            await poly.add_mark(
                USER_STATE_SESSION_ID,
                "pin",
                target_type="message",
                message_id="missing-message",
            )
        with pytest.raises(ValueError, match="not in session"):
            await poly.save_annotation(
                "ann-missing",
                USER_STATE_SESSION_ID,
                "Missing",
                target_type="message",
                message_id="missing-message",
            )


@pytest.mark.asyncio
async def test_recall_pack_items_resolve_and_degrade_explicitly(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    _session_id, message_id = _seed_user_state_session(archive_root)

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        assert await poly.add_mark(USER_STATE_SESSION_ID, "pin", target_type="message", message_id=message_id)
        assert await poly.save_annotation(
            "ann-msg", USER_STATE_SESSION_ID, "Message note", target_type="message", message_id=message_id
        )
        created = await poly.create_recall_pack(
            "pack-user-state",
            "User state pack",
            (
                '{"items":['
                f'{{"target_type":"session","session_id":"{USER_STATE_SESSION_ID}"}},'
                '{"target_type":"session","session_id":"missing-conv"},'
                f'{{"target_type":"message","session_id":"{USER_STATE_SESSION_ID}","message_id":"{message_id}"}},'
                f'{{"target_type":"message","session_id":"{USER_STATE_SESSION_ID}","message_id":"missing-msg"}},'
                f'{{"target_type":"mark","mark_target_type":"message","mark_target_id":"{message_id}","mark_type":"pin","session_id":"{USER_STATE_SESSION_ID}"}},'
                '{"target_type":"annotation","annotation_id":"ann-msg"},'
                '{"target_type":"annotation","annotation_id":"missing-ann"},'
                '{"target_type":"topology_edge","target_id":"edge-1"}'
                '],"summary":"handoff"}'
            ),
        )
        saved = await poly.get_recall_pack("pack-user-state")

    assert created is True
    assert saved is not None
    session_ids = json.loads(saved["session_ids_json"])
    payload = json.loads(saved["payload_json"])
    assert session_ids == [USER_STATE_SESSION_ID]
    assert payload["schema_version"] == 1
    assert payload["summary"] == "handoff"
    assert payload["resolved_count"] == 4
    assert payload["degraded_count"] == 4
    item_statuses = {(item["target_type"], item["target_id"]): item["status"] for item in payload["items"]}
    assert item_statuses[("session", USER_STATE_SESSION_ID)] == "resolved"
    assert item_statuses[("session", "missing-conv")] == "missing"
    assert item_statuses[("message", message_id)] == "resolved"
    assert item_statuses[("message", "missing-msg")] == "missing"
    assert item_statuses[("annotation", "ann-msg")] == "resolved"
    assert item_statuses[("annotation", "missing-ann")] == "missing"
    assert item_statuses[("topology_edge", "edge-1")] == "unsupported"


@pytest.mark.asyncio
async def test_reader_workspaces_preserve_resolved_and_degraded_targets(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    _session_id, message_id = _seed_user_state_session(archive_root)

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        before_hash = _session_content_hash(archive_root / "index.db", USER_STATE_SESSION_ID)
        created = await poly.save_workspace(
            "workspace-user-state",
            "Investigation",
            "compare",
            (
                "["
                f'{{"target_type":"session","session_id":"{USER_STATE_SESSION_ID}"}},'
                f'{{"target_type":"message","session_id":"{USER_STATE_SESSION_ID}","message_id":"{message_id}"}},'
                f'{{"target_type":"message","session_id":"{USER_STATE_SESSION_ID}","message_id":"missing-msg"}},'
                '{"target_type":"topology_edge","target_id":"edge-1"}'
                "]"
            ),
            '{"panes":[{"width":0.5},{"width":0.5}]}',
            f'{{"target_type":"message","session_id":"{USER_STATE_SESSION_ID}","message_id":"{message_id}"}}',
        )
        saved = await poly.get_workspace("workspace-user-state")
        listed = await poly.list_workspaces()
        after_hash = _session_content_hash(archive_root / "index.db", USER_STATE_SESSION_ID)

    assert created is True
    assert saved is not None
    assert listed[0]["workspace_id"] == "workspace-user-state"
    assert after_hash == before_hash
    assert json.loads(saved["layout_json"]) == {"panes": [{"width": 0.5}, {"width": 0.5}]}
    targets = json.loads(saved["open_targets_json"])
    statuses = {(item["target_type"], item["target_id"]): item["status"] for item in targets}
    assert statuses[("session", USER_STATE_SESSION_ID)] == "resolved"
    assert statuses[("message", message_id)] == "resolved"
    assert statuses[("message", "missing-msg")] == "missing"
    assert statuses[("topology_edge", "edge-1")] == "unsupported"
    active = json.loads(saved["active_target_json"])
    assert active["identity_key"] == f"message:{USER_STATE_SESSION_ID}:{message_id}"
