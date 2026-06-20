"""Durable user-state storage contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, list_assertions_for_target

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
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
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
        mark = conn.execute("SELECT target_ref, key, value_json FROM assertions WHERE kind = 'mark'").fetchone()
        annotation = conn.execute(
            "SELECT target_ref, key, body_text FROM assertions WHERE kind = 'annotation' AND key = 'ann-v1'"
        ).fetchone()
        saved_view = conn.execute(
            "SELECT target_ref, key, value_json FROM assertions WHERE kind = 'saved_query' AND target_ref = 'saved_view:view-v1'"
        ).fetchone()
        recall_pack = conn.execute(
            "SELECT target_ref, key, value_json FROM assertions WHERE kind = 'recall_pack' AND target_ref = 'recall_pack:pack-v1'"
        ).fetchone()
        workspace = conn.execute(
            "SELECT target_ref, key, value_json FROM assertions WHERE kind = 'workspace_note' AND target_ref = 'workspace:workspace-v1'"
        ).fetchone()
        assertion_rows = conn.execute(
            """
            SELECT target_ref, kind, key, value_json
            FROM assertions
            WHERE target_ref IN ('recall_pack:pack-v1', 'saved_view:view-v1', 'workspace:workspace-v1')
            ORDER BY target_ref
            """
        ).fetchall()
        correction_row = conn.execute(
            "SELECT target_ref, key, value_json FROM assertions WHERE kind = 'correction'"
        ).fetchone()

    assert mark is not None
    assert mark[0:2] == (f"session:{ARCHIVE_USER_STATE_SESSION_ID}", "star")
    assert (json.loads(mark[2]) if mark[2] is not None else {}) == {}
    assert annotation is not None
    assert annotation[0:2] == (f"session:{ARCHIVE_USER_STATE_SESSION_ID}", "ann-v1")
    assert annotation[2] == "Stored in user.db"
    assert saved_view is not None
    assert saved_view[0:2] == ("saved_view:view-v1", "Archive view")
    assert json.loads(saved_view[2]) == {"query": "storage", "limit": 5}
    assert recall_pack is not None
    assert recall_pack[0:2] == ("recall_pack:pack-v1", "Archive pack")
    assert json.loads(json.loads(recall_pack[2])["session_ids_json"]) == [ARCHIVE_USER_STATE_SESSION_ID]
    assert workspace is not None
    assert workspace[0:2] == ("workspace:workspace-v1", "Archive workspace")
    assert json.loads(workspace[2])["mode"] == "tabs"
    assert [(row[0], row[1], row[2]) for row in assertion_rows] == [
        ("recall_pack:pack-v1", "recall_pack", "Archive pack"),
        ("saved_view:view-v1", "saved_query", "Archive view"),
        ("workspace:workspace-v1", "workspace_note", "Archive workspace"),
    ]
    assert json.loads(assertion_rows[0][3])["session_ids_json"]
    assert json.loads(assertion_rows[1][3]) == {"query": "storage", "limit": 5}
    assert json.loads(assertion_rows[2][3])["mode"] == "tabs"
    assert correction_row is not None
    assert correction_row[0:2] == (f"insight:{ARCHIVE_USER_STATE_SESSION_ID}", "tag_accept")
    assert json.loads(correction_row[2])["payload"] == {"tag": "archive"}


@pytest.mark.asyncio
async def test_blackboard_public_surface_writes_assertion_metadata(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    session_id, message_id = _seed_user_state_session(
        archive_root,
        native_id="conv-blackboard-assertion",
        message_native_id="msg-blackboard-assertion",
    )

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        note = await poly.post_blackboard_note(
            kind="finding",
            title="Assertion-backed blackboard",
            content="The public blackboard surface should persist assertion metadata.",
            scope_session=session_id,
            author_ref="agent:codex-session:unit",
            author_kind="agent",
            evidence_refs=(f"message:{message_id}",),
            staleness={"expires_after_days": 7},
            context_policy={"inject": False, "promotion_required": True},
        )
        notes = await poly.list_blackboard_notes(kind="finding", limit=5)

    assert [listed.note_id for listed in notes] == [note.note_id]
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        mirrored = list_assertions_for_target(conn, f"session:{session_id}", kind=AssertionKind.NOTE)

    assert len(mirrored) == 1
    assertion = mirrored[0]
    assert assertion.key == note.note_id
    assert assertion.author_ref == "agent:codex-session:unit"
    assert assertion.author_kind == "agent"
    assert assertion.evidence_refs == [f"message:{message_id}"]
    assert assertion.staleness == {"expires_after_days": 7}
    assert assertion.context_policy == {"inject": False, "promotion_required": True}


@pytest.mark.asyncio
async def test_tags_and_metadata_are_assertion_backed_user_metadata(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    session_id, _message_id = _seed_user_state_session(
        archive_root,
        native_id="conv-tag-metadata",
        message_native_id="msg-tag-metadata",
    )

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        tag_result = await poly.add_tag(session_id, "Planning")
        metadata_result = await poly.set_metadata(session_id, "owner", {"name": "sinity"})

    assert tag_result.outcome == "added"
    assert metadata_result.outcome == "set"
    with sqlite3.connect(archive_root / "user.db") as conn:
        assertion_rows = conn.execute(
            "SELECT kind, target_ref, key, status, value_json FROM assertions WHERE kind IN ('tag', 'metadata')"
        ).fetchall()

    rows_by_kind = {row[0]: row for row in assertion_rows}
    assert rows_by_kind["tag"][:4] == ("tag", f"session:{session_id}", "planning", "active")
    assert json.loads(rows_by_kind["tag"][4]) == {
        "tag_source": "user",
        "method": "cli",
        "evidence": {"source": "archive_query"},
    }
    assert rows_by_kind["metadata"][:4] == ("metadata", f"session:{session_id}", "owner", "active")
    assert json.loads(rows_by_kind["metadata"][4]) == {"name": "sinity"}


@pytest.mark.asyncio
async def test_user_state_target_resolution_reads_archive_file_set_from_archive_tiers(tmp_path: Path) -> None:
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
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="mark me")],
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

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        assert await poly.add_mark(session_id, "star", target_type="session")
        assert await poly.add_mark(session_id, "pin", target_type="message", message_id=message_id)
    with sqlite3.connect(archive_root / "user.db") as conn:
        rows = conn.execute(
            """
            SELECT target_ref, key
            FROM assertions
            WHERE kind = 'mark'
            ORDER BY target_ref, key
            """
        ).fetchall()
    assert rows == [
        (f"message:{message_id}", "pin"),
        (f"session:{session_id}", "star"),
    ]


@pytest.mark.asyncio
async def test_user_state_targets_reject_contradictory_identifiers(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    _session_id, message_id = _seed_user_state_session(archive_root)

    async with Polylogue(db_path=archive_root / "index.db", archive_root=archive_root) as poly:
        with pytest.raises(ValueError, match="message target_id must match message_id"):
            await poly.add_mark(
                USER_STATE_SESSION_ID,
                "pin",
                target_type="message",
                target_id="other-message",
                message_id=message_id,
            )
        with pytest.raises(ValueError, match="canonical non-negative block_index"):
            await poly.add_mark(
                USER_STATE_SESSION_ID,
                "pin",
                target_type="block",
                target_id=f"{message_id}:00",
            )


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
