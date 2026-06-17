"""Archive storage contracts for user-state target kinds (#1113, archive).

The user-state target registry (``session``, ``message``, ``session``,
``work_event``, ``thread``, ``block``, ``attachment``, ``paste_span``)
is exercised here against the archive: seeding writes through
``ArchiveStore`` / ``SessionBuilder``, insight rows are seeded into the native
``index.db`` tables, and marks/annotations/recall-packs/workspaces are driven
through the async ``Polylogue`` facade.

Archive target ids are the deterministic public ids:
  * session  -> archive session id (``origin:provider_session_id``)
  * message  -> ``session_id:provider_message_id``
  * thread   -> native thread_id (the root session id for a single-session thread)
  * block -> ``message_id:block_index``

All insight kinds are now writable as marks/annotations directly:
  * ``block`` — the resolver emits ``block`` and assertions persist the same
    public target vocabulary.
  * ``work_event`` — the resolver probes the native ``session_work_events``
    generated ``event_id`` column (``session_id || ':work_event:' ||
    position``).
The recall-pack / workspace resolution path resolves ``block`` correctly and is
asserted.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.user_state_targets import (
    MARK_TYPE_NAMES,
    TARGET_KIND_NAMES,
    identity_key,
    is_mark_type_supported,
    is_supported,
    validate_mark_type,
)
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from tests.infra.storage_records import SessionBuilder, db_setup


def _seed_session_profile(db_path: Path, session_id: str) -> None:
    """Materialize a minimal session_profiles row for the native session."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO session_profiles (session_id, search_text) VALUES (?, ?)",
            (session_id, ""),
        )
        conn.commit()


def _native_thread_id(db_path: Path, session_id: str) -> str:
    """The builder materializes one thread rooted at the session; return its id."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT thread_id FROM threads WHERE thread_id = ?",
            (session_id,),
        ).fetchone()
    assert row is not None, "expected a materialized thread for the seeded session"
    return str(row[0])


# ---------------------------------------------------------------------------
# Registry / identity-key contracts (pure)
# ---------------------------------------------------------------------------


def test_target_kinds_registry_admits_documented_kinds() -> None:
    expected = {
        "session",
        "message",
        "work_event",
        "thread",
        "block",
        "attachment",
        "paste_span",
    }
    assert set(TARGET_KIND_NAMES) == expected
    for name in expected:
        assert is_supported(name)
    assert not is_supported("topology_edge")


def test_mark_type_registry_is_the_public_validation_source() -> None:
    assert MARK_TYPE_NAMES == ("star", "pin", "archive")
    assert is_mark_type_supported("star")
    assert validate_mark_type("pin") == "pin"
    with pytest.raises(ValueError, match="mark_type must be one of"):
        validate_mark_type("bogus")


def test_user_tier_assertions_store_public_target_refs() -> None:
    assert "CREATE TABLE IF NOT EXISTS assertions" in USER_DDL
    assert "target_ref          TEXT NOT NULL" in USER_DDL


def test_identity_key_renders_distinct_keys_per_kind() -> None:
    assert identity_key("session", session_id="conv", target_id="conv") == "session:conv"
    assert identity_key("work_event", session_id="conv", target_id="evt-1") == "work_event:conv:evt-1"
    assert identity_key("thread", session_id="conv", target_id="thr-1") == "thread:thr-1"
    assert identity_key("block", session_id="conv", target_id="msg-1:0") == "block:conv:msg-1:0"


# ---------------------------------------------------------------------------
# Marks: directly-writable insight target kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marks_admit_native_insight_target_kinds_without_changing_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-target-kinds")
        .provider("claude-code")
        .add_message(
            message_id="msg-1",
            text="Block payload",
        )
    )
    builder.save()
    session_id = builder.native_session_id()
    _seed_session_profile(db_path, session_id)
    thread_id = _native_thread_id(db_path, session_id)

    with sqlite3.connect(db_path) as conn:
        before_hash = conn.execute(
            "SELECT content_hash FROM sessions WHERE origin || ':' || native_id = ?",
            (session_id,),
        ).fetchone()[0]

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star", target_type="session")
        assert await poly.add_mark(session_id, "archive", target_type="thread", target_id=thread_id)
        assert await poly.add_mark(session_id, "pin", target_type="attachment", target_id="att-1")
        assert await poly.add_mark(session_id, "archive", target_type="paste_span", target_id="paste-1")

        # Idempotency on the new kinds.
        assert not await poly.add_mark(session_id, "archive", target_type="thread", target_id=thread_id)

        # Unfiltered listing surfaces every kind; the session_id filter is
        # scoped to session/session targets only (thread/attachment/
        # paste_span marks are standalone target rows in the user tier).
        marks = await poly.list_marks()

    with sqlite3.connect(db_path) as conn:
        after_hash = conn.execute(
            "SELECT content_hash FROM sessions WHERE origin || ':' || native_id = ?",
            (session_id,),
        ).fetchone()[0]

    target_types = {(row["target_type"], row["target_id"]) for row in marks}
    assert target_types == {
        ("session", session_id),  # 'session' kind is re-projected to 'session'
        ("thread", thread_id),
        ("attachment", "att-1"),
        ("paste_span", "paste-1"),
    }
    assert after_hash == before_hash


@pytest.mark.asyncio
async def test_marks_reject_unsupported_target_type(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    builder = SessionBuilder(db_path, "conv-reject").provider("claude-code").add_message(message_id="msg-1", text="Hi")
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        with pytest.raises(ValueError, match="target_type must be one of"):
            await poly.add_mark(session_id, "star", target_type="topology_edge")


@pytest.mark.asyncio
async def test_session_mark_does_not_require_materialized_profile(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-no-profile").provider("claude-code").add_message(message_id="msg-1", text="Hi")
    )
    builder.save()
    session_id = builder.native_session_id()
    # No session_profiles row is seeded. In archive, marking
    # ``target_type="session"`` resolves the session row directly and does not
    # require a materialized profile — the profile requirement applies only to
    # insight-kind targets (which route through ``resolve_insight_target``).

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star", target_type="session") is True
        # Idempotent: re-marking the same session is a no-op.
        assert await poly.add_mark(session_id, "star", target_type="session") is False


# ---------------------------------------------------------------------------
# Annotations: directly-writable insight target kinds with CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotations_admit_native_insight_target_kinds_with_crud(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-ann")
        .provider("claude-code")
        .add_message(
            message_id="msg-1",
            text="Block payload",
        )
    )
    builder.save()
    session_id = builder.native_session_id()
    _seed_session_profile(db_path, session_id)
    thread_id = _native_thread_id(db_path, session_id)

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.save_annotation("ann-session", session_id, "Session note", target_type="session")
        assert await poly.save_annotation(
            "ann-thread", session_id, "Thread note", target_type="thread", target_id=thread_id
        )
        assert await poly.save_annotation(
            "ann-attach", session_id, "Attachment note", target_type="attachment", target_id="att-1"
        )

        # Updating an existing annotation returns False (not newly created).
        assert (
            await poly.save_annotation("ann-session", session_id, "Session note (updated)", target_type="session")
            is False
        )

        # The session kind is re-projected to the public 'session' token on read.
        rows = await poly.list_annotations()
        assert {row["target_type"] for row in rows} == {"session", "thread", "attachment"}

        assert await poly.delete_annotation("ann-thread") is True
        rows_after = await poly.list_annotations()
        assert {row["target_type"] for row in rows_after} == {"session", "attachment"}


# ---------------------------------------------------------------------------
# Recall packs: resolution + explicit degradation across kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_pack_resolves_insight_targets_and_degrades_explicitly(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    builder = SessionBuilder(db_path, "conv-pack").provider("claude-code").add_message(message_id="msg-1", text="Hi")
    builder.save()
    session_id = builder.native_session_id()
    message_id = f"{session_id}:msg-1"
    _seed_session_profile(db_path, session_id)
    thread_id = _native_thread_id(db_path, session_id)

    items_json = json.dumps(
        {
            "items": [
                {"target_type": "session", "session_id": session_id},
                {"target_type": "thread", "session_id": session_id, "target_id": thread_id},
                {"target_type": "block", "session_id": session_id, "target_id": f"{message_id}:0"},
                {"target_type": "session", "session_id": "missing-conv"},
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
    assert statuses[("session", session_id)] == "resolved"
    assert statuses[("thread", thread_id)] == "resolved"
    assert statuses[("block", f"{message_id}:0")] == "resolved"
    assert statuses[("session", "missing-conv")] == "missing"
    assert statuses[("topology_edge", "edge-1")] == "unsupported"

    identity_keys = {
        (item["target_type"], item["target_id"]): item.get("identity_key")
        for item in payload["items"]
        if item["status"] == "resolved"
    }
    assert identity_keys[("session", session_id)] == f"session:{session_id}"
    assert identity_keys[("thread", thread_id)] == f"thread:{thread_id}"
    assert identity_keys[("block", f"{message_id}:0")] == f"block:{session_id}:{message_id}:0"


# ---------------------------------------------------------------------------
# Workspaces: open-target round-trip across kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_open_targets_round_trip_insight_kinds(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    builder = SessionBuilder(db_path, "conv-ws").provider("claude-code").add_message(message_id="msg-1", text="Hi")
    builder.save()
    session_id = builder.native_session_id()
    _seed_session_profile(db_path, session_id)
    thread_id = _native_thread_id(db_path, session_id)

    open_targets_json = json.dumps(
        [
            {"target_type": "session", "session_id": session_id},
            {"target_type": "thread", "session_id": session_id, "target_id": thread_id},
            {"target_type": "session", "session_id": "missing-conv"},
        ]
    )

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.save_workspace(
            "workspace-kinds",
            "Kinds workspace",
            "tabs",
            open_targets_json,
            "{}",
            json.dumps({"target_type": "session", "session_id": session_id}),
        )
        saved = await poly.get_workspace("workspace-kinds")

    assert saved is not None
    targets = json.loads(saved["open_targets_json"])
    statuses = {(item["target_type"], item["target_id"]): item["status"] for item in targets}
    assert statuses[("session", session_id)] == "resolved"
    assert statuses[("thread", thread_id)] == "resolved"
    assert statuses[("session", "missing-conv")] == "missing"
    active = json.loads(saved["active_target_json"])
    assert active["identity_key"] == f"session:{session_id}"
