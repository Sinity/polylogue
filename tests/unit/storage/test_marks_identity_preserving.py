"""Native marks/annotations identity-preservation laws (#1114, archive).

These tests pin the surface contract: user marks and annotations live in the
archive ``user.db`` tier as assertions, keyed by stable public target ids, and
survive a hard delete + re-import of the underlying session. Because the native
session id is deterministic (``origin:provider_session_id``), re-ingesting the
same logical session rebinds to the identical target id with no repoint pass.

Seeding writes the archive through ``SessionBuilder`` (which routes
to ``ArchiveStore``); reads go through the async ``Polylogue`` facade. The
assertions are inspected directly in ``user.db``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.user_state_targets import identity_key
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, db_setup


def _user_db_path(workspace_env: dict[str, Path]) -> Path:
    return workspace_env["archive_root"] / "user.db"


def _mark_row(user_db: Path, mark_type: str = "star") -> tuple[str, str, str, dict[str, object]]:
    with sqlite3.connect(user_db) as conn:
        row = conn.execute(
            "SELECT target_ref, key, value_json FROM assertions WHERE kind = 'mark' AND key = ?",
            (mark_type,),
        ).fetchone()
    assert row is not None
    target_type, target_id = str(row[0]).split(":", 1)
    metadata = json.loads(row[2]) if row[2] is not None else {}
    assert isinstance(metadata, dict)
    return target_type, target_id, row[1], metadata


def _annotation_row(user_db: Path, annotation_id: str) -> tuple[str, str, str]:
    with sqlite3.connect(user_db) as conn:
        row = conn.execute(
            "SELECT target_ref, body_text FROM assertions WHERE kind = 'annotation' AND key = ?",
            (annotation_id,),
        ).fetchone()
    assert row is not None
    target_type, target_id = str(row[0]).split(":", 1)
    return target_type, target_id, row[1]


# ---------------------------------------------------------------------------
# Pure identity-key contracts (storage-token convention)
# ---------------------------------------------------------------------------


def test_identity_key_matches_payload_convention() -> None:
    """The canonical ``identity_key`` token matches the surface ``TargetRefPayload``.

    Marks/annotations store deterministic public target ids (``origin:native_id``);
    the recall-pack/workspace identity token is derived from the same canonical
    builder so storage- and surface-level tokens are wire-compatible.
    """
    assert (
        identity_key(
            "session",
            session_id="chatgpt:1",
            target_id="chatgpt:1",
        )
        == "session:chatgpt:1"
    )
    assert (
        identity_key(
            "message",
            session_id="chatgpt:1",
            target_id="chatgpt:1:m1",
        )
        == "message:chatgpt:1:chatgpt:1:m1"
    )


# ---------------------------------------------------------------------------
# Native marks: storage shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_mark_records_native_session_target(workspace_env: dict[str, Path]) -> None:
    """A session mark is stored against the native session target id."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star") is True

    target_type, target_id, mark_type, metadata = _mark_row(_user_db_path(workspace_env))
    # The user tier stores session marks under the 'session' target
    # kind, addressed by the deterministic archive session id.
    assert target_type == "session"
    assert target_id == session_id
    assert mark_type == "star"
    assert metadata == {}


@pytest.mark.asyncio
async def test_list_marks_projects_session_vocabulary(workspace_env: dict[str, Path]) -> None:
    """``list_marks`` re-projects the stored 'session' kind to the public 'session'."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star") is True
        rows = await poly.list_marks(session_id=None, mark_type="star")

    assert {(row["target_type"], row["target_id"]) for row in rows} == {("session", session_id)}


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque_marker", [":n:", ":p:"])
async def test_message_user_state_projects_owner_for_opaque_session_native_ids(
    workspace_env: dict[str, Path],
    opaque_marker: str,
) -> None:
    """Message user state keeps exact ownership after index rows disappear."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, f"opaque{opaque_marker}session")
        .provider("claude-code")
        .add_message(
            message_id="message-native",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()
    message_id = f"{session_id}:n:message-native"

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert (
            await poly.add_mark(
                session_id,
                "pin",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        assert (
            await poly.save_annotation(
                "opaque-session-message-note",
                session_id,
                "important",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        marks = await poly.list_marks(mark_type="pin")
        annotations = await poly.list_annotations()

        with sqlite3.connect(_user_db_path(workspace_env)) as conn:
            durable_scopes = conn.execute(
                """
                SELECT kind, scope_ref
                FROM assertions
                WHERE target_ref = ?
                ORDER BY kind
                """,
                (f"message:{message_id}",),
            ).fetchall()
        assert durable_scopes == [
            ("annotation", f"session:{session_id}"),
            ("mark", f"session:{session_id}"),
        ]

        assert await poly.delete_session(session_id) is True
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None
            assert conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (message_id,)).fetchone() is None
        filtered_marks = await poly.list_marks(session_id=session_id, mark_type="pin")
        filtered_annotations = await poly.list_annotations(session_id=session_id)

    assert marks[0]["target_id"] == message_id
    assert marks[0]["session_id"] == session_id
    assert annotations[0]["target_id"] == message_id
    assert annotations[0]["session_id"] == session_id
    assert filtered_marks == [marks[0]]
    assert filtered_annotations == [annotations[0]]


# ---------------------------------------------------------------------------
# The core #1114 acceptance: marks survive hard delete and rebind on reimport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marks_survive_session_delete_and_rebind_on_reimport(
    workspace_env: dict[str, Path],
) -> None:
    """Mark + annotation survive hard delete and rebind on deterministic reimport."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star") is True
        assert await poly.save_annotation("ann-1", session_id, "important") is True

        # Hard-delete the session; user state must remain.
        assert await poly.delete_session(session_id) is True

        marks_after_delete = await poly.list_marks(session_id=None, mark_type="star")
        annotations_after_delete = await poly.list_annotations()
        assert {row["target_id"] for row in marks_after_delete} == {session_id}
        assert {row["annotation_id"] for row in annotations_after_delete} == {"ann-1"}

    # The stored assertions persist through delete (no orphaning, no repoint needed).
    target_type, target_id, _, metadata = _mark_row(_user_db_path(workspace_env))
    assert target_type == "session"
    assert target_id == session_id
    assert metadata == {}

    # Reimport the same logical session. The native id is deterministic
    # (origin:provider_session_id), so the mark stays bound to it.
    SessionBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    target_type, target_id, _, metadata = _mark_row(_user_db_path(workspace_env))
    assert target_type == "session"
    assert target_id == session_id
    assert metadata == {}

    annotation_target_type, annotation_target_id, body = _annotation_row(_user_db_path(workspace_env), "ann-1")
    assert annotation_target_type == "session"
    assert annotation_target_id == session_id
    assert body == "important"


@pytest.mark.asyncio
async def test_message_target_marks_survive_reimport(workspace_env: dict[str, Path]) -> None:
    """Message-target marks stay bound to the deterministic native message id."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()
    message_id = f"{session_id}:n:msg-id"

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert (
            await poly.add_mark(
                session_id,
                "pin",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        assert await poly.delete_session(session_id) is True

    # Reimport the same session with the same message.
    SessionBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()

    target_type, target_id, _, _ = _mark_row(_user_db_path(workspace_env), "pin")
    assert target_type == "message"
    assert target_id == message_id


@pytest.mark.asyncio
async def test_legacy_message_owner_reads_batch_index_lookup(workspace_env: dict[str, Path]) -> None:
    """Legacy message marks resolve through one indexed-message batch."""
    db_path = db_setup(workspace_env)
    builder = SessionBuilder(db_path, "batch-owner")
    builder.provider("claude-code")
    for index in range(1, 4):
        builder.add_message(message_id=f"msg-{index}", text=f"message {index}")
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        for index in range(1, 4):
            assert (
                await poly.add_mark(
                    session_id,
                    "pin",
                    target_type="message",
                    message_id=f"{session_id}:n:msg-{index}",
                )
                is True
            )

    with sqlite3.connect(_user_db_path(workspace_env)) as conn:
        conn.execute("UPDATE assertions SET scope_ref = NULL WHERE kind = 'mark'")
        conn.commit()

    with ArchiveStore.open_existing(workspace_env["archive_root"]) as archive:
        statements: list[str] = []
        archive._conn.set_trace_callback(statements.append)
        try:
            rows = archive.list_marks(mark_type="pin")
        finally:
            archive._conn.set_trace_callback(None)

    message_lookup_statements = [
        statement for statement in statements if "FROM messages WHERE message_id IN" in statement
    ]
    assert len(message_lookup_statements) == 1
    assert {row["session_id"] for row in rows} == {session_id}


@pytest.mark.asyncio
async def test_message_target_mark_survives_when_message_disappears(
    workspace_env: dict[str, Path],
) -> None:
    """A message-target mark persists even if the message is absent on reimport."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()
    message_id = f"{session_id}:n:msg-id"

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert (
            await poly.add_mark(
                session_id,
                "pin",
                target_type="message",
                message_id=message_id,
            )
            is True
        )
        assert await poly.delete_session(session_id) is True

    # Reimport WITHOUT the original message — the mark is keyed by the stable
    # public message id and is not deleted just because the message is gone.
    SessionBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="other-msg",
        text="different",
    ).save()

    target_type, target_id, _, _ = _mark_row(_user_db_path(workspace_env), "pin")
    assert target_type == "message"
    assert target_id == message_id


@pytest.mark.asyncio
async def test_reimport_is_idempotent_for_user_state(workspace_env: dict[str, Path]) -> None:
    """Re-ingesting unchanged content does not duplicate or re-touch marks."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star") is True

    user_db = _user_db_path(workspace_env)
    before = _mark_row(user_db)
    SessionBuilder(db_path, "conv-id").provider("claude-code").add_message(
        message_id="msg-id",
        text="hello",
    ).save()
    after = _mark_row(user_db)
    assert after == before


@pytest.mark.asyncio
async def test_user_state_does_not_affect_session_content_hash(
    workspace_env: dict[str, Path],
) -> None:
    """User state remains outside the content-hash boundary post-#1114."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "conv-id")
        .provider("claude-code")
        .add_message(
            message_id="msg-id",
            text="hello",
        )
    )
    builder.save()
    session_id = builder.native_session_id()

    def _session_hash() -> object:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT content_hash FROM sessions WHERE origin || ':' || native_id = ?",
                (session_id,),
            ).fetchone()
        assert row is not None
        return row[0]

    before_hash = _session_hash()

    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.add_mark(session_id, "star") is True
        assert await poly.save_annotation("ann-x", session_id, "note") is True

    after_hash = _session_hash()

    assert before_hash == after_hash
