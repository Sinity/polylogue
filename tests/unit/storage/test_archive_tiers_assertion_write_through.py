"""Write-through parity tests: legacy user-overlay upserts mirror an assertion row (#1883)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    AssertionKind,
    assertion_id_for_annotation,
    assertion_id_for_blackboard_note,
    assertion_id_for_correction,
    assertion_id_for_mark,
    assertion_id_for_recall_pack,
    assertion_id_for_saved_view,
    assertion_id_for_suppression,
    assertion_id_for_workspace,
    list_archive_blackboard_note_envelopes,
    list_assertions_for_target,
    read_archive_annotation_envelope,
    read_archive_blackboard_note_envelope,
    read_archive_correction_envelope,
    read_archive_mark_envelope,
    read_archive_recall_pack_envelope,
    read_archive_saved_view_envelope,
    read_archive_suppression_envelope,
    read_archive_workspace_envelope,
    read_assertion_envelope,
    upsert_annotation,
    upsert_assertion,
    upsert_blackboard_note,
    upsert_correction,
    upsert_mark,
    upsert_recall_pack,
    upsert_saved_view,
    upsert_suppression,
    upsert_workspace,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _assertion_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0])


def test_mark_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    mark = upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="wip",
        metadata={"scope": "alpha"},
        now_ms=1_700_000_000_000,
    )

    # Legacy row still authoritative.
    legacy = conn.execute("SELECT mark_id, label FROM marks WHERE mark_id = ?", (mark.mark_id,)).fetchone()
    assert legacy is not None
    assert legacy["label"] == "wip"

    mirrored = list_assertions_for_target(conn, "session:session-1", kind=AssertionKind.MARK)
    assert len(mirrored) == 1
    env = mirrored[0]
    assert env.kind == AssertionKind.MARK
    assert env.target_ref == "session:session-1"
    assert env.key == "star"
    assert env.body_text == "wip"
    assert env.value == {"scope": "alpha"}
    assert env.author_kind == "user"
    assert env.created_at_ms == 1_700_000_000_000


def test_mark_write_through_is_idempotent(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_mark(conn, "session", "session-1", "star", label="v1", now_ms=1_700_000_000_000)
    before_id = list_assertions_for_target(conn, "session:session-1", kind=AssertionKind.MARK)[0].assertion_id
    before_count = _assertion_count(conn)

    upsert_mark(conn, "session", "session-1", "star", label="v2", now_ms=1_700_000_005_000)
    after = list_assertions_for_target(conn, "session:session-1", kind=AssertionKind.MARK)

    assert _assertion_count(conn) == before_count  # no duplicate
    assert len(after) == 1
    assert after[0].assertion_id == before_id  # same row updated
    assert after[0].body_text == "v2"
    assert after[0].created_at_ms == 1_700_000_000_000  # preserved
    assert after[0].updated_at_ms == 1_700_000_005_000


def test_annotation_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    annotation = upsert_annotation(
        conn,
        "message",
        "msg-7",
        "needs follow-up",
        now_ms=1_700_000_010_000,
    )

    assert (
        conn.execute("SELECT 1 FROM annotations WHERE annotation_id = ?", (annotation.annotation_id,)).fetchone()
        is not None
    )

    mirrored = list_assertions_for_target(conn, "message:msg-7", kind=AssertionKind.ANNOTATION)
    assert len(mirrored) == 1
    assert mirrored[0].body_text == "needs follow-up"
    assert mirrored[0].kind == AssertionKind.ANNOTATION
    assert mirrored[0].author_kind == "user"

    # Idempotent re-upsert (same deterministic annotation id → same assertion).
    upsert_annotation(conn, "message", "msg-7", "needs follow-up", now_ms=1_700_000_011_000)
    again = list_assertions_for_target(conn, "message:msg-7", kind=AssertionKind.ANNOTATION)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id


def test_correction_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    correction = upsert_correction(
        conn,
        "insight",
        "session-1",
        "tag_reject",
        {"tag": "rust"},
        now_ms=1_700_000_020_000,
    )

    assert (
        conn.execute("SELECT 1 FROM corrections WHERE correction_id = ?", (correction.correction_id,)).fetchone()
        is not None
    )

    mirrored = list_assertions_for_target(conn, "insight:session-1", kind=AssertionKind.CORRECTION)
    assert len(mirrored) == 1
    env = mirrored[0]
    assert env.key == "tag_reject"
    assert env.value == {"tag": "rust"}
    assert env.author_kind == "user"

    upsert_correction(conn, "insight", "session-1", "tag_reject", {"tag": "python"}, now_ms=1_700_000_021_000)
    again = list_assertions_for_target(conn, "insight:session-1", kind=AssertionKind.CORRECTION)
    assert len(again) == 1
    assert again[0].assertion_id == env.assertion_id
    assert again[0].value == {"tag": "python"}


def test_blackboard_note_write_through_scoped_and_unscoped(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_blackboard_note(
        conn,
        "scoped body",
        target_type="session",
        target_id="session-9",
        now_ms=1_700_000_030_000,
    )
    scoped_mirror = list_assertions_for_target(conn, "session:session-9", kind=AssertionKind.NOTE)
    assert len(scoped_mirror) == 1
    assert scoped_mirror[0].body_text == "scoped body"
    assert scoped_mirror[0].author_kind == "user"

    unscoped = upsert_blackboard_note(conn, "global body", now_ms=1_700_000_031_000)
    # Unscoped note falls back to its own note_id as target_ref.
    unscoped_env = read_assertion_envelope(
        conn,
        next(
            r["assertion_id"]
            for r in conn.execute(
                "SELECT assertion_id FROM assertions WHERE kind = ? AND target_ref = ?",
                (AssertionKind.NOTE, unscoped.note_id),
            )
        ),
    )
    assert unscoped_env is not None
    assert unscoped_env.body_text == "global body"

    # Idempotency on the scoped note.
    upsert_blackboard_note(
        conn,
        "scoped body",
        target_type="session",
        target_id="session-9",
        now_ms=1_700_000_032_000,
    )
    assert len(list_assertions_for_target(conn, "session:session-9", kind=AssertionKind.NOTE)) == 1


def test_blackboard_note_assertion_metadata_is_preserved(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_blackboard_note(
        conn,
        "agent finding",
        target_type="session",
        target_id="session-9",
        author_ref="agent:codex-session:abc",
        author_kind="agent",
        evidence_refs=("message:m1", "block:m1:2"),
        staleness={"expires_after_days": 14},
        context_policy={"inject": False},
        now_ms=1_700_000_033_000,
    )

    mirrored = list_assertions_for_target(conn, "session:session-9", kind=AssertionKind.NOTE)
    assert len(mirrored) == 1
    env = mirrored[0]
    assert env.author_ref == "agent:codex-session:abc"
    assert env.author_kind == "agent"
    assert env.evidence_refs == ["message:m1", "block:m1:2"]
    assert env.staleness == {"expires_after_days": 14}
    assert env.context_policy == {"inject": False}


def test_user_overlay_reads_prefer_assertion_mirrors(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    suppression = upsert_suppression(conn, "session-2", "legacy reason", mode="hide", now_ms=1_700_000_034_000)
    mark = upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="legacy label",
        metadata={"scope": "legacy"},
        now_ms=1_700_000_034_000,
    )
    annotation = upsert_annotation(conn, "message", "msg-1", "legacy body", now_ms=1_700_000_034_000)
    correction = upsert_correction(
        conn,
        "insight",
        "session-1",
        "tag_reject",
        {"tag": "legacy"},
        now_ms=1_700_000_034_000,
    )
    saved_view = upsert_saved_view(conn, "recent", {"limit": 10}, now_ms=1_700_000_034_000)
    recall_pack = upsert_recall_pack(conn, "handoff", {"sessions": ["legacy"]}, now_ms=1_700_000_034_000)
    workspace = upsert_workspace(conn, "main", {"repo": "legacy"}, now_ms=1_700_000_034_000)

    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_suppression(suppression.session_id),
        target_ref=suppression.session_id,
        kind=AssertionKind.SUPPRESSION,
        value={"mode": "freeze"},
        body_text="asserted reason",
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_mark(mark.target_type, mark.target_id, mark.mark_type),
        target_ref="session:session-1",
        kind=AssertionKind.MARK,
        key=mark.mark_type,
        value={"scope": "asserted"},
        body_text="asserted label",
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_annotation(annotation.annotation_id),
        target_ref="message:msg-1",
        kind=AssertionKind.ANNOTATION,
        body_text="asserted annotation",
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_correction(correction.correction_id),
        target_ref="insight:session-1",
        kind=AssertionKind.CORRECTION,
        key=correction.correction_type,
        value={"tag": "asserted"},
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_saved_view(saved_view.view_id),
        target_ref=f"saved_view:{saved_view.view_id}",
        kind=AssertionKind.SAVED_QUERY,
        key=saved_view.name,
        value={"limit": 20},
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_recall_pack(recall_pack.recall_pack_id),
        target_ref=f"recall_pack:{recall_pack.recall_pack_id}",
        kind=AssertionKind.RECALL_PACK,
        key=recall_pack.name,
        value={"sessions": ["asserted"]},
        body_text=recall_pack.name,
        now_ms=1_700_000_035_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_workspace(workspace.workspace_id),
        target_ref=f"workspace:{workspace.workspace_id}",
        kind=AssertionKind.WORKSPACE_NOTE,
        key=workspace.name,
        value={"repo": "asserted"},
        body_text=workspace.name,
        now_ms=1_700_000_035_000,
    )

    suppression_read = read_archive_suppression_envelope(conn, suppression.session_id)
    mark_read = read_archive_mark_envelope(conn, mark.mark_id)
    annotation_read = read_archive_annotation_envelope(conn, annotation.annotation_id)
    correction_read = read_archive_correction_envelope(conn, correction.correction_id)
    saved_view_read = read_archive_saved_view_envelope(conn, saved_view.name)
    recall_pack_read = read_archive_recall_pack_envelope(conn, recall_pack.recall_pack_id)
    workspace_read = read_archive_workspace_envelope(conn, workspace.name)

    assert suppression_read.reason == "asserted reason"
    assert suppression_read.mode == "freeze"
    assert mark_read.mark_id == mark.mark_id
    assert mark_read.label == "asserted label"
    assert mark_read.metadata == {"scope": "asserted"}
    assert annotation_read.body == "asserted annotation"
    assert correction_read.payload == {"tag": "asserted"}
    assert saved_view_read.query == {"limit": 20}
    assert recall_pack_read.payload == {"sessions": ["asserted"]}
    assert workspace_read.settings == {"repo": "asserted"}
    assert workspace_read.updated_at_ms == 1_700_000_035_000


def test_blackboard_note_read_prefers_assertion_mirror(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    note = upsert_blackboard_note(conn, "legacy body", note_id="note-1", now_ms=1_700_000_034_000)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_blackboard_note(note.note_id),
        target_ref=note.note_id,
        kind=AssertionKind.NOTE,
        body_text="[decision] asserted title\n\nassertion body",
        now_ms=1_700_000_035_000,
    )

    envelope = read_archive_blackboard_note_envelope(conn, note.note_id)

    assert envelope.note_id == note.note_id
    assert envelope.target_type is None
    assert envelope.target_id is None
    assert envelope.body == "[decision] asserted title\n\nassertion body"
    assert envelope.created_at_ms == 1_700_000_034_000
    assert envelope.updated_at_ms == 1_700_000_035_000


def test_blackboard_note_list_is_assertion_backed_with_legacy_fallback(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    mirrored = upsert_blackboard_note(
        conn,
        "legacy mirrored body",
        note_id="note-mirrored",
        target_type="session",
        target_id="session-1",
        now_ms=1_700_000_036_000,
    )
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_blackboard_note(mirrored.note_id),
        target_ref="session:session-1",
        kind=AssertionKind.NOTE,
        body_text="[finding] assertion title\n\nassertion body",
        now_ms=1_700_000_037_000,
    )
    conn.execute(
        """
        INSERT INTO blackboard_notes (note_id, target_type, target_id, body, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("note-legacy", None, None, "[question] legacy title\n\nlegacy body", 1_700_000_038_000, 1_700_000_038_000),
    )

    notes = list_archive_blackboard_note_envelopes(conn)

    assert [note.note_id for note in notes] == ["note-legacy", "note-mirrored"]
    assert notes[0].body == "[question] legacy title\n\nlegacy body"
    assert notes[1].target_type == "session"
    assert notes[1].target_id == "session-1"
    assert notes[1].body == "[finding] assertion title\n\nassertion body"
    assert notes[1].updated_at_ms == 1_700_000_037_000


def test_suppression_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_suppression(conn, "session-2", "noise", mode="hide", now_ms=1_700_000_040_000)
    assert conn.execute("SELECT 1 FROM suppressions WHERE session_id = ?", ("session-2",)).fetchone() is not None

    mirrored = list_assertions_for_target(conn, "session-2", kind=AssertionKind.SUPPRESSION)
    assert len(mirrored) == 1
    assert mirrored[0].value == {"mode": "hide"}
    assert mirrored[0].body_text == "noise"

    upsert_suppression(conn, "session-2", "still noise", mode="freeze", now_ms=1_700_000_041_000)
    again = list_assertions_for_target(conn, "session-2", kind=AssertionKind.SUPPRESSION)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id
    assert again[0].value == {"mode": "freeze"}


def test_saved_view_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    saved_view = upsert_saved_view(conn, "recent", {"limit": 10}, now_ms=1_700_000_050_000)
    assert conn.execute("SELECT 1 FROM saved_views WHERE name = ?", ("recent",)).fetchone() is not None

    mirrored = list_assertions_for_target(conn, f"saved_view:{saved_view.view_id}", kind=AssertionKind.SAVED_QUERY)
    assert len(mirrored) == 1
    assert mirrored[0].key == "recent"
    assert mirrored[0].value == {"limit": 10}

    upsert_saved_view(conn, "recent", {"limit": 20}, now_ms=1_700_000_051_000)
    again = list_assertions_for_target(conn, f"saved_view:{saved_view.view_id}", kind=AssertionKind.SAVED_QUERY)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id
    assert again[0].value == {"limit": 20}
