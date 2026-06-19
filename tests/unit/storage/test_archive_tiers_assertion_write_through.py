"""Assertion-backed user-overlay storage contracts (#1883)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    AssertionKind,
    assertion_id_for_annotation,
    assertion_id_for_mark,
    assertion_id_for_session_tag,
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
    upsert_blackboard_note,
    upsert_correction,
    upsert_mark,
    upsert_recall_pack,
    upsert_saved_view,
    upsert_suppression,
    upsert_workspace,
)
from polylogue.storage.sqlite.archive_tiers.write import upsert_session_tag


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _assertion_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0])


def test_mark_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="wip",
        metadata={"scope": "alpha"},
        now_ms=1_700_000_000_000,
    )

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


def test_user_session_tag_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_session_tag(
        conn,
        session_id="session-1",
        tag="Review",
        tag_source="user",
        method="cli",
        confidence=0.9,
        evidence={"source": "mark"},
    )

    mirrored = list_assertions_for_target(conn, "session:session-1", kind=AssertionKind.TAG)
    assert len(mirrored) == 1
    env = mirrored[0]
    assert env.assertion_id == assertion_id_for_session_tag("session-1", "review", "user")
    assert env.kind == AssertionKind.TAG
    assert env.target_ref == "session:session-1"
    assert env.key == "review"
    assert env.body_text == "review"
    assert env.value == {"evidence": {"source": "mark"}, "method": "cli", "tag_source": "user"}
    assert env.confidence == 0.9
    assert env.author_kind == "user"


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

    upsert_annotation(
        conn,
        "message",
        "msg-7",
        "needs follow-up",
        now_ms=1_700_000_010_000,
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


def test_block_target_overlays_mirror_to_block_assertions(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    mark = upsert_mark(
        conn,
        "block",
        "msg-7:0",
        "pin",
        label="important block",
        metadata={"reason": "decision"},
        now_ms=1_700_000_015_000,
    )
    annotation = upsert_annotation(
        conn,
        "block",
        "msg-7:0",
        "carry this into the work packet",
        now_ms=1_700_000_016_000,
    )

    mark_mirror = list_assertions_for_target(conn, "block:msg-7:0", kind=AssertionKind.MARK)
    annotation_mirror = list_assertions_for_target(conn, "block:msg-7:0", kind=AssertionKind.ANNOTATION)

    assert mark_mirror[0].assertion_id == assertion_id_for_mark("block", "msg-7:0", "pin")
    assert mark_mirror[0].key == "pin"
    assert mark_mirror[0].body_text == "important block"
    assert mark_mirror[0].value == {"reason": "decision"}
    assert annotation_mirror[0].assertion_id == assertion_id_for_annotation(annotation.annotation_id)
    assert annotation_mirror[0].body_text == "carry this into the work packet"

    refreshed_mark = read_archive_mark_envelope(conn, mark.mark_id)
    refreshed_annotation = read_archive_annotation_envelope(conn, annotation.annotation_id)
    assert refreshed_mark.target_type == "block"
    assert refreshed_annotation.target_type == "block"


def test_correction_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_correction(
        conn,
        "insight",
        "session-1",
        "tag_reject",
        {"tag": "rust"},
        now_ms=1_700_000_020_000,
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
    unscoped_target_ref = f"assertion:{unscoped.note_id}"
    unscoped_env = read_assertion_envelope(
        conn,
        next(
            r["assertion_id"]
            for r in conn.execute(
                "SELECT assertion_id FROM assertions WHERE kind = ? AND target_ref = ?",
                (AssertionKind.NOTE, unscoped_target_ref),
            )
        ),
    )
    assert unscoped_env is not None
    assert unscoped_env.target_ref == unscoped_target_ref
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


def test_user_overlay_reads_project_assertion_envelopes(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    suppression = upsert_suppression(conn, "session-2", "asserted reason", mode="freeze", now_ms=1_700_000_035_000)
    mark = upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="asserted label",
        metadata={"scope": "asserted"},
        now_ms=1_700_000_035_000,
    )
    annotation = upsert_annotation(conn, "message", "msg-1", "asserted annotation", now_ms=1_700_000_035_000)
    correction = upsert_correction(
        conn,
        "insight",
        "session-1",
        "tag_reject",
        {"tag": "asserted"},
        now_ms=1_700_000_035_000,
    )
    saved_view = upsert_saved_view(conn, "recent", {"limit": 20}, now_ms=1_700_000_035_000)
    recall_pack = upsert_recall_pack(conn, "handoff", {"sessions": ["asserted"]}, now_ms=1_700_000_035_000)
    workspace = upsert_workspace(conn, "main", {"repo": "asserted"}, now_ms=1_700_000_035_000)

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


def test_blackboard_note_read_projects_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    note = upsert_blackboard_note(
        conn,
        "[decision] asserted title\n\nassertion body",
        note_id="note-1",
        now_ms=1_700_000_035_000,
    )

    envelope = read_archive_blackboard_note_envelope(conn, note.note_id)

    assert envelope.note_id == note.note_id
    assert envelope.target_type is None
    assert envelope.target_id is None
    assert envelope.body == "[decision] asserted title\n\nassertion body"
    assert envelope.created_at_ms == 1_700_000_035_000
    assert envelope.updated_at_ms == 1_700_000_035_000


def test_blackboard_note_list_projects_assertions(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_blackboard_note(
        conn,
        "[finding] assertion title\n\nassertion body",
        note_id="note-mirrored",
        target_type="session",
        target_id="session-1",
        now_ms=1_700_000_037_000,
    )
    upsert_blackboard_note(
        conn, "[question] global title\n\nglobal body", note_id="note-global", now_ms=1_700_000_038_000
    )

    notes = list_archive_blackboard_note_envelopes(conn)

    assert [note.note_id for note in notes] == ["note-global", "note-mirrored"]
    assert notes[0].body == "[question] global title\n\nglobal body"
    assert notes[1].target_type == "session"
    assert notes[1].target_id == "session-1"
    assert notes[1].body == "[finding] assertion title\n\nassertion body"
    assert notes[1].updated_at_ms == 1_700_000_037_000


def test_suppression_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    upsert_suppression(conn, "session-2", "noise", mode="hide", now_ms=1_700_000_040_000)

    mirrored = list_assertions_for_target(conn, "session:session-2", kind=AssertionKind.SUPPRESSION)
    assert len(mirrored) == 1
    assert mirrored[0].value == {"mode": "hide"}
    assert mirrored[0].body_text == "noise"

    upsert_suppression(conn, "session-2", "still noise", mode="freeze", now_ms=1_700_000_041_000)
    again = list_assertions_for_target(conn, "session:session-2", kind=AssertionKind.SUPPRESSION)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id
    assert again[0].value == {"mode": "freeze"}


def test_saved_view_write_through_mirrors_assertion(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    saved_view = upsert_saved_view(conn, "recent", {"limit": 10}, now_ms=1_700_000_050_000)

    mirrored = list_assertions_for_target(conn, f"saved_view:{saved_view.view_id}", kind=AssertionKind.SAVED_QUERY)
    assert len(mirrored) == 1
    assert mirrored[0].key == "recent"
    assert mirrored[0].value == {"limit": 10}

    upsert_saved_view(conn, "recent", {"limit": 20}, now_ms=1_700_000_051_000)
    again = list_assertions_for_target(conn, f"saved_view:{saved_view.view_id}", kind=AssertionKind.SAVED_QUERY)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id
    assert again[0].value == {"limit": 20}
