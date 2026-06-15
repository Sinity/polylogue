"""Write-through parity tests: legacy user-overlay upserts mirror an assertion row (#1883)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    AssertionKind,
    list_assertions_for_target,
    read_assertion_envelope,
    upsert_annotation,
    upsert_blackboard_note,
    upsert_correction,
    upsert_mark,
    upsert_saved_view,
    upsert_suppression,
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

    upsert_saved_view(conn, "recent", {"limit": 10}, now_ms=1_700_000_050_000)
    assert conn.execute("SELECT 1 FROM saved_views WHERE name = ?", ("recent",)).fetchone() is not None

    mirrored = list_assertions_for_target(conn, "saved_view:recent", kind=AssertionKind.SAVED_QUERY)
    assert len(mirrored) == 1
    assert mirrored[0].key == "recent"
    assert mirrored[0].value == {"limit": 10}

    upsert_saved_view(conn, "recent", {"limit": 20}, now_ms=1_700_000_051_000)
    again = list_assertions_for_target(conn, "saved_view:recent", kind=AssertionKind.SAVED_QUERY)
    assert len(again) == 1
    assert again[0].assertion_id == mirrored[0].assertion_id
    assert again[0].value == {"limit": 20}
