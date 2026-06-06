from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAnnotationEnvelope,
    ArchiveBlackboardNoteEnvelope,
    ArchiveCorrectionEnvelope,
    ArchiveMarkEnvelope,
    ArchiveRecallPackEnvelope,
    ArchiveSavedViewEnvelope,
    ArchiveSuppressionEnvelope,
    ArchiveWorkspaceEnvelope,
    read_archive_annotation_envelope,
    read_archive_blackboard_note_envelope,
    read_archive_correction_envelope,
    read_archive_mark_envelope,
    read_archive_recall_pack_envelope,
    read_archive_saved_view_envelope,
    read_archive_suppression_envelope,
    read_archive_workspace_envelope,
    upsert_annotation,
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


def test_archive_tiers_user_write_minimal_upserts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")

    suppression = upsert_suppression(
        conn,
        "session-1",
        "manual review",
        mode="freeze",
        now_ms=1_700_000_000_000,
    )
    suppression_refreshed = upsert_suppression(
        conn,
        "session-1",
        "follow-up",
        mode="hide",
        now_ms=1_700_000_001_000,
    )

    mark = upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="work-in-progress",
        metadata={"scope": "alpha"},
        now_ms=1_700_000_002_000,
    )
    mark_refresh = upsert_mark(
        conn,
        "session",
        "session-1",
        "star",
        label="revised",
        metadata={"scope": "beta"},
        now_ms=1_700_000_003_000,
    )

    annotation = upsert_annotation(
        conn,
        "session",
        "session-1",
        "needs review",
        now_ms=1_700_000_004_000,
    )
    annotation_refresh = upsert_annotation(
        conn,
        "session",
        "session-1",
        "reviewed",
        annotation_id=annotation.annotation_id,
        now_ms=1_700_000_005_000,
    )

    correction = upsert_correction(
        conn,
        "session",
        "session-1",
        "summary_override",
        payload={"summary": "initial"},
        now_ms=1_700_000_006_000,
    )
    correction_refresh = upsert_correction(
        conn,
        "session",
        "session-1",
        "summary_override",
        payload={"summary": "updated"},
        now_ms=1_700_000_007_000,
    )

    saved_view = upsert_saved_view(
        conn,
        "recent-codex",
        {"origin": "codex-session", "limit": 10},
        now_ms=1_700_000_008_000,
    )
    saved_view_refresh = upsert_saved_view(
        conn,
        "recent-codex",
        {"origin": "codex-session", "limit": 20},
        now_ms=1_700_000_009_000,
    )

    recall_pack = upsert_recall_pack(
        conn,
        "launch-pack",
        {"session_ids": ["session-1"]},
        recall_pack_id="pack-1",
        now_ms=1_700_000_010_000,
    )
    recall_pack_refresh = upsert_recall_pack(
        conn,
        "launch-pack",
        {"session_ids": ["session-1", "session-2"]},
        recall_pack_id="pack-1",
        now_ms=1_700_000_011_000,
    )

    workspace = upsert_workspace(
        conn,
        "polylogue",
        {"root": "/realm/project/polylogue"},
        now_ms=1_700_000_012_000,
    )
    workspace_refresh = upsert_workspace(
        conn,
        "polylogue",
        {"root": "/realm/project/polylogue", "branch": "feature"},
        now_ms=1_700_000_013_000,
    )

    blackboard_note = upsert_blackboard_note(
        conn,
        "keep this with the session",
        target_type="session",
        target_id="session-1",
        now_ms=1_700_000_014_000,
    )
    blackboard_note_refresh = upsert_blackboard_note(
        conn,
        "updated note",
        target_type="session",
        target_id="session-1",
        note_id=blackboard_note.note_id,
        now_ms=1_700_000_015_000,
    )

    refreshed_suppression: ArchiveSuppressionEnvelope = read_archive_suppression_envelope(conn, "session-1")
    refreshed_mark: ArchiveMarkEnvelope = read_archive_mark_envelope(conn, mark.mark_id)
    refreshed_annotation: ArchiveAnnotationEnvelope = read_archive_annotation_envelope(conn, annotation.annotation_id)
    refreshed_correction: ArchiveCorrectionEnvelope = read_archive_correction_envelope(conn, correction.correction_id)
    refreshed_saved_view: ArchiveSavedViewEnvelope = read_archive_saved_view_envelope(conn, "recent-codex")
    refreshed_recall_pack: ArchiveRecallPackEnvelope = read_archive_recall_pack_envelope(
        conn, recall_pack.recall_pack_id
    )
    refreshed_workspace: ArchiveWorkspaceEnvelope = read_archive_workspace_envelope(conn, "polylogue")
    refreshed_note: ArchiveBlackboardNoteEnvelope = read_archive_blackboard_note_envelope(conn, blackboard_note.note_id)

    assert suppression.session_id == "session-1"
    assert suppression.created_at_ms == 1_700_000_000_000
    assert suppression_refreshed.created_at_ms == suppression.created_at_ms
    assert suppression_refreshed.updated_at_ms == 1_700_000_001_000
    assert refreshed_suppression.reason == "follow-up"
    assert refreshed_suppression.mode == "hide"

    assert mark.mark_id == mark_refresh.mark_id
    assert mark.created_at_ms == 1_700_000_002_000
    assert mark_refresh.updated_at_ms == 1_700_000_003_000
    assert refreshed_mark.label == "revised"
    assert refreshed_mark.metadata == {"scope": "beta"}

    assert annotation.annotation_id == annotation_refresh.annotation_id
    assert annotation.created_at_ms == 1_700_000_004_000
    assert annotation_refresh.updated_at_ms == 1_700_000_005_000
    assert refreshed_annotation.body == "reviewed"

    assert correction.correction_id == correction_refresh.correction_id
    assert correction.created_at_ms == 1_700_000_006_000
    assert correction_refresh.updated_at_ms == 1_700_000_007_000
    assert refreshed_correction.payload == {"summary": "updated"}

    assert saved_view.view_id == saved_view_refresh.view_id
    assert saved_view.created_at_ms == 1_700_000_008_000
    assert saved_view_refresh.updated_at_ms == 1_700_000_009_000
    assert refreshed_saved_view.query == {"origin": "codex-session", "limit": 20}

    assert recall_pack.recall_pack_id == recall_pack_refresh.recall_pack_id
    assert recall_pack.created_at_ms == 1_700_000_010_000
    assert recall_pack_refresh.updated_at_ms == 1_700_000_011_000
    assert refreshed_recall_pack.payload == {"session_ids": ["session-1", "session-2"]}

    assert workspace.workspace_id == workspace_refresh.workspace_id
    assert workspace.created_at_ms == 1_700_000_012_000
    assert workspace_refresh.updated_at_ms == 1_700_000_013_000
    assert refreshed_workspace.settings == {"root": "/realm/project/polylogue", "branch": "feature"}

    assert blackboard_note.note_id == blackboard_note_refresh.note_id
    assert blackboard_note.created_at_ms == 1_700_000_014_000
    assert blackboard_note_refresh.updated_at_ms == 1_700_000_015_000
    assert refreshed_note.body == "updated note"
    assert refreshed_note.target_type == "session"
    assert refreshed_note.target_id == "session-1"
