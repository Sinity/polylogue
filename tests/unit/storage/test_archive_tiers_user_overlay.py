from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_overlay import find_archive_user_overlay_orphans
from polylogue.storage.sqlite.archive_tiers.write import (
    upsert_session_phase,
    upsert_session_work_event,
    write_parsed_session_to_archive,
)
from polylogue.types import BlockType, Provider


def _connect(path: Path, tier: ArchiveTier) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, tier)
    return conn


def test_archive_tiers_user_overlay_orphan_check_resolves_archive_targets(tmp_path: Path) -> None:
    archive_conn = _connect(tmp_path / "index.db", ArchiveTier.INDEX)
    user_conn = _connect(tmp_path / "user.db", ArchiveTier.USER)
    session_id = write_parsed_session_to_archive(
        archive_conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="codex-overlay-1",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    content_blocks=[ParsedContentBlock(type=BlockType.TEXT, text="mark me")],
                )
            ],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="att-1",
                    message_provider_id="m1",
                    name="note.txt",
                    upload_origin="paste",
                )
            ],
        ),
    )
    message_id = f"{session_id}:m1"
    block_id = f"{message_id}:0"
    # Attachments are content-addressed; the stored attachment_id is the blob
    # hash, not a generated token. Read it back so the "ok" mark resolves.
    attachment_id = str(archive_conn.execute("SELECT attachment_id FROM attachments").fetchone()["attachment_id"])
    work_event = upsert_session_work_event(
        archive_conn,
        session_id=session_id,
        position=0,
        work_event_type="implementation",
        summary="Overlay target",
    )
    phase = upsert_session_phase(
        archive_conn,
        session_id=session_id,
        position=0,
        phase_type="build",
    )

    user_conn.executemany(
        """
        INSERT INTO marks (
            mark_id, target_type, target_id, mark_type, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("mark-session-ok", "session", session_id, "pin", 1, 1),
            ("mark-message-ok", "message", message_id, "pin", 1, 1),
            ("mark-block-ok", "block", block_id, "pin", 1, 1),
            ("mark-attachment-ok", "attachment", attachment_id, "pin", 1, 1),
            ("mark-work-event-ok", "work_event", work_event.event_id, "pin", 1, 1),
            ("mark-phase-ok", "phase", phase.phase_id, "pin", 1, 1),
            ("mark-thread-ok", "thread", session_id, "pin", 1, 1),
            ("mark-missing", "message", f"{session_id}:missing", "pin", 1, 1),
        ],
    )
    user_conn.executemany(
        """
        INSERT INTO annotations (
            annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("annotation-missing", "paste_span", f"{message_id}:0:4", "lost paste", 1, 1),
            ("annotation-attachment-missing", "attachment", f"{session_id}:attachment:missing", "lost", 1, 1),
            ("annotation-phase-missing", "phase", f"{session_id}:phase:7", "lost phase", 1, 1),
            ("annotation-thread-missing", "thread", f"{session_id}:missing-thread", "lost thread", 1, 1),
            (
                "annotation-work-event-missing",
                "work_event",
                f"{session_id}:work_event:7",
                "lost event",
                1,
                1,
            ),
        ],
    )
    user_conn.executemany(
        """
        INSERT INTO blackboard_notes (
            note_id, target_type, target_id, body, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("note-global-ok", None, None, "global note", 1, 1),
            ("note-session-ok", "session", session_id, "session note", 1, 1),
            ("note-work-event-ok", "work_event", work_event.event_id, "event note", 1, 1),
            ("note-work-event-missing", "work_event", f"{session_id}:work_event:8", "lost event", 1, 1),
        ],
    )

    orphans = find_archive_user_overlay_orphans(user_conn, archive_conn)

    assert [(item.table, item.row_id, item.target_type, item.target_id) for item in orphans] == [
        ("marks", "mark-missing", "message", f"{session_id}:missing"),
        ("annotations", "annotation-attachment-missing", "attachment", f"{session_id}:attachment:missing"),
        ("annotations", "annotation-missing", "paste_span", f"{message_id}:0:4"),
        ("annotations", "annotation-phase-missing", "phase", f"{session_id}:phase:7"),
        ("annotations", "annotation-thread-missing", "thread", f"{session_id}:missing-thread"),
        ("annotations", "annotation-work-event-missing", "work_event", f"{session_id}:work_event:7"),
        ("blackboard_notes", "note-work-event-missing", "work_event", f"{session_id}:work_event:8"),
    ]
