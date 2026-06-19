from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_overlay import find_archive_user_overlay_orphans
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from polylogue.storage.sqlite.archive_tiers.write import (
    upsert_session_phase,
    upsert_session_work_event,
    write_parsed_session_to_archive,
)


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
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="mark me")],
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

    for assertion_id, target_ref, kind in [
        ("mark-session-ok", f"session:{session_id}", AssertionKind.MARK),
        ("mark-message-ok", f"message:{message_id}", AssertionKind.MARK),
        ("mark-block-ok", f"block:{block_id}", AssertionKind.MARK),
        ("mark-attachment-ok", f"attachment:{attachment_id}", AssertionKind.MARK),
        ("mark-work-event-ok", f"work_event:{work_event.event_id}", AssertionKind.MARK),
        ("mark-phase-ok", f"phase:{phase.phase_id}", AssertionKind.MARK),
        ("mark-thread-ok", f"thread:{session_id}", AssertionKind.MARK),
        ("mark-missing", f"message:{session_id}:missing", AssertionKind.MARK),
        ("annotation-missing", f"paste_span:{message_id}:0:4", AssertionKind.ANNOTATION),
        ("annotation-attachment-missing", f"attachment:{session_id}:attachment:missing", AssertionKind.ANNOTATION),
        ("annotation-phase-missing", f"phase:{session_id}:phase:7", AssertionKind.ANNOTATION),
        ("annotation-thread-missing", f"thread:{session_id}:missing-thread", AssertionKind.ANNOTATION),
        ("annotation-work-event-missing", f"work_event:{session_id}:work_event:7", AssertionKind.ANNOTATION),
        ("note-global-ok", "assertion:note-global-ok", AssertionKind.NOTE),
        ("note-session-ok", f"session:{session_id}", AssertionKind.NOTE),
        ("note-work-event-ok", f"work_event:{work_event.event_id}", AssertionKind.NOTE),
        ("note-work-event-missing", f"work_event:{session_id}:work_event:8", AssertionKind.NOTE),
    ]:
        upsert_assertion(
            user_conn,
            assertion_id=assertion_id,
            target_ref=target_ref,
            kind=kind,
            body_text=assertion_id,
            now_ms=1,
        )

    orphans = find_archive_user_overlay_orphans(user_conn, archive_conn)

    assert [(item.table, item.row_id, item.target_type, item.target_id) for item in orphans] == [
        ("assertions", "annotation-attachment-missing", "attachment", f"{session_id}:attachment:missing"),
        ("assertions", "annotation-missing", "paste_span", f"{message_id}:0:4"),
        ("assertions", "annotation-phase-missing", "phase", f"{session_id}:phase:7"),
        ("assertions", "annotation-thread-missing", "thread", f"{session_id}:missing-thread"),
        ("assertions", "annotation-work-event-missing", "work_event", f"{session_id}:work_event:7"),
        ("assertions", "mark-missing", "message", f"{session_id}:missing"),
        ("assertions", "note-work-event-missing", "work_event", f"{session_id}:work_event:8"),
    ]
